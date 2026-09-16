"""PhysLoc clips as LikePhys evaluation groups.

LikePhys scores a model by how often it prefers a valid video over an invalid
one of the same scene. A PhysLoc *pair* is exactly that group: one valid clip
and every invalid clip rendered from its scene, sharing a bit-identical prefix
up to the violation. Each invalid clip's variation type is
`<family>_<severity bin>`, so `compute_misrank_normalized` reports one mis-rank
per family and bin.

Reading is done by PhysLoc's own dataloader, `physloc/loader.py` from the
PhysLoc repository (`--physloc_repo`, `$PHYSLOC_REPO`, or a `physloc` checkout
next to this repository). It imports nothing but numpy and the standard
library, so it is imported by path and the generator need not be installed.
`iter_groups` yields the loader's own `Pair` and `Clip` objects, so anything
else the loader offers (masks, severity, timelines, ...) is one attribute away.

    python evaluator.py --model ltx-0.9.5 --data physloc \
        --physloc_root /path/to/physloc_release [--physloc_family permanence]

Run this file directly to list the groups a release yields, without a GPU:

    python -m utils.physloc_dataset --physloc_root /path/to/release
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

#: The PhysLoc checkout whose `physloc/loader.py` is used when neither
#: --physloc_repo nor $PHYSLOC_REPO is set: one beside this repository.
DEFAULT_REPO = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "physloc")

#: Path filters `iter_groups` forwards to the loader. `split` is handled
#: separately because it selects clips rather than filtering fields.
FILTERS = ("family", "scenario", "level", "condition", "severity_bin")


def _import_by_path(path):
    spec = importlib.util.spec_from_file_location("physloc_loader", path)
    module = importlib.util.module_from_spec(spec)
    # Registered so that anything the loader pickles or reflects on resolves.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_loader(physloc_repo=None):
    """Import `physloc/loader.py` from the PhysLoc checkout, by path."""
    repo = physloc_repo or os.environ.get("PHYSLOC_REPO") or DEFAULT_REPO
    path = os.path.join(repo, "physloc", "loader.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no PhysLoc loader at %s; pass --physloc_repo or set PHYSLOC_REPO" % path)
    return _import_by_path(path)


def download(hub_repo, cache="data/physloc"):
    """Snapshot a PhysLoc release from the Hub into `<cache>/<owner>__<name>`."""
    from huggingface_hub import snapshot_download

    local = os.path.join(cache, hub_repo.replace("/", "__"))
    return snapshot_download(repo_id=hub_repo, repo_type="dataset", local_dir=local)


def _is_temp(path, root):
    """True when the loader had to unpack `path` out of a shard for us."""
    return not os.path.realpath(path).startswith(os.path.realpath(root) + os.sep)


def _check_schema(ds, root, loader):
    """Fail early, and in words, on a release the loader cannot read.

    A release exported before schema v2 names its files `rgb.mp4` and
    `meta.json`, which the current loader does not look for. Left alone that
    surfaces as a bare `KeyError` deep inside the loader, part-way through an
    evaluation; this asks one clip for the two things `iter_groups` needs.
    """
    if not ds.clips:
        return
    clip = ds.clips[0]
    try:
        path = clip.video_path
        _ = clip.uid
    except KeyError as exc:
        raise RuntimeError(
            "%s has no %s, so it is not a schema v%s PhysLoc release and cannot "
            "be evaluated (a release exported before v%s stores clips as "
            "rgb.mp4 and meta.json). Re-export it with a current "
            "`physloc export`, or point --physloc_root at a generator run."
            % (root, exc.args[0], loader.SCHEMA_VERSION, loader.SCHEMA_VERSION)) from exc
    else:
        # A shard-backed clip unpacks its video to a temp file just to be asked.
        if _is_temp(path, root):
            try:
                os.remove(path)
            except OSError:
                pass


def iter_groups(root, physloc_repo=None, split=None, **filters):
    """Yield `(pair, clips)` for every complete pair under `root`.

    `pair` is the loader's `Pair` (`pair_uid`, `prompt`, `valid`, `invalids`)
    and `clips` a list of `(variation_type, clip, video_path)` with the pair's
    valid clip first, `clip` being the loader's `Clip`. `filters` are the
    loader's path filters (see `FILTERS`); they select which *invalid* clips are
    scored, while a pair's valid clip is always kept, since it is the reference
    the invalid ones are ranked against. A pair with no surviving invalid clip
    is skipped.

    Once the consumer moves past a pair, its clips' cached arrays are released
    and any `video.mp4` the loader unpacked from a shard to a temporary file is
    deleted, so a long evaluation grows neither memory nor the temp directory.
    """
    loader = load_loader(physloc_repo)
    want = {k: {v} if isinstance(v, str) else set(v)
            for k, v in filters.items() if v}
    unknown = sorted(set(want) - set(FILTERS))
    if unknown:
        raise TypeError("unknown filter(s) %s; known: %s" % (unknown, list(FILTERS)))

    ds = loader.PhysLocDataset(root, split=split)
    _check_schema(ds, root, loader)

    # The loader's path fields decide what is kept, so filtering opens no
    # metadata. Filtering the dataset itself would drop the valid clips.
    fields = {id(clip): ds.fields(i) for i, clip in enumerate(ds.clips)}

    def variation_type(clip):
        f = fields[id(clip)]
        return "%s_%s" % (f["family"], f["severity_bin"])

    for pair in ds.pairs():
        invalids = [c for c in pair.invalids
                    if all(fields[id(c)].get(k) in v for k, v in want.items())]
        if pair.valid is None or not invalids:
            continue

        clips = [("valid", pair.valid)] + [(variation_type(c), c) for c in invalids]
        clips = [(vt, c, c.video_path) for vt, c in clips]
        try:
            yield pair, clips
        finally:
            for _, clip, path in clips:
                clip.release()
                if _is_temp(path, root):
                    try:
                        os.remove(path)
                    except OSError:
                        pass


def main():
    import argparse

    ap = argparse.ArgumentParser(description="List the groups a PhysLoc release yields.")
    ap.add_argument("--physloc_root", required=True)
    ap.add_argument("--physloc_repo", default=None)
    ap.add_argument("--physloc_split", default=None)
    for name in FILTERS:
        ap.add_argument("--physloc_" + name, default=None)
    a = ap.parse_args()

    filters = {name: getattr(a, "physloc_" + name) for name in FILTERS}
    groups = 0
    for pair, clips in iter_groups(
            a.physloc_root, a.physloc_repo, split=a.physloc_split, **filters):
        groups += 1
        print("%s  %s" % (pair.pair_uid, (pair.prompt or "")[:60]))
        for variation_type, clip, _ in clips:
            print("    %-24s %s" % (variation_type, os.path.basename(clip.uid)))
    print("\n%d group(s)" % groups)


if __name__ == "__main__":
    main()
