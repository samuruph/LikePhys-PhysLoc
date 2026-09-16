"""PhysLoc clips as LikePhys evaluation groups.

LikePhys scores a model by how often it prefers a valid video over an invalid
one of the same scene. A PhysLoc *pair* is exactly that group: one valid clip
and every invalid clip rendered from its scene, sharing a bit-identical prefix
up to the violation. Each invalid clip's variation type is
`<family>_<severity bin>`, so `compute_misrank_normalized` reports one mis-rank
per family and bin.

Reading is delegated to PhysLoc's own `loader.py`, which imports nothing but
numpy and the standard library. A downloaded (or exported) release ships that
file at its root and it is used from there, so evaluating a release needs no
PhysLoc checkout at all. A generator run -- a bare `clips/` tree -- has no
shipped loader, so for those one is imported by path from a checkout
(`--physloc_repo`, `$PHYSLOC_REPO`, or a `physloc` checkout next to this
repository).

    python evaluator.py --model ltx-0.9.5 --data physloc \
        --physloc_root /path/to/physloc_release [--physloc_family permanence]

Run this file directly to list the groups a release yields, without a GPU:

    python -m utils.physloc_dataset --physloc_root /path/to/release
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Where to look for `physloc/loader.py` when a root ships none and neither
#: --physloc_repo nor $PHYSLOC_REPO is set: a checkout beside this repository.
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


def load_loader(root, physloc_repo=None):
    """The `loader.py` to read `root` with: the one shipped inside it if any.

    A release carries the loader matching the schema it was written with, so
    reading it with its own copy is what a downloaded release supports. Only a
    generator run (`clips/`, nothing shipped) falls back to a checkout.
    """
    shipped = os.path.join(root, "loader.py")
    if os.path.exists(shipped):
        return _import_by_path(shipped)

    repo = physloc_repo or os.environ.get("PHYSLOC_REPO") or DEFAULT_REPO
    path = os.path.join(repo, "physloc", "loader.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s ships no loader.py and there is no PhysLoc loader at %s; pass "
            "--physloc_repo or set PHYSLOC_REPO" % (root, path))
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
    """Yield `(pair_uid, prompt, videos)` for every complete pair under `root`.

    `videos` is a list of `(variation_type, video_path, clip_uid)` with the
    pair's valid clip first. `filters` are the loader's path filters (see
    `FILTERS`); they select which *invalid* clips are scored, while a pair's
    valid clip is always kept, since it is the reference the invalid ones are
    ranked against. A pair with no surviving invalid clip is skipped.

    On a release read from shards the loader unpacks each `video.mp4` to a
    temporary file; those are deleted once the consumer moves past the pair,
    so a long evaluation does not fill the temp directory.
    """
    loader = load_loader(root, physloc_repo)
    want = {k: {v} if isinstance(v, str) else set(v)
            for k, v in filters.items() if v}
    unknown = sorted(set(want) - set(FILTERS))
    if unknown:
        raise TypeError("unknown filter(s) %s; known: %s" % (unknown, list(FILTERS)))

    ds = loader.PhysLocDataset(root, split=split)
    _check_schema(ds, root, loader)

    # One index pass, grouped by pair. The fields come from each clip's path,
    # so filtering opens no metadata; only a kept pair's prompt is read.
    groups, order = {}, []
    for i, clip in enumerate(ds.clips):
        f = ds.fields(i)
        pair_uid = f["pair_uid"]
        if pair_uid not in groups:
            groups[pair_uid] = {"valid": None, "invalid": []}
            order.append(pair_uid)
        group = groups[pair_uid]
        if f["label"] == "valid":
            group["valid"] = clip
        elif all(f.get(k) in v for k, v in want.items()):
            group["invalid"].append(("%s_%s" % (f["family"], f["severity_bin"]), clip))

    for pair_uid in order:
        group = groups[pair_uid]
        if group["valid"] is None or not group["invalid"]:
            continue

        clips = [("valid", group["valid"])] + group["invalid"]
        videos = [(variation_type, clip.video_path, clip.uid)
                  for variation_type, clip in clips]
        try:
            yield pair_uid, group["valid"].prompt, videos
        finally:
            for _, path, _ in videos:
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
    for pair_uid, prompt, videos in iter_groups(
            a.physloc_root, a.physloc_repo, split=a.physloc_split, **filters):
        groups += 1
        print("%s  %s" % (pair_uid, (prompt or "")[:60]))
        for variation_type, path, clip_uid in videos:
            print("    %-24s %s" % (variation_type, os.path.basename(clip_uid)))
    print("\n%d group(s)" % groups)


if __name__ == "__main__":
    main()
