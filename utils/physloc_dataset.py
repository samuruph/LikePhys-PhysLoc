"""PhysLoc clips as LikePhys evaluation groups.

LikePhys scores a model by how often it prefers a valid video over an invalid
one of the same scene. A PhysLoc *pair* is exactly that group: one valid clip
and every invalid clip rendered from its scene, sharing a bit-identical prefix
up to the violation. Each invalid clip's variation type is
`<family>_<severity bin>`, so `compute_misrank_normalized` reports one mis-rank
per family and bin.

Reading is done by PhysLoc's own `PhysLocDataset` in pair mode -- this file
adds no reader of its own. The loader comes, in order, from:

  1. `<release>/loader.py`, the copy every exported PhysLoc release ships, so a
     download is read by the code it was exported with;
  2. `physloc/loader.py` in a PhysLoc checkout (`--physloc_repo`,
     `$PHYSLOC_REPO`, or a `physloc` checkout next to this repository), for a
     generator run, which has no shipped copy.

Either way it imports nothing but numpy and the standard library, so it is
imported by path and the generator need not be installed. A release on disk is
plain folders, `clips/<release>/<level>/<scenario>/<seed>_<condition>/<clip>/`,
whether it was downloaded or generated.

    python evaluator.py --model ltx-0.9.5 --data physloc \
        --physloc_root /path/to/physloc_release [--physloc_family permanence]

Run this file directly to list the groups a release yields, without a GPU:

    python -m utils.physloc_dataset --physloc_root /path/to/release
"""
import importlib.util
import os
import sys
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))

#: The PhysLoc checkout whose `physloc/loader.py` is used for a release that
#: ships no loader, when neither --physloc_repo nor $PHYSLOC_REPO is set.
DEFAULT_REPO = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "physloc")

#: Filters `iter_groups` forwards to the loader, each exposed by evaluator.py
#: as `--physloc_<name>`. They choose the INVALID clips; a pair's valid clip is
#: always kept. `split` is passed separately.
FILTERS = ("family", "scenario", "level", "condition", "severity_bin")


def _import_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered so that anything the loader pickles or reflects on resolves.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def loader_path(root=None, physloc_repo=None):
    """Which `loader.py` reads `root`: its shipped copy, else the checkout's."""
    if root:
        shipped = os.path.join(root, "loader.py")
        if os.path.exists(shipped):
            return shipped
    repo = physloc_repo or os.environ.get("PHYSLOC_REPO") or DEFAULT_REPO
    path = os.path.join(repo, "physloc", "loader.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "%s ships no loader.py and there is no PhysLoc loader at %s; pass "
            "--physloc_repo or set PHYSLOC_REPO" % (root, path))
    return path


def load_loader(root=None, physloc_repo=None):
    """Import the loader that reads `root` (see `loader_path`), by path."""
    path = os.path.realpath(loader_path(root, physloc_repo))
    # One module name per file, so a shipped loader and a checkout's never
    # shadow each other within a process.
    name = "physloc_loader_%08x" % zlib.crc32(path.encode())
    return sys.modules.get(name) or _import_by_path(path, name)


def download(hub_repo, cache="data/physloc", **kwargs):
    """Snapshot a PhysLoc release from the Hub into `<cache>/<owner>__<name>`.

    `kwargs` go to `snapshot_download`; `allow_patterns` limits the download,
    e.g. to what a video model is scored on:
    `["*.py", "*.txt", "*/metadata.json", "*/video.mp4"]`.
    """
    from huggingface_hub import snapshot_download

    local = os.path.join(cache, hub_repo.replace("/", "__"))
    return snapshot_download(repo_id=hub_repo, repo_type="dataset",
                             local_dir=local, **kwargs)


def iter_groups(root, physloc_repo=None, split=None, **filters):
    """Yield `(pair, clips)` for every complete pair under `root`.

    `pair` is the loader's `Pair` (`pair_uid`, `prompt`, `valid`, `invalids`)
    and `clips` a list of `(variation_type, clip, video_path)` with the pair's
    valid clip first, `clip` being the loader's `Clip`. `filters` (see
    `FILTERS`) select which invalid clips are scored; a pair with none left is
    skipped.

    Once the consumer moves past a pair, its clips' cached arrays are released,
    so a long evaluation does not grow in memory. A release exported before
    folders is still read, by the loader it shipped with.
    """
    loader = load_loader(root, physloc_repo)
    want = {k: v for k, v in filters.items() if v}
    unknown = sorted(set(want) - set(FILTERS))
    if unknown:
        raise TypeError("unknown filter(s) %s; known: %s" % (unknown, list(FILTERS)))

    for pair in _pairs(loader, root, split, want):
        clips = [("valid", pair.valid)] + [
            ("%s_%s" % (c.family, c.severity_bin), c) for c in pair.invalids]
        clips = [(vt, c, c.video_path) for vt, c in clips]
        try:
            yield pair, clips
        finally:
            for _, clip, path in clips:
                clip.release()
                # Only an old shard-backed release unpacks a video to a
                # temporary file; a folder release's path is its own file.
                if not os.path.realpath(path).startswith(
                        os.path.realpath(root) + os.sep):
                    try:
                        os.remove(path)
                    except OSError:
                        pass


def _pairs(loader, root, split, want):
    """The complete pairs `iter_groups` scores, with only the wanted invalids.

    A release exported before pair mode ships a loader without it (and reads
    its tar shards in place), so for that one the grouping is done here, on the
    path fields that loader exposes as `fields(i)`.
    """
    if hasattr(loader.PhysLocDataset, "UNITS"):
        return loader.PhysLocDataset(root, unit="pair", split=split,
                                     fields=("video_path",), **want).pairs()
    ds = loader.PhysLocDataset(root, split=split)
    info = {id(c): ds.fields(i) for i, c in enumerate(ds.clips)}
    out = []
    for pair in ds.pairs():
        pair.invalids = [c for c in pair.invalids
                         if all(info[id(c)].get(k) == v for k, v in want.items())]
        if pair.valid is not None and pair.invalids:
            out.append(pair)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description="List the groups a PhysLoc release yields.")
    ap.add_argument("--physloc_root", required=True)
    ap.add_argument("--physloc_repo", default=None)
    ap.add_argument("--physloc_split", default=None)
    for name in FILTERS:
        ap.add_argument("--physloc_" + name, default=None)
    a = ap.parse_args()

    print("loader: %s" % loader_path(a.physloc_root, a.physloc_repo))
    filters = {name: getattr(a, "physloc_" + name) for name in FILTERS}
    groups = 0
    for pair, clips in iter_groups(
            a.physloc_root, a.physloc_repo, split=a.physloc_split, **filters):
        groups += 1
        print("%s  %s" % (pair.pair_uid, (pair.prompt or "")[:60]))
        for variation_type, clip, _ in clips:
            print("    %-24s %s" % (variation_type, clip.uid.rsplit("/", 1)[-1]))
    print("\n%d group(s)" % groups)


if __name__ == "__main__":
    main()
