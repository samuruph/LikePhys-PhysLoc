"""PhysLoc clips as LikePhys evaluation groups.

LikePhys scores a model by how often it prefers a valid video over an invalid
one of the same scene. A PhysLoc *pair* is exactly that group: one valid clip
and every invalid clip rendered from its scene, sharing a bit-identical prefix
up to the violation. Each invalid clip's type is `<family>_<severity bin>`, so
`compute_misrank_normalized` reports one mis-rank per family and bin.

The PhysLoc loader (`physloc/loader.py`) imports nothing but numpy and the
standard library, so it is loaded straight from a PhysLoc checkout by path --
this environment does not need the generator installed.

    python evaluator.py --model ltx-0.9.5 --data physloc \
        --physloc_root /path/to/physloc_release [--physloc_family permanence]
"""
import importlib.util
import os
import sys

#: Where to find `physloc/loader.py` when neither --physloc_repo nor
#: PHYSLOC_REPO is given: a PhysLoc checkout next to this repository.
DEFAULT_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "physloc")


def load_loader(physloc_repo=None):
    """Import `physloc/loader.py` from a PhysLoc checkout, by path."""
    repo = physloc_repo or os.environ.get("PHYSLOC_REPO") or DEFAULT_REPO
    path = os.path.join(repo, "physloc", "loader.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no PhysLoc loader at %s; pass --physloc_repo or set PHYSLOC_REPO" % path)
    spec = importlib.util.spec_from_file_location("physloc_loader", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def iter_groups(root, physloc_repo=None, **filters):
    """Yield `(pair_uid, prompt, videos)` for every complete pair under `root`.

    `videos` is a list of `(variation_type, video_path, clip_uid)`, valid first.
    `filters` are the loader's: `family`, `scenario`, `level`, `condition`,
    `severity_bin`, and `split` on an exported release. Invalid clips are
    filtered by them; a pair's valid clip is always kept, since it is the
    reference every invalid clip is ranked against.
    """
    loader = load_loader(physloc_repo)
    keep = {k: v for k, v in filters.items() if v}
    split = keep.pop("split", None)
    wanted = {c.uid for c in loader.PhysLocDataset(root, split=split, **keep).clips}
    for pair in loader.PhysLocDataset(root, split=split).pairs():
        invalids = [c for c in pair.invalids if c.uid in wanted]
        if pair.valid is None or not invalids:
            continue
        videos = [("valid", pair.valid.video_path, pair.valid.uid)]
        videos += [("%s_%s" % (c.family, c.severity_bin), c.video_path, c.uid)
                   for c in invalids]
        yield pair.pair_uid, pair.prompt, videos
