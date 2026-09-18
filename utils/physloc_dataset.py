"""Bridge to PhysLoc's canonical schema-v3 loader.

The dataset repository owns its format and derivation semantics.  This project
therefore imports ``physloc/loader.py`` from a sibling PhysLoc checkout instead
of maintaining a second copy.  Set ``PHYSLOC_LOADER`` or pass
``--physloc_loader`` when the checkout lives elsewhere.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
from functools import lru_cache
from types import ModuleType
from typing import Optional


DEFAULT_LOADER = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "physloc", "physloc", "loader.py"))


def resolve_loader_path(path: Optional[str] = None) -> str:
    """Return and validate the configured canonical PhysLoc loader path."""
    resolved = os.path.abspath(
        path or os.environ.get("PHYSLOC_LOADER") or DEFAULT_LOADER)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            "PhysLoc loader not found at %s; pass --physloc_loader or set "
            "PHYSLOC_LOADER" % resolved)
    return resolved


@lru_cache(maxsize=None)
def _load_resolved_loader(resolved: str) -> ModuleType:
    """Import and cache one already-resolved canonical loader path."""
    spec = importlib.util.spec_from_file_location(
        "_likephys_canonical_physloc_loader", resolved)
    if spec is None or spec.loader is None:
        raise ImportError("could not create an import spec for %s" % resolved)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if int(getattr(module, "SCHEMA_VERSION", 0)) != 3:
        raise RuntimeError(
            "canonical loader %s supports schema v%s, expected v3"
            % (resolved, getattr(module, "SCHEMA_VERSION", "unknown")))
    if not hasattr(module, "PhysLocDataset"):
        raise AttributeError("%s does not expose PhysLocDataset" % resolved)
    return module


def load_physloc_loader(path: Optional[str] = None) -> ModuleType:
    """Resolve, import, and cache PhysLoc's loader from its checkout."""
    return _load_resolved_loader(resolve_loader_path(path))


def validate_release(root: str, loader_path: Optional[str] = None) -> str:
    """Validate a schema-v3 root and dependencies before loading a model."""
    absolute = os.path.abspath(root)
    samples = glob.glob(os.path.join(
        absolute, "samples", "**", "sample.json"), recursive=True)
    if not samples:
        legacy = glob.glob(os.path.join(
            absolute, "clips", "**", "metadata.json"), recursive=True)
        if legacy:
            raise ValueError(
                "%s is a schema-v2 clips/NPZ release (%d metadata.json "
                "files), not schema v3. Point --physloc_root at the release "
                "containing samples/**/sample.json and data.h5"
                % (absolute, len(legacy)))
        raise FileNotFoundError(
            "no schema-v3 samples under %s; expected "
            "samples/**/sample.json and data.h5" % absolute)
    missing_h5 = [path for path in samples
                  if not os.path.isfile(os.path.join(os.path.dirname(path),
                                                     "data.h5"))]
    if missing_h5:
        raise FileNotFoundError(
            "schema-v3 release is incomplete: %d sample(s) have no data.h5; "
            "first: %s" % (len(missing_h5), missing_h5[0]))
    try:
        import h5py  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PhysLoc schema v3 requires h5py in this evaluator environment. "
            "Install h5py in the active environment before loading the model"
        ) from exc
    resolved = resolve_loader_path(loader_path)
    load_physloc_loader(resolved)
    return resolved


def open_dataset(root: str, loader_path: Optional[str] = None, **kwargs):
    """Construct ``PhysLocDataset`` using the canonical external loader."""
    loader = load_physloc_loader(loader_path)
    return loader.PhysLocDataset(root, **kwargs)


def __getattr__(name: str):
    """Forward legacy loader symbol imports to the canonical module."""
    loader = load_physloc_loader()
    try:
        return getattr(loader, name)
    except AttributeError as exc:
        raise AttributeError("module %r has no attribute %r"
                             % (__name__, name)) from exc


def _main() -> None:
    """Validate and summarize one schema-v3 release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physloc_root", required=True)
    parser.add_argument("--physloc_loader", default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    loader_path = validate_release(args.physloc_root, args.physloc_loader)
    dataset = open_dataset(
        args.physloc_root, loader_path=loader_path, unit="pair",
        split=args.split, fields=("observations.rgb_path",))
    invalids = sum(len(pair.invalids) for pair in dataset.pairs())
    print("PhysLoc schema-v3 via %s" % loader_path)
    print("%d complete pairs, %d invalid clips, %d total samples"
          % (len(dataset), invalids, len(dataset.samples)))
    for pair in dataset.pairs()[:10]:
        variants = ["%s/%s" % (sample.family, sample.severity_bin)
                    for sample in pair.invalids]
        print("- %s: %s" % (pair.pair_uid, ", ".join(variants)))
    if len(dataset) > 10:
        print("... %d more pairs" % (len(dataset) - 10))
    dataset.release()


if __name__ == "__main__":
    _main()
