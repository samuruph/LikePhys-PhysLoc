"""Backward-compatible import location for PhysLoc dataset helpers.

Dataset adapters now live in :mod:`datasets.physloc`. This module remains so
existing notebooks and commands importing `utils.physloc_dataset` continue to
work without changes.
"""

from datasets.physloc import *  # noqa: F401,F403
from datasets import physloc as _physloc


def __getattr__(name: str):
    """Forward legacy symbols, including canonical-loader attributes."""
    return getattr(_physloc, name)


if __name__ == "__main__":
    from datasets.physloc import _main

    _main()
