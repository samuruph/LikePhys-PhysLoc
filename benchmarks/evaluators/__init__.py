"""Benchmark adapters sharing the PPE model-scoring implementation."""

from .common import compute_misrank_normalized
from .likephys import DATASETS as LIKEPHYS_DATASETS
from .likephys import PROMPTS as LIKEPHYS_PROMPTS
from .likephys import evaluate as evaluate_likephys
from .physloc import FILTERS as PHYSLOC_FILTERS
from .physloc import NAME as PHYSLOC
from .physloc import evaluate as evaluate_physloc
from .physloc import iter_groups as iter_physloc_groups
from .physloc import resolve_root as resolve_physloc_root

__all__ = (
    "LIKEPHYS_DATASETS",
    "LIKEPHYS_PROMPTS",
    "PHYSLOC",
    "PHYSLOC_FILTERS",
    "compute_misrank_normalized",
    "evaluate_likephys",
    "evaluate_physloc",
    "iter_physloc_groups",
    "resolve_physloc_root",
)
