"""Benchmark datasets and evaluation adapters."""

from .evaluators.common import compute_misrank_normalized
from .evaluators.likephys import DATASETS as LIKEPHYS_DATASETS
from .evaluators.likephys import PROMPTS as LIKEPHYS_PROMPTS
from .evaluators.likephys import evaluate as evaluate_likephys
from .evaluators.physloc import FILTERS as PHYSLOC_FILTERS
from .evaluators.physloc import NAME as PHYSLOC
from .evaluators.physloc import evaluate as evaluate_physloc
from .evaluators.physloc import iter_groups as iter_physloc_groups
from .evaluators.physloc import resolve_root as resolve_physloc_root

__all__ = (
    "LIKEPHYS_DATASETS", "LIKEPHYS_PROMPTS", "PHYSLOC", "PHYSLOC_FILTERS",
    "compute_misrank_normalized", "evaluate_likephys", "evaluate_physloc",
    "iter_physloc_groups", "resolve_physloc_root",
)
