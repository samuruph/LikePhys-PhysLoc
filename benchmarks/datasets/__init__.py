"""Dataset-specific configuration, loading, and traversal helpers."""

from .likephys import DATASETS as LIKEPHYS_DATASETS
from .likephys import PROMPTS as LIKEPHYS_PROMPTS


def open_physloc_dataset(*args, **kwargs):
    """Load PhysLoc lazily, only when a PhysLoc dataset is requested."""
    from .physloc import open_dataset
    return open_dataset(*args, **kwargs)


def resolve_physloc_loader_path(*args, **kwargs):
    """Resolve the external PhysLoc loader lazily."""
    from .physloc import resolve_loader_path
    return resolve_loader_path(*args, **kwargs)


def validate_physloc_release(*args, **kwargs):
    """Validate a PhysLoc release lazily."""
    from .physloc import validate_release
    return validate_release(*args, **kwargs)

__all__ = (
    "LIKEPHYS_DATASETS",
    "LIKEPHYS_PROMPTS",
    "open_physloc_dataset",
    "resolve_physloc_loader_path",
    "validate_physloc_release",
)
