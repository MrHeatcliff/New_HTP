"""Read-only CoRe-WM evaluation infrastructure."""

from .config import ALPHA_GRID, HORIZONS, LONG_HORIZONS, PERFORMANCE_THRESHOLDS
from .metrics import linear_cka, multivariate_r2, reconstruction_metrics
from .slicing import split_prefixes_blocks

__all__ = [
    'ALPHA_GRID', 'HORIZONS', 'LONG_HORIZONS', 'PERFORMANCE_THRESHOLDS',
    'linear_cka', 'multivariate_r2', 'reconstruction_metrics',
    'split_prefixes_blocks',
]
