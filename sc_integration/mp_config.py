"""Re-export from scmp_kernels.mp — the canonical implementation now lives there.

All classes and functions are re-exported here so existing Q-DiT imports
(sc_attention.py, sc_mlp.py, sc_controller.py, sc_modelutils.py) keep working
via ``from .mp_config import ...`` without changes.
"""
from scmp_kernels.mp import (  # noqa: F401
    MPConfig,
    AdaptiveMPConfig,
    RangeMPConfig,
    RowAssignment,
    classify_rows_by_metric,
    adaptive_classify_rows,
    classify_groups_by_range,
    MPDistributionLogger,
    MetricProfiler,
)
