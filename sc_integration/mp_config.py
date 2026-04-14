"""Re-export from SC/mp_config.py — the canonical implementation now lives in SC/.

All classes and functions are re-exported here so that existing Q-DiT imports
(sc_attention.py, sc_mlp.py, sc_controller.py, __init__.py, quant_sc_main.py)
continue to work without changes.
"""
import sys
from pathlib import Path

SC_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "SC")
if SC_PATH not in sys.path:
    sys.path.insert(0, SC_PATH)

from mp_config import (  # noqa: F401, E402
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
