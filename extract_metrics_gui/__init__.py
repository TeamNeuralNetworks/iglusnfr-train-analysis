"""Interactive GUI for the extract_metrics fitting workflow."""

from __future__ import annotations

import sys
from pathlib import Path

# The package sits next to Feature_extraction/ and Model_Calibration/; make sure
# the repository root is importable however the GUI is launched.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = ["core", "loaders", "options", "registry", "theme", "widgets"]
