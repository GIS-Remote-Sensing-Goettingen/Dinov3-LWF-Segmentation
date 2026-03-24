"""Compatibility wrapper for the utility folder-merge helper."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility import merge_folder_prediction_tifs as _impl  # noqa: E402

for _name in dir(_impl):
    if _name.startswith("__") and _name not in {"__all__"}:
        continue
    globals()[_name] = getattr(_impl, _name)


if __name__ == "__main__":
    _impl.main()
