"""Compatibility wrapper for the utility folder-merge helper."""

from __future__ import annotations

from utility import merge_folder_prediction_tifs as _impl

for _name in dir(_impl):
    if _name.startswith("__") and _name not in {"__all__"}:
        continue
    globals()[_name] = getattr(_impl, _name)


if __name__ == "__main__":
    _impl.main()
