"""Compatibility wrapper for the utility batch launcher."""

from __future__ import annotations

from utility import launch_batched_inference as _impl

for _name in dir(_impl):
    if _name.startswith("__") and _name not in {"__all__"}:
        continue
    globals()[_name] = getattr(_impl, _name)


if __name__ == "__main__":
    _impl.main()
