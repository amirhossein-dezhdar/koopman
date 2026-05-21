"""Simple run logger.

Accumulates per-step quantities in memory then dumps to ``.npz`` at end of
run. Designed for the metrics the paper actually plots: tracking error,
barrier value, residual bound, trigger flags, solve times, conformal
coverage.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


class RunLogger:
    """Per-step buffer that flushes to disk at end of run."""

    def __init__(self, out_dir: str | Path, run_name: str | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
        self._buffer: dict[str, list[Any]] = defaultdict(list)
        self._meta: dict[str, Any] = {}

    def log(self, **kwargs: Any) -> None:
        """Append one step's worth of named scalars / vectors."""
        for k, v in kwargs.items():
            self._buffer[k].append(v)

    def set_meta(self, **kwargs: Any) -> None:
        self._meta.update(kwargs)

    def flush(self) -> Path:
        """Write ``<run_name>.npz`` + ``<run_name>.json``; return npz path."""
        npz_path = self.out_dir / f"{self.run_name}.npz"
        json_path = self.out_dir / f"{self.run_name}.json"

        arrays: dict[str, np.ndarray] = {}
        for k, v in self._buffer.items():
            try:
                arrays[k] = np.asarray(v)
            except (ValueError, TypeError):
                arrays[k] = np.asarray(v, dtype=object)
        np.savez_compressed(npz_path, **arrays)

        with open(json_path, "w") as f:
            json.dump(self._meta, f, indent=2, default=str)
        return npz_path

    def __len__(self) -> int:
        if not self._buffer:
            return 0
        return max(len(v) for v in self._buffer.values())
