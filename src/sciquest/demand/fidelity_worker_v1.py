"""Standalone array-structure check, copied into an empty-root sandbox.

No SciQuest imports and no execution of scientist Python. Untrusted NPZ parsing
occurs only here, under a bounded subprocess, with pickle disabled.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text())
    arrays = []
    opened = {}
    total = 0
    try:
        for spec in manifest["arrays"]:
            filename = spec["file"]
            if filename not in opened:
                opened[filename] = np.load(filename, allow_pickle=False)
            archive = opened[filename]
            value = archive[spec["key"]]
            if (
                not isinstance(value, np.ndarray)
                or value.dtype.kind not in "biuf"
                or list(value.shape) != spec["shape"]
                or not np.isfinite(value).all()
            ):
                raise ValueError("Declared numeric array shape/type/finite check failed")
            total += value.nbytes
            if total > 16 * 1024 * 1024:
                raise ValueError("Decoded declared arrays exceed16MiB")
            if spec["role"] == "particle_weights":
                if value.ndim != 1 or (value < 0).any() or not math.isclose(float(value.sum()), 1, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError("Particle weights must be a normalized nonnegative vector")
            arrays.append(dict(
                family_id=spec["family_id"], file=filename, key=spec["key"],
                shape=list(value.shape), dtype=str(value.dtype), bytes=int(value.nbytes),
            ))
    finally:
        for archive in opened.values():
            archive.close()
    Path(sys.argv[2]).write_text(json.dumps(dict(status="succeeded", arrays=arrays)))


if __name__ == "__main__":
    main()
