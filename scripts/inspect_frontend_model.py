"""Inspect a TF.js client model dumped by the frontend.

The frontend sends its trained model over the WebSocket as
``SAVE_CLIENT_MODEL``; ``scripts/server.py`` writes two paired files into
``data/models/``:

    frontend_client_<ts>.json          # topology + weight_specs + training meta
    frontend_client_<ts>.weights.bin   # concatenated Float32 weight bytes

This script reads them and prints, per layer:

  * shape, mean, std, abs-max, fraction of near-zero entries

If ``data/models/server_mnist.onnx`` exists, it also prints stats for the
matching initializers from the server's PyTorch half so you can sanity-check
both sides on the same run.

Usage:

    python scripts/inspect_frontend_model.py                 # latest dump
    python scripts/inspect_frontend_model.py path/to/.json   # specific dump
"""

import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
except ImportError:  # pragma: no cover - onnx is in environment.yml
    onnx = None
    numpy_helper = None

from split_learning.utils import utils


_logger = logging.getLogger(__name__)


_DTYPE_TO_NP = {
    "float32": np.float32,
    "float16": np.float16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def _latest_dump() -> Path | None:
    candidates = sorted(
        (utils.workspace_root_path() / "data/models").glob("frontend_client_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _array_stats(arr: np.ndarray) -> dict:
    flat = arr.ravel().astype(np.float64)
    near_zero = np.mean(np.abs(flat) < 1e-6)
    return {
        "shape": list(arr.shape),
        "n": int(arr.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "abs_max": float(np.abs(flat).max()),
        "frac_near_zero": float(near_zero),
    }


def _fmt_stats(s: dict) -> str:
    return (
        f"shape={s['shape']!s:<20s}  "
        f"mean={s['mean']:+.4f}  std={s['std']:.4f}  "
        f"min={s['min']:+.4f}  max={s['max']:+.4f}  "
        f"|max|={s['abs_max']:.4f}  ~0={s['frac_near_zero']:.2%}"
    )


def _iter_frontend_weights(
    meta: dict, blob: bytes
) -> Iterable[tuple[str, np.ndarray]]:
    """Walk the weight blob in the order described by ``weight_specs``."""
    offset = 0
    for spec in meta.get("weight_specs", []):
        name = spec["name"]
        shape = tuple(spec["shape"])
        dtype_name = spec.get("dtype", "float32")
        dtype = _DTYPE_TO_NP.get(dtype_name)
        if dtype is None:
            _logger.warning("Skipping weight %s with unsupported dtype %s", name, dtype_name)
            continue
        n_elements = 1
        for d in shape:
            n_elements *= d
        n_bytes = n_elements * dtype().itemsize
        chunk = blob[offset : offset + n_bytes]
        offset += n_bytes
        arr = np.frombuffer(chunk, dtype=dtype).reshape(shape)
        yield name, arr
    if offset != len(blob):
        _logger.warning(
            "Trailing %d bytes in weights.bin after consuming all weight_specs",
            len(blob) - offset,
        )


def _server_initializers() -> dict[str, np.ndarray] | None:
    if onnx is None:
        return None
    path = utils.workspace_root_path() / "data/models/server_mnist.onnx"
    if not path.exists():
        return None
    model = onnx.load(str(path))
    return {
        init.name: np.array(numpy_helper.to_array(init))
        for init in model.graph.initializer
    }


def main() -> int:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        force=True,
    )

    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        latest = _latest_dump()
        if latest is None:
            print(
                "No data/models/frontend_client_*.json found. Run training and "
                "click 'Send model to server for inspection' first.",
                file=sys.stderr,
            )
            return 2
        json_path = latest

    bin_path = json_path.with_suffix("").with_suffix(".weights.bin")
    if not json_path.exists():
        print(f"Missing {json_path}", file=sys.stderr)
        return 2
    if not bin_path.exists():
        # Earlier dumps may have used `.weights.bin` as a single suffix.
        alt = json_path.parent / (json_path.stem + ".weights.bin")
        if alt.exists():
            bin_path = alt
        else:
            print(f"Missing weights binary alongside {json_path}", file=sys.stderr)
            return 2

    meta = json.loads(json_path.read_text())
    blob = bin_path.read_bytes()

    _logger.info("Frontend model: %s", json_path.name)
    training = meta.get("training") or {}
    if training:
        _logger.info("Training meta: %s", json.dumps(training, indent=2))
    _logger.info("Weights blob: %d bytes", len(blob))
    _logger.info("---- Per-weight stats (TF.js client) ----")
    for name, arr in _iter_frontend_weights(meta, blob):
        _logger.info("%-40s %s", name, _fmt_stats(_array_stats(arr)))

    server_inits = _server_initializers()
    if server_inits:
        _logger.info("---- Per-weight stats (server: data/models/server_mnist.onnx) ----")
        for name, arr in server_inits.items():
            _logger.info("%-40s %s", name, _fmt_stats(_array_stats(arr)))
    else:
        _logger.info(
            "data/models/server_mnist.onnx not found (or onnx not installed); "
            "skipping server-side comparison."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
