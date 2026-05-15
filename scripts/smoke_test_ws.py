"""End-to-end smoke test for the split-learning WebSocket protocol.

Mimics what the browser does for both the training flow and the inference
flow, without bringing a real browser into the loop. Useful for catching
protocol regressions without needing to spin up Vite + a UI.

Run against a running server:

    python scripts/server.py --learning-rate=0.01 &
    python scripts/smoke_test_ws.py

The script:

  1. Connects, sends ``REQUEST_BATCH``, verifies a ``BATCH`` reply of shape
     ``(B, 1, 28, 28)`` images + ``(B,)`` int64 labels and that the bytes
     decode to the expected element counts.
  2. Re-uses those labels to send an ``ACTIVATIONS_AND_LABELS`` message with
     random activations of shape ``(B, 16, 7, 7)``, verifies a ``GRADS``
     reply of the same shape with a finite scalar loss.
  3. Sends a fresh ``ACTIVATIONS`` message (inference path), verifies a
     ``LOGITS`` reply of shape ``(B, 10)``.

Exits non-zero on any assertion failure.
"""

import asyncio
import base64
import json
import logging
import math
import sys
from typing import Any

import click
import numpy as np
import websockets


_logger = logging.getLogger(__name__)


def _encode(message: dict) -> bytes:
    """Mirror the server's envelope: base64(JSON) as UTF-8 bytes."""
    return base64.b64encode(json.dumps(message).encode("utf-8"))


def _decode(payload: Any) -> dict:
    if isinstance(payload, (bytes, bytearray)):
        decoded = base64.b64decode(payload).decode("utf-8")
    else:
        decoded = base64.b64decode(payload.encode()).decode("utf-8")
    return json.loads(decoded)


async def _request_batch(ws) -> tuple[np.ndarray, np.ndarray, bytes]:
    await ws.send(
        _encode({"type": "request_batch", "data": {}, "raw": {}})
    )
    envelope = _decode(await asyncio.wait_for(ws.recv(), timeout=30))
    assert envelope["type"] == "batch", f"expected batch, got {envelope['type']}"

    images_shape = envelope["data"]["images_shape"]
    labels_shape = envelope["data"]["labels_shape"]
    images_raw = base64.b64decode(envelope["raw"]["images"])
    labels_raw = base64.b64decode(envelope["raw"]["labels"])
    images = np.frombuffer(images_raw, dtype=np.float32).reshape(images_shape)
    labels = np.frombuffer(labels_raw, dtype=np.int64).reshape(labels_shape)

    expected_image_elems = int(np.prod(images_shape))
    expected_label_elems = int(np.prod(labels_shape))
    assert images.size == expected_image_elems, (
        f"image element mismatch: got {images.size}, expected {expected_image_elems}"
    )
    assert labels.size == expected_label_elems, (
        f"label element mismatch: got {labels.size}, expected {expected_label_elems}"
    )
    assert images.shape[1:] == (1, 28, 28), f"unexpected image shape {images.shape}"

    _logger.info(
        "BATCH ok: images=%s labels=%s pixel-range=[%.3f, %.3f]",
        images.shape,
        labels.shape,
        float(images.min()),
        float(images.max()),
    )
    return images, labels, labels_raw


async def _round_trip_grads(ws, batch_size: int, labels_raw: bytes) -> None:
    activations = np.random.randn(batch_size, 16, 7, 7).astype(np.float32)
    await ws.send(
        _encode(
            {
                "type": "activations_and_labels",
                "data": {"tensor_shape": [batch_size, 16, 7, 7]},
                "raw": {
                    "tensor": base64.b64encode(activations.tobytes()).decode(),
                    "labels": base64.b64encode(labels_raw).decode(),
                },
            }
        )
    )
    envelope = _decode(await asyncio.wait_for(ws.recv(), timeout=30))
    assert envelope["type"] == "grads", f"expected grads, got {envelope['type']}"

    grads_shape = envelope["data"]["tensor_shape"]
    loss = float(envelope["data"]["loss"])
    grads_raw = base64.b64decode(envelope["raw"]["tensor"])
    grads = np.frombuffer(grads_raw, dtype=np.float32).reshape(grads_shape)

    assert tuple(grads_shape) == (batch_size, 16, 7, 7), (
        f"unexpected grads shape {grads_shape}"
    )
    assert math.isfinite(loss), f"non-finite loss {loss}"
    _logger.info(
        "GRADS ok: shape=%s loss=%.4f grad-range=[%.4f, %.4f]",
        grads.shape,
        loss,
        float(grads.min()),
        float(grads.max()),
    )


async def _round_trip_logits(ws, batch_size: int) -> None:
    activations = np.random.randn(batch_size, 16, 7, 7).astype(np.float32)
    await ws.send(
        _encode(
            {
                "type": "activations",
                "data": {"tensor_shape": [batch_size, 16, 7, 7]},
                "raw": {"tensor": base64.b64encode(activations.tobytes()).decode()},
            }
        )
    )
    envelope = _decode(await asyncio.wait_for(ws.recv(), timeout=30))
    assert envelope["type"] == "logits", f"expected logits, got {envelope['type']}"

    logits_shape = envelope["data"]["tensor_shape"]
    logits_raw = base64.b64decode(envelope["raw"]["tensor"])
    logits = np.frombuffer(logits_raw, dtype=np.float32).reshape(logits_shape)

    assert tuple(logits_shape) == (batch_size, 10), (
        f"unexpected logits shape {logits_shape} (expected (B, 10))"
    )
    _logger.info(
        "LOGITS ok: shape=%s, first-sample argmax=%d",
        logits.shape,
        int(logits[0].argmax()),
    )


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--endpoint", default="/ws", show_default=True)
@click.option("-q", "--quiet", "log_level", flag_value=logging.WARNING)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO, default=True)
def main(host: str, port: int, endpoint: str, log_level: int) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="[%(asctime)s] %(levelname)s: %(message)s",
        force=True,
    )

    uri = f"ws://{host}:{port}{endpoint}"

    async def run() -> None:
        _logger.info("Connecting to %s ...", uri)
        async with websockets.connect(uri, max_size=64 * 1024 * 1024) as ws:
            _, labels, labels_raw = await _request_batch(ws)
            await _round_trip_grads(ws, batch_size=int(labels.shape[0]), labels_raw=labels_raw)
            await _round_trip_logits(ws, batch_size=int(labels.shape[0]))

        _logger.info("All round-trips passed.")

    try:
        asyncio.run(run())
    except AssertionError as e:
        _logger.error("ASSERTION FAILED: %s", e)
        sys.exit(2)
    except (ConnectionRefusedError, OSError) as e:
        _logger.error(
            "Could not connect to %s: %s. Is `python scripts/server.py` running?",
            uri,
            e,
        )
        sys.exit(3)


if __name__ == "__main__":
    main()
