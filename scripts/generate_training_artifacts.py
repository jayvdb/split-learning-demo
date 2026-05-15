"""Generate ONNX Runtime Training (Web) artifacts for the split-learning client.

Produces four files under ``apps/web/public/models/training/``:

    training_model.onnx   forward + dot-product fake-loss + backward
    eval_model.onnx       forward only (used for inference after training)
    optimizer_model.onnx  SGD optimizer (lr is supplied per-step in the browser)
    checkpoint            initial parameter state

Loss is the split-learning chain-rule trick:

    fake_loss = ReduceSum( activations * upstream_signal )

so that d(fake_loss)/d(params) = d(activations)/d(params) * upstream_signal
— i.e. exactly the gradient the Python client computes via
``torch.autograd.backward(activations, grads)``. ``upstream_signal`` becomes a
regular graph input that the browser supplies as a feed.

Re-run any time ``CNN2DClient`` changes shape.
"""

import io
import logging
import sys

import onnx
import torch
from onnxruntime.training import artifacts
from onnxruntime.training.onnxblock import blocks
from torch import nn

from split_learning.models.vision.cnn_2d import CNN2DClient
from split_learning.utils import utils


_logger = logging.getLogger(__name__)


class DotProductLoss(blocks.Block):
    """Sum-of-products fake loss for split learning.

    Expects ``upstream_signal`` to already be declared as a graph input on the
    forward model (we add it manually before calling ``generate_artifacts``,
    because ORT's runtime determines user-inputs from the model's input
    list at load time and won't pick up inputs added by a loss block).
    """

    def __init__(self):
        super().__init__()
        self._mul = blocks.Mul()
        # keepdims=False so the loss is a 0-d scalar (ORT training requires it).
        self._reduce = blocks.ReduceSum(keepdims=False)

    def build(self, activations_name: str, upstream_signal_name: str):
        return self._reduce(self._mul(activations_name, upstream_signal_name))


def export_client_forward() -> onnx.ModelProto:
    """Export ``CNN2DClient`` and inject ``upstream_signal`` as a dangling input.

    The ONNX TorchScript exporter elides unused module arguments, so we can't
    plumb ``upstream_signal`` through Python forward. Instead we export the
    forward as-is (one input ``image`` → one output ``activations``) and
    then append ``upstream_signal`` directly to ``graph.input`` with the same
    type as ``activations``. ORT's training runtime then recognises it as a
    second user input alongside ``image``.
    """
    model = CNN2DClient(in_channels=1, dim_out=10, img_size=28)
    model.eval()
    example_image = torch.zeros(1, 1, 28, 28)

    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        example_image,
        buffer,
        input_names=["image"],
        output_names=["activations"],
        dynamic_axes={"image": {0: "batch"}, "activations": {0: "batch"}},
        # Force the legacy TorchScript exporter — see scripts/client.py:227 for
        # the matching Python-3.14 / onnxscript rationale.
        dynamo=False,
    )
    buffer.seek(0)
    forward = onnx.load_model_from_string(buffer.getvalue())

    activations_out = next(
        o for o in forward.graph.output if o.name == "activations"
    )
    upstream_signal = onnx.ValueInfoProto()
    upstream_signal.CopyFrom(activations_out)
    upstream_signal.name = "upstream_signal"
    forward.graph.input.append(upstream_signal)
    return forward


def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M",
        format="[%(asctime)s] %(levelname)s: %(message)s",
        # ORT imports its own logging handlers on first import; without
        # force=True the basicConfig is a no-op and we get no script output.
        force=True,
    )

    forward_model = export_client_forward()
    initializer_names = [init.name for init in forward_model.graph.initializer]
    _logger.info("Trainable initializers: %s", initializer_names)

    output_dir = utils.workspace_root_path() / "apps/web/public/models/training"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts.generate_artifacts(
        forward_model,
        requires_grad=initializer_names,
        frozen_params=[],
        loss=DotProductLoss(),
        loss_input_names=["activations", "upstream_signal"],
        # Expose the client activations as a training/eval output too — the
        # browser needs them to send over the WebSocket. Without this the
        # only output is the scalar fake-loss.
        additional_output_names=["activations"],
        optimizer=artifacts.OptimType.SGD,
        artifact_directory=str(output_dir),
    )
    _logger.info("Wrote training artifacts to %s", output_dir)

    # onnx>=1.19 stamps IR_VERSION=13 on emitted models, but
    # onnxruntime-training-cpu==1.19 (and onnxruntime-web ^1.17/^1.19) only
    # support IR up to 10. We don't use any new IR features in these
    # graphs; rewrite the field so the runtimes can load them.
    MAX_SUPPORTED_IR = 10
    for stem in ("training_model", "eval_model", "optimizer_model"):
        path = output_dir / f"{stem}.onnx"
        mp = onnx.load(str(path))
        if mp.ir_version > MAX_SUPPORTED_IR:
            mp.ir_version = MAX_SUPPORTED_IR
            onnx.save(mp, str(path))
            _logger.info("Pinned %s to IR %s", path.name, MAX_SUPPORTED_IR)

    # End-to-end training-API sanity: load all four files and run a single
    # train step + optimizer step on zero inputs. Confirms upstream_signal is
    # wired in as an input, the loss reduces to a scalar, and the optimizer
    # model accepts the resulting parameter grads.
    from onnxruntime.training import api as ort_train

    checkpoint = ort_train.CheckpointState.load_checkpoint(
        str(output_dir / "checkpoint")
    )
    module = ort_train.Module(
        train_model_uri=str(output_dir / "training_model.onnx"),
        state=checkpoint,
        eval_model_uri=str(output_dir / "eval_model.onnx"),
    )
    optimizer = ort_train.Optimizer(
        optimizer_uri=str(output_dir / "optimizer_model.onnx"), module=module
    )
    optimizer.set_learning_rate(0.01)

    image = torch.zeros(1, 1, 28, 28).numpy()
    upstream_signal = torch.zeros(1, 16, 7, 7).numpy()
    module.lazy_reset_grad()
    module.train()
    loss = module(image, upstream_signal)
    optimizer.step()
    _logger.info("Smoke train+opt step OK; loss = %s", loss)


if __name__ == "__main__":
    main()
