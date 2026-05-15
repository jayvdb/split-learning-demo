import logging
import sys
from pathlib import Path
from typing import Any

import click
import lightning as L
import onnx
import torch
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from onnx import numpy_helper
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from uvicorn import Config, Server

import split_learning
from split_learning.models.vision.cnn_2d import CNN2D, CNN2DServer
from split_learning.schemas.message import MessageType, WSMessage
from split_learning.utils import datasets as datasets
from split_learning.utils import utils
from split_learning.utils.serde import (
    decode_message_b64,
    deserialize_tensor,
    encode_message_b64,
    serialize_tensor,
)


def load_onnx_weights(model: nn.Module, onnx_path: Path) -> list[str]:
    onnx_model = onnx.load(str(onnx_path))
    onnx_weights = {
        init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
        for init in onnx_model.graph.initializer
    }
    state_dict = model.state_dict()
    unmatched: list[str] = []
    for key, current in state_dict.items():
        candidate = onnx_weights.get(key)
        if candidate is not None and candidate.shape == current.shape:
            state_dict[key] = candidate
        else:
            unmatched.append(key)
    model.load_state_dict(state_dict)
    return unmatched


def export_onnx(model: nn.Module, onnx_path: Path, example_input: torch.Tensor) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    try:
        torch.onnx.export(
            model,
            example_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            # Force the legacy TorchScript exporter: the dynamo path pulls in
            # onnxscript's torchlib registry, which on Python 3.14 trips a
            # `typing.Union` typeinfo check in onnxscript 0.5.6.dev*.
            dynamo=False,
        )
    finally:
        model.train(was_training)


class MnistBatchStream:
    """Endless iterator over MNIST training batches, lazily reset on exhaustion.

    Each wraparound of the underlying ``DataLoader`` corresponds to one full
    MNIST pass — i.e. one browser-side training epoch, since the browser
    consumes batches one-for-one via ``REQUEST_BATCH``. We log on wraparound
    so the server-side console shows epoch boundaries.
    """

    def __init__(self, batch_size: int = 128, num_workers: int = 0) -> None:
        mnist_normalize = transforms.Normalize((0.1307,), (0.3081,))
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=4),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                mnist_normalize,
            ]
        )
        self._dataset = datasets.mnist(split="train", transform=train_transform)
        # num_workers=0 keeps the DataLoader in-process. Background workers
        # would need to pickle the dataset, but the HuggingFace transform is
        # a local closure (defined inside `datasets.dataset_loader`); under
        # Python 3.14's new `forkserver` POSIX default that pickle fails.
        # We serve at most one batch per WebSocket request so prefetching
        # isn't worth the workaround.
        self._loader = DataLoader(
            self._dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self._iter = iter(self._loader)
        self._epoch = 0
        self._batches_this_epoch = 0

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            data = next(self._iter)
        except StopIteration:
            self._epoch += 1
            _logger.info(
                "Browser epoch %d complete (%d batches delivered)",
                self._epoch,
                self._batches_this_epoch,
            )
            self._batches_this_epoch = 0
            self._iter = iter(self._loader)
            data = next(self._iter)
        self._batches_this_epoch += 1
        return data["image"], data["label"]


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send(self, websocket: WebSocket, message: Any):
        await websocket.send(message)

    async def send_bytes(self, websocket: WebSocket, message: bytes):
        await websocket.send_bytes(message)

    async def broadcast(self, message: Any):
        for connection in self.active_connections:
            await connection.send(message)

    async def broadcast_bytes(self, message: bytes):
        for connection in self.active_connections:
            await connection.send_bytes(message)


# logger
_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-c",
    "--config",
    "cfg_path",
    type=click.Path(exists=True),
    help="path to config file",
)
# training
@click.option("--learning-rate", "learning_rate", type=float, default=1e-4)
@click.option("--grad-clip", "grad_clip", type=float, default=0.5)
# runtime
@click.option(
    "--accelerator",
    "accelerator",
    type=click.Choice(["auto", "cpu", "gpu", "cuda", "mps", "tpu"]),
    default="auto",
    show_default=True,
    help="Lightning Fabric accelerator backend.",
)
# logging
@click.option("--grad-accumulate-every", "grad_accumulate_every", type=int, default=4)
@click.option("--validate-every", "validate_every", type=int, default=100)
@click.option("--generate-every", "generate_every", type=int, default=500)
# log levels
@click.option("-q", "--quiet", "log_level", flag_value=logging.WARNING)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO, default=True)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
# version
@click.version_option(split_learning.__version__)
def main(
    cfg_path: Path,
    # training
    learning_rate: float,
    grad_clip: float,
    # runtime
    accelerator: str,
    # logging
    grad_accumulate_every: int,
    validate_every: int,
    generate_every: int,
    # log levels
    log_level: int,
):
    # logging
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    # accelerator
    fabric = L.Fabric(accelerator=accelerator, precision="32-true")
    fabric.launch()

    # webserver
    api_prefix = "/api/v1"
    app = FastAPI(title="split-learning", openapi_url=f"{api_prefix}/openapi.json")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    manager = ConnectionManager()

    # model
    model = CNN2D(
        in_channels=1,
        dim_out=10,
        img_size=28,
        dropout=0.15,
    )
    server_onnx_path = utils.workspace_root_path() / "data/models/server_mnist.onnx"

    model = CNN2DServer(in_channels=1, dim_out=10, img_size=28, model=model)
    if server_onnx_path.exists():
        unmatched = load_onnx_weights(model, server_onnx_path)
        if unmatched:
            _logger.warning(f"ONNX weights not loaded for: {unmatched}")
        else:
            _logger.info(f"Loaded server weights from {server_onnx_path}")
    else:
        _logger.info(f"No server weights at {server_onnx_path}; starting from random")

    unwrapped_model = model
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model, optimizer = fabric.setup(model, optimizer)

    batch_stream: MnistBatchStream | None = None

    def get_batch_stream() -> MnistBatchStream:
        nonlocal batch_stream
        if batch_stream is None:
            _logger.info("Initialising MNIST batch stream for browser training...")
            batch_stream = MnistBatchStream(batch_size=128)
        return batch_stream

    @app.websocket("/ws", api_prefix)
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        trained_this_session = False
        try:
            while True:
                optimizer.zero_grad()

                messages_bytes = await websocket.receive_bytes()
                message = decode_message_b64(messages_bytes)

                if message.type == MessageType.ACTIVATIONS_AND_LABELS:
                    activations = deserialize_tensor(
                        message.raw["tensor"], dtype=torch.float32
                    )
                    labels = deserialize_tensor(
                        message.raw["labels"], dtype=torch.int64
                    )

                    activations = activations.to(fabric.device)
                    activations = activations.reshape(*message.data["tensor_shape"])
                    labels = labels.to(fabric.device)

                    model.train()
                    trained_this_session = True
                    activations.requires_grad = True
                    outputs = model(activations)
                    loss = criterion(outputs, labels)
                    fabric.backward(loss)

                    optimizer.step()

                    # send grads
                    grads = activations.grad
                    client_grads = grads.detach().clone()
                    serialized_grads = serialize_tensor(client_grads.cpu())
                    response_message = WSMessage(
                        type=MessageType.GRADS,
                        data={"tensor_shape": grads.shape, "loss": loss.item()},
                        raw={"tensor": serialized_grads},
                    )
                    encoded_response = encode_message_b64(response_message)
                    await websocket.send_bytes(encoded_response)
                elif message.type == MessageType.REQUEST_BATCH:
                    images, labels = get_batch_stream().next()

                    serialized_images = serialize_tensor(images.cpu())
                    serialized_labels = serialize_tensor(labels.cpu())
                    response_message = WSMessage(
                        type=MessageType.BATCH,
                        data={
                            "images_shape": list(images.shape),
                            "labels_shape": list(labels.shape),
                        },
                        raw={
                            "images": serialized_images,
                            "labels": serialized_labels,
                        },
                    )
                    encoded_response = encode_message_b64(response_message)
                    await websocket.send_bytes(encoded_response)
                elif message.type == MessageType.ACTIVATIONS:
                    activations = deserialize_tensor(
                        message.raw["tensor"], dtype=torch.float32
                    )

                    activations = activations.to(fabric.device)
                    activations = activations.reshape(*message.data["tensor_shape"])

                    model.eval()
                    outputs = model(activations)

                    # send logits
                    logits = outputs.detach().clone()
                    serialized_logits = serialize_tensor(logits.cpu())
                    response_message = WSMessage(
                        type=MessageType.LOGITS,
                        data={"tensor_shape": logits.shape},
                        raw={"tensor": serialized_logits},
                    )
                    encoded_response = encode_message_b64(response_message)
                    await websocket.send_bytes(encoded_response)
        except WebSocketDisconnect:
            manager.disconnect(websocket)
            if trained_this_session:
                example_input = torch.zeros(1, 16, 7, 7, device=fabric.device)
                export_onnx(unwrapped_model, server_onnx_path, example_input)
                _logger.info(f"Saved trained server weights to {server_onnx_path}")
        except Exception as e:
            _logger.error(e)
            raise e

    server_config = Config(
        app=app, host="127.0.0.1", port=8000, ws_max_size=64 * 1024 * 1024
    )
    server = Server(config=server_config)
    server.run()


if __name__ == "__main__":
    main()
