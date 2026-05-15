# Split Learning Demo

Simple implementation of split learning/split inference. A model is split at specified "cut" layer, and trained by a set of clients (holding input data), and a parameter server.

![Split Inference Web](https://github.com/evanwrm/split-learning-demo/assets/66049888/b79d461b-5178-4280-a5ba-4119e02bf7ff)

We provide a few example implementations based on the following scenarios:

| Type            | Network Communication | Client Framework | Server Framework |
| --------------- | --------------------- | ---------------- | ---------------- |
| Split Inference | Websockets            | Onnx Runtime     | PyTorch          |
| Split Learning  | Websockets            | PyTorch          | PyTorch          |
| Split Learning  | MPI                   | PyTorch          | PyTorch          |

## Installation

The core implementation is in Python (3.11). To install we can create a virtual environment using Conda:

```sh
conda create -f environment.yml
conda activate split-learning-demo
```

If your operating system is not Linux or Windows (64bit), or doesnt have an Nvidia GPU,
remove the line containing `pytorch::pytorch-cuda` from `environment.yml` before creating
the environment.

### Web

To install the webapp:

```sh
cd apps/web
pnpm install
```

## Usage

Demos scripts can be found in the `scripts` directory.

### Server/Client

To run a simple client/server split learning setup:

```sh
python scripts/server.py --learning-rate=0.01
```

In a different terminal:

```sh
python scripts/client.py --learning-rate=0.01
```

After completion, the client will write `apps/web/public/models/client_mnist.onnx`
and the server will write `data/models/server_mnist.onnx`.

### Server/Web

Make sure the webapp is installed (see above)

```sh
python scripts/server.py --learning-rate=0.01
```

In a different terminal:

```sh
cd apps/web
pnpm run dev
```

Navigate to `http://localhost:5173` in your browser. The websocket server will default to `ws://127.0.0.1:8000/ws`.

To perform split inference, change the model to LeNet-5 SplitNN,

The other models do not communicate with the backend.

### MPI

To run the MPI demo with 1 server and 1 client:

```sh
mpirun -n 2 python scripts/mpi.py --learning-rate=0.01
```

## End-to-end smoke tests

`scripts/smoke_test_ws.py` exercises the WebSocket protocol the way the
browser does — useful for catching protocol regressions without a UI:

```sh
python scripts/server.py --learning-rate=0.01 &
python scripts/smoke_test_ws.py
```

It connects, then verifies three round-trips:

- `REQUEST_BATCH` → `BATCH` (a `(128,1,28,28)` MNIST image tensor plus
  `(128,)` int64 labels — the data path the browser-trained client uses).
- `ACTIVATIONS_AND_LABELS` → `GRADS` (the training path the server already
  runs for `scripts/client.py`).
- `ACTIVATIONS` → `LOGITS` (the inference path that powers the drawing
  canvas).

Exits 0 on success, 2 on a shape / loss assertion failure, 3 if it can't
reach the server.

## TODO

-   [x] Add a simple local baseline model for comparisons
-   [x] Add split inference model
-   [ ] Introduce an adversarial attack
-   [ ] Add a serform a simple defence
-   [ ] Show and address SplitNN communication overheads with compression

## Citations

```bibtex
@article{vepakomma2018split,
    title={Split learning for health: Distributed deep learning without sharing raw patient data},
    author={Vepakomma, Praneeth and Gupta, Otkrist and Swedish, Tristan and Raskar, Ramesh},
    journal={arXiv preprint arXiv:1812.00564},
    year={2018}
}
```
