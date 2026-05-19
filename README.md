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

### Ubuntu system packages (CPU-only)

PyTorch and torchvision wheels are self-contained, but they dynamically link
against a few shared libraries that minimal Ubuntu installs (and slim Docker
images) do not ship. To run on CPU without a GPU, install:

```sh
sudo apt install libgomp1 libgl1 libglib2.0-0
```

- `libgomp1` — GNU OpenMP runtime that PyTorch links against for CPU threading.
- `libgl1` — OpenGL loader. torchvision's image ops `dlopen` it at import time;
  without it you get `ImportError: libGL.so.1: cannot open shared object file`.
  On Ubuntu ≤ 22.04 the equivalent (transitional) package was
  `libgl1-mesa-glx`, which was dropped in 23.10+.
- `libglib2.0-0` — pulled in alongside `libgl1` by several image-processing
  paths; cheap to install and avoids a second round-trip if you hit it.

No NVIDIA / CUDA / Mesa-driver packages are needed for CPU-only use.

### Web

To install the webapp:

```sh
cd apps/web
pnpm install
```

## Usage

Demos scripts can be found in the `scripts` directory.

### Fetch the dataset

The server, MPI client, and standalone client all read MNIST from disk —
nothing is downloaded on the fly. The
[Hugging Face Hub CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)
is provisioned by `mise install` (see `mise.toml`); prefetch the dataset
once with:

```sh
python scripts/fetch_data.py            # downloads ylecun/mnist (parquet)
```

The parquet files land under `data/external/mnist/` and are gitignored.
Running any of the training scripts without this step will fail fast with
a "MNIST parquet not found" error rather than silently hitting the
network.

Alternatively run:

```sh
hf download ylecun/mnist --repo-type dataset --local-dir data/external/mnist
```

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
and the server will write `apps/web/public/models/server_mnist.onnx`. Browser-
trained client weights (when invoked via `pnpm run train` from the webapp) land
at `apps/web/public/models/client_mnist.{json,weights.bin}` (TF.js native).

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

### Headless training + e2e digit verification

`pnpm run train-and-verify` runs a full split-learning training pass driven by
a headless browser, then classifies 10 hand-written-style digit images
through the resulting model to prove the trained halves work end-to-end.

Prerequisites (in addition to the usual `pnpm install`):

```sh
cd apps/web
# ~377MB. Install the *full* chromium binary; --no-shell skips the
# headless-shell variant. The full binary is what gets you real hardware
# WebGPU when Playwright runs in headed mode (the default here).
pnpm exec playwright install chromium --no-shell

python ../../scripts/generate_test_images.py   # renders data/test-images/digit-{0..9}.png
python ../../scripts/server.py                 # leave running in another terminal
```

Playwright ships two separate Chromium binaries and the CLI installs
them independently. List what's currently installed with:

```sh
pnpm exec playwright install --list
```

Sample output (paths under `~/.cache/ms-playwright/`):

```
Playwright version: 1.60.0
  Browsers:
    /home/<you>/.cache/ms-playwright/chromium-1223
    /home/<you>/.cache/ms-playwright/chromium_headless_shell-1223
    /home/<you>/.cache/ms-playwright/ffmpeg-1011
```

The directory names tell you which binaries are present (`chromium-*`
for the full binary, `chromium_headless_shell-*` for the headless-shell
variant). The mapping:

| Binary                       | Installed by                                          | Size   | Used when                                          | Hardware GPU? |
| ---------------------------- | ----------------------------------------------------- | ------ | -------------------------------------------------- | ------------- |
| `chromium`                   | `playwright install chromium --no-shell`              | ~377MB | `headless: false` (default here, headed window)    | yes           |
| `chromium-headless-shell`    | `playwright install chromium-headless-shell`          | ~260MB | `headless: true` (`HEADLESS=true pnpm run train`)  | software only |
| both                         | `playwright install chromium` (no flag)               | ~640MB | either mode                                        | mixed         |

If you only install `chromium-headless-shell`, Playwright falls through
to Chrome's bundled SwiftShader software Vulkan when the test runs and
WebGPU training OOMs at the first non-trivial allocation. So the
recommended install command for this repo is the explicit
`--no-shell` form above.

Then, with the Python server running:

```sh
cd apps/web
EPOCHS=25 pnpm run train-and-verify         # default 25 epochs; set EPOCHS=N to override
```

The runner probes `webgpu → webgl → cpu` and uses whichever the browser
can initialise. On a Linux laptop with a working Vulkan driver WebGPU is
picked automatically; otherwise it falls through to WebGL or CPU. Force a
specific backend with `TFJS_BACKEND=cpu` (or `webgl`, `webgpu`) if you
need to.

By default the Playwright config runs Chromium in **headed** mode (a
browser window appears for the duration of training) because Playwright's
`headless: true` uses the cut-down `chromium-headless-shell` binary,
which on Linux falls back to Chrome's bundled SwiftShader (software
Vulkan) regardless of flags or PRIME env vars. Headed mode launches the
full `chromium` binary against your X/Wayland session and gets real
hardware WebGPU. Set `HEADLESS=true pnpm run train` to force the
software-Vulkan headless path (CI / no-display systems only — expect
~10× slower training).

If WebGPU dies mid-training with `vkAllocateMemory failed with
VK_ERROR_OUT_OF_DEVICE_MEMORY` / `[Device] is lost`, the cause is almost
always Chromium running on the wrong GPU (an integrated iGPU with tiny
shared VRAM, or worse, Mesa lavapipe — a software Vulkan rasteriser).
The headless runner now logs the actual adapter on startup:

```
[headless] WebGPU adapter: {"vendor":"nvidia","device":"NVIDIA RTX ...","maxBufferSize":...}
```

If the `device` says `llvmpipe` / `lavapipe` / `SwiftShader`, you're on
software Vulkan — install your GPU vendor's Vulkan driver
(`mesa-vulkan-drivers` for Intel/AMD, the NVIDIA proprietary driver
package on Pop!_OS) and re-run.

On hybrid-graphics laptops (Intel iGPU + NVIDIA dGPU) the Playwright
config already exports `__NV_PRIME_RENDER_OFFLOAD=1`,
`__GLX_VENDOR_LIBRARY_NAME=nvidia`, and `__VK_LAYER_NV_optimus=NVIDIA_only`
to route the browser to the discrete GPU. If you're on an AMD-only or
Intel-only box and the wrong adapter is still being picked, set
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/<your_icd>.json` (e.g.
`radeon_icd.json`, `intel_icd.json` — *not* `lvp_icd.json`, which is
software).

If you've confirmed Chromium is on real hardware but VRAM is still
tight, drop the server batch size:

```sh
python scripts/server.py --batch-size 32
```

The default 128 is fine for CPU / CUDA / WebGL; 32 leaves ~4× more
headroom for WebGPU's heavier intermediate buffers.

**Buffer-pool leak on long runs.** Even on a real hardware GPU, TF.js's
WebGPU backend leaks `GPUBuffer` allocations over thousands of training
ops and eventually trips `createBuffer failed, size N is too large for
the implementation` — typically after a few thousand batches. There is
no public API to flush its pool. The headless runner treats this as a
graceful early-stop: if at least one epoch finished before the crash,
it saves the in-memory model and reports success. One MNIST epoch with
momentum SGD already converges to ~97% test accuracy, so the verifier
still passes.

This:

1. Starts (or reuses) `pnpm run dev` and opens
   `http://localhost:5173/?headless=true&epochs=$EPOCHS&backend=cpu` in headless
   Chromium. The Vue app's auto-runner sets the epoch count, kicks off
   training, awaits completion, then ships the trained TF.js client model to
   the server over the WebSocket.
2. The server writes the trained client to
   `apps/web/public/models/client_mnist.{json,weights.bin}` (TF.js layers
   format) and, on WebSocket disconnect, its own half to
   `apps/web/public/models/server_mnist.onnx`.
3. `scripts/verify-digits.ts` loads both halves in Node
   (`@tensorflow/tfjs-node` + `onnxruntime-node`), classifies each
   `data/test-images/digit-{0..9}.png`, and exits 0 only if all 10 digits
   are predicted correctly. Per-digit logits are printed for debugging.

You can also run the two stages independently: `pnpm run train` (Playwright
only) and `pnpm run verify-digits` (verifier only).

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
