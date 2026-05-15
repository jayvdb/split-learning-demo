# Why ORT Web Training didn't work for this demo

This file records the attempt to use **ONNX Runtime Training (Web)** —
`onnxruntime-web/training` — for the in-browser half of split learning.
The attempt is preserved in git history; this is the autopsy so the next
person doesn't repeat the same dead-end.

## What we wanted

Run the `CNN2DClient` half (`Conv → ReLU → MaxPool → Conv → ReLU →
MaxPool`, input `(B, 1, 28, 28)` → activations `(B, 16, 7, 7)`) in the
browser:

- Forward to compute activations.
- Send activations + labels to the Python server over a WebSocket
  (`ACTIVATIONS_AND_LABELS`).
- Receive the gradient w.r.t. activations back (`GRADS`) — same wire
  protocol the existing `scripts/client.py` uses.
- Backprop those upstream gradients through the client params and apply
  an SGD step.

The Python `scripts/client.py` does exactly this via
`torch.autograd.backward(activations, grads)`. We wanted parity in JS,
auto-started from a dropdown entry, no user input needed.

## What we tried

`scripts/generate_training_artifacts.py` builds the four ORT-Training
artifact files (`training_model.onnx`, `eval_model.onnx`,
`optimizer_model.onnx`, `checkpoint`) using
`onnxruntime.training.artifacts.generate_artifacts` from
`onnxruntime-training-cpu==1.19.2`. The key tricks:

1. **`upstream_signal` as an explicit graph input.** ORT-web's wasm
   runtime drops `_grad`-suffixed inputs from its
   "user-input" list, so we named the upstream-gradient input
   `upstream_signal` (not `upstream_grad`) and appended it to the
   forward model's `graph.input` directly. The TorchScript exporter
   elides unused module args, so a Python `def forward(image,
   upstream_grad)` wrapper wouldn't have carried it through.

2. **Custom `DotProductLoss` block** that emits
   `loss = ReduceSum(Mul(activations, upstream_signal))`. This makes
   the chain-rule trick the actual loss the autograd machinery
   differentiates, so `d(loss)/d(params) = d(activations)/d(params) ·
   upstream_signal`. Wired in via `generate_artifacts(...,
   loss=DotProductLoss(), loss_input_names=["activations",
   "upstream_signal"])`.

3. **`additional_output_names=["activations"]`** so `runEvalStep`
   returns the activations alongside the scalar loss — we need them
   to send over the WebSocket.

4. **`ir_version = 10` post-pin** on the emitted models. `onnx>=1.19`
   stamps `ir_version=13`, but `onnxruntime-training-cpu==1.19.2`
   refuses to load anything above 10. ORT-web 1.19 has the same cap.

5. **COOP / COEP headers** on the Vite dev server (
   `Cross-Origin-Embedder-Policy: require-corp`,
   `Cross-Origin-Opener-Policy: same-origin`). The training wasm
   variant is `ort-training-wasm-simd-threaded.wasm` — threaded only
   — which requires `SharedArrayBuffer`, which requires
   cross-origin isolation.

6. **Vite alias + `optimizeDeps.exclude`** so the bare
   `onnxruntime-web` import resolves to its ESM bundle and the
   pre-bundler doesn't choke on the runtime-fetched
   `ort-wasm-simd-threaded.mjs` worker.

The Python smoke test in `scripts/generate_training_artifacts.py` —
load all four artifacts under `onnxruntime.training.api.Module` and
`Optimizer`, run one train+optimizer step on zero inputs — passes
green.

## The wall

In the browser, `TrainingSession.create(...)` throws an opaque emscripten
exception pointer (e.g. `wasm exception (pointer 10250368)`) the moment
it tries to load `optimizer_model.onnx`. The training_model and
eval_model both load fine; the failure is specifically the second
internal session-create (visible as a second `Using global/env
threadpools` log line in the ORT verbose output, immediately followed by
the exception).

The optimizer model uses `com.microsoft.SGDOptimizerV2`. That's a
contrib op stripped from the ORT-web wasm build for size reasons.
`onnxblock.optim.SGD` always emits V2, and `artifacts.generate_artifacts`
only ships built-in SGD and AdamW — AdamW is `com.microsoft.AdamWOptimizer`,
also contrib, so it has the same problem.

We confirmed this by retrying `TrainingSession.create` without
`optimizerModel`:

- Session loads.
- `runEvalStep({image, upstream_signal: zeros})` returns the
  activations correctly.
- `runTrainStep({image, upstream_signal: realGrads})` runs forward +
  loss + backward, accumulates gradients into the internal grad-buffers.
- `runOptimizerStep()` throws
  `"This TrainingSession has no OptimizerModel loaded."` — the JS
  guard, not a wasm fault.

So forward and backward both work end-to-end; only the param-update
step is missing.

## Why we couldn't work around the missing optimizer

The obvious workaround would be to skip the optimizer model and apply
SGD by hand:

1. Run `runTrainStep` → gradients populate the internal accumulator
   buffers.
2. Read the gradients out.
3. Compute `param -= learning_rate * grad` in JS.
4. Push the new params back with `loadParametersBuffer`.

Step 2 is the blocker. ORT-web's `TrainingSession` exposes:

- `getContiguousParameters(trainableOnly)` — **parameters**, not gradients.
- `loadParametersBuffer(buffer, trainableOnly)` — parameter write-side.
- `getParametersSize(trainableOnly)`.

There is **no public API to read the gradient accumulator buffers**.
They're owned by the `CheckpointState`, populated in-place by
`InPlaceAccumulatorV2` nodes inside the training model, and not
surfaced as model outputs (the gradient-related outputs are just `BOOL`
"accumulator updated" flags). So the manual-SGD path is closed.

A theoretical alternative — write a custom optimizer block in `onnxblock`
that uses only standard `ai.onnx` ops on the param/grad sequences — is
speculative. ORT-web's `runOptimizerStep` has internal logic to bind the
sequence inputs from the CheckpointState, and we don't know whether it'd
do that for a user-defined graph the same way it does for the built-in
SGD/AdamW shapes. Not worth the next round-trip.

## Decision

The next reasonable step is the **TensorFlow.js fallback** that was
already flagged in `IDEAS.md`:

> *TensorFlow.js — most mature path for in-browser training; means
> re-expressing CNN2DClient as a tfjs model. The two halves live in
> different frameworks anyway, so this isn't worse than today.*

The plan:

- Re-express `CNN2DClient` as a `tf.sequential` model of two
  `Conv2d → ReLU → MaxPool` blocks.
- Use `tf.variableGrads(...)` (or equivalent) so we can pass the
  server-supplied upstream gradient as the `dy` for the activations
  tensor, then apply via `tf.train.sgd(0.01).applyGradients(...)`.
- Keep the protocol, the WebSocket flow, the `LeNet-5 SplitNN (Train
  in Browser)` dropdown entry, the `<TrainingProgress>` panel, and the
  "Start training" button.
- Drop `scripts/generate_training_artifacts.py`, the
  `apps/web/public/models/training/` directory, the Vite alias /
  `optimizeDeps.exclude` for `onnxruntime-web/training`, and the
  COOP/COEP plugin if nothing else needs it.

## Files that exist today because of this dead end

If you're cleaning up after, these are the pieces that exist solely
for the ORT-web Training attempt and can go:

- `scripts/generate_training_artifacts.py`
- `apps/web/public/models/training/` (four artifact files)
- `apps/web/src/lib/utils/training.ts`'s ORT-specific bits (the file
  itself can stay; just swap its impl)
- The `onnxruntime-training-cpu` dependency in `environment.yml`
- The `ort-*.mjs` rule in `vite.config.ts`'s `viteStaticCopy`
- The `crossOriginIsolation` plugin in `vite.config.ts`
- The `onnxruntime-web` alias + `optimizeDeps.exclude` entries in
  `vite.config.ts`

The Pinia training store, `TrainingProgress.vue`, the splitnn-train
model entry, the `REQUEST_BATCH` / `BATCH` protocol additions, and
`scripts/smoke_test_ws.py` are all framework-agnostic and survive the
switch.
