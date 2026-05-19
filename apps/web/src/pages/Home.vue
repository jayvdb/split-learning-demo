<script setup lang="ts">
import PlotFigure from "@/components/charts/PlotFigure.vue";
import TrainingProgress from "@/components/TrainingProgress.vue";
import DrawingCanvas from "@/components/inputs/DrawingCanvas.vue";
import Button from "@/components/ui/Button.vue";
import Input from "@/components/ui/Input.vue";
import Select from "@/components/ui/Select.vue";
import { models, type ModelConfig } from "@/config/models";
import {
    f32Normalize,
    imageDataRescale,
    imageDataToF32,
    imageDataToGrayscale
} from "@/lib/utils/image";
import { argmax, softmax } from "@/lib/utils/math";
import { deserializeTensor, executionProviderConfig, serializeTensor } from "@/lib/utils/onnx";
import type { ONNXBackend } from "@/lib/utils/onnx";
import { tf, TFJS_BACKENDS, type TfjsBackend } from "@/lib/utils/training";
import { useOnnxStore } from "@/stores/onnx";
import { useTrainingStore } from "@/stores/training";
import { useWebsocketStore } from "@/stores/websocket";
import * as Plot from "@observablehq/plot";
import { Tensor } from "onnxruntime-web";
import { computed, ref, watch, watchEffect } from "vue";

const canvasRef = ref<InstanceType<typeof DrawingCanvas> | null>(null);
const imageData = ref<ImageData | null>(null);
const probabilities = ref<number[]>([]);
const prediction = ref<number | null>(null);

const dataset = ref<"mnist" | "quickdraw">("mnist");
const model = ref<ModelConfig | null>(null);
const parameterServer = ref<string>("ws://127.0.0.1:8000/ws");

const onnx = useOnnxStore();
const websocket = useWebsocketStore();
const training = useTrainingStore();

const isSplitnnTrain = computed(() => model.value?.type === "splitnn-train");
// Panel stays visible the whole time the user is on splitnn-train —
// during training, after training (Re-train / Send-model controls), and
// before training (Start button). It renders *below* the drawing canvas
// so the inference UI never moves.
const showTrainingPanel = computed(() => isSplitnnTrain.value);
const isTraining = computed(
    () => training.status === "training" || training.status === "loading"
);

// Backend dropdown is context-aware: TF.js backends when on splitnn-train,
// ORT execution providers otherwise. Disabled during training so the user
// can't swap the runtime out from under an in-flight loop.
const backendOptions = computed(() => {
    if (isSplitnnTrain.value) {
        return TFJS_BACKENDS.map(b => ({ value: b, label: `TF.js: ${b}` }));
    }
    return Object.entries(executionProviderConfig).map(([k, v]) => ({
        value: k,
        label: v.name
    }));
});
const currentBackend = computed(() =>
    isSplitnnTrain.value ? training.tfjsBackend : onnx.sessionBackend
);
const onBackendChange = (b: string) => {
    if (isSplitnnTrain.value) {
        training.setBackend(b as TfjsBackend);
    } else {
        onnx.setBackend(b as ONNXBackend);
    }
};
const trainedBadge = computed(() => {
    switch (training.status) {
        case "idle":
            return { label: "TF.js: not trained", tone: "neutral" as const };
        case "loading":
            return { label: "TF.js: loading…", tone: "active" as const };
        case "training":
            return {
                label: `TF.js: training (${training.epoch}/${training.epochs} epochs done)`,
                tone: "active" as const
            };
        case "error":
            return { label: "TF.js: training stopped", tone: "error" as const };
        case "done":
            return {
                label: `TF.js: trained ✓ (${training.epoch} epochs${
                    training.loss !== null ? `, loss ${training.loss.toFixed(3)}` : ""
                })`,
                tone: "ok" as const
            };
    }
    return { label: "TF.js: unknown", tone: "neutral" as const };
});

// prediction
const displayPrediction = (output: Tensor) => {
    const outputData = [...output.data].map(Number);
    const proba = softmax(outputData);
    if (!proba.some(v => v !== 0)) return;
    const predicted = argmax(proba);
    probabilities.value = proba;
    prediction.value = predicted;
};
const sendActivationsForInference = (output: Tensor) => {
    // eslint-disable-next-line no-console
    console.info(
        "[inference] sending ACTIVATIONS via ORT — modelType=%s shape=%o",
        model.value?.type,
        output.dims
    );
    const message = {
        type: "activations",
        data: { tensor_shape: output.dims },
        raw: { tensor: serializeTensor(output) }
    };
    const json = JSON.stringify(message);
    const b64 = btoa(json);
    const encoder = new TextEncoder();
    const bytes = encoder.encode(b64);
    websocket.sendMessage(bytes);
};

const sendRawActivationsForInference = (data: Float32Array, shape: number[]) => {
    // eslint-disable-next-line no-console
    console.info(
        "[inference] sending ACTIVATIONS via TF.js — modelType=%s shape=%o",
        model.value?.type,
        shape
    );
    const message = {
        type: "activations",
        data: { tensor_shape: shape },
        raw: { tensor: serializeTensor(new Tensor("float32", data, shape)) }
    };
    const json = JSON.stringify(message);
    const b64 = btoa(json);
    const encoder = new TextEncoder();
    websocket.sendMessage(encoder.encode(b64));
};

watch(
    [imageData, () => onnx.session, () => training.status],
    async () => {
        if (!imageData.value) return;

        // preprocess
        const rescaled = imageDataRescale(imageData.value, 28, 28);
        const grayscale = imageDataToGrayscale(rescaled);
        const f32array = imageDataToF32(grayscale);
        const normalized = f32Normalize(f32array, [0.1307], [0.3081]);
        const input = new Tensor("float32", normalized, [1, 1, 28, 28]);

        const modelType = model.value?.type;
        if (modelType === "local") {
            if (!onnx.session || onnx.modelLoading) return;
            const { output } = await onnx.runModel(input);
            displayPrediction(output);
        } else if (modelType === "splitnn") {
            if (!onnx.session || onnx.modelLoading) return;
            const { output } = await onnx.runModel(input);
            sendActivationsForInference(output);
        } else if (modelType === "splitnn-train") {
            if (training.status !== "done") return;
            const { activations, shape } = await training.runInference(
                normalized,
                [1, 1, 28, 28]
            );
            sendRawActivationsForInference(activations, shape);
        }
    }
);
const saveImage = (data: ImageData) => {
    imageData.value = data;
};

// websocket
const connect = (url: string) => {
    if (websocket.status === "open") websocket.disconnect();
    websocket.connect(url, {
        onMessage: message => {
            const json = atob(message);
            const response = JSON.parse(json);
            if (response.type === "logits") {
                const tensorShape = response.data.tensor_shape;
                const output = deserializeTensor(response.raw.tensor, tensorShape);
                console.log("Received message prediction response...", tensorShape);
                displayPrediction(output);
            }
        }
    });
};
watch(
    [parameterServer],
    () => {
        if (!parameterServer.value || websocket.url === parameterServer.value) return;
        connect(parameterServer.value);
    },
    { immediate: true }
);

// model loading
const selectModel = (path?: string) => {
    const availableModels = models[dataset.value];
    const defaultModel = availableModels[0];
    const validModel = availableModels.find(m => m.path === path);
    const modelConfig = validModel ? validModel : defaultModel;

    model.value = modelConfig;
};
// Headless auto-runner: `?headless=true&epochs=25` lets Playwright drive a
// full training run with no UI clicks. Triggers training.start(), awaits
// completion, then ships the trained model to the server. Surfaces its
// result on window.__headlessDone / __headlessError so the spec can poll.
//
// `backend=auto` (the default) probes webgpu → webgl → cpu and uses
// whichever the running browser actually grants — so a Linux laptop with
// a working Vulkan driver gets WebGPU, headed Chrome falls back to webgl,
// and CPU-only environments use the universal cpu backend.
//
// WebGPU caveat: TF.js training kernels are buffer-heavy and can hit
// VK_ERROR_OUT_OF_DEVICE_MEMORY (device-lost) with the default
// server-side batch size of 128 under headless Chromium's small VRAM
// budget. Run `scripts/server.py --batch-size 32` if you see device-lost
// errors. We also flip `WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE` to 1 below so
// GPU buffers are released between ops instead of every 15.
const headlessParams = new URLSearchParams(window.location.search);
if (headlessParams.get("headless") === "true") {
    const headlessEpochs = Number(headlessParams.get("epochs") ?? "25") || 25;
    const requested = headlessParams.get("backend") ?? "auto";
    const candidates: TfjsBackend[] =
        requested === "auto"
            ? ["webgpu", "webgl", "cpu"]
            : [requested as TfjsBackend];
    (async () => {
        type HeadlessWin = {
            __headlessDone?: boolean;
            __headlessError?: string;
        };
        const w = window as unknown as HeadlessWin;
        try {
            while (websocket.status !== "open") {
                await new Promise(r => setTimeout(r, 50));
            }
            let chosen: TfjsBackend | null = null;
            let lastErr: unknown = null;
            for (const b of candidates) {
                try {
                    if (b === "webgpu") {
                        // Eager flush keeps the WebGPU buffer pool from
                        // growing between submissions on memory-constrained
                        // GPUs. (Don't set WEBGPU_USE_NAIVE_CONV2D_DEBUG —
                        // its WGSL output is broken in TF.js 4.22: emits
                        // `main();;` which Dawn rejects.)
                        tf.env().set("WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE", 1);
                    }
                    await training.setBackend(b);
                    chosen = b;
                    break;
                } catch (e) {
                    lastErr = e;
                    console.warn(
                        "[headless] backend %s unavailable: %s",
                        b,
                        e instanceof Error ? e.message : String(e)
                    );
                }
            }
            if (!chosen) {
                throw lastErr ?? new Error("no TF.js backend available");
            }

            // Diagnostic: print the actual WebGPU adapter the browser
            // granted, plus key memory limits. If you see "llvmpipe" /
            // "lavapipe" / "swiftshader" as the device name, Chromium is
            // running software Vulkan with tiny VRAM — install the real
            // GPU driver (mesa-vulkan-drivers / nvidia-vulkan-icd-loader)
            // and re-run. Real hardware names look like "Intel UHD",
            // "AMD Radeon", "NVIDIA GeForce ...".
            if (chosen === "webgpu") {
                try {
                    const gpu = (
                        navigator as unknown as {
                            gpu?: {
                                requestAdapter: (
                                    opts?: { powerPreference?: string }
                                ) => Promise<{
                                    info?: {
                                        vendor?: string;
                                        device?: string;
                                        architecture?: string;
                                        description?: string;
                                    };
                                    limits?: Record<string, number>;
                                } | null>;
                            };
                        }
                    ).gpu;
                    const adapter = (await gpu?.requestAdapter({
                        powerPreference: "high-performance"
                    })) as {
                        info?: {
                            vendor?: string;
                            device?: string;
                            architecture?: string;
                            description?: string;
                        };
                        limits?: Record<string, number>;
                    } | null;
                    if (adapter) {
                        console.info(
                            "[headless] WebGPU adapter:",
                            JSON.stringify({
                                vendor: adapter.info?.vendor,
                                device: adapter.info?.device,
                                architecture: adapter.info?.architecture,
                                description: adapter.info?.description,
                                maxBufferSize: adapter.limits?.maxBufferSize,
                                maxStorageBufferBindingSize:
                                    adapter.limits?.maxStorageBufferBindingSize,
                                maxComputeWorkgroupStorageSize:
                                    adapter.limits?.maxComputeWorkgroupStorageSize
                            })
                        );
                    } else {
                        console.warn("[headless] requestAdapter returned null");
                    }
                } catch (e) {
                    console.warn("[headless] adapter info probe failed:", e);
                }
            }
            training.epochs = headlessEpochs;
            console.info(
                "[headless] starting training: epochs=%d backend=%s (requested=%s)",
                headlessEpochs,
                chosen,
                requested
            );
            await training.start();
            // Graceful early-stop: TF.js's WebGPU backend leaks GPU
            // buffers over thousands of ops and eventually trips
            // createBuffer/OOM mid-run. There's no public API to flush
            // the pool. If we got at least one full epoch in, the
            // in-memory model is already well-trained — capture it
            // rather than throwing the whole run away. CPU/WebGL runs
            // won't hit this branch.
            if (training.status === "error" && training.epoch >= 1) {
                console.warn(
                    "[headless] training stopped at epoch %d/%d (%s) — saving the partial model anyway",
                    training.epoch,
                    headlessEpochs,
                    training.error ?? "(no message)"
                );
            } else if (training.status !== "done") {
                throw new Error(
                    `training ended status=${training.status} error=${training.error ?? "(none)"}`
                );
            }
            await training.saveModelToServer();
            console.info("[headless] saved model; marking done");
            w.__headlessDone = true;
        } catch (e) {
            const msg = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
            console.error("[headless] failed:", msg);
            w.__headlessError = msg;
            w.__headlessDone = true;
        }
    })();
}

watchEffect(() => {
    selectModel(model.value?.path);
    if (!model.value) return;

    if (model.value.type !== "splitnn-train") {
        // If training was active but the selected model isn't splitnn-train
        // (can only happen via a Vite HMR — Pinia preserves store state
        // across reloads, but Home.vue's `model` ref resets), the training
        // loop is orphaned. Force a clean reset so the dropdowns unlock
        // and the page is consistent again.
        if (training.status === "training" || training.status === "loading") {
            // eslint-disable-next-line no-console
            console.warn(
                "[home] orphaned training detected after non-splitnn-train selection; resetting"
            );
            training.reset();
        }
        // Load the inference-only ONNX session for the inference paths.
        // The TF.js training state is otherwise left intact so switching
        // back to splitnn-train lands on the trained model.
        onnx.loadModel(model.value.path);
    }
});
</script>

<template>
    <main class="flex h-full w-full flex-1 flex-col items-center justify-center overflow-hidden">
        <section
            class="flex h-full w-full flex-1 flex-col items-center justify-center overflow-hidden px-4"
        >
            <div class="mt-6 flex flex-col items-center justify-center gap-4 md:mt-2 md:flex-row">
                <!-- Model first — the available Backend options depend on
                     which Model is selected (ORT execution providers for
                     ORT models, TF.js backends for splitnn-train). -->
                <Select
                    :key="dataset"
                    label="Model"
                    :options="models[dataset].map(m => ({ value: m.path, label: m.name }))"
                    :selected-option="model ? model.path : models[dataset][0].path"
                    :disabled="isTraining"
                    @change="selectModel"
                    class="mb-4"
                />
                <Select
                    :key="isSplitnnTrain ? 'tfjs' : 'ort'"
                    label="Backend"
                    :options="backendOptions"
                    :selected-option="currentBackend"
                    :disabled="isTraining"
                    @change="onBackendChange"
                    class="mb-4"
                />
                <Select
                    label="Dataset"
                    :options="[
                        { value: 'mnist', label: 'MNIST' },
                        { value: 'quickdraw', label: 'QuickDraw' }
                    ]"
                    :selected-option="dataset"
                    :disabled="isTraining || isSplitnnTrain"
                    @change="d => (dataset = d)"
                    class="mb-4"
                />
                <Input
                    label="Parameter Server"
                    placeholder="Enter server URL"
                    :error="websocket.status === 'closed' ? 'Connection failed' : ''"
                    v-model:value="parameterServer"
                    class="mb-4"
                />
                <Button class="mb-4" @click="() => connect(parameterServer)">Reconnect</Button>
            </div>
            <!-- Permanent TF.js trained-state indicator, only when the
                 training panel itself isn't already on screen (otherwise
                 it just duplicates the panel's epoch counter). -->
            <div v-if="!showTrainingPanel" class="mb-3 flex items-center justify-center">
                <span
                    class="rounded-full border px-3 py-1 text-xs font-semibold"
                    :class="{
                        'border-base-300 text-base-content text-opacity-70':
                            trainedBadge.tone === 'neutral',
                        'border-primary text-primary': trainedBadge.tone === 'active',
                        'border-success text-success': trainedBadge.tone === 'ok',
                        'border-error text-error': trainedBadge.tone === 'error'
                    }"
                >
                    {{ trainedBadge.label }}
                </span>
            </div>
            <!-- Inference UI is always rendered in the same spot. When
                 training is running we dim it and block strokes via the
                 DrawingCanvas `disabled` prop; the user always sees the
                 canvas, predictions stay in place, and switching modes
                 doesn't cause the page to reflow. -->
            <div
                class="flex flex-col items-center justify-center transition-opacity md:flex-row"
                :class="{
                    'pointer-events-none opacity-50': isTraining
                }"
                :aria-disabled="isTraining"
            >
                <div class="flex flex-col items-start justify-end">
                    <div class="flex items-center justify-center">
                        <DrawingCanvas
                            ref="canvasRef"
                            class="rounded-md border border-base-300 outline-none ring-base-300 ring-offset-0 transition"
                            :class="{
                                'ring-2 ring-secondary ring-offset-2 ring-offset-base-100':
                                    canvasRef?.dirty
                            }"
                            :width="400"
                            :height="400"
                            :line-width="20"
                            stroke="currentColor"
                            background-color="#ffffff00"
                            save-as="data"
                            :disabled="isTraining"
                            @update:image="saveImage"
                        />
                        <span
                            v-if="!canvasRef?.dirty"
                            class="pointer-events-none absolute select-none font-semibold text-base-content text-opacity-80"
                        >
                            {{
                                isTraining
                                    ? "Training in progress…"
                                    : "Start drawing here..."
                            }}
                        </span>
                    </div>
                    <div class="flex gap-2">
                        <Button
                            class="mt-2"
                            :disabled="isTraining"
                            @click="() => canvasRef?.reset()"
                        >
                            Clear
                        </Button>
                    </div>
                </div>
                <div class="flex flex-col items-end justify-center">
                    <span class="text-5xl font-bold">
                        {{ prediction }}
                    </span>
                    <PlotFigure
                        :options="{
                            style: { background: 'transparent' },
                            marks: [Plot.barX(probabilities), Plot.ruleX([0, 1], { opacity: 0 })]
                        }"
                    />
                </div>
            </div>
            <TrainingProgress v-if="showTrainingPanel" class="mt-4" />
        </section>
    </main>
</template>
