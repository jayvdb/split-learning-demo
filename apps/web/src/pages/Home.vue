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
import { TFJS_BACKENDS, type TfjsBackend } from "@/lib/utils/training";
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
// before training (Start button).
const showTrainingPanel = computed(() => isSplitnnTrain.value);
// Drawing canvas shows for the inference paths (`splitnn` and `local`) and
// for `splitnn-train` once training has produced a usable model.
const showDrawingCanvas = computed(
    () => !isSplitnnTrain.value || training.status === "done"
);
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
watchEffect(() => {
    selectModel(model.value?.path);
    if (!model.value) return;

    if (model.value.type !== "splitnn-train") {
        // Load the inference-only ONNX session for the inference paths.
        // The TF.js training state is left intact in memory so switching
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
                    :key="dataset"
                    label="Model"
                    :options="models[dataset].map(m => ({ value: m.path, label: m.name }))"
                    :selected-option="model ? model.path : models[dataset][0].path"
                    :disabled="isTraining"
                    @change="selectModel"
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
            <TrainingProgress v-if="showTrainingPanel" />
            <div
                v-if="showDrawingCanvas"
                class="flex flex-col items-center justify-center md:flex-row"
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
                            @update:image="saveImage"
                        />
                        <span
                            v-if="!canvasRef?.dirty"
                            class="pointer-events-none absolute select-none font-semibold text-base-content text-opacity-80"
                        >
                            Start drawing here...
                        </span>
                    </div>
                    <div class="flex gap-2">
                        <Button class="mt-2" @click="() => canvasRef?.reset()">Clear</Button>
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
        </section>
    </main>
</template>
