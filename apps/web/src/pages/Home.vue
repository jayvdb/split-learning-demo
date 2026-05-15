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
// We keep the progress panel visible for every state except "done" — that
// includes "idle" so the Start button is reachable, and "error" so the
// user can retry without having to switch models away and back.
const inTrainingPhase = computed(
    () => isSplitnnTrain.value && training.status !== "done"
);

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
            const activations = await training.runInference(input as any);
            sendActivationsForInference(activations as any);
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

    if (model.value.type === "splitnn-train") {
        // The browser-trained client loads its weights via the ORT
        // TrainingSession, not the inference ONNX store. We do NOT auto-start
        // training — TrainingProgress.vue surfaces a "Start training" button
        // so the action is discoverable.
    } else {
        // Drop any in-memory training state when switching to a non-training
        // model so the user can flip back and forth without stale UI.
        if (training.status !== "idle") training.reset();
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
                    label="Backend"
                    :options="
                        Object.entries(executionProviderConfig).map(([k, v]) => ({
                            value: k,
                            label: v.name
                        }))
                    "
                    :selected-option="onnx.sessionBackend"
                    @change="d => onnx.setBackend(d)"
                    class="mb-4"
                />
                <Select
                    label="Model"
                    :options="models[dataset].map(m => ({ value: m.path, label: m.name }))"
                    :selected-option="model ? model.path : models[dataset][0].path"
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
                <Input
                    v-if="isSplitnnTrain"
                    label="Epochs"
                    type="number"
                    :value="String(training.epochs)"
                    @update:value="
                        (v: string) => {
                            const n = Number(v);
                            if (Number.isFinite(n) && n >= 1 && n <= 50) training.epochs = n;
                        }
                    "
                    class="mb-4"
                />
                <Button class="mb-4" @click="() => connect(parameterServer)">Reconnect</Button>
            </div>
            <TrainingProgress v-if="inTrainingPhase" />
            <div v-else class="flex flex-col items-center justify-center md:flex-row">
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
