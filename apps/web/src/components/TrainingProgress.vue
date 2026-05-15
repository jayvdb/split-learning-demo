<script setup lang="ts">
import PlotFigure from "@/components/charts/PlotFigure.vue";
import Input from "@/components/ui/Input.vue";
import { useTrainingStore } from "@/stores/training";
import * as Plot from "@observablehq/plot";
import { computed } from "vue";

const training = useTrainingStore();

const epochsDone = computed(() => training.epoch);
const totalBatches = computed(() => training.epochs * (training.batchesPerEpoch || 0));
const batchesDone = computed(
    () => training.epoch * (training.batchesPerEpoch || 0) + training.batch
);
const progressPct = computed(() => {
    if (totalBatches.value === 0) return 0;
    return Math.min(100, Math.round((100 * batchesDone.value) / totalBatches.value));
});

const formattedLoss = computed(() =>
    training.loss === null ? "—" : training.loss.toFixed(4)
);

const canStart = computed(
    () => training.status === "idle" || training.status === "error"
);
// Lock both widgets the moment the user clicks Start — covers both the
// brief "loading" stage and the actual "training" loop. They re-open on
// done / error / idle.
const widgetsEditable = computed(
    () => training.status !== "training" && training.status !== "loading"
);

const startLabel = computed(() => {
    switch (training.status) {
        case "idle":
            return "Start training";
        case "loading":
            return "Loading model…";
        case "training":
            return "Training…";
        case "error":
            return "Try again";
        case "done":
            return "Restart training";
    }
    return "Start training";
});

const onStart = () => {
    if (training.status === "error") training.reset();
    training.start();
};

const onEpochs = (v: string) => {
    const n = Number(v);
    if (Number.isFinite(n) && n >= 1 && n <= 200) training.epochs = Math.floor(n);
};
const onLearningRate = (v: string) => {
    const n = Number(v);
    if (Number.isFinite(n) && n > 0 && n <= 1) training.learningRate = n;
};
</script>

<template>
    <section
        class="flex w-full max-w-xl flex-col gap-3 rounded-lg border border-base-300 bg-base-100 p-4"
    >
        <header class="flex items-center justify-between">
            <h2 class="text-lg font-semibold">Training split model</h2>
            <span class="text-xs text-base-content text-opacity-70">
                {{ training.status }}
            </span>
        </header>

        <div class="mt-3 flex flex-wrap items-end gap-4">
            <Input
                label="Learning rate"
                type="number"
                :value="String(training.learningRate)"
                :disabled="!widgetsEditable"
                @update:value="onLearningRate"
            />
            <button
                type="button"
                class="rounded-lg border border-base-300 bg-primary px-4 py-2 font-semibold text-primary-content transition hover:scale-95 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:scale-100"
                :disabled="!canStart"
                @click="onStart"
            >
                {{ startLabel }}
            </button>
        </div>
        <p class="text-xs text-base-content text-opacity-60">
            Set <em>Learning rate</em> to match the server's
            <code>--learning-rate</code> CLI flag (default
            <code>1e-4</code>).
        </p>

        <div v-if="training.error" class="text-sm text-error">
            {{ training.error }}
        </div>

        <div class="grid grid-cols-3 gap-2 text-sm">
            <div>
                <div class="text-xs text-base-content text-opacity-60">Epochs done</div>
                <div class="flex items-baseline gap-1 font-semibold">
                    <span>{{ epochsDone }} /</span>
                    <input
                        type="number"
                        min="1"
                        max="200"
                        :value="training.epochs"
                        :disabled="!widgetsEditable"
                        @input="
                            (e: Event) =>
                                onEpochs((e.target as HTMLInputElement).value)
                        "
                        class="w-16 rounded border border-base-300 bg-base-100 px-1 py-0.5 font-semibold disabled:border-transparent disabled:bg-transparent disabled:opacity-100"
                    />
                </div>
            </div>
            <div>
                <div class="text-xs text-base-content text-opacity-60">
                    Batch in current epoch
                </div>
                <div class="font-semibold">
                    {{ training.batch
                    }}<span v-if="training.batchesPerEpoch">
                        / {{ training.batchesPerEpoch }}</span
                    >
                </div>
            </div>
            <div>
                <div class="text-xs text-base-content text-opacity-60">Server loss</div>
                <div class="font-semibold">{{ formattedLoss }}</div>
            </div>
        </div>

        <div class="h-2 w-full overflow-hidden rounded bg-base-200">
            <div
                class="h-full bg-primary transition-all"
                :style="{ width: `${progressPct}%` }"
            ></div>
        </div>

        <PlotFigure
            v-if="training.lossHistory.length > 1"
            :options="{
                width: 480,
                height: 160,
                style: { background: 'transparent' },
                y: { grid: true, label: 'loss' },
                x: { label: 'step' },
                marks: [Plot.lineY(training.lossHistory)]
            }"
        />
    </section>
</template>
