<script setup lang="ts">
import PlotFigure from "@/components/charts/PlotFigure.vue";
import { useTrainingStore } from "@/stores/training";
import * as Plot from "@observablehq/plot";
import { computed } from "vue";

const training = useTrainingStore();

const progressPct = computed(() => {
    if (training.batchesPerEpoch === 0) return 0;
    const totalBatches = training.epochs * training.batchesPerEpoch;
    const done = training.epoch * training.batchesPerEpoch + training.batch;
    return Math.min(100, Math.round((100 * done) / totalBatches));
});

const formattedLoss = computed(() =>
    training.loss === null ? "—" : training.loss.toFixed(4)
);

const canStart = computed(
    () => training.status === "idle" || training.status === "error"
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
</script>

<template>
    <section
        class="flex w-full max-w-xl flex-col gap-3 rounded-lg border border-base-300 bg-base-100 p-4"
    >
        <header class="flex items-center justify-between">
            <h2 class="text-lg font-semibold">Training split client</h2>
            <span class="text-xs text-base-content text-opacity-70">
                {{ training.status }}
            </span>
        </header>

        <button
            type="button"
            class="self-start rounded-lg border border-base-300 bg-primary px-4 py-2 font-semibold text-primary-content transition hover:scale-95 disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:scale-100"
            :disabled="!canStart"
            @click="onStart"
        >
            {{ startLabel }}
        </button>

        <div v-if="training.error" class="text-sm text-error">
            {{ training.error }}
        </div>

        <div class="grid grid-cols-3 gap-2 text-sm">
            <div>
                <div class="text-xs text-base-content text-opacity-60">Epoch</div>
                <div class="font-semibold">
                    {{ training.epoch + 1 }} / {{ training.epochs }}
                </div>
            </div>
            <div>
                <div class="text-xs text-base-content text-opacity-60">Batch</div>
                <div class="font-semibold">
                    {{ training.batch }}<span
                        v-if="training.batchesPerEpoch"
                    >
                        / {{ training.batchesPerEpoch }}</span
                    >
                </div>
            </div>
            <div>
                <div class="text-xs text-base-content text-opacity-60">Loss</div>
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
