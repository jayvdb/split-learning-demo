import { arrayToB64, b64ToArray } from "@/lib/utils/serde";
import {
    createTrainingSession,
    defaultArtifactPaths,
    zerosLikeActivations,
    Tensor
} from "@/lib/utils/training";
import { useWebsocketStore } from "@/stores/websocket";
import type { TrainingSession } from "onnxruntime-web/training";
import { defineStore } from "pinia";
import { ref } from "vue";

type Status = "idle" | "loading" | "training" | "done" | "error";

const describeError = (e: unknown): string => {
    // ORT's wasm runtime sometimes throws raw numbers (emscripten exception
    // pointers) or strings, not Error objects. Extract anything we can —
    // and always dump the original to the console so the stack survives.
    // eslint-disable-next-line no-console
    console.error("[training] error:", e);
    if (e instanceof Error) {
        try {
            // eslint-disable-next-line no-console
            console.error("[training] stack:", e.stack);
        } catch {
            /* ignore */
        }
        return `${e.name}: ${e.message}`;
    }
    if (typeof e === "number") {
        return `wasm exception (pointer ${e}) — check the browser console for ORT runtime output`;
    }
    if (typeof e === "string") return e;
    try {
        return `${typeof e}: ${JSON.stringify(e)}`;
    } catch {
        return String(e);
    }
};

const decodeEnvelope = (raw: string): { type: string; data: any; raw: Record<string, string> } => {
    const json = atob(raw);
    return JSON.parse(json);
};

const encodeEnvelope = (envelope: {
    type: string;
    data: Record<string, unknown>;
    raw: Record<string, string>;
}) => {
    const json = JSON.stringify(envelope);
    const b64 = btoa(json);
    return new TextEncoder().encode(b64);
};

const tensorToB64 = (tensor: Tensor) =>
    arrayToB64(tensor.data as Float32Array | BigInt64Array);

const waitFor = <T>(
    subscribe: (cb: (data: string) => void) => () => boolean,
    matcher: (envelope: ReturnType<typeof decodeEnvelope>) => T | undefined
): Promise<T> =>
    new Promise<T>(resolve => {
        const unsubscribe = subscribe(raw => {
            try {
                const envelope = decodeEnvelope(raw);
                const value = matcher(envelope);
                if (value !== undefined) {
                    unsubscribe();
                    resolve(value);
                }
            } catch {
                /* not a JSON envelope; ignore */
            }
        });
    });

export const useTrainingStore = defineStore("training", () => {
    const websocket = useWebsocketStore();

    const session = ref<TrainingSession | null>(null);
    const status = ref<Status>("idle");
    const error = ref<string | null>(null);

    const epochs = ref(3);
    const epoch = ref(0);
    const batch = ref(0);
    const batchesPerEpoch = ref(0); // discovered on first batch
    const loss = ref<number | null>(null);
    const lossHistory = ref<number[]>([]);

    const ensureSession = async () => {
        if (session.value) return session.value;
        status.value = "loading";
        // eslint-disable-next-line no-console
        console.info("[training] loading ORT TrainingSession from %o", defaultArtifactPaths());
        try {
            session.value = await createTrainingSession(defaultArtifactPaths());
            // eslint-disable-next-line no-console
            console.info(
                "[training] session ready. trainingInputNames=%o evalInputNames=%o",
                session.value.trainingInputNames,
                session.value.evalInputNames
            );
            return session.value;
        } catch (e) {
            status.value = "error";
            error.value = `Failed to load training session: ${describeError(e)}`;
            throw e;
        }
    };

    const requestBatch = async (): Promise<{
        images: Tensor;
        labels: Tensor;
    }> => {
        const subscribe = websocket.subscribe;
        websocket.sendMessage(
            encodeEnvelope({ type: "request_batch", data: {}, raw: {} })
        );
        const { images, labels } = await waitFor(subscribe, envelope => {
            if (envelope.type !== "batch") return undefined;
            const imagesShape = envelope.data.images_shape as number[];
            const labelsShape = envelope.data.labels_shape as number[];
            const images = new Tensor(
                "float32",
                b64ToArray(envelope.raw.images, "float32") as Float32Array,
                imagesShape
            );
            const labels = new Tensor(
                "int64",
                b64ToArray(envelope.raw.labels, "int64") as BigInt64Array,
                labelsShape
            );
            return { images, labels };
        });
        return { images, labels };
    };

    const sendActivationsAndAwaitGrads = async (
        activations: Tensor,
        labels: Tensor
    ): Promise<{ upstreamGrad: Tensor; loss: number }> => {
        websocket.sendMessage(
            encodeEnvelope({
                type: "activations_and_labels",
                data: { tensor_shape: activations.dims },
                raw: {
                    tensor: tensorToB64(activations),
                    labels: tensorToB64(labels)
                }
            })
        );

        const result = await waitFor(websocket.subscribe, envelope => {
            if (envelope.type !== "grads") return undefined;
            const shape = envelope.data.tensor_shape as number[];
            const upstreamGrad = new Tensor(
                "float32",
                b64ToArray(envelope.raw.tensor, "float32") as Float32Array,
                shape
            );
            const lossValue = Number(envelope.data.loss);
            return { upstreamGrad, loss: lossValue };
        });

        return result;
    };

    const start = async () => {
        if (status.value === "training") return;
        let phase: string = "init";
        try {
            const s = await ensureSession();
            status.value = "training";
            error.value = null;
            epoch.value = 0;
            batch.value = 0;
            lossHistory.value = [];

            while (epoch.value < epochs.value) {
                phase = "request_batch";
                const { images, labels } = await requestBatch();
                const batchSize = images.dims[0];

                phase = "lazy_reset_grad";
                await s.lazyResetGrad();

                // Forward only — we need the activations to ship to the server.
                phase = "run_eval_step";
                const evalOut = await s.runEvalStep({
                    image: images,
                    upstream_signal: zerosLikeActivations(batchSize)
                });
                const activations = evalOut["activations"] as Tensor;
                if (!activations) {
                    throw new Error(
                        `eval_model produced no "activations" output. Outputs: ${Object.keys(
                            evalOut
                        ).join(", ")}`
                    );
                }

                phase = "send_activations_and_labels";
                const { upstreamGrad, loss: lossValue } =
                    await sendActivationsAndAwaitGrads(activations, labels);

                // Backward + optimizer using the real upstream gradient.
                phase = "run_train_step";
                await s.runTrainStep({
                    image: images,
                    upstream_signal: upstreamGrad
                });
                phase = "run_optimizer_step";
                await s.runOptimizerStep();

                loss.value = lossValue;
                lossHistory.value.push(lossValue);
                batch.value += 1;

                // We don't know the true MNIST batches-per-epoch until the
                // server tells us via the labels' shape — but since the
                // server's MNIST loader is fixed at 60k samples and
                // batch_size=128, that's 469 batches. We track for display.
                if (batchesPerEpoch.value === 0) {
                    batchesPerEpoch.value = Math.ceil(60000 / batchSize);
                }
                if (batch.value >= batchesPerEpoch.value) {
                    epoch.value += 1;
                    batch.value = 0;
                }
            }

            status.value = "done";
        } catch (e) {
            status.value = "error";
            // eslint-disable-next-line no-console
            console.error(
                "[training] failed during phase=%s epoch=%d batch=%d",
                phase,
                epoch.value,
                batch.value
            );
            error.value = `phase=${phase} epoch=${epoch.value} batch=${batch.value}: ${describeError(
                e
            )}`;
        }
    };

    const runInference = async (input: Tensor): Promise<Tensor> => {
        const s = await ensureSession();
        const batchSize = input.dims[0];
        const evalOut = await s.runEvalStep({
            image: input,
            upstream_signal: zerosLikeActivations(batchSize)
        });
        const activations = evalOut["activations"] as Tensor;
        if (!activations) {
            throw new Error(
                `eval_model produced no "activations" output. Outputs: ${Object.keys(
                    evalOut
                ).join(", ")}`
            );
        }
        return activations;
    };

    const reset = () => {
        session.value = null;
        status.value = "idle";
        error.value = null;
        epoch.value = 0;
        batch.value = 0;
        loss.value = null;
        lossHistory.value = [];
    };

    return {
        session,
        status,
        error,
        epochs,
        epoch,
        batch,
        batchesPerEpoch,
        loss,
        lossHistory,
        start,
        runInference,
        reset
    };
});
