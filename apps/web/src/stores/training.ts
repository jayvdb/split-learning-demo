import { arrayToB64, b64ToArray } from "@/lib/utils/serde";
import { createClientModel, tensorFromFloat32, tf, type ClientModel } from "@/lib/utils/training";
import { useWebsocketStore } from "@/stores/websocket";
import { defineStore } from "pinia";
import { ref, watch } from "vue";

type Status = "idle" | "loading" | "training" | "done" | "error";

const describeError = (e: unknown): string => {
    // eslint-disable-next-line no-console
    console.error("[training] error:", e);
    if (e instanceof Error) {
        // eslint-disable-next-line no-console
        try {
            console.error("[training] stack:", e.stack);
        } catch {
            /* ignore */
        }
        return `${e.name}: ${e.message}`;
    }
    if (typeof e === "string") return e;
    try {
        return `${typeof e}: ${JSON.stringify(e)}`;
    } catch {
        return String(e);
    }
};

interface Envelope {
    type: string;
    data: Record<string, unknown>;
    raw: Record<string, string>;
}

const decodeEnvelope = (raw: string): Envelope => {
    const json = atob(raw);
    return JSON.parse(json);
};

const encodeEnvelope = (envelope: Envelope) => {
    const json = JSON.stringify(envelope);
    return new TextEncoder().encode(btoa(json));
};

const waitFor = <T>(
    subscribe: (cb: (data: string) => void) => () => boolean,
    matcher: (envelope: Envelope) => T | undefined,
    signal: AbortSignal
): Promise<T> =>
    new Promise<T>((resolve, reject) => {
        if (signal.aborted) {
            reject(new Error("aborted"));
            return;
        }
        const unsubscribe = subscribe(raw => {
            try {
                const envelope = decodeEnvelope(raw);
                const value = matcher(envelope);
                if (value !== undefined) {
                    unsubscribe();
                    signal.removeEventListener("abort", onAbort);
                    resolve(value);
                }
            } catch {
                /* not a JSON envelope; ignore */
            }
        });
        const onAbort = () => {
            unsubscribe();
            signal.removeEventListener("abort", onAbort);
            reject(new Error("aborted"));
        };
        signal.addEventListener("abort", onAbort);
    });

const float32FromB64 = (b64: string) => b64ToArray(b64, "float32") as Float32Array;
const int64FromB64 = (b64: string) => b64ToArray(b64, "int64") as BigInt64Array;
const float32ToB64 = (a: Float32Array) => arrayToB64(a);
const int64ToB64 = (a: BigInt64Array) => arrayToB64(a);

const meanAbs = (arr: Float32Array): number => {
    if (arr.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += Math.abs(arr[i]);
    return sum / arr.length;
};

export const useTrainingStore = defineStore("training", () => {
    const websocket = useWebsocketStore();

    const client = ref<ClientModel | null>(null);
    const status = ref<Status>("idle");
    const error = ref<string | null>(null);

    const epochs = ref(3);
    const epoch = ref(0);
    const batch = ref(0);
    const batchesPerEpoch = ref(0);
    const loss = ref<number | null>(null);
    const lossHistory = ref<number[]>([]);
    // Client-side SGD learning rate. The user is expected to set this so it
    // matches `scripts/server.py --learning-rate`; we don't forward it
    // over the protocol because the server's optimizer is fixed at process
    // start.
    const learningRate = ref(0.01);

    const ensureModel = (): ClientModel => {
        if (!client.value) {
            status.value = "loading";
            // eslint-disable-next-line no-console
            console.info("[training] building TF.js client model (backend=%s)", tf.getBackend());
            client.value = createClientModel(learningRate.value);
        }
        return client.value;
    };

    // Live-update the optimizer's LR when the user edits the widget. Takes
    // effect on the next train step; no need to dispose the model.
    watch(learningRate, lr => {
        if (client.value) client.value.setLearningRate(lr);
    });

    // Each `start()` owns an AbortController so a WebSocket drop (or a
    // user-triggered reset) can cancel any in-flight `waitFor` and let the
    // loop exit cleanly instead of hanging forever.
    let abortController: AbortController | null = null;

    // If the WebSocket drops while training is running, surface it as an
    // error — the in-flight `waitFor` would otherwise hang forever and the
    // panel would just sit there spinning with no explanation. The
    // `inTrainingPhase` derivation in Home.vue includes the "error" state,
    // so the panel stays visible.
    watch(
        () => websocket.status,
        wsStatus => {
            if (wsStatus === "closed" && status.value === "training") {
                // eslint-disable-next-line no-console
                console.warn("[training] WS dropped mid-training; halting loop");
                abortController?.abort();
            }
        }
    );

    const requestBatch = async (
        signal: AbortSignal
    ): Promise<{
        images: Float32Array;
        imagesShape: number[];
        labels: BigInt64Array;
    }> => {
        websocket.sendMessage(
            encodeEnvelope({ type: "request_batch", data: {}, raw: {} })
        );
        return await waitFor(
            websocket.subscribe,
            envelope => {
                if (envelope.type !== "batch") return undefined;
                const imagesShape = envelope.data.images_shape as number[];
                return {
                    images: float32FromB64(envelope.raw.images),
                    imagesShape,
                    labels: int64FromB64(envelope.raw.labels)
                };
            },
            signal
        );
    };

    const sendActivationsAndAwaitGrads = async (
        activationsNchw: Float32Array,
        activationsShape: number[],
        labels: BigInt64Array,
        signal: AbortSignal
    ): Promise<{ upstreamGrad: Float32Array; loss: number }> => {
        websocket.sendMessage(
            encodeEnvelope({
                type: "activations_and_labels",
                data: { tensor_shape: activationsShape },
                raw: {
                    tensor: float32ToB64(activationsNchw),
                    labels: int64ToB64(labels)
                }
            })
        );
        return await waitFor(
            websocket.subscribe,
            envelope => {
                if (envelope.type !== "grads") return undefined;
                return {
                    upstreamGrad: float32FromB64(envelope.raw.tensor),
                    loss: Number(envelope.data.loss)
                };
            },
            signal
        );
    };

    const start = async () => {
        if (status.value === "training") return;
        let phase = "init";
        abortController?.abort(); // belt+braces: cancel anything stale
        abortController = new AbortController();
        const { signal } = abortController;
        try {
            const model = ensureModel();
            status.value = "training";
            error.value = null;
            epoch.value = 0;
            batch.value = 0;
            lossHistory.value = [];

            while (epoch.value < epochs.value) {
                if (signal.aborted) throw new Error("aborted");
                phase = "request_batch";
                const { images, imagesShape, labels } = await requestBatch(signal);
                const batchSize = imagesShape[0];
                if (batchesPerEpoch.value === 0) {
                    batchesPerEpoch.value = Math.ceil(60000 / batchSize);
                }

                // Build the input tensor once for both forward + backward.
                // tf.tidy() in trainStep/forward handles intermediate cleanup.
                phase = "forward";
                const imagesTensor = tensorFromFloat32(images, imagesShape);
                let activationsNchw: Float32Array;
                let activationsShape: number[];
                const activationsTensor = model.forward(imagesTensor);
                try {
                    activationsShape = activationsTensor.shape.slice();
                    activationsNchw = (await activationsTensor.data()) as Float32Array;
                } finally {
                    activationsTensor.dispose();
                }

                phase = "send_activations_and_labels";
                const { upstreamGrad, loss: lossValue } =
                    await sendActivationsAndAwaitGrads(
                        activationsNchw,
                        activationsShape,
                        labels,
                        signal
                    );

                phase = "backward";
                const upstreamShape: [number, number, number, number] = [
                    batchSize,
                    ...model.activationDimsNchw
                ];
                const upstreamTensor = tensorFromFloat32(upstreamGrad, upstreamShape);
                try {
                    model.trainStep(imagesTensor, upstreamTensor);
                } finally {
                    upstreamTensor.dispose();
                    imagesTensor.dispose();
                }

                loss.value = lossValue;
                lossHistory.value.push(lossValue);
                batch.value += 1;

                // Diagnostics — log every 50 batches and at epoch boundaries
                // so a frontend run is comparable line-for-line to the
                // server's running stats and a scripts/client.py run.
                const totalBatch = epoch.value * batchesPerEpoch.value + batch.value;
                if (totalBatch % 50 === 0) {
                    const actAbs = meanAbs(activationsNchw);
                    const gradAbs = meanAbs(upstreamGrad);
                    // eslint-disable-next-line no-console
                    console.info(
                        "[training] e=%d b=%d/%d loss=%.4f |act|=%.4f |grad|=%.6f",
                        epoch.value + 1,
                        batch.value,
                        batchesPerEpoch.value,
                        lossValue,
                        actAbs,
                        gradAbs
                    );
                }
                if (batch.value >= batchesPerEpoch.value) {
                    epoch.value += 1;
                    batch.value = 0;
                    // eslint-disable-next-line no-console
                    console.info(
                        "[training] epoch %d/%d done; last loss=%.4f",
                        epoch.value,
                        epochs.value,
                        lossValue
                    );
                }
            }

            status.value = "done";
        } catch (e) {
            status.value = "error";
            const aborted = e instanceof Error && e.message === "aborted";
            if (aborted) {
                error.value = `Training stopped — WebSocket dropped at epoch ${epoch.value + 1}/${epochs.value}, batch ${batch.value}/${batchesPerEpoch.value || "?"}. Reconnect, then click Try again. (Click increases the chance of training-graph drift; for a clean restart click Reset first.)`;
            } else {
                // eslint-disable-next-line no-console
                console.error(
                    "[training] failed during phase=%s epoch=%d batch=%d",
                    phase,
                    epoch.value,
                    batch.value
                );
                error.value = `phase=${phase} epoch=${epoch.value} batch=${batch.value}: ${describeError(e)}`;
            }
        }
    };

    /** Forward-only pass for post-training inference. Returns NCHW activations. */
    const runInference = async (
        imageNchw: Float32Array,
        shape: number[]
    ): Promise<{ activations: Float32Array; shape: number[] }> => {
        const model = ensureModel();
        const input = tensorFromFloat32(imageNchw, shape);
        const out = model.forward(input);
        try {
            const data = (await out.data()) as Float32Array;
            return { activations: data, shape: out.shape.slice() };
        } finally {
            out.dispose();
            input.dispose();
        }
    };

    const saveModelToServer = async (): Promise<{ bytes: number }> => {
        const model = client.value;
        if (!model) throw new Error("No client model in memory — train first");
        const artifacts = await model.exportArtifacts();
        const weightBytes = new Uint8Array(artifacts.weightData);
        websocket.sendMessage(
            encodeEnvelope({
                type: "save_client_model",
                data: {
                    topology: artifacts.topology,
                    weight_specs: artifacts.weightSpecs,
                    format: artifacts.format ?? "tfjs-layers-model",
                    generated_by: artifacts.generatedBy,
                    converted_by: artifacts.convertedBy,
                    training: {
                        epochs_target: epochs.value,
                        epochs_done: epoch.value,
                        learning_rate: learningRate.value,
                        last_loss: loss.value,
                        loss_history_summary: {
                            count: lossHistory.value.length,
                            first: lossHistory.value[0] ?? null,
                            last: lossHistory.value[lossHistory.value.length - 1] ?? null
                        }
                    }
                },
                raw: {
                    weights: arrayToB64(weightBytes)
                }
            })
        );
        // eslint-disable-next-line no-console
        console.info(
            "[training] sent SAVE_CLIENT_MODEL (%d bytes of weights, %d specs)",
            weightBytes.byteLength,
            artifacts.weightSpecs.length
        );
        return { bytes: weightBytes.byteLength };
    };

    const reset = () => {
        abortController?.abort();
        abortController = null;
        if (client.value) {
            client.value.dispose();
            client.value = null;
        }
        status.value = "idle";
        error.value = null;
        epoch.value = 0;
        batch.value = 0;
        batchesPerEpoch.value = 0;
        loss.value = null;
        lossHistory.value = [];
    };

    return {
        client,
        status,
        error,
        epochs,
        epoch,
        batch,
        batchesPerEpoch,
        loss,
        lossHistory,
        learningRate,
        start,
        runInference,
        saveModelToServer,
        reset
    };
});
