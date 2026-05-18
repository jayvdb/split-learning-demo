import { arrayToB64, b64ToArray } from "@/lib/utils/serde";
import {
    createClientModel,
    setTfjsBackend,
    tensorFromFloat32,
    tf,
    type ClientModel,
    type TfjsBackend
} from "@/lib/utils/training";
import { useWebsocketStore } from "@/stores/websocket";
import { defineStore } from "pinia";
import { ref, watch } from "vue";

type Status = "idle" | "loading" | "training" | "done" | "error";

const describeError = (e: unknown): string => {
    // eslint-disable-next-line no-console
    console.error("[training] error:", e);
    if (e instanceof Error) return `${e.name}: ${e.message}`;
    if (typeof e === "string") return e;
    try {
        return JSON.stringify(e);
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

    const epochs = ref(25);
    const epoch = ref(0);
    const batch = ref(0);
    const batchesPerEpoch = ref(0);
    const loss = ref<number | null>(null);
    const lossHistory = ref<number[]>([]);
    // Must be set to match `scripts/server.py --learning-rate`; not
    // forwarded over the protocol (server's optimizer is fixed at startup).
    const learningRate = ref(0.01);
    // TF.js backend used for the client-side model. webgl is the default
    // (GPU); cpu is the universal fallback.
    const tfjsBackend = ref<TfjsBackend>("webgl");

    const ensureModel = (): ClientModel => {
        if (!client.value) {
            status.value = "loading";
            // eslint-disable-next-line no-console
            console.info("[training] building TF.js client model (backend=%s)", tf.getBackend());
            client.value = createClientModel(learningRate.value);
        }
        return client.value;
    };

    // TF.js's "highest priority registered backend" defaults to webgpu once
    // we side-effect-import it, regardless of what the user picked in the
    // dropdown. Without calling tf.setBackend explicitly the first tensor
    // op triggers a lazy webgpu init that throws if webgpu isn't ready.
    // Call this before any tensor work to keep TF.js in sync with the
    // store's `tfjsBackend.value`.
    const ensureBackend = async () => {
        if (tf.getBackend() !== tfjsBackend.value) {
            await setTfjsBackend(tfjsBackend.value);
        }
    };

    watch(learningRate, lr => {
        if (client.value) client.value.setLearningRate(lr);
    });

    let abortController: AbortController | null = null;
    // Tracks why the most recent abort was issued so the catch in start()
    // can pick the right user-facing message.
    let abortReason: "user-cancelled" | "ws-dropped" | null = null;

    watch(
        () => websocket.status,
        wsStatus => {
            if (wsStatus === "closed" && status.value === "training") {
                // eslint-disable-next-line no-console
                console.warn("[training] WS dropped mid-training; halting loop");
                abortReason = "ws-dropped";
                abortController?.abort();
            }
        }
    );

    const cancel = () => {
        if (status.value !== "training" && status.value !== "loading") return;
        abortReason = "user-cancelled";
        abortController?.abort();
    };

    // Monotonic id so a slow-failing setBackend can't overwrite the state
    // produced by a newer attempt that already finished.
    let backendChangeId = 0;
    const setBackend = async (backend: TfjsBackend) => {
        // Always clear any prior backend error on a fresh user action,
        // including the no-op same-backend case (otherwise an error from
        // a previous failed click is sticky until something else succeeds).
        error.value = null;
        if (tfjsBackend.value === backend) return;
        // TF.js tensors don't migrate across backends — dispose any model
        // that exists so the next start() rebuilds on the new backend.
        if (client.value) reset();
        const myId = ++backendChangeId;
        try {
            await setTfjsBackend(backend);
            if (myId !== backendChangeId) return;
            tfjsBackend.value = backend;
            // eslint-disable-next-line no-console
            console.info("[training] TF.js backend set to %s", backend);
        } catch (e) {
            if (myId !== backendChangeId) return;
            error.value = e instanceof Error ? e.message : String(e);
            // eslint-disable-next-line no-console
            console.error("[training] setBackend failed:", e);
        }
    };

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
        abortController?.abort();
        abortController = new AbortController();
        abortReason = null;
        const { signal } = abortController;
        try {
            phase = "set_backend";
            await ensureBackend();
            phase = "init";
            const model = ensureModel();
            status.value = "training";
            error.value = null;
            epoch.value = 0;
            batch.value = 0;
            lossHistory.value = [];

            // Tell the server to reset its half too — otherwise a
            // previously-trained server gets clobbered by the freshly-
            // initialised client's random activations during the first
            // few batches. Fire-and-forget; the server doesn't reply.
            websocket.sendMessage(
                encodeEnvelope({ type: "reset_server", data: {}, raw: {} })
            );

            while (epoch.value < epochs.value) {
                if (signal.aborted) throw new Error("aborted");
                phase = "request_batch";
                const { images, imagesShape, labels } = await requestBatch(signal);
                const batchSize = imagesShape[0];
                if (batchesPerEpoch.value === 0) {
                    batchesPerEpoch.value = Math.ceil(60000 / batchSize);
                }

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

                const totalBatch = epoch.value * batchesPerEpoch.value + batch.value;
                if (totalBatch % 50 === 0) {
                    // eslint-disable-next-line no-console
                    console.info(
                        "[training] e=%d b=%d/%d loss=%.4f |act|=%.4f |grad|=%.6f",
                        epoch.value + 1,
                        batch.value,
                        batchesPerEpoch.value,
                        lossValue,
                        meanAbs(activationsNchw),
                        meanAbs(upstreamGrad)
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
            if (e instanceof Error && e.message === "aborted") {
                const where = `epoch ${epoch.value + 1}/${epochs.value}, batch ${batch.value}/${batchesPerEpoch.value || "?"}`;
                if (abortReason === "user-cancelled") {
                    status.value = "idle";
                    error.value = null;
                    // eslint-disable-next-line no-console
                    console.info("[training] cancelled by user at %s", where);
                } else {
                    status.value = "error";
                    error.value = `Training stopped — WebSocket dropped at ${where}. Reconnect, then click Try again.`;
                }
            } else {
                status.value = "error";
                error.value = `phase=${phase} epoch=${epoch.value} batch=${batch.value}: ${describeError(e)}`;
            }
        }
    };

    const runInference = async (
        imageNchw: Float32Array,
        shape: number[]
    ): Promise<{ activations: Float32Array; shape: number[] }> => {
        await ensureBackend();
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
        tfjsBackend,
        start,
        cancel,
        setBackend,
        runInference,
        saveModelToServer,
        reset
    };
});
