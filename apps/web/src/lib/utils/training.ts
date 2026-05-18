import * as tf from "@tensorflow/tfjs";
// Side-effect import — registers the webgpu backend with the global TF.js
// registry so `tf.setBackend('webgpu')` can find it. We deliberately do
// NOT register `@tensorflow/tfjs-backend-wasm`: its kernel set is
// inference-only (no Conv2DBackpropFilter etc.) so training would always
// fail, and this app's only TF.js consumer is the split-learning training
// flow.
import "@tensorflow/tfjs-backend-webgpu";

// eslint-disable-next-line no-console
console.info(
    "[training] TF.js backends registered:",
    Object.keys((tf.engine() as unknown as { registryFactory: Record<string, unknown> }).registryFactory)
);

// TF.js layers default to NHWC; the rest of the app (and the WebSocket
// protocol) speaks NCHW. The transpose is local to this file.

const ACTIVATION_DIMS_NCHW: [number, number, number] = [16, 7, 7];

export interface ClientModel {
    forward(imagesNchw: tf.Tensor4D): tf.Tensor4D;
    trainStep(imagesNchw: tf.Tensor4D, upstreamGradNchw: tf.Tensor4D): number;
    dispose(): void;
    setLearningRate(lr: number): void;
    exportArtifacts(): Promise<{
        topology: unknown;
        weightSpecs: tf.io.WeightsManifestEntry[];
        weightData: ArrayBuffer;
        format?: string | null;
        generatedBy?: string | null;
        convertedBy?: string | null;
    }>;
    readonly activationDimsNchw: readonly [number, number, number];
}

// Mirrors `CNN2DClient` (first two children of `CNN2D` in
// packages/.../models/vision/cnn_2d.py): Conv→ReLU→MaxPool→Dropout ×2.
// Dropout is only active under `model.apply(x, {training: true})`,
// matching PyTorch's `train()` vs `eval()`.
const CLIENT_DROPOUT = 0.4;

const buildSequential = (): tf.LayersModel =>
    tf.sequential({
        layers: [
            tf.layers.conv2d({
                inputShape: [28, 28, 1],
                filters: 6,
                kernelSize: 5,
                padding: "same"
            }),
            tf.layers.activation({ activation: "relu" }),
            tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
            tf.layers.dropout({ rate: CLIENT_DROPOUT }),
            tf.layers.conv2d({ filters: 16, kernelSize: 5, padding: "same" }),
            tf.layers.activation({ activation: "relu" }),
            tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
            tf.layers.dropout({ rate: CLIENT_DROPOUT })
        ]
    });

export const createClientModel = (learningRate: number = 0.01): ClientModel => {
    const model = buildSequential();
    // Match `scripts/client.py:torch.optim.SGD(..., momentum=0.9)`.
    const optimizer = tf.train.momentum(learningRate, 0.9);

    const forward = (imagesNchw: tf.Tensor4D): tf.Tensor4D =>
        tf.tidy(() => {
            const imagesNhwc = tf.transpose(imagesNchw, [0, 2, 3, 1]);
            const activationsNhwc = model.predict(imagesNhwc) as tf.Tensor4D;
            return tf.transpose(activationsNhwc, [0, 3, 1, 2]) as tf.Tensor4D;
        });

    const trainStep = (
        imagesNchw: tf.Tensor4D,
        upstreamGradNchw: tf.Tensor4D
    ): number => {
        const imagesNhwc = tf.transpose(imagesNchw, [0, 2, 3, 1]) as tf.Tensor4D;
        const upstreamGradNhwc = tf.transpose(
            upstreamGradNchw,
            [0, 2, 3, 1]
        ) as tf.Tensor4D;

        let lossValue = 0;
        const lossScalar = optimizer.minimize(
            () => {
                const activationsNhwc = model.apply(imagesNhwc, {
                    training: true
                }) as tf.Tensor4D;
                // Split-learning chain-rule loss:
                //   d(sum(a*g))/d(params) = d(a)/d(params) · g
                return tf.sum(tf.mul(activationsNhwc, upstreamGradNhwc)) as tf.Scalar;
            },
            true
        );

        if (lossScalar) {
            lossValue = lossScalar.dataSync()[0];
            lossScalar.dispose();
        }
        imagesNhwc.dispose();
        upstreamGradNhwc.dispose();
        return lossValue;
    };

    const dispose = () => model.dispose();

    const setLearningRate = (lr: number) => {
        // `setLearningRate` is on the concrete optimizer but not the abstract base.
        (optimizer as unknown as { setLearningRate(lr: number): void }).setLearningRate(lr);
    };

    const exportArtifacts: ClientModel["exportArtifacts"] = async () => {
        let captured: tf.io.ModelArtifacts | null = null;
        await model.save(
            tf.io.withSaveHandler(async (artifacts: tf.io.ModelArtifacts) => {
                captured = artifacts;
                return {
                    modelArtifactsInfo: {
                        dateSaved: new Date(),
                        modelTopologyType: "JSON"
                    }
                };
            })
        );
        if (!captured) throw new Error("TF.js model.save did not invoke the handler");
        const a = captured as tf.io.ModelArtifacts;
        const weightData =
            a.weightData instanceof ArrayBuffer
                ? a.weightData
                : a.weightData
                  ? (a.weightData as ArrayBuffer[]).reduce((acc, buf) => {
                        const merged = new Uint8Array(acc.byteLength + buf.byteLength);
                        merged.set(new Uint8Array(acc), 0);
                        merged.set(new Uint8Array(buf), acc.byteLength);
                        return merged.buffer;
                    }, new ArrayBuffer(0))
                  : new ArrayBuffer(0);
        return {
            topology: a.modelTopology,
            weightSpecs: a.weightSpecs ?? [],
            weightData,
            format: a.format,
            generatedBy: a.generatedBy,
            convertedBy: a.convertedBy
        };
    };

    return {
        forward,
        trainStep,
        dispose,
        setLearningRate,
        exportArtifacts,
        activationDimsNchw: ACTIVATION_DIMS_NCHW
    };
};

export const tensorFromFloat32 = (
    data: Float32Array,
    shape: number[]
): tf.Tensor4D => tf.tensor4d(data, shape as [number, number, number, number]);

// webgl first — in practice it's the fastest on most laptops we've
// tested, since TF.js's webgpu kernel coverage for small conv models is
// still catching up. webgpu is exposed for users on machines/browsers
// where it's actually faster. cpu is the universal fallback.
// `wasm` is omitted on purpose — see the side-effect import comment above.
export const TFJS_BACKENDS = ["webgl", "webgpu", "cpu"] as const;
export type TfjsBackend = (typeof TFJS_BACKENDS)[number];

export const setTfjsBackend = async (backend: TfjsBackend) => {
    const engine = tf.engine() as unknown as {
        registryFactory: Record<string, unknown>;
    };
    const registered = Object.keys(engine.registryFactory);

    if (!registered.includes(backend)) {
        throw new Error(
            `TF.js backend "${backend}" is not registered. ` +
                `Registered backends: [${registered.join(", ")}]. ` +
                `If this is webgpu, the side-effect import in lib/utils/training.ts may have failed — check the Network and Console tabs for a 404 or module-load error.`
        );
    }

    // WebGPU pre-check: TF.js's webgpu factory dereferences the result of
    // `navigator.gpu.requestAdapter()` without a null check (it reads
    // `.features` on it), which throws an opaque `TypeError: Cannot read
    // properties of null (reading 'features')` when the browser refuses
    // to grant an adapter. Probe ourselves first so the user gets a
    // useful message and a fix, not the TF.js call-stack.
    if (backend === "webgpu") {
        const gpu = (navigator as unknown as { gpu?: { requestAdapter: () => Promise<unknown> } }).gpu;
        if (!gpu) {
            throw new Error(
                "WebGPU is not available in this browser (navigator.gpu is undefined). " +
                    "Try a Chromium-based browser ≥ 113, Safari ≥ 18, or Firefox ≥ 141. " +
                    "Pick webgl or cpu instead."
            );
        }
        let adapter: unknown;
        try {
            adapter = await gpu.requestAdapter();
        } catch (e) {
            const detail = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
            throw new Error(
                `navigator.gpu.requestAdapter() threw: ${detail}. Pick webgl or cpu instead.`
            );
        }
        if (!adapter) {
            throw new Error(
                "navigator.gpu.requestAdapter() returned null — the browser exposes WebGPU " +
                    "but refuses to grant an adapter for this page. On Linux Chrome, the usual " +
                    "fix is to enable chrome://flags/#enable-unsafe-webgpu and restart. " +
                    "(chrome://flags/#enable-vulkan is occasionally also required, depending on " +
                    "the driver/GPU combo, but usually isn't.) On macOS/Windows, the more " +
                    "likely cause is a blocklisted driver (chrome://gpu shows it) or an " +
                    "extension denying the adapter. http://localhost is already a secure " +
                    "context, so that's not the blocker. Pick webgl or cpu in the meantime."
            );
        }
    }

    let ok: boolean;
    try {
        ok = await tf.setBackend(backend);
    } catch (e) {
        const detail = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
        throw new Error(
            `TF.js threw while initialising backend "${backend}": ${detail}. ` +
                `Currently active backend: "${tf.getBackend()}". ` +
                `Look just above this in the JS console for the underlying TF.js / browser API error.`
        );
    }

    if (!ok) {
        throw new Error(
            `TF.js backend "${backend}" failed to initialise (setBackend returned false). ` +
                `Currently active backend: "${tf.getBackend()}". ` +
                `TF.js typically logs a warning with the real reason just above this — common causes: ` +
                `webgpu without navigator.gpu, wasm without the .wasm blob being served at "/", ` +
                `or a browser policy blocking the API.`
        );
    }

    await tf.ready();
    const actual = tf.getBackend();
    if (actual !== backend) {
        throw new Error(
            `TF.js backend "${backend}" was selected but TF.js fell back to "${actual}". ` +
                `Inspect the JS console for the reason.`
        );
    }
    // eslint-disable-next-line no-console
    console.info("[training] TF.js backend now active: %s", actual);
};

export { tf };
