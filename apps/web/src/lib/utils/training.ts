import * as tf from "@tensorflow/tfjs";

// CNN2DClient (cut after conv2 + relu + pool ×2):
//   Input  NCHW (B, 1, 28, 28)
//   Conv2d(1→6, 5×5, padding=same) → ReLU → MaxPool(2×2)
//   Conv2d(6→16, 5×5, padding=same) → ReLU → MaxPool(2×2)
//   Output NCHW (B, 16, 7, 7)
//
// TF.js layers default to NHWC. We expose the public API in NCHW so the
// rest of the app (and the existing WebSocket protocol) doesn't have to
// know about the layout difference — the transpose is local to this file.

const ACTIVATION_DIMS_NCHW: [number, number, number] = [16, 7, 7];
const ACTIVATION_DIMS_NHWC: [number, number, number] = [7, 7, 16];

export interface ClientModel {
    /** Forward pass. Input NCHW (B,1,28,28) → activations NCHW (B,16,7,7). */
    forward(imagesNchw: tf.Tensor4D): tf.Tensor4D;
    /**
     * Split-learning train step. Builds the chain-rule loss
     * `sum(activations · upstreamGrad)`, lets tfjs autograd backprop, and
     * applies an SGD update. Returns the chain-rule loss scalar (mainly
     * useful as a "did the gradient flow" liveness check; the meaningful
     * server-side cross-entropy loss arrives separately in the GRADS
     * envelope).
     */
    trainStep(imagesNchw: tf.Tensor4D, upstreamGradNchw: tf.Tensor4D): number;
    /** Sequential model — exposed so the store can `.dispose()` on reset. */
    dispose(): void;
    /** Update the optimizer's learning rate without re-creating the model. */
    setLearningRate(lr: number): void;
    /**
     * Dump the model weights for inspection. Returns the artifacts in the
     * tfjs-layers `IOHandler` save format: JSON-able topology + weight
     * specs, and a single concatenated Float32 ArrayBuffer of weight bytes.
     */
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

// Mirror `CNN2DClient` (= first two children of `CNN2D` in
// packages/.../models/vision/cnn_2d.py) layer-for-layer:
//   Conv2d(1→6, 5×5, padding=2) → ReLU → MaxPool(2×2) → Dropout(0.4)
//   Conv2d(6→16, 5×5, padding=2) → ReLU → MaxPool(2×2) → Dropout(0.4)
// The Dropout layers are only active under `model.apply(x, {training: true})`
// — they short-circuit in `model.predict` / `model.apply(x)`. That matches
// PyTorch's `model.train()` vs `model.eval()` behaviour. Without them we
// optimize a different objective than `scripts/client.py` does.
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
    // `scripts/client.py` uses `torch.optim.SGD(..., momentum=0.9)` — plain
    // `tf.train.sgd` is *not* equivalent. Without momentum, 50 epochs of
    // chain-rule training is much slower to converge.
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
        // Hold the NHWC inputs across the minimize() closure so they're
        // not freed mid-callback. We dispose them manually after.
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
                // — exactly the upstream-gradient propagation
                // `torch.autograd.backward(activations, grads)` does in
                // scripts/client.py.
                return tf.sum(tf.mul(activationsNhwc, upstreamGradNhwc)) as tf.Scalar;
            },
            true /* returnCost */
        );

        if (lossScalar) {
            lossValue = lossScalar.dataSync()[0];
            lossScalar.dispose();
        }
        imagesNhwc.dispose();
        upstreamGradNhwc.dispose();
        return lossValue;
    };

    const dispose = () => {
        model.dispose();
    };

    const setLearningRate = (lr: number) => {
        // `tf.train.momentum(...)` extends `SGDOptimizer`, which has a
        // `setLearningRate` method; the abstract base type doesn't declare
        // it, hence the narrowing cast.
        (optimizer as unknown as { setLearningRate(lr: number): void }).setLearningRate(lr);
    };

    const exportArtifacts: ClientModel["exportArtifacts"] = async () => {
        // `tf.io.withSaveHandler` lets us intercept the save call and grab
        // the artifacts directly instead of writing them to local-storage
        // or a download. We dispatch the same artifacts the server expects.
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

export { tf };
void ACTIVATION_DIMS_NHWC;
