import * as tf from "@tensorflow/tfjs";

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

export { tf };
