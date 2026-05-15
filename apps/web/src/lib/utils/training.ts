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
    readonly activationDimsNchw: readonly [number, number, number];
}

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
            tf.layers.conv2d({ filters: 16, kernelSize: 5, padding: "same" }),
            tf.layers.activation({ activation: "relu" }),
            tf.layers.maxPooling2d({ poolSize: 2, strides: 2 })
        ]
    });

export const createClientModel = (learningRate: number = 0.01): ClientModel => {
    const model = buildSequential();
    const optimizer = tf.train.sgd(learningRate);

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
        // `tf.train.sgd(...)` returns an `SGDOptimizer` with a
        // `setLearningRate` method; the abstract base type doesn't declare
        // it, hence the narrowing cast.
        (optimizer as unknown as { setLearningRate(lr: number): void }).setLearningRate(lr);
    };

    return {
        forward,
        trainStep,
        dispose,
        setLearningRate,
        activationDimsNchw: ACTIVATION_DIMS_NCHW
    };
};

export const tensorFromFloat32 = (
    data: Float32Array,
    shape: number[]
): tf.Tensor4D => tf.tensor4d(data, shape as [number, number, number, number]);

export { tf };
void ACTIVATION_DIMS_NHWC;
