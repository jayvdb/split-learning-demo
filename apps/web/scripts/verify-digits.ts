/**
 * E2E: classify the 10 hand-written-style digit PNGs through the
 * frontend-trained split model.
 *
 *   client_mnist.{json,weights.bin}  →  TF.js Sequential (NHWC, conv stack)
 *   server_mnist.onnx                →  CNN2DServer  (NCHW, fc head)
 *
 * For each digit-{0..9}.png: preprocess to MNIST-normalised (1,1,28,28),
 * run the client to get (1,16,7,7) activations, hand them to ORT to get
 * (1,10) logits, argmax → predicted digit, assert == expected.
 *
 * Exits 0 if all 10 are correct, 1 otherwise (prints per-digit diagnostics).
 */
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

import * as tf from "@tensorflow/tfjs-node";
import * as ort from "onnxruntime-node";

const __dirname = dirname(fileURLToPath(import.meta.url));
const WORKSPACE = resolve(__dirname, "../../..");
const MODELS_DIR = resolve(WORKSPACE, "apps/web/public/models");
const IMAGES_DIR = resolve(WORKSPACE, "data/test-images");

const MNIST_MEAN = 0.1307;
const MNIST_STD = 0.3081;

async function loadClientModel(): Promise<tf.LayersModel> {
    const meta = JSON.parse(
        readFileSync(resolve(MODELS_DIR, "client_mnist.json"), "utf-8")
    );
    const weights = readFileSync(resolve(MODELS_DIR, "client_mnist.weights.bin"));
    // Node's Buffer is a Uint8Array view into a larger pool — slice out the
    // exact subrange so tf.io sees the right ArrayBuffer.
    const weightData = weights.buffer.slice(
        weights.byteOffset,
        weights.byteOffset + weights.byteLength
    );
    const ioHandler = tf.io.fromMemory({
        modelTopology: meta.topology,
        weightSpecs: meta.weight_specs,
        weightData,
        format: meta.format ?? "tfjs-layers-model"
    });
    return await tf.loadLayersModel(ioHandler);
}

function preprocessPng(pngPath: string): tf.Tensor4D {
    return tf.tidy(() => {
        const raw = readFileSync(pngPath);
        // decodeImage with channels=1 → (H, W, 1) uint8 grayscale
        const decoded = tf.node.decodeImage(raw, 1) as tf.Tensor3D;
        const f = decoded.toFloat().div(255);
        const normalized = f.sub(MNIST_MEAN).div(MNIST_STD);
        // (28, 28, 1) → (1, 28, 28, 1) NHWC, which is what the TF.js client wants
        return normalized.expandDims(0) as tf.Tensor4D;
    });
}

async function classify(
    clientModel: tf.LayersModel,
    serverSession: ort.InferenceSession,
    inputNhwc: tf.Tensor4D
): Promise<{ predicted: number; logits: Float32Array }> {
    // Client forward (NHWC throughout); transpose to NCHW for the server.
    const actNhwc = clientModel.predict(inputNhwc) as tf.Tensor4D;
    const actNchw = tf.transpose(actNhwc, [0, 3, 1, 2]) as tf.Tensor4D;
    const actData = (await actNchw.data()) as Float32Array;
    const actShape = actNchw.shape; // [1, 16, 7, 7]
    actNhwc.dispose();
    actNchw.dispose();

    const ortInput = new ort.Tensor("float32", actData, actShape);
    const ortOut = await serverSession.run({ input: ortInput });
    const logits = ortOut.output.data as Float32Array;

    let predicted = 0;
    let best = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        if (logits[i] > best) {
            best = logits[i];
            predicted = i;
        }
    }
    return { predicted, logits };
}

async function waitForArtifacts(timeoutMs = 15_000): Promise<void> {
    // The server writes client_mnist.* after processing the WS SAVE_CLIENT_MODEL
    // message, and server_mnist.onnx on the subsequent WebSocket disconnect.
    // Both are racy with `pnpm run train` returning, so poll briefly.
    const paths = [
        resolve(MODELS_DIR, "client_mnist.json"),
        resolve(MODELS_DIR, "client_mnist.weights.bin"),
        resolve(MODELS_DIR, "server_mnist.onnx")
    ];
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (paths.every(p => existsSync(p))) return;
        await sleep(200);
    }
    const missing = paths.filter(p => !existsSync(p));
    throw new Error(
        `Timed out waiting for trained artifacts. Missing: ${missing.join(", ")}`
    );
}

async function main(): Promise<void> {
    console.log(`models: ${MODELS_DIR}`);
    console.log(`images: ${IMAGES_DIR}`);

    await waitForArtifacts();

    const clientModel = await loadClientModel();
    const serverSession = await ort.InferenceSession.create(
        resolve(MODELS_DIR, "server_mnist.onnx")
    );

    const failures: number[] = [];
    for (let digit = 0; digit < 10; digit++) {
        const input = preprocessPng(resolve(IMAGES_DIR, `digit-${digit}.png`));
        try {
            const { predicted, logits } = await classify(
                clientModel,
                serverSession,
                input
            );
            const ok = predicted === digit;
            if (!ok) failures.push(digit);
            const logitsStr = Array.from(logits)
                .map(v => v.toFixed(2))
                .join(", ");
            console.log(
                `digit ${digit}: predicted ${predicted} ${ok ? "OK" : "FAIL"}  logits=[${logitsStr}]`
            );
        } finally {
            input.dispose();
        }
    }

    console.log(`\n${10 - failures.length}/10 correct`);
    if (failures.length > 0) {
        console.log(`failed: ${failures.join(", ")}`);
        process.exit(1);
    }
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
