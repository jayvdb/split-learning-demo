import { TrainingSession, Tensor, env } from "onnxruntime-web/training";

const TRAINING_ARTIFACT_DIR = "/models/training";

export interface TrainingArtifactPaths {
    checkpointState: string;
    trainModel: string;
    evalModel: string;
    optimizerModel: string;
}

export const defaultArtifactPaths = (
    base: string = TRAINING_ARTIFACT_DIR
): TrainingArtifactPaths => ({
    checkpointState: `${base}/checkpoint`,
    trainModel: `${base}/training_model.onnx`,
    evalModel: `${base}/eval_model.onnx`,
    optimizerModel: `${base}/optimizer_model.onnx`
});

export const createTrainingSession = async (
    artifactPaths: TrainingArtifactPaths = defaultArtifactPaths(),
    opts: { skipOptimizer?: boolean } = {}
): Promise<TrainingSession> => {
    // onnxruntime-web/training loads the wasm runtime + worker scripts from
    // the same origin; vite.config.ts copies them to the site root.
    env.wasm.wasmPaths = "/";
    // Bubble ORT's own error/warning messages to the JS console so wasm-side
    // failures (which otherwise surface as opaque numeric exception
    // pointers) come with a readable message attached.
    env.logLevel = "warning";
    env.debug = true;

    const create = (paths: TrainingArtifactPaths) =>
        TrainingSession.create(paths, {
            logSeverityLevel: 0, // VERBOSE
            logVerbosityLevel: 0
        });

    if (opts.skipOptimizer) {
        const { optimizerModel: _drop, ...withoutOpt } = artifactPaths;
        return await create(withoutOpt as TrainingArtifactPaths);
    }

    try {
        return await create(artifactPaths);
    } catch (e) {
        // ORT-web's stripped wasm build doesn't include all `com.microsoft`
        // training-only contrib ops, and SGDOptimizerV2 — what onnxblock
        // emits for the SGD optimizer — has been observed to throw an
        // opaque emscripten exception here. Retry without the optimizer
        // model so the caller can do the param update by hand. The session
        // still exposes runTrainStep / runEvalStep.
        // eslint-disable-next-line no-console
        console.warn(
            "[training] TrainingSession.create failed with optimizer model; retrying without it",
            e
        );
        const { optimizerModel: _drop, ...withoutOpt } = artifactPaths;
        return await create(withoutOpt as TrainingArtifactPaths);
    }
};

export const zerosLikeActivations = (batch: number): Tensor =>
    new Tensor("float32", new Float32Array(batch * 16 * 7 * 7), [batch, 16, 7, 7]);

export { Tensor };
