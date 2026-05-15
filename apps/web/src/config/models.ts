export type ModelType = "splitnn" | "splitnn-train" | "local";
export interface ModelConfig {
    type: ModelType;
    name: string;
    path: string;
}

export const models: Record<string, ModelConfig[]> = {
    mnist: [
        { name: "LeNet-5 SplitNN", path: "/models/client_mnist.onnx", type: "splitnn" },
        {
            name: "LeNet-5 SplitNN (Train in Browser)",
            path: "/models/training/",
            type: "splitnn-train"
        },
        { name: "LeNet-5", path: "/models/mnist.onnx", type: "local" },
        { name: "ORT Demo", path: "/models/mnist_default.onnx", type: "local" }
    ],
    quickdraw: [{ name: "LeNet-5", path: "/models/quickdraw.onnx", type: "local" }]
};
