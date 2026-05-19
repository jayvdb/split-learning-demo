import { defineConfig, devices } from "@playwright/test";

// Long timeout: 25 epochs of MNIST split training over WebSocket
// regularly takes 10+ minutes on CPU.
const TRAIN_TIMEOUT_MS = 30 * 60 * 1000;

export default defineConfig({
    testDir: "./tests",
    timeout: TRAIN_TIMEOUT_MS,
    expect: { timeout: TRAIN_TIMEOUT_MS },
    fullyParallel: false,
    workers: 1,
    reporter: [["list"]],
    use: {
        baseURL: "http://localhost:5173",
        actionTimeout: 30_000,
        navigationTimeout: 30_000,
        trace: "retain-on-failure",
        // Headed mode is required for real hardware WebGPU on Linux:
        // Playwright's `headless: true` uses the `chromium-headless-shell`
        // binary, which falls back to Chrome's bundled SwiftShader software
        // Vulkan no matter what flags or PRIME env vars you set. Headed mode
        // launches the full `chromium` binary against your X/Wayland session
        // and goes through the real GPU. A browser window will appear for
        // the duration of training. Override with `HEADLESS=true pnpm run
        // train` if you must run without a display (CPU/CPU-Wasm only).
        headless: process.env.HEADLESS === "true"
    },
    projects: [
        {
            name: "chromium",
            use: {
                ...devices["Desktop Chrome"],
                // Enable real GPU + WebGPU in headless Chromium. Requires
                // a working host driver (Mesa Vulkan / NVIDIA Vulkan); the
                // headless-shell build Playwright ships by default is GPU-
                // less, so without these the auto-detect in Home.vue falls
                // through to CPU. With `channel: "chromium"` + these flags
                // and the full chromium binary (`pnpm exec playwright
                // install chromium`), webgpu/webgl come up on Linux GPUs.
                channel: "chromium",
                launchOptions: {
                    // PRIME / Optimus offload: on hybrid-graphics Linux
                    // laptops (Intel iGPU + NVIDIA dGPU) Chromium defaults
                    // to the iGPU, which has tiny shared VRAM and OOMs at
                    // the first non-trivial WebGPU allocation. These env
                    // vars route GL *and* Vulkan to the NVIDIA discrete
                    // GPU. They're no-ops on systems without NVIDIA.
                    //
                    // If you're on an AMD-only or Intel-only box and want
                    // to force a specific Vulkan ICD, set VK_ICD_FILENAMES
                    // to the matching json under /usr/share/vulkan/icd.d/
                    // (e.g. radeon_icd.json for AMD, intel_icd.json for
                    // Intel). DO NOT point it at lvp_icd.json — that's
                    // lavapipe, a software rasteriser.
                    env: {
                        ...process.env,
                        __NV_PRIME_RENDER_OFFLOAD: "1",
                        __GLX_VENDOR_LIBRARY_NAME: "nvidia",
                        __VK_LAYER_NV_optimus: "NVIDIA_only"
                    } as Record<string, string>,
                    args: [
                        // Enable Vulkan + WebGPU; allow them outside their
                        // usual security feature gates.
                        "--enable-features=Vulkan,WebGPU,VulkanFromANGLE,DefaultANGLEVulkan",
                        "--enable-unsafe-webgpu",
                        "--use-vulkan",
                        "--enable-gpu",
                        "--ignore-gpu-blocklist",
                        // The GPU sandbox aggressively caps VRAM allocations
                        // per renderer. Disabling it lets WebGPU claim more
                        // device memory — the difference between hitting
                        // VK_ERROR_OUT_OF_DEVICE_MEMORY at batch 1 and a
                        // clean training run on hybrid laptop GPUs.
                        "--disable-gpu-sandbox",
                        "--no-sandbox",
                        "--disable-software-rasterizer"
                    ]
                }
            }
        }
    ],
    webServer: {
        command: "pnpm run dev",
        url: "http://localhost:5173",
        timeout: 60_000,
        reuseExistingServer: true,
        stdout: "pipe",
        stderr: "pipe"
    }
});
