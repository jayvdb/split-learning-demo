import vue from "@vitejs/plugin-vue";
import { fileURLToPath, URL } from "node:url";
import { defineConfig, type Plugin } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

// `ort-training-wasm-simd-threaded.wasm` is ORT's only training wasm
// variant; it requires SharedArrayBuffer, which the browser only exposes
// in a "cross-origin-isolated" context. That means the dev server has to
// send these two headers on every response. Without them, the wasm
// runtime throws an opaque emscripten exception pointer at
// `TrainingSession.create` time (visible only as a numeric ERR).
const crossOriginIsolation = (): Plugin => ({
    name: "cross-origin-isolation",
    configureServer(server) {
        server.middlewares.use((_req, res, next) => {
            res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
            res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
            next();
        });
    },
    configurePreviewServer(server) {
        server.middlewares.use((_req, res, next) => {
            res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
            res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
            next();
        });
    }
});

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [
        vue(),
        crossOriginIsolation(),
        viteStaticCopy({
            targets: [
                {
                    src: "node_modules/onnxruntime-web/dist/*.wasm",
                    dest: "."
                },
                // ORT 1.19's wasm runtime fetches its own worker .mjs files
                // (e.g. `ort-wasm-simd-threaded.mjs`,
                // `ort-training-wasm-simd-threaded.mjs`) from the same
                // origin as the .wasm. Without these, TrainingSession.create
                // fails with a 404 looking for the worker.
                {
                    src: "node_modules/onnxruntime-web/dist/ort-*.mjs",
                    dest: "."
                }
            ]
        })
    ],
    resolve: {
        // Array-form alias so we can use a regex for an exact-match alias on
        // `onnxruntime-web`. The package declares no `module` field, so Vite
        // would otherwise resolve the bare import to `dist/ort.node.min.js`
        // (CJS) and `InferenceSession` wouldn't be a named ESM export. We
        // pin it to the browser ESM bundle directly. The `/training`
        // subpath still flows through the package's `exports` map so it
        // continues to resolve to `dist/ort.training.wasm.min.mjs`.
        alias: [
            { find: "@", replacement: fileURLToPath(new URL("./src", import.meta.url)) },
            {
                find: /^onnxruntime-web$/,
                // Use an absolute file URL — a package-relative target
                // (`onnxruntime-web/dist/…`) re-enters Vite's `exports`
                // resolution and `onnxruntime-web`'s `exports` map doesn't
                // expose `./dist/*`, so it would fail with
                // `Missing "./dist/ort.bundle.min.mjs" specifier`.
                replacement: fileURLToPath(
                    new URL(
                        "./node_modules/onnxruntime-web/dist/ort.bundle.min.mjs",
                        import.meta.url
                    )
                )
            }
        ]
    },
    optimizeDeps: {
        // The wasm runtime (used by both the inference and training entries)
        // dynamically imports `ort-wasm-simd-threaded.mjs` at runtime; Vite's
        // pre-bundler can't track that and 504s with "Outdated Optimize Dep"
        // → the user-visible "no available backend found" error. Excluding
        // both entries keeps the optimizer's hands off the wasm loader so
        // it can fetch its worker .mjs directly from the site root, where
        // viteStaticCopy publishes the dist files.
        exclude: ["oh-vue-icons/icons", "onnxruntime-web", "onnxruntime-web/training"]
    }
});
