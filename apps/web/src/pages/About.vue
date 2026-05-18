<template>
    <main class="mx-auto flex h-full w-full max-w-3xl flex-col gap-4 overflow-y-auto px-6 py-8">
        <h1 class="text-2xl font-semibold">About this demo</h1>

        <p>
            A small in-browser demo of <strong>split learning</strong> /
            <strong>split inference</strong>: a neural network is cut at a
            chosen layer so that the input-facing half runs in the browser and
            the output-facing half runs on a Python server. The two halves
            communicate over a WebSocket; the raw drawing never leaves the
            browser, only the intermediate activations do.
        </p>

        <h2 class="mt-2 text-lg font-semibold">Models</h2>

        <p>
            The <em>Model</em> dropdown on the Home page lists a few options.
            Two of them are split between the browser and the server — the
            other two run entirely in the browser as a baseline.
        </p>

        <ul class="ml-6 list-disc space-y-2">
            <li>
                <strong>LeNet-5 SplitNN (Train)</strong> — <em>special</em>.
                Split-learning model with the client half built fresh in the
                browser as a TensorFlow.js model. Click <em>Start training</em>
                to jointly train both halves over WebSocket — the server
                streams MNIST batches, the browser does forward passes locally
                and sends activations, the server computes the loss + backward
                and returns the upstream gradient. Both sides start fresh on
                each training run (the server resets its half when the browser
                begins a new run).
            </li>
            <li>
                <strong>LeNet-5 SplitNN (Static)</strong> — <em>special</em>.
                Loads a pre-trained client half (<code>client_mnist.onnx</code>)
                into ONNX Runtime Web. When you draw, the browser computes
                activations locally and ships them to the Python server, which
                runs the rest of the network and returns the predicted class.
                Training of this pair is done with <code>scripts/client.py</code>
                against <code>scripts/server.py</code>.
            </li>
            <li>
                <strong>LeNet-5</strong> — full LeNet-5 ONNX model running
                entirely in the browser. Purely local inference, no WebSocket
                in the loop.
            </li>
            <li>
                <strong>ORT Demo</strong> — Microsoft's MNIST sample ONNX model,
                also fully local.
            </li>
        </ul>

        <h2 class="mt-2 text-lg font-semibold">Why the two SplitNN entries matter</h2>

        <p>
            They are the only models in the dropdown where the browser is
            <strong>not</strong> doing the full inference end-to-end. The
            split-learning property — that the raw input pixels stay on the
            client, only the cut-layer activations cross the wire — only
            applies to these two entries. The other two run the full network
            locally; the server isn't involved.
        </p>

        <p>
            See
            <a
                href="https://github.com/evanwrm/split-learning-demo"
                class="text-primary underline"
                rel="noopener noreferrer"
                target="_blank"
                >the repository</a
            >
            for the model architectures, the WebSocket protocol, and the
            Python server/client scripts.
        </p>
    </main>
</template>
