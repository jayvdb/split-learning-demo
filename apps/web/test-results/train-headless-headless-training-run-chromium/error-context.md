# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: train-headless.spec.ts >> headless training run
- Location: tests/train-headless.spec.ts:24:1

# Error details

```
Error: headless runner reported: AbortError: Failed to execute 'mapAsync' on 'GPUBuffer': A valid external Instance reference no longer exists.

expect(received).toBeNull()

Received: "AbortError: Failed to execute 'mapAsync' on 'GPUBuffer': A valid external Instance reference no longer exists."
```

# Page snapshot

```yaml
- generic [ref=e3]:
  - banner [ref=e4]:
    - navigation [ref=e5]:
      - list [ref=e6]:
        - listitem [ref=e7]:
          - link "Home" [ref=e8] [cursor=pointer]:
            - /url: /
        - listitem [ref=e9]:
          - link "About" [ref=e10] [cursor=pointer]:
            - /url: /about
      - generic [ref=e11]:
        - link [ref=e12] [cursor=pointer]:
          - /url: https://github.com/evanwrm/split-learning-demo
          - img [ref=e13]
        - button "Toggle Dark Mode" [ref=e15] [cursor=pointer]:
          - img [ref=e17]
  - main [ref=e19]:
    - generic [ref=e20]:
      - generic [ref=e21]:
        - generic [ref=e22]:
          - generic [ref=e23]: Model
          - button "Model" [ref=e24] [cursor=pointer]:
            - text: LeNet-5 SplitNN (Train)
            - img [ref=e25]
        - generic [ref=e27]:
          - generic [ref=e28]: Backend
          - button "Backend" [ref=e29] [cursor=pointer]:
            - text: "TF.js: webgpu"
            - img [ref=e30]
        - generic:
          - generic:
            - generic: Dataset
          - button "Dataset" [disabled]:
            - text: MNIST
            - img
        - generic [ref=e32]:
          - generic [ref=e33]: Parameter Server
          - textbox "Enter server URL" [ref=e34]: ws://127.0.0.1:8000/ws
        - button "Reconnect" [ref=e35] [cursor=pointer]
      - generic [ref=e36]:
        - generic [ref=e37]:
          - generic [ref=e38]:
            - generic [ref=e39]: Your browser does not support the canvas element.
            - generic: Start drawing here...
          - button "Clear" [ref=e41] [cursor=pointer]
        - img [ref=e44]:
          - generic "x-axis tick" [ref=e45]
          - generic "x-axis tick label" [ref=e46]:
            - generic [ref=e47]: "0.0"
            - generic [ref=e48]: "0.1"
            - generic [ref=e49]: "0.2"
            - generic [ref=e50]: "0.3"
            - generic [ref=e51]: "0.4"
            - generic [ref=e52]: "0.5"
            - generic [ref=e53]: "0.6"
            - generic [ref=e54]: "0.7"
            - generic [ref=e55]: "0.8"
            - generic [ref=e56]: "0.9"
            - generic [ref=e57]: "1.0"
          - generic "rule" [ref=e58]
      - generic [ref=e59]:
        - generic [ref=e60]:
          - heading "Training split model" [level=2] [ref=e61]
          - generic [ref=e62]: error
        - generic [ref=e63]:
          - generic [ref=e64]:
            - generic [ref=e65]: Learning rate
            - spinbutton [ref=e66]: "0.01"
          - button "Try again" [ref=e67] [cursor=pointer]
        - paragraph [ref=e68]:
          - text: Set
          - emphasis [ref=e69]: Learning rate
          - text: to match the server's
          - code [ref=e70]: "--learning-rate"
          - text: CLI flag.
        - generic [ref=e71]: "phase=forward epoch=1 batch=560: RangeError: Failed to execute 'createBuffer' on 'GPUDevice': createBuffer failed, size (100352) is too large for the implementation when mappedAtCreation == true"
        - generic [ref=e72]:
          - generic [ref=e73]:
            - generic [ref=e74]: Epochs done
            - generic [ref=e75]:
              - generic [ref=e76]: 1 /
              - spinbutton [ref=e77]: "25"
          - generic [ref=e78]:
            - generic [ref=e79]: Batch in current epoch
            - generic [ref=e80]: 560 / 1875
          - generic [ref=e81]:
            - generic [ref=e82]: Server loss
            - generic [ref=e83]: "0.3198"
        - img [ref=e87]:
          - generic "y-grid" [ref=e88]
          - generic "y-axis tick" [ref=e89]
          - generic "y-axis tick label" [ref=e90]:
            - generic [ref=e91]: "1"
            - generic [ref=e92]: "2"
          - generic "y-axis label" [ref=e93]:
            - generic [ref=e94]: ↑ loss
          - generic "x-axis tick" [ref=e95]
          - generic "x-axis tick label" [ref=e96]:
            - generic [ref=e97]: "0"
            - generic [ref=e98]: "500"
            - generic [ref=e99]: 1,000
            - generic [ref=e100]: 1,500
            - generic [ref=e101]: 2,000
          - generic "x-axis label" [ref=e102]:
            - generic [ref=e103]: step →
          - generic "line" [ref=e104]
        - generic [ref=e106]:
          - button "Send model to server for inspection" [ref=e108] [cursor=pointer]
          - paragraph [ref=e109]:
            - text: Dumps the in-memory TF.js client weights over the WS to
            - code [ref=e110]: "data/models/frontend_client_<ts>.{json,weights.bin}"
            - text: . Inspect with
            - code [ref=e111]: scripts/inspect_frontend_model.py
            - text: .
```

# Test source

```ts
  1  | import { test, expect } from "@playwright/test";
  2  | 
  3  | const EPOCHS = process.env.EPOCHS ?? "25";
  4  | const BACKEND = process.env.TFJS_BACKEND ?? "auto";
  5  | 
  6  | // Playwright 1.60's msg.text() returns the printf-style format string
  7  | // verbatim ("epoch %d/%d done") with the arg values appended on the end,
  8  | // so the log is unreadable. Read the JSHandle args and substitute the
  9  | // %d / %s / %f / %.Nf specifiers ourselves.
  10 | function formatBrowserMessage(fmt: unknown, rest: unknown[]): string {
  11 |     if (typeof fmt !== "string") {
  12 |         return [fmt, ...rest].map(v => String(v)).join(" ");
  13 |     }
  14 |     if (!/%(?:\.\d+)?[sdjifo]/.test(fmt)) {
  15 |         return [fmt, ...rest].map(v => String(v)).join(" ");
  16 |     }
  17 |     let i = 0;
  18 |     return fmt.replace(
  19 |         /%(?:\.\d+)?[sdjifo]/g,
  20 |         () => (i < rest.length ? String(rest[i++]) : "")
  21 |     );
  22 | }
  23 | 
  24 | test("headless training run", async ({ page }) => {
  25 |     page.on("console", async msg => {
  26 |         // Mirror browser logs into the test output so a failure isn't
  27 |         // a silent screen — easier to diagnose protocol / WS issues.
  28 |         try {
  29 |             const args = await Promise.all(
  30 |                 msg.args().map(a => a.jsonValue().catch(() => undefined))
  31 |             );
  32 |             const text = formatBrowserMessage(args[0], args.slice(1));
  33 |             console.log(`[browser] ${msg.type()}: ${text}`);
  34 |         } catch {
  35 |             console.log(`[browser] ${msg.type()}: ${msg.text()}`);
  36 |         }
  37 |     });
  38 | 
  39 |     await page.goto(`/?headless=true&epochs=${EPOCHS}&backend=${BACKEND}`);
  40 | 
  41 |     await page.waitForFunction(
  42 |         () => (window as { __headlessDone?: boolean }).__headlessDone === true,
  43 |         null,
  44 |         { timeout: 28 * 60 * 1000 }
  45 |     );
  46 | 
  47 |     const err = await page.evaluate(
  48 |         () => (window as { __headlessError?: string }).__headlessError ?? null
  49 |     );
> 50 |     expect(err, `headless runner reported: ${err}`).toBeNull();
     |                                                     ^ Error: headless runner reported: AbortError: Failed to execute 'mapAsync' on 'GPUBuffer': A valid external Instance reference no longer exists.
  51 | });
  52 | 
```