import { test, expect } from "@playwright/test";

const EPOCHS = process.env.EPOCHS ?? "25";
const BACKEND = process.env.TFJS_BACKEND ?? "auto";

// Playwright 1.60's msg.text() returns the printf-style format string
// verbatim ("epoch %d/%d done") with the arg values appended on the end,
// so the log is unreadable. Read the JSHandle args and substitute the
// %d / %s / %f / %.Nf specifiers ourselves.
function formatBrowserMessage(fmt: unknown, rest: unknown[]): string {
    if (typeof fmt !== "string") {
        return [fmt, ...rest].map(v => String(v)).join(" ");
    }
    if (!/%(?:\.\d+)?[sdjifo]/.test(fmt)) {
        return [fmt, ...rest].map(v => String(v)).join(" ");
    }
    let i = 0;
    return fmt.replace(
        /%(?:\.\d+)?[sdjifo]/g,
        () => (i < rest.length ? String(rest[i++]) : "")
    );
}

test("headless training run", async ({ page }) => {
    page.on("console", async msg => {
        // Mirror browser logs into the test output so a failure isn't
        // a silent screen — easier to diagnose protocol / WS issues.
        try {
            const args = await Promise.all(
                msg.args().map(a => a.jsonValue().catch(() => undefined))
            );
            const text = formatBrowserMessage(args[0], args.slice(1));
            console.log(`[browser] ${msg.type()}: ${text}`);
        } catch {
            console.log(`[browser] ${msg.type()}: ${msg.text()}`);
        }
    });

    await page.goto(`/?headless=true&epochs=${EPOCHS}&backend=${BACKEND}`);

    await page.waitForFunction(
        () => (window as { __headlessDone?: boolean }).__headlessDone === true,
        null,
        { timeout: 28 * 60 * 1000 }
    );

    const err = await page.evaluate(
        () => (window as { __headlessError?: string }).__headlessError ?? null
    );
    expect(err, `headless runner reported: ${err}`).toBeNull();
});
