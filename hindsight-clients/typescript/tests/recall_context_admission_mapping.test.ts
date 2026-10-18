/**
 * Tests that recall() forwards purpose and context (context admission).
 *
 * Kept in step with the Python wrapper's test_recall_context_admission.py: a parameter only one
 * wrapper forwards is silently dropped for every consumer of the other. No server required:
 * sdk.recallMemories is mocked.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

describe("recall context admission", () => {
  let spy: jest.SpyInstance;

  beforeEach(() => {
    spy = jest.spyOn(sdk, "recallMemories").mockResolvedValue({ data: { results: [] } } as any);
  });

  afterEach(() => {
    spy.mockRestore();
  });

  test("purpose and context reach the request body", async () => {
    await new HindsightClient({ baseUrl: "http://localhost:8888" }).recall("bank", "q", {
      purpose: "auto_context",
      context: { repository: "github.com/acme/payments", branch: "main", module: "auth" },
    });
    const body = (spy.mock.calls[0][0] as any).body;
    expect(body.purpose).toBe("auto_context");
    expect(body.context).toEqual({
      repository: "github.com/acme/payments",
      branch: "main",
      module: "auth",
    });
  });

  test("both are left out by default", async () => {
    await new HindsightClient({ baseUrl: "http://localhost:8888" }).recall("bank", "q");
    const body = (spy.mock.calls[0][0] as any).body;
    expect([body.purpose, body.context]).toEqual([undefined, undefined]);
  });
});
