/**
 * Tests that recall() forwards lifecycleView and asOf (adaptive memory lifecycle mode).
 *
 * Kept in step with the Python wrapper's test_recall_lifecycle_view.py: a parameter only one
 * wrapper forwards is silently dropped for every consumer of the other. No server required:
 * sdk.recallMemories is mocked.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

describe("recall lifecycle view", () => {
  let spy: jest.SpyInstance;

  beforeEach(() => {
    spy = jest.spyOn(sdk, "recallMemories").mockResolvedValue({ data: { results: [] } } as any);
  });

  afterEach(() => {
    spy.mockRestore();
  });

  test("lifecycleView and asOf reach the request body", async () => {
    await new HindsightClient({ baseUrl: "http://localhost:8888" }).recall("bank", "q", {
      lifecycleView: "historical",
      asOf: "2026-01-15T00:00:00Z",
    });
    const body = (spy.mock.calls[0][0] as any).body;
    expect([body.lifecycle_view, body.as_of]).toEqual(["historical", "2026-01-15T00:00:00Z"]);
  });

  test("both are left out by default", async () => {
    await new HindsightClient({ baseUrl: "http://localhost:8888" }).recall("bank", "q");
    const body = (spy.mock.calls[0][0] as any).body;
    expect([body.lifecycle_view, body.as_of]).toEqual([undefined, undefined]);
  });
});
