/**
 * Tests that retain()/retainBatch() forward importance and forgetting_exempt.
 *
 * The wrapper builds each item field by field, so a field it does not list is dropped
 * before the request is sent. No server required: sdk.retainMemories is mocked.
 */

import { HindsightClient } from "../src";
import * as sdk from "../generated/sdk.gen";

function capturedItems(spy: jest.SpyInstance): any[] {
  return (spy.mock.calls[0][0] as any).body.items;
}

describe("retain retention policy", () => {
  let spy: jest.SpyInstance;

  beforeEach(() => {
    spy = jest.spyOn(sdk, "retainMemories").mockResolvedValue({ data: { success: true } } as any);
  });

  afterEach(() => {
    spy.mockRestore();
  });

  test("retainBatch passes importance and forgetting_exempt through", async () => {
    await new HindsightClient({ baseUrl: "http://localhost:8888" }).retainBatch("bank", [
      { content: "Pinned fact.", importance: 0.8, forgetting_exempt: true },
      { content: "Plain fact." },
    ]);
    const [pinned, plain] = capturedItems(spy);
    expect([pinned.importance, pinned.forgetting_exempt]).toEqual([0.8, true]);
    expect([plain.importance, plain.forgetting_exempt]).toEqual([undefined, undefined]);
  });

  test("retain maps importance and forgettingExempt onto its item", async () => {
    await new HindsightClient({ baseUrl: "http://localhost:8888" }).retain("bank", "Pinned fact.", {
      importance: 0.2,
      forgettingExempt: false,
    });
    const [item] = capturedItems(spy);
    expect([item.importance, item.forgetting_exempt]).toEqual([0.2, false]);
  });
});
