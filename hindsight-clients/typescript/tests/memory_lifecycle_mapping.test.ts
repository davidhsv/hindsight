/**
 * Tests that confirmMemory() and updateMemoryLifecycle() map their camelCase options onto the
 * snake_case request body and the path, and surface server refusals as HindsightError.
 *
 * Kept in step with the Python wrapper's confirm_memory / update_memory_lifecycle. No server
 * required: the generated SDK calls are mocked.
 */

import { HindsightClient, HindsightError } from "../src";
import * as sdk from "../generated/sdk.gen";

const client = () => new HindsightClient({ baseUrl: "http://localhost:8888" });

describe("confirmMemory", () => {
  let spy: jest.SpyInstance;

  beforeEach(() => {
    spy = jest.spyOn(sdk, "confirmMemory").mockResolvedValue({
      data: { memory_id: "m1", recorded: true, deduplicated: false, outcome: "applied" },
    } as any);
  });

  afterEach(() => {
    spy.mockRestore();
  });

  test("every option reaches the request body and the ids reach the path", async () => {
    const result = await client().confirmMemory("bank", "m1", {
      sourceKey: "ticket-42",
      eventTime: "2026-01-15T00:00:00Z",
      reason: "verified in prod",
    });

    const arg = spy.mock.calls[0][0] as any;
    expect(arg.path).toEqual({ bank_id: "bank", memory_id: "m1" });
    expect(arg.body).toEqual({
      source_key: "ticket-42",
      event_time: "2026-01-15T00:00:00Z",
      reason: "verified in prod",
    });
    expect(result.outcome).toBe("applied");
  });

  test("no options sends an empty confirmation", async () => {
    await client().confirmMemory("bank", "m1");

    const body = (spy.mock.calls[0][0] as any).body;
    expect([body.source_key, body.event_time, body.reason]).toEqual([
      undefined,
      undefined,
      undefined,
    ]);
  });

  test("a mode refusal throws HindsightError with the status", async () => {
    spy.mockResolvedValue({
      error: { detail: "adaptive memory mode is off" },
      response: { status: 409 },
    } as any);

    await expect(client().confirmMemory("bank", "m1")).rejects.toMatchObject({
      name: "HindsightError",
      statusCode: 409,
    });
    await expect(client().confirmMemory("bank", "m1")).rejects.toBeInstanceOf(HindsightError);
  });
});

describe("updateMemoryLifecycle", () => {
  let spy: jest.SpyInstance;

  beforeEach(() => {
    spy = jest.spyOn(sdk, "updateMemoryLifecycle").mockResolvedValue({
      data: { memory_id: "m1", archived: false, stored: true, retention: {} },
    } as any);
  });

  afterEach(() => {
    spy.mockRestore();
  });

  test("every option maps to its snake_case field", async () => {
    await client().updateMemoryLifecycle("bank", "m1", {
      importance: 0.9,
      forgettingExempt: true,
      exemptionReason: "compliance",
      truthStatus: "superseded",
      supersededBy: "m2",
      reason: "upgraded to Java 25",
      hold: false,
      retentionState: "active",
    });

    const arg = spy.mock.calls[0][0] as any;
    expect(arg.path).toEqual({ bank_id: "bank", memory_id: "m1" });
    expect(arg.body).toEqual({
      importance: 0.9,
      forgetting_exempt: true,
      exemption_reason: "compliance",
      truth_status: "superseded",
      superseded_by: "m2",
      reason: "upgraded to Java 25",
      hold: false,
      retention_state: "active",
    });
  });

  test("omitted options stay out of the body", async () => {
    await client().updateMemoryLifecycle("bank", "m1", { truthStatus: "refuted" });

    const body = (spy.mock.calls[0][0] as any).body;
    expect(JSON.parse(JSON.stringify(body))).toEqual({ truth_status: "refuted" });
  });

  test("a 404 throws HindsightError", async () => {
    spy.mockResolvedValue({
      error: { detail: "Memory unit 'm1' not found" },
      response: { status: 404 },
    } as any);

    await expect(
      client().updateMemoryLifecycle("bank", "m1", { importance: 0.2 })
    ).rejects.toMatchObject({ statusCode: 404 });
  });
});
