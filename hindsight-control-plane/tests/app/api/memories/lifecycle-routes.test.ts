import type { NextRequest } from "next/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { confirmMemory, updateMemoryLifecycle } = vi.hoisted(() => ({
  confirmMemory: vi.fn(),
  updateMemoryLifecycle: vi.fn(),
}));

vi.mock("@/lib/hindsight-client", () => ({
  sdk: { confirmMemory, updateMemoryLifecycle },
  lowLevelClient: {},
}));

import { POST as confirm } from "@/app/api/memories/[memoryId]/confirm/route";
import { PATCH as updateLifecycle } from "@/app/api/memories/[memoryId]/lifecycle/route";

function makeRequest(url: string, method: string, body?: unknown): NextRequest {
  const request = new Request(url, {
    method,
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  // The routes read `request.nextUrl.searchParams` for a bank_id outside the body.
  return Object.assign(request, { nextUrl: new URL(url) }) as unknown as NextRequest;
}

const params = (memoryId: string) => ({ params: Promise.resolve({ memoryId }) });

describe("POST /api/memories/[memoryId]/confirm", () => {
  beforeEach(() => {
    confirmMemory.mockReset();
  });

  it("forwards the confirmation fields and keeps bank_id out of the body", async () => {
    const upstream = { memory_id: "m1", recorded: true, deduplicated: false, outcome: "applied" };
    confirmMemory.mockResolvedValue({ data: upstream });

    const response = await confirm(
      makeRequest("http://localhost/api/memories/m1/confirm", "POST", {
        bank_id: "b",
        source_key: "ticket-42",
        event_time: "2026-01-15T00:00:00Z",
        reason: "checked",
      }),
      params("m1")
    );

    const arg = confirmMemory.mock.calls[0][0];
    expect(arg.path).toEqual({ bank_id: "b", memory_id: "m1" });
    expect(arg.body).toEqual({
      source_key: "ticket-42",
      event_time: "2026-01-15T00:00:00Z",
      reason: "checked",
    });
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual(upstream);
  });

  it("accepts an empty body with bank_id in the query string", async () => {
    confirmMemory.mockResolvedValue({ data: { memory_id: "m1", recorded: true } });

    const response = await confirm(
      makeRequest("http://localhost/api/memories/m1/confirm?bank_id=b", "POST"),
      params("m1")
    );

    expect(confirmMemory.mock.calls[0][0].path).toEqual({ bank_id: "b", memory_id: "m1" });
    expect(response.status).toBe(200);
  });

  it("answers 400 without a bank_id and never calls the dataplane", async () => {
    const response = await confirm(
      makeRequest("http://localhost/api/memories/m1/confirm", "POST", {}),
      params("m1")
    );

    expect(response.status).toBe(400);
    expect(confirmMemory).not.toHaveBeenCalled();
  });

  it("keeps a mode refusal's status and the server's reason", async () => {
    confirmMemory.mockResolvedValue({
      error: { detail: "adaptive memory mode 'off' does not reinforce" },
      response: { status: 409 },
    });

    const response = await confirm(
      makeRequest("http://localhost/api/memories/m1/confirm?bank_id=b", "POST"),
      params("m1")
    );

    expect(response.status).toBe(409);
    expect((await response.json()).details).toBe("adaptive memory mode 'off' does not reinforce");
  });
});

describe("PATCH /api/memories/[memoryId]/lifecycle", () => {
  beforeEach(() => {
    updateMemoryLifecycle.mockReset();
  });

  it("forwards every lifecycle field and keeps bank_id out of the body", async () => {
    updateMemoryLifecycle.mockResolvedValue({ data: { memory_id: "m1", stored: true } });

    const fields = {
      importance: 0.9,
      forgetting_exempt: false,
      exemption_reason: "compliance",
      truth_status: "superseded",
      superseded_by: "m2",
      reason: "upgraded to Java 25",
      hold: true,
      retention_state: "active",
    };
    const response = await updateLifecycle(
      makeRequest("http://localhost/api/memories/m1/lifecycle", "PATCH", {
        bank_id: "b",
        ...fields,
      }),
      params("m1")
    );

    const arg = updateMemoryLifecycle.mock.calls[0][0];
    expect(arg.path).toEqual({ bank_id: "b", memory_id: "m1" });
    expect(arg.body).toEqual(fields);
    expect(response.status).toBe(200);
  });

  it("keeps a 400 for a successor in another scope, with the reason", async () => {
    updateMemoryLifecycle.mockResolvedValue({
      error: { detail: "superseded_by has different tags" },
      response: { status: 400 },
    });

    const response = await updateLifecycle(
      makeRequest("http://localhost/api/memories/m1/lifecycle?bank_id=b", "PATCH", {
        truth_status: "superseded",
        superseded_by: "m2",
      }),
      params("m1")
    );

    expect(response.status).toBe(400);
    expect((await response.json()).details).toBe("superseded_by has different tags");
  });
});
