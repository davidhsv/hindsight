import type { NextRequest } from "next/server";
import { describe, expect, it, vi } from "vitest";

const { recallMemories } = vi.hoisted(() => ({
  recallMemories: vi.fn(),
}));

vi.mock("@/lib/hindsight-client", () => ({
  sdk: { recallMemories },
  lowLevelClient: {},
}));

import { POST } from "@/app/api/recall/route";

describe("POST /api/recall", () => {
  it("returns source_facts and source_facts_truncated from the dataplane", async () => {
    const upstream = {
      results: [{ id: "o1", type: "observation", source_fact_ids: ["f1"] }],
      source_facts: { f1: { id: "f1", text: "evidence" } },
      source_facts_truncated: true,
    };
    recallMemories.mockResolvedValue({ data: upstream });

    const request = new Request("http://localhost/api/recall", {
      method: "POST",
      body: JSON.stringify({
        bank_id: "b",
        query: "q",
        include: { source_facts: { max_tokens: 100 } },
      }),
    });
    const response = await POST(request as unknown as NextRequest);

    expect(recallMemories.mock.calls[0][0].body.include).toEqual({
      source_facts: { max_tokens: 100 },
    });
    expect(await response.json()).toEqual(upstream);
  });

  it("forwards the adaptive memory lifecycle view and as_of", async () => {
    recallMemories.mockResolvedValue({ data: { results: [] } });

    const request = new Request("http://localhost/api/recall", {
      method: "POST",
      body: JSON.stringify({
        bank_id: "b",
        query: "q",
        lifecycle_view: "historical",
        as_of: "2026-01-15T00:00:00Z",
      }),
    });
    await POST(request as unknown as NextRequest);

    const body = recallMemories.mock.calls[recallMemories.mock.calls.length - 1][0].body;
    expect([body.lifecycle_view, body.as_of]).toEqual(["historical", "2026-01-15T00:00:00Z"]);
  });

  it("forwards the context admission purpose and context", async () => {
    recallMemories.mockResolvedValue({ data: { results: [] } });

    const request = new Request("http://localhost/api/recall", {
      method: "POST",
      body: JSON.stringify({
        bank_id: "b",
        query: "q",
        purpose: "auto_context",
        context: { repository: "github.com/acme/payments", module: "auth" },
      }),
    });
    await POST(request as unknown as NextRequest);

    const body = recallMemories.mock.calls[recallMemories.mock.calls.length - 1][0].body;
    expect(body.purpose).toBe("auto_context");
    expect(body.context).toEqual({ repository: "github.com/acme/payments", module: "auth" });
  });
});
