// @vitest-environment jsdom
/**
 * The adaptive memory views: the claim annotation on a recall result and the
 * lifecycle block of the memory detail dialog.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string, values?: Record<string, unknown>) =>
    values ? `${key}:${JSON.stringify(values)}` : key,
}));

import { LifecycleBadge, MemoryLifecycleSection } from "@/components/memory-lifecycle";
import type { MemoryLifecycle } from "@/lib/api";

afterEach(cleanup);

const retention = {
  materialized: true,
  retrievability: 0.42,
  stability_days: 31.25,
  importance: 0.5,
  last_reinforced_at: "2026-01-01T00:00:00Z",
  reinforcement_count: 3,
  access_count: 0,
  forgetting_exempt: false,
};

describe("LifecycleBadge", () => {
  it("renders nothing for a plain active result, which carries no annotation", () => {
    const { container } = render(<LifecycleBadge lifecycle={undefined} />);
    expect(container.innerHTML).toBe("");
  });

  it("names the status and the end of a superseded claim's validity", () => {
    render(
      <LifecycleBadge
        lifecycle={{
          truth_status: "superseded",
          valid_to: "2026-02-01T00:00:00Z",
          superseded_by: "m2",
        }}
      />
    );
    const badge = screen.getByText(/status\.superseded/);
    expect(badge.textContent).toContain("until:");
    expect(badge.getAttribute("title")).toContain("m2");
  });
});

describe("MemoryLifecycleSection", () => {
  it("shows a retired claim beside its retention, which stays independent", () => {
    const lifecycle: MemoryLifecycle = {
      memory_id: "m1",
      archived: false,
      stored: true,
      retention,
      claim: {
        truth_status: "superseded",
        valid_from: null,
        valid_to: "2026-02-01T00:00:00Z",
        retired_via: "supersedes",
        superseded_by_id: "m2",
        reason: "upgraded to Java 25",
      },
    };
    render(<MemoryLifecycleSection lifecycle={lifecycle} />);

    expect(screen.getByText("status.superseded")).toBeTruthy();
    expect(screen.getByText("m2")).toBeTruthy();
    expect(screen.getByText("supersedes")).toBeTruthy();
    expect(screen.getByText("upgraded to Java 25")).toBeTruthy();
    expect(screen.getByText(/^unknownStart – /)).toBeTruthy();
    expect(screen.getByText("42%")).toBeTruthy();
    expect(screen.getByText('stabilityValue:{"days":"31.3"}')).toBeTruthy();
    expect(screen.queryByText("archived")).toBeNull();
  });

  it("reads a memory without a claim row as active and shows its exemption and archive", () => {
    render(
      <MemoryLifecycleSection
        lifecycle={{
          memory_id: "m1",
          archived: true,
          stored: true,
          retention: {
            ...retention,
            forgetting_exempt: true,
            exemption_reason: "hold",
            archived_at: null,
          },
          claim: null,
          shadow: { would_status: "superseded" },
        }}
      />
    );

    expect(screen.getByText("status.active")).toBeTruthy();
    expect(screen.queryByText("validity")).toBeNull();
    expect(screen.getByText("hold")).toBeTruthy();
    expect(screen.getByText("archived")).toBeTruthy();
    expect(screen.getByText("observeShadow")).toBeTruthy();
    expect(screen.getByText("status.superseded")).toBeTruthy();
  });
});
