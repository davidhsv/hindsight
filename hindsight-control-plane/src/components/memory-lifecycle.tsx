"use client";

import type { ReactNode } from "react";
import { useTranslations } from "next-intl";
import type { MemoryLifecycle, RecallLifecycle } from "@/lib/api";
import { cn } from "@/lib/utils";

type TruthStatus = "active" | "uncertain" | "superseded" | "refuted";

const STATUS_CLASS: Record<TruthStatus, string> = {
  active: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  uncertain: "bg-amber-500/10 text-amber-700 dark:text-amber-400",
  superseded: "bg-slate-500/10 text-slate-700 dark:text-slate-300",
  refuted: "bg-red-500/10 text-red-700 dark:text-red-400",
};

function asTruthStatus(value: string | null | undefined): TruthStatus {
  return value === "uncertain" || value === "superseded" || value === "refuted" ? value : "active";
}

function fmtDate(value: string | null | undefined): string | null {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString();
}

/**
 * The claim annotation a lifecycle-mode recall puts on a result whose claim is not
 * plain active: superseded and refuted memories are history, uncertain ones are
 * contested. Plain active results carry no annotation and render nothing.
 */
export function LifecycleBadge({
  lifecycle,
  className,
}: {
  lifecycle: RecallLifecycle | null | undefined;
  className?: string;
}) {
  const t = useTranslations("memoryLifecycle");
  if (!lifecycle) return null;

  const status = asTruthStatus(lifecycle.truth_status);
  const until = fmtDate(lifecycle.valid_to);
  return (
    <span
      className={cn("px-2 py-0.5 rounded whitespace-nowrap", STATUS_CLASS[status], className)}
      title={
        lifecycle.superseded_by
          ? t("supersededByTitle", { id: lifecycle.superseded_by })
          : undefined
      }
    >
      {t(`status.${status}`)}
      {until && ` · ${t("until", { date: until })}`}
    </span>
  );
}

/**
 * What adaptive memory knows about one memory, as `GET /memories/{id}` returns it
 * under `lifecycle`: what should be believed about it (claim) and how readily it
 * comes back (retention). The two are independent, so they are shown side by side
 * rather than merged into one score.
 */
export function MemoryLifecycleSection({ lifecycle }: { lifecycle: MemoryLifecycle }) {
  const t = useTranslations("memoryLifecycle");
  const { claim, retention } = lifecycle;
  const status = asTruthStatus(claim?.truth_status);
  const from = fmtDate(claim?.valid_from);
  const to = fmtDate(claim?.valid_to);

  const rows: { label: string; value: ReactNode }[] = [
    {
      label: t("truthStatus"),
      value: (
        <span className={cn("px-2 py-0.5 rounded text-xs", STATUS_CLASS[status])}>
          {t(`status.${status}`)}
        </span>
      ),
    },
  ];
  if (from || to) {
    rows.push({
      label: t("validity"),
      value: `${from ?? t("unknownStart")} – ${to ?? t("openEnd")}`,
    });
  }
  if (claim?.superseded_by_id) {
    rows.push({
      label: t("supersededBy"),
      value: <code className="text-xs font-mono break-all">{claim.superseded_by_id}</code>,
    });
  }
  if (claim?.retired_via) rows.push({ label: t("retiredVia"), value: claim.retired_via });
  if (claim?.reason) rows.push({ label: t("reason"), value: claim.reason });
  rows.push(
    {
      label: t("retrievability"),
      value: `${Math.round(retention.retrievability * 100)}%`,
    },
    {
      label: t("stability"),
      value: t("stabilityValue", { days: retention.stability_days.toFixed(1) }),
    },
    { label: t("reinforcements"), value: String(retention.reinforcement_count) },
    { label: t("importance"), value: retention.importance.toFixed(2) }
  );
  if (retention.forgetting_exempt) {
    rows.push({
      label: t("exempt"),
      value: retention.exemption_reason || t("yes"),
    });
  }
  if (lifecycle.archived) {
    rows.push({
      label: t("archived"),
      value: fmtDate(retention.archived_at) ?? t("yes"),
    });
  }
  if (lifecycle.shadow) {
    rows.push({
      label: t("observeShadow"),
      value: t(`status.${asTruthStatus(lifecycle.shadow.would_status)}`),
    });
  }

  return (
    <div>
      <div className="text-xs font-bold text-muted-foreground uppercase mb-1">
        {t("sectionTitle")}
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        {rows.map((row) => (
          <div key={row.label} className="contents">
            <dt className="text-muted-foreground">{row.label}</dt>
            <dd>{row.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
