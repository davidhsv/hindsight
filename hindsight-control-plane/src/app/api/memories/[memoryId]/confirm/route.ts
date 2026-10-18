import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ memoryId: string }> }
) {
  try {
    const { memoryId } = await params;
    // Every confirmation field is optional, so an empty body is a valid request.
    const raw = await request.text();
    const body = raw ? JSON.parse(raw) : {};
    const bankId = body.bank_id || request.nextUrl.searchParams.get("bank_id");

    if (!bankId) {
      return NextResponse.json(
        localizeApiErrorPayload(request, {
          error: "bank_id is required",
          errorKey: "api.errors.validation.bankIdRequired",
        }),
        { status: 400 }
      );
    }

    // bank_id is a routing param, not part of the body.
    const { source_key, event_time, reason } = body;

    const result = await sdk.confirmMemory({
      client: lowLevelClient,
      path: { bank_id: bankId, memory_id: memoryId },
      body: { source_key, event_time, reason },
    });

    // A 409 (the bank's adaptive memory mode does not reinforce) or 404 keeps its
    // status and the server's reason in `details`.
    return respondWithSdk(result, "Failed to confirm memory", {
      request,
      errorKey: "api.errors.memories.confirm",
    });
  } catch (error) {
    console.error("Error confirming memory:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to confirm memory",
        errorKey: "api.errors.memories.confirm",
      }),
      { status: 500 }
    );
  }
}
