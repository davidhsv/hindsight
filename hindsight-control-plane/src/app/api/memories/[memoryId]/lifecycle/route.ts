import { NextRequest, NextResponse } from "next/server";
import { localizeApiErrorPayload } from "@/lib/i18n/api-errors";
import { lowLevelClient, sdk } from "@/lib/hindsight-client";
import { respondWithSdk } from "@/lib/sdk-response";

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ memoryId: string }> }
) {
  try {
    const { memoryId } = await params;
    const body = await request.json();
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

    // Lifecycle fields only; bank_id is a routing param, not part of the body.
    const {
      importance,
      forgetting_exempt,
      exemption_reason,
      truth_status,
      superseded_by,
      reason,
      hold,
      retention_state,
    } = body;

    const result = await sdk.updateMemoryLifecycle({
      client: lowLevelClient,
      path: { bank_id: bankId, memory_id: memoryId },
      body: {
        importance,
        forgetting_exempt,
        exemption_reason,
        truth_status,
        superseded_by,
        reason,
        hold,
        retention_state,
      },
    });

    // A 409 (the mode does not allow the change, or the memory is already retired),
    // 400 (a successor in another scope) or 404 keeps its status and the server's
    // reason in `details`.
    return respondWithSdk(result, "Failed to update memory lifecycle", {
      request,
      errorKey: "api.errors.memories.lifecycle",
    });
  } catch (error) {
    console.error("Error updating memory lifecycle:", error);
    return NextResponse.json(
      localizeApiErrorPayload(request, {
        error: "Failed to update memory lifecycle",
        errorKey: "api.errors.memories.lifecycle",
      }),
      { status: 500 }
    );
  }
}
