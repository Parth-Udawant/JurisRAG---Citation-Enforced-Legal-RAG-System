import type { ActKey, ChatRequest, ChatResponse } from "@/types/api";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export async function sendChat(
  question: string,
  act: ActKey,
  conversationId: string | null,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const payload: ChatRequest = {
    question,
    act,
    conversation_id: conversationId,
  };

  const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
    }
    throw new Error(detail);
  }

  return response.json() as Promise<ChatResponse>;
}
