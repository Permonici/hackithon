import type { AgentStep, ChatRequestPayload, ChatResponse, PriceInfoResponse, StatsResponse } from "./types";

const API_URL = import.meta.env.VITE_API_URL || "/api";

export async function fetchStats(): Promise<StatsResponse> {
  const response = await fetch(`${API_URL}/stats`);
  if (!response.ok) {
    throw new Error("Statistiky se nepodařilo načíst.");
  }
  return response.json();
}

export async function fetchPricing(): Promise<PriceInfoResponse> {
  const response = await fetch(`${API_URL}/pricing`);
  if (!response.ok) {
    throw new Error("Ceník se nepodařilo načíst.");
  }
  return response.json();
}

export async function sendChatStream(
  payload: ChatRequestPayload,
  onStep: (step: AgentStep) => void
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Stream se nepodařilo spustit.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: ChatResponse | null = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const rawEvent of events) {
      const parsed = parseSse(rawEvent);
      if (!parsed) continue;
      if (parsed.event === "step") {
        onStep(parsed.data as AgentStep);
      }
      if (parsed.event === "final") {
        finalResponse = parsed.data as ChatResponse;
      }
      if (parsed.event === "error") {
        const body = parsed.data as { message?: string; detail?: string };
        throw new Error(body.message || body.detail || "Backend při odpovědi vrátil chybu.");
      }
    }
  }

  if (!finalResponse) {
    throw new Error("Stream skončil bez finální odpovědi.");
  }
  return finalResponse;
}

export async function ingestData(): Promise<void> {
  const response = await fetch(`${API_URL}/ingest`, { method: "POST" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Indexaci se nepodařilo spustit.");
  }
}

function parseSse(rawEvent: string): { event: string; data: unknown } | null {
  const eventLine = rawEvent.split("\n").find((line) => line.startsWith("event:"));
  const dataLine = rawEvent.split("\n").find((line) => line.startsWith("data:"));
  if (!eventLine || !dataLine) return null;
  return {
    event: eventLine.replace("event:", "").trim(),
    data: JSON.parse(dataLine.replace("data:", "").trim())
  };
}
