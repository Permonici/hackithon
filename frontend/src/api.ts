import type { AgentStep, CacheStats, ChatRequestPayload, ChatResponse, StatsResponse } from "./types";

const API_URL = import.meta.env.VITE_API_URL || "/api";

export async function fetchStats(): Promise<StatsResponse> {
  const response = await fetch(`${API_URL}/stats`);
  if (!response.ok) {
    throw new Error("Statistiky se nepodarilo nacist.");
  }
  return response.json();
}

export async function sendChatStream(
  payload: ChatRequestPayload,
  onStep: (step: AgentStep) => void,
  signal?: AbortSignal
): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal
  });

  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Stream se nepodarilo spustit.");
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
        throw new Error(body.message || body.detail || "Backend pri odpovedi vratil chybu.");
      }
    }
  }

  if (!finalResponse) {
    throw new Error("Stream skoncil bez finalni odpovedi.");
  }
  return finalResponse;
}

export async function fetchCacheStats(): Promise<CacheStats> {
  const response = await fetch(`${API_URL}/cache/stats`);
  if (!response.ok) {
    throw new Error("Cache statistiky se nepodarilo nacist.");
  }
  return response.json();
}

export async function ingestData(): Promise<void> {
  const response = await fetch(`${API_URL}/ingest`, { method: "POST" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "Indexaci se nepodarilo spustit.");
  }
}

function parseSse(rawEvent: string): { event: string; data: unknown } | null {
  const lines = rawEvent.split("\n");
  const eventLine = lines.find((line) => line.startsWith("event:"));
  const dataLine = lines.find((line) => line.startsWith("data:"));
  if (!eventLine || !dataLine) return null;
  const rawData = dataLine.replace("data:", "").trim();
  try {
    return {
      event: eventLine.replace("event:", "").trim(),
      data: JSON.parse(rawData)
    };
  } catch {
    return null;
  }
}
