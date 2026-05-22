export type AgentStep = {
  id: string;
  label: string;
  status: "queued" | "running" | "done" | "warning" | "error";
  detail: string;
  payload?: Record<string, unknown>;
};

export type Source = {
  source: string;
  topic: string | null;
  score: number;
  excerpt: string;
  summary?: string | null;
  intent?: string | null;
  resolution?: string | null;
};

export type ChatResponse = {
  answer: string;
  topic: string;
  topic_label: string;
  confidence: number;
  sources: Source[];
  steps: AgentStep[];
  escalation_packet?: string | null;
  used_llm: boolean;
};

export type StatsResponse = {
  collection: string;
  points_count: number;
  topics: Array<{ topic: string; label: string; chunks: number }>;
  api_ready: boolean;
  qdrant_ready: boolean;
};
