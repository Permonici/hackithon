export type RetrievalTolerance = "strict" | "balanced" | "broad";

export type UserInfo = {
  name?: string | null;
  clinic?: string | null;
  role?: string | null;
  software_version?: string | null;
  contact?: string | null;
};

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

export type UsageEstimate = {
  chat_model: string;
  embedding_model: string;
  estimated_chat_input_tokens: number;
  estimated_chat_output_tokens: number;
  estimated_embedding_tokens: number;
  estimated_chat_cost_usd: number;
  estimated_embedding_cost_usd: number;
  total_estimated_cost_usd: number;
  note: string;
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
  session_id?: string | null;
  user?: UserInfo | null;
  retrieval_tolerance: RetrievalTolerance;
  usage?: UsageEstimate | null;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  response?: ChatResponse;
};

export type StatsResponse = {
  collection: string;
  points_count: number;
  topics: Array<{ topic: string; label: string; chunks: number }>;
  api_ready: boolean;
  qdrant_ready: boolean;
};

export type PriceInfoResponse = {
  currency: string;
  chat_model: string;
  embedding_model: string;
  chat_input_price_per_1m: number;
  chat_output_price_per_1m: number;
  embedding_price_per_1m: number;
  note: string;
  reference_url: string;
};

export type ChatRequestPayload = {
  message: string;
  strict_mode: boolean;
  top_k: number;
  retrieval_tolerance: RetrievalTolerance;
  session_id: string;
  user?: UserInfo | null;
};
