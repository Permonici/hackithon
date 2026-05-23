export type RetrievalTolerance = "strict" | "balanced" | "broad";

export type UserInfo = {
  name?: string | null;
  clinic?: string | null;
  role?: string | null;
  software_version?: string | null;
  contact?: string | null;
  patient_name?: string | null;
  patient_identifier?: string | null;
  patient_age?: string | null;
  patient_city?: string | null;
  patient_address?: string | null;
  patient_phone?: string | null;
  patient_email?: string | null;
  preferred_contact_method?: "phone" | "email" | "sms" | "any" | null;
  urgency?: "low" | "normal" | "high" | "critical" | null;
  problem_summary?: string | null;
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

export type TriageResult = {
  urgency: "low" | "normal" | "high" | "critical";
  label: string;
  confidence: number;
  reasons: string[];
  recommendation: string;
  needs_immediate_care: boolean;
};

export type ClinicOption = {
  name: string;
  city: string;
  address: string;
  distance_km?: number | null;
  accepting_new_patients: boolean;
  services: string[];
  map_x: number;
  map_y: number;
  phone: string;
  email: string;
  earliest_slot?: string | null;
  note: string;
};

export type AppointmentProposal = {
  status: "pre_reserved" | "needs_contact" | "unavailable";
  clinic_name?: string | null;
  slot_start?: string | null;
  reservation_id?: string | null;
  message: string;
  confirmation_required: boolean;
};

export type ChatResponse = {
  answer: string;
  topic: string;
  topic_label: string;
  confidence: number;
  answer_confidence?: number | null;
  sources: Source[];
  steps: AgentStep[];
  escalation_packet?: string | null;
  used_llm: boolean;
  session_id?: string | null;
  user?: UserInfo | null;
  retrieval_tolerance: RetrievalTolerance;
  usage?: UsageEstimate | null;
  triage?: TriageResult | null;
  clinics: ClinicOption[];
  appointment?: AppointmentProposal | null;
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

export type ChatRequestPayload = {
  message: string;
  strict_mode: boolean;
  top_k: number;
  retrieval_tolerance: RetrievalTolerance;
  session_id: string;
  user?: UserInfo | null;
};

export type FrequentQuery = {
  query: string;
  count: number;
};

export type CacheStats = {
  active_entries: number;
  total_tracked_queries: number;
  top_frequent: FrequentQuery[];
};
