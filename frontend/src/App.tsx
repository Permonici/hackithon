/* eslint-disable @typescript-eslint/no-explicit-any */
declare global {
  interface Window {
    webkitSpeechRecognition?: new () => SpeechRecognition;
    SpeechRecognition?: new () => SpeechRecognition;
  }
}

import {
  Activity,
  AlertTriangle,
  Bot,
  CheckCircle2,
  Clock3,
  Copy,
  Database,
  DollarSign,
  FileSearch,
  Gauge,
  History,
  Loader2,
  MessageSquare,
  Mic,
  MicOff,
  RefreshCw,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  TrendingUp,
  User,
  Volume2,
  VolumeX,
  Zap
} from "lucide-react";
import { type FormEvent, type KeyboardEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchCacheStats, fetchPricing, fetchStats, ingestData, sendChatStream } from "./api";
import type {
  AgentStep,
  CacheStats,
  ChatMessage,
  ChatResponse,
  PriceInfoResponse,
  RetrievalTolerance,
  Source,
  StatsResponse,
  UserInfo
} from "./types";

const demoQuestions = [
  "Nejde mi odeslat ePoukaz, systém píše chybu s úhradou. Co mám zkontrolovat?",
  "Po instalaci certifikátu se uživatel nemůže přihlásit do XDENTu.",
  "Dokument se netiskne správně a potřebuji upravit šablonu tisku.",
  "Kde v kalendáři změním termín objednaného pacienta?"
];

const emptySteps: AgentStep[] = [
  { id: "classify", label: "Téma", status: "queued", detail: "Čekám na dotaz." },
  { id: "retrieve", label: "Retrieval", status: "queued", detail: "Připraveno vyhledat podobné hovory." },
  { id: "validate", label: "Jistota", status: "queued", detail: "Ověřím, jestli zdroje stačí." },
  { id: "answer", label: "Odpověď", status: "queued", detail: "Výsledek bude krátký a se zdrojem." }
];

const defaultUser: UserInfo = {
  name: "",
  clinic: "",
  role: "",
  software_version: "",
  contact: ""
};

const toleranceOptions: Array<{ value: RetrievalTolerance; label: string; hint: string }> = [
  { value: "strict", label: "Přesné", hint: "vyšší jistota" },
  { value: "balanced", label: "Vyvážené", hint: "doporučeno" },
  { value: "broad", label: "Širší", hint: "víc tolerance" }
];

const STORAGE_MESSAGES = "xdent.chat.messages";
const STORAGE_USER = "xdent.chat.user";
const STORAGE_TOLERANCE = "xdent.chat.tolerance";
const STORAGE_SESSION = "xdent.chat.session";
const STORAGE_VOICE_OUTPUT = "xdent.chat.voiceOutput";

function App() {
  const [activeTab, setActiveTab] = useState<"chat" | "history" | "price">("chat");
  const [message, setMessage] = useState(demoQuestions[0]);
  const [strictMode, setStrictMode] = useState(false);
  const [retrievalTolerance, setRetrievalTolerance] = useState<RetrievalTolerance>(() =>
    loadJson<RetrievalTolerance>(STORAGE_TOLERANCE, "balanced")
  );
  const [userInfo, setUserInfo] = useState<UserInfo>(() => loadJson<UserInfo>(STORAGE_USER, defaultUser));
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadJson<ChatMessage[]>(STORAGE_MESSAGES, []));
  const [sessionId] = useState(() => loadOrCreateSessionId());
  const [loading, setLoading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [pricing, setPricing] = useState<PriceInfoResponse | null>(null);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [liveSteps, setLiveSteps] = useState<AgentStep[]>(emptySteps);

  // ── voice ─────────────────────────────────────────────────────────────────
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState(() =>
    loadJson<boolean>(STORAGE_VOICE_OUTPUT, false)
  );
  const [ttsVoices, setTtsVoices] = useState<SpeechSynthesisVoice[]>([]);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const voiceInputSupported = typeof window !== "undefined" &&
    !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  const voiceOutputSupported = typeof window !== "undefined" && !!window.speechSynthesis;

  // ── refs ──────────────────────────────────────────────────────────────────
  const abortRef = useRef<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // ── derived ───────────────────────────────────────────────────────────────
  const isIndexed = Boolean(stats?.points_count && stats.points_count > 0);
  const savedResponse = useMemo(() => [...messages].reverse().find((item) => item.response)?.response ?? null, [messages]);
  const visibleResponse = response ?? savedResponse;
  const steps = loading ? liveSteps : visibleResponse?.steps ?? liveSteps;
  const sources = visibleResponse?.sources ?? sourcesFromSteps(liveSteps);
  const topScore = Math.max(0, ...sources.map((source) => source.score));
  const sessionCost = useMemo(
    () => messages.reduce((sum, item) => sum + (item.response?.usage?.total_estimated_cost_usd ?? 0), 0),
    [messages]
  );

  const topicCoverage = useMemo(() => {
    const counts = stats?.topics.map((topic) => topic.chunks) ?? [];
    const max = Math.max(1, ...counts);
    return (stats?.topics ?? []).slice(0, 7).map((topic) => ({
      ...topic,
      width: Math.max(6, Math.round((topic.chunks / max) * 100))
    }));
  }, [stats]);

  // ── effects ───────────────────────────────────────────────────────────────
  useEffect(() => { refreshStats(); refreshPricing(); refreshCacheStats(); }, []);

  useEffect(() => { saveJson(STORAGE_MESSAGES, messages); }, [messages]);
  useEffect(() => { saveJson(STORAGE_USER, userInfo); }, [userInfo]);
  useEffect(() => { saveJson(STORAGE_TOLERANCE, retrievalTolerance); }, [retrievalTolerance]);
  useEffect(() => { saveJson(STORAGE_VOICE_OUTPUT, voiceOutputEnabled); }, [voiceOutputEnabled]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Load available TTS voices (async on some browsers).
  useEffect(() => {
    if (!voiceOutputSupported) return;
    const load = () => setTtsVoices(window.speechSynthesis.getVoices());
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", load);
  }, [voiceOutputSupported]);

  // Stop speaking when voice output is disabled mid-stream.
  useEffect(() => {
    if (!voiceOutputEnabled && isSpeaking) {
      window.speechSynthesis?.cancel();
      setIsSpeaking(false);
    }
  }, [voiceOutputEnabled, isSpeaking]);

  // ── data fetchers ─────────────────────────────────────────────────────────
  async function refreshStats() {
    try { setStats(await fetchStats()); } catch { setStats(null); }
  }

  async function refreshPricing() {
    try { setPricing(await fetchPricing()); } catch { setPricing(null); }
  }

  async function refreshCacheStats() {
    try { setCacheStats(await fetchCacheStats()); } catch { setCacheStats(null); }
  }

  // ── voice input ───────────────────────────────────────────────────────────
  function startListening() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;

    recognitionRef.current?.abort();
    const recognition = new SR();
    recognition.lang = "cs-CZ";
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = Array.from(event.results)
        .map((r) => r[0].transcript)
        .join("");
      setMessage(transcript);
      if (event.results[event.results.length - 1].isFinal) {
        setIsListening(false);
      }
    };
    recognition.onerror = () => setIsListening(false);
    recognition.onend = () => setIsListening(false);

    recognitionRef.current = recognition;
    recognition.start();
    setIsListening(true);
  }

  function stopListening() {
    recognitionRef.current?.stop();
    setIsListening(false);
  }

  // ── voice output ──────────────────────────────────────────────────────────
  const speak = useCallback((text: string) => {
    if (!voiceOutputEnabled || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();

    // Strip markdown/source annotations for cleaner TTS.
    const cleaned = text.replace(/Zdroj:.*$/m, "").replace(/\[.*?\]/g, "").trim();
    const utterance = new SpeechSynthesisUtterance(cleaned);
    utterance.lang = "cs-CZ";
    utterance.rate = 1.05;

    const czechVoice = ttsVoices.find((v) => v.lang.startsWith("cs"));
    if (czechVoice) utterance.voice = czechVoice;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  }, [voiceOutputEnabled, ttsVoices]);

  function stopSpeaking() {
    window.speechSynthesis?.cancel();
    setIsSpeaking(false);
  }

  // ── send message ──────────────────────────────────────────────────────────
  async function handleSend(event?: FormEvent) {
    event?.preventDefault();
    const text = message.trim();
    if (!text || !isIndexed || loading) return;

    // Abort any in-flight stream.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    // Stop ongoing TTS.
    if (isSpeaking) stopSpeaking();

    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      content: text,
      created_at: new Date().toISOString()
    };

    setMessages((current) => [...current, userMessage]);
    setMessage("");
    setLoading(true);
    setError(null);
    setResponse(null);
    setLiveSteps(emptySteps);

    try {
      const result = await sendChatStream(
        {
          message: text,
          strict_mode: strictMode,
          top_k: retrievalTolerance === "broad" ? 8 : 6,
          retrieval_tolerance: retrievalTolerance,
          session_id: sessionId,
          user: compactUser(userInfo)
        },
        (step) => setLiveSteps((current) => mergeStep(current, step)),
        controller.signal
      );

      const assistantMessage: ChatMessage = {
        id: createId(),
        role: "assistant",
        content: result.answer,
        created_at: new Date().toISOString(),
        response: result
      };
      setResponse(result);
      setMessages((current) => [...current, assistantMessage]);

      // Auto-speak the response if voice output is enabled.
      if (voiceOutputEnabled) speak(result.answer);

      // Refresh cache stats to reflect new hit counts.
      void refreshCacheStats();
    } catch (err) {
      // Ignore abort errors – they are intentional.
      if (err instanceof Error && err.name === "AbortError") return;
      const errorText = err instanceof Error ? err.message : "Něco se nepodařilo.";
      setError(errorText);
      setMessages((current) => [
        ...current,
        {
          id: createId(),
          role: "assistant",
          content: errorText,
          created_at: new Date().toISOString()
        }
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function handleIngest() {
    setIndexing(true);
    setError(null);
    try {
      await ingestData();
      await refreshStats();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Indexace se nepodařila.");
    } finally {
      setIndexing(false);
    }
  }

  function clearHistory() {
    setMessages([]);
    setResponse(null);
    setLiveSteps(emptySteps);
  }

  async function copyHistory() {
    const transcript = messages
      .map((item) => `${item.role === "user" ? "Uživatel" : "Asistent"}: ${item.content}`)
      .join("\n\n");
    await navigator.clipboard?.writeText(transcript);
  }

  return (
    <div className="min-h-screen bg-mist text-ink">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-md bg-mint text-white">
              <Bot size={24} />
            </div>
            <div>
              <h1 className="text-xl font-semibold">XDENT AI Support</h1>
              <p className="text-sm text-slate-500">RAG operátor pro první úroveň podpory</p>
            </div>
          </div>

          <nav className="flex flex-wrap items-center gap-2">
            <TabButton active={activeTab === "chat"} icon={<MessageSquare size={17} />} label="Chat" onClick={() => setActiveTab("chat")} />
            <TabButton active={activeTab === "history"} icon={<History size={17} />} label="Historie" onClick={() => setActiveTab("history")} />
            <TabButton active={activeTab === "price"} icon={<DollarSign size={17} />} label="Cena & Cache" onClick={() => setActiveTab("price")} />
          </nav>

          <div className="flex flex-wrap items-center gap-2">
            <StatusPill icon={<Database size={16} />} label={isIndexed ? `${stats?.points_count} chunků` : "Index prázdný"} ok={isIndexed} />
            <StatusPill icon={<Sparkles size={16} />} label={stats?.api_ready ? "OpenAI ready" : "Chybí klíč"} ok={Boolean(stats?.api_ready)} />
            <button className="icon-button" onClick={() => { void refreshStats(); void refreshCacheStats(); }} title="Obnovit stav">
              <RefreshCw size={18} />
            </button>
            <button className="primary-button" onClick={handleIngest} disabled={indexing || !stats?.api_ready}>
              {indexing ? <Loader2 className="animate-spin" size={17} /> : <Zap size={17} />}
              {isIndexed ? "Přeindexovat" : "Indexovat"}
            </button>
          </div>
        </div>
      </header>

      {activeTab === "chat" && (
        <main className="mx-auto grid max-w-7xl gap-5 px-5 py-5 xl:grid-cols-[minmax(0,1fr)_390px]">
          <section className="space-y-5">
            <IndexBanner stats={stats} isIndexed={isIndexed} indexing={indexing} />

            <section className="panel flex min-h-[560px] flex-col">
              <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
                <PanelTitle icon={<MessageSquare size={19} />} title="Konverzace" subtitle={`${messages.length} zpráv v relaci`} />
                <div className="flex items-center gap-2">
                  {isSpeaking && (
                    <button
                      className="flex items-center gap-1 rounded-md border border-mint/40 bg-mint/10 px-2 py-1 text-xs text-mint"
                      onClick={stopSpeaking}
                      title="Zastavit přehrávání"
                    >
                      <Volume2 size={13} className="animate-pulse" />
                      Zastavit
                    </button>
                  )}
                  <div className="flex items-center gap-2 rounded-md bg-slate-50 px-3 py-2 text-xs text-slate-500">
                    <Clock3 size={15} />
                    {sessionId.slice(0, 8)}
                  </div>
                </div>
              </div>

              <div className="thin-scroll flex-1 space-y-4 overflow-y-auto pr-1">
                {messages.length === 0 && <WelcomeBlock voiceOutputEnabled={voiceOutputEnabled} />}
                {messages.map((item) => (
                  <ChatBubble key={item.id} message={item} />
                ))}
                {loading && <PendingBubble steps={liveSteps} />}
                <div ref={messagesEndRef} />
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                {demoQuestions.map((question) => (
                  <button key={question} className="sample-chip" onClick={() => setMessage(question)}>
                    {question}
                  </button>
                ))}
              </div>

              <form className="mt-4 flex flex-col gap-3 md:flex-row" onSubmit={handleSend}>
                <div className="relative flex-1">
                  <textarea
                    className="field-input min-h-24 w-full resize-none"
                    value={message}
                    onChange={(event) => setMessage(event.target.value)}
                    onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
                      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                        event.preventDefault();
                        void handleSend();
                      }
                    }}
                    placeholder="Dotaz zákazníka… (Ctrl+Enter odešle)"
                    maxLength={4000}
                  />
                  <span className="pointer-events-none absolute bottom-2 right-3 text-xs text-slate-400">
                    {message.length}/4000
                  </span>
                </div>
                <div className="flex gap-2 md:flex-col">
                  {voiceInputSupported && (
                    <button
                      type="button"
                      className={`flex items-center justify-center gap-2 rounded-md border px-3 py-2 text-sm font-medium transition-colors ${
                        isListening
                          ? "animate-pulse border-red-300 bg-red-50 text-red-600"
                          : "border-slate-200 bg-white text-slate-600 hover:border-mint hover:text-mint"
                      }`}
                      onClick={isListening ? stopListening : startListening}
                      title={isListening ? "Zastavit nahrávání" : "Zadat hlasem (cs-CZ)"}
                    >
                      {isListening ? <MicOff size={17} /> : <Mic size={17} />}
                      {isListening ? "Nahrávám…" : "Hlasem"}
                    </button>
                  )}
                  <button
                    className="primary-button h-auto min-h-12 md:w-40"
                    type="submit"
                    disabled={loading || !isIndexed || !message.trim()}
                  >
                    {loading ? <Loader2 className="animate-spin" size={17} /> : <Send size={17} />}
                    Odeslat
                  </button>
                </div>
              </form>

              {error && (
                <div className="mt-4 flex gap-2 rounded-md border border-coral/30 bg-coral/10 p-3 text-sm text-coral">
                  <AlertTriangle size={18} />
                  <span>{error}</span>
                </div>
              )}
            </section>

            <EvidencePanel response={visibleResponse} loading={loading} sources={sources} topScore={topScore} />
          </section>

          <aside className="space-y-5">
            <UserPanel userInfo={userInfo} onChange={setUserInfo} />
            <RetrievalPanel
              strictMode={strictMode}
              onStrictModeChange={setStrictMode}
              retrievalTolerance={retrievalTolerance}
              onRetrievalToleranceChange={setRetrievalTolerance}
            />
            <VoicePanel
              voiceInputSupported={voiceInputSupported}
              voiceOutputSupported={voiceOutputSupported}
              voiceOutputEnabled={voiceOutputEnabled}
              onVoiceOutputChange={setVoiceOutputEnabled}
              isListening={isListening}
              isSpeaking={isSpeaking}
              onStartListening={startListening}
              onStopListening={stopListening}
              onStopSpeaking={stopSpeaking}
            />
            <AgentPanel steps={steps} loading={loading} />
            <CoveragePanel topicCoverage={topicCoverage} />
            <EscalationPanel response={visibleResponse} />
          </aside>
        </main>
      )}

      {activeTab === "history" && (
        <HistoryView messages={messages} onClear={clearHistory} onCopy={copyHistory} sessionCost={sessionCost} sessionId={sessionId} />
      )}

      {activeTab === "price" && (
        <PriceView
          pricing={pricing}
          response={visibleResponse}
          messages={messages}
          sessionCost={sessionCost}
          cacheStats={cacheStats}
          onRefreshCache={refreshCacheStats}
        />
      )}
    </div>
  );
}

// ── helpers ────────────────────────────────────────────────────────────────

function mergeStep(current: AgentStep[], next: AgentStep): AgentStep[] {
  const exists = current.some((step) => step.id === next.id);
  if (!exists) return [...current, next];
  return current.map((step) => (step.id === next.id ? { ...step, ...next } : step));
}

function sourcesFromSteps(steps: AgentStep[]): Source[] {
  const retrieveStep = steps.find((step) => step.id === "retrieve");
  const payloadSources = retrieveStep?.payload?.sources;
  return Array.isArray(payloadSources) ? (payloadSources as Source[]) : [];
}

// ── components ─────────────────────────────────────────────────────────────

function IndexBanner({ stats, isIndexed, indexing }: { stats: StatsResponse | null; isIndexed: boolean; indexing: boolean }) {
  if (isIndexed) {
    return (
      <div className="success-banner">
        <CheckCircle2 size={19} />
        Index je připravený. Chat může odpovídat nad transkripcemi.
      </div>
    );
  }

  return (
    <div className="warning-banner">
      <AlertTriangle size={19} />
      {indexing
        ? "Probíhá indexace transkripcí."
        : stats?.api_ready
          ? "Index je prázdný."
          : "Chybí OPENAI_API_KEY v .env."}
    </div>
  );
}

function WelcomeBlock({ voiceOutputEnabled }: { voiceOutputEnabled: boolean }) {
  return (
    <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 p-5">
      <div className="mb-2 flex items-center gap-2 font-semibold">
        <Bot size={18} />
        Připraveno na dotaz
      </div>
      <p className="text-sm leading-6 text-slate-600">
        Asistent vrací krátkou odpověď, zdroje z transkripcí, téma, jistotu a eskalační balíček.
        {voiceOutputEnabled && " Hlasový výstup je zapnutý – odpovědi budou přečteny."}
      </p>
    </div>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const usage = message.response?.usage;
  const [copied, setCopied] = useState(false);

  async function copyContent() {
    await navigator.clipboard?.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <article className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`chat-bubble ${isUser ? "chat-bubble-user" : "chat-bubble-assistant"} group relative`}>
        <div className="mb-2 flex items-center justify-between gap-3 text-xs">
          <span className="font-semibold">{isUser ? "Uživatel" : message.response?.topic_label ?? "Asistent"}</span>
          <div className="flex items-center gap-2">
            <span className={isUser ? "text-white/70" : "text-slate-400"}>{formatTime(message.created_at)}</span>
            <button
              className={`opacity-0 transition-opacity group-hover:opacity-100 ${isUser ? "text-white/70 hover:text-white" : "text-slate-400 hover:text-slate-600"}`}
              onClick={copyContent}
              title="Kopírovat zprávu"
            >
              {copied ? <CheckCircle2 size={13} /> : <Copy size={13} />}
            </button>
          </div>
        </div>
        <p className="whitespace-pre-wrap text-sm leading-6">{message.content}</p>
        {!isUser && message.response && (
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            <span className="rounded-md bg-white/70 px-2 py-1 text-slate-600">jistota {Math.round(message.response.confidence * 100)} %</span>
            <span className="rounded-md bg-white/70 px-2 py-1 text-slate-600">{message.response.sources.length} zdrojů</span>
            {usage && <span className="rounded-md bg-white/70 px-2 py-1 text-slate-600">{formatUsd(usage.total_estimated_cost_usd)}</span>}
          </div>
        )}
      </div>
    </article>
  );
}

function PendingBubble({ steps }: { steps: AgentStep[] }) {
  const running = steps.find((step) => step.status === "running") ?? steps[0];
  return (
    <article className="flex justify-start">
      <div className="chat-bubble chat-bubble-assistant">
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold">
          <Loader2 className="animate-spin" size={15} />
          Agent pracuje
        </div>
        <p className="text-sm leading-6 text-slate-600">{running.detail}</p>
      </div>
    </article>
  );
}

function UserPanel({ userInfo, onChange }: { userInfo: UserInfo; onChange: (value: UserInfo) => void }) {
  function update(field: keyof UserInfo, value: string) {
    onChange({ ...userInfo, [field]: value });
  }

  return (
    <section className="panel">
      <PanelTitle icon={<User size={19} />} title="Uživatel" subtitle="Kontext pro odpověď" />
      <div className="grid gap-3">
        <Input label="Jméno" value={userInfo.name ?? ""} onChange={(value) => update("name", value)} />
        <Input label="Ordinace" value={userInfo.clinic ?? ""} onChange={(value) => update("clinic", value)} />
        <Input label="Role" value={userInfo.role ?? ""} onChange={(value) => update("role", value)} />
        <Input label="Verze XDENT" value={userInfo.software_version ?? ""} onChange={(value) => update("software_version", value)} />
        <Input label="Kontakt" value={userInfo.contact ?? ""} onChange={(value) => update("contact", value)} />
      </div>
    </section>
  );
}

function RetrievalPanel({
  strictMode,
  onStrictModeChange,
  retrievalTolerance,
  onRetrievalToleranceChange
}: {
  strictMode: boolean;
  onStrictModeChange: (value: boolean) => void;
  retrievalTolerance: RetrievalTolerance;
  onRetrievalToleranceChange: (value: RetrievalTolerance) => void;
}) {
  return (
    <section className="panel">
      <PanelTitle icon={<SlidersHorizontal size={19} />} title="Hledání" subtitle="Tolerance v chuncích" />
      <div className="grid grid-cols-3 gap-2">
        {toleranceOptions.map((option) => (
          <button
            key={option.value}
            className={`mode-button ${retrievalTolerance === option.value ? "mode-button-active" : ""}`}
            onClick={() => onRetrievalToleranceChange(option.value)}
            title={option.hint}
          >
            <span>{option.label}</span>
            <small>{option.hint}</small>
          </button>
        ))}
      </div>
      <label className="mt-4 flex items-center justify-between gap-3 rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600">
        Strict mode
        <input
          type="checkbox"
          className="h-4 w-4 rounded border-slate-300 text-mint focus:ring-mint"
          checked={strictMode}
          onChange={(event) => onStrictModeChange(event.target.checked)}
        />
      </label>
    </section>
  );
}

function VoicePanel({
  voiceInputSupported,
  voiceOutputSupported,
  voiceOutputEnabled,
  onVoiceOutputChange,
  isListening,
  isSpeaking,
  onStartListening,
  onStopListening,
  onStopSpeaking
}: {
  voiceInputSupported: boolean;
  voiceOutputSupported: boolean;
  voiceOutputEnabled: boolean;
  onVoiceOutputChange: (v: boolean) => void;
  isListening: boolean;
  isSpeaking: boolean;
  onStartListening: () => void;
  onStopListening: () => void;
  onStopSpeaking: () => void;
}) {
  return (
    <section className="panel">
      <PanelTitle icon={<Mic size={19} />} title="Hlas" subtitle="Hlasový vstup & výstup (cs-CZ)" />
      <div className="space-y-3">
        {/* Voice input */}
        <div className="rounded-md border border-slate-200 bg-white p-3">
          <div className="mb-2 flex items-center justify-between text-xs font-medium uppercase text-slate-400">
            <span>Vstup (mikrofon)</span>
            {voiceInputSupported
              ? <span className="rounded-md bg-mint/10 px-1.5 py-0.5 text-mint">k dispozici</span>
              : <span className="rounded-md bg-slate-100 px-1.5 py-0.5 text-slate-500">nepodporováno</span>}
          </div>
          {voiceInputSupported ? (
            <button
              className={`w-full rounded-md border px-3 py-2 text-sm font-medium transition-colors ${
                isListening
                  ? "animate-pulse border-red-300 bg-red-50 text-red-600"
                  : "border-slate-200 bg-slate-50 text-slate-600 hover:border-mint hover:text-mint"
              }`}
              onClick={isListening ? onStopListening : onStartListening}
            >
              <span className="flex items-center justify-center gap-2">
                {isListening ? <><MicOff size={15} /> Zastavit nahrávání</> : <><Mic size={15} /> Mluvit</>}
              </span>
            </button>
          ) : (
            <p className="text-xs text-slate-500">Prohlížeč nepodporuje Web Speech API.</p>
          )}
        </div>

        {/* Voice output */}
        <div className="rounded-md border border-slate-200 bg-white p-3">
          <div className="mb-2 flex items-center justify-between text-xs font-medium uppercase text-slate-400">
            <span>Výstup (TTS)</span>
            {voiceOutputSupported
              ? <span className="rounded-md bg-mint/10 px-1.5 py-0.5 text-mint">k dispozici</span>
              : <span className="rounded-md bg-slate-100 px-1.5 py-0.5 text-slate-500">nepodporováno</span>}
          </div>
          {voiceOutputSupported ? (
            <div className="space-y-2">
              <label className="flex cursor-pointer items-center justify-between gap-3 text-sm text-slate-600">
                Automaticky číst odpovědi
                <button
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${voiceOutputEnabled ? "bg-mint" : "bg-slate-200"}`}
                  onClick={() => onVoiceOutputChange(!voiceOutputEnabled)}
                  role="switch"
                  aria-checked={voiceOutputEnabled}
                >
                  <span className={`inline-block h-4 w-4 translate-x-1 rounded-full bg-white shadow transition-transform ${voiceOutputEnabled ? "translate-x-6" : ""}`} />
                </button>
              </label>
              {isSpeaking && (
                <button
                  className="flex w-full items-center justify-center gap-2 rounded-md border border-mint/40 bg-mint/10 px-3 py-1.5 text-xs text-mint"
                  onClick={onStopSpeaking}
                >
                  <Volume2 size={13} className="animate-pulse" /> Zastavit přehrávání
                </button>
              )}
            </div>
          ) : (
            <p className="text-xs text-slate-500">Prohlížeč nepodporuje SpeechSynthesis.</p>
          )}
        </div>
      </div>
    </section>
  );
}

function AgentPanel({ steps, loading }: { steps: AgentStep[]; loading: boolean }) {
  return (
    <section className="panel">
      <PanelTitle icon={<Activity size={19} />} title="Průběh agenta" subtitle="Realtime stav" />
      <div className="space-y-3">
        {steps.map((step, index) => (
          <StepItem key={step.id} step={step} index={index} active={loading && step.status === "running"} />
        ))}
      </div>
    </section>
  );
}

function EvidencePanel({
  response,
  loading,
  sources,
  topScore
}: {
  response: ChatResponse | null;
  loading: boolean;
  sources: Source[];
  topScore: number;
}) {
  const confidence = Math.round((response?.confidence ?? 0) * 100);
  const scoreWidth = Math.min(100, Math.round(topScore * 100));

  return (
    <section className="panel">
      <PanelTitle icon={<FileSearch size={19} />} title="Relevantní informace" subtitle="Zdroje a signály" />
      <div className="grid gap-3 md:grid-cols-3">
        <Metric label="Téma" value={response?.topic_label ?? (loading ? "počítám" : "bez dotazu")} />
        <Metric label="Jistota tématu" value={`${confidence} %`} bar={confidence} />
        <Metric label="Síla zdroje" value={topScore ? `${topScore}` : "bez zdroje"} bar={scoreWidth} />
      </div>

      <div className="mt-4 grid gap-3">
        {sources.length === 0 && <EmptyState text="Zdroje se zobrazí po odpovědi asistenta." />}
        {sources.map((source, index) => (
          <SourceCard key={`${source.source}-${index}`} source={source} index={index} maxScore={topScore} />
        ))}
      </div>
    </section>
  );
}

function CoveragePanel({ topicCoverage }: { topicCoverage: Array<{ topic: string; label: string; chunks: number; width: number }> }) {
  return (
    <section className="panel">
      <PanelTitle icon={<Gauge size={19} />} title="Pokrytí" subtitle="Témata v indexu" />
      <div className="space-y-3">
        {topicCoverage.length === 0 && <EmptyState text="Po indexaci se zobrazí témata." />}
        {topicCoverage.map((topic) => (
          <div key={topic.topic}>
            <div className="mb-1 flex justify-between gap-3 text-xs text-slate-500">
              <span className="truncate">{topic.label}</span>
              <span>{topic.chunks}</span>
            </div>
            <div className="h-2 rounded-full bg-slate-100">
              <div className="h-2 rounded-full bg-mint" style={{ width: `${topic.width}%` }} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function EscalationPanel({ response }: { response: ChatResponse | null }) {
  return (
    <section className="panel">
      <PanelTitle icon={<ShieldCheck size={19} />} title="Eskalace" subtitle="Fallback výstup" />
      <pre className="thin-scroll max-h-52 overflow-auto whitespace-pre-wrap rounded-md border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-700">
        {response?.escalation_packet ?? "Bez eskalace."}
      </pre>
    </section>
  );
}

function HistoryView({
  messages,
  onClear,
  onCopy,
  sessionCost,
  sessionId
}: {
  messages: ChatMessage[];
  onClear: () => void;
  onCopy: () => void;
  sessionCost: number;
  sessionId: string;
}) {
  const answered = messages.filter((item) => item.role === "assistant" && item.response).length;
  return (
    <main className="mx-auto max-w-7xl px-5 py-5">
      <section className="panel">
        <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
          <PanelTitle icon={<History size={19} />} title="Historie chatu" subtitle={`Relace ${sessionId.slice(0, 8)}`} />
          <div className="flex gap-2">
            <button className="icon-button" onClick={onCopy} title="Kopírovat historii" disabled={messages.length === 0}>
              <Copy size={18} />
            </button>
            <button className="icon-button text-coral hover:border-coral hover:text-coral" onClick={onClear} title="Vymazat historii" disabled={messages.length === 0}>
              <Trash2 size={18} />
            </button>
          </div>
        </div>

        <div className="mb-5 grid gap-3 md:grid-cols-3">
          <Metric label="Zprávy" value={`${messages.length}`} />
          <Metric label="Odpovědi" value={`${answered}`} />
          <Metric label="Odhad relace" value={formatUsd(sessionCost)} />
        </div>

        <div className="space-y-3">
          {messages.length === 0 && <EmptyState text="Historie je prázdná." />}
          {messages.map((item) => (
            <article key={item.id} className="rounded-md border border-slate-200 bg-white p-4">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
                <span className="font-semibold text-ink">{item.role === "user" ? "Uživatel" : item.response?.topic_label ?? "Asistent"}</span>
                <span>{formatTime(item.created_at)}</span>
              </div>
              <p className="whitespace-pre-wrap text-sm leading-6 text-slate-700">{item.content}</p>
              {item.response && (
                <div className="mt-3 flex flex-wrap gap-2 text-xs">
                  <span className="rounded-md bg-slate-100 px-2 py-1">zdroje: {item.response.sources.length}</span>
                  <span className="rounded-md bg-slate-100 px-2 py-1">tolerance: {toleranceName(item.response.retrieval_tolerance)}</span>
                  <span className="rounded-md bg-slate-100 px-2 py-1">cena: {formatUsd(item.response.usage?.total_estimated_cost_usd ?? 0)}</span>
                </div>
              )}
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}

function PriceView({
  pricing,
  response,
  messages,
  sessionCost,
  cacheStats,
  onRefreshCache
}: {
  pricing: PriceInfoResponse | null;
  response: ChatResponse | null;
  messages: ChatMessage[];
  sessionCost: number;
  cacheStats: CacheStats | null;
  onRefreshCache: () => void;
}) {
  const usage = response?.usage ?? null;
  const chatCost = usage?.estimated_chat_cost_usd ?? 0;
  const embeddingCost = usage?.estimated_embedding_cost_usd ?? 0;
  const maxCost = Math.max(chatCost, embeddingCost, 0.000001);

  return (
    <main className="mx-auto grid max-w-7xl gap-5 px-5 py-5 lg:grid-cols-[minmax(0,0.95fr)_minmax(360px,0.7fr)]">
      <div className="space-y-5">
        <section className="panel">
          <PanelTitle icon={<DollarSign size={19} />} title="Price info" subtitle="Odhad nákladů API" />
          <div className="grid gap-3 md:grid-cols-3">
            <Metric label="Chat model" value={pricing?.chat_model ?? "nenačteno"} />
            <Metric label="Embedding model" value={pricing?.embedding_model ?? "nenačteno"} />
            <Metric label="Relace" value={formatUsd(sessionCost)} />
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-3">
            <PriceCard label="Input chat" value={pricing ? formatRate(pricing.chat_input_price_per_1m, pricing.currency) : "-"} />
            <PriceCard label="Output chat" value={pricing ? formatRate(pricing.chat_output_price_per_1m, pricing.currency) : "-"} />
            <PriceCard label="Embedding" value={pricing ? formatRate(pricing.embedding_price_per_1m, pricing.currency) : "-"} />
          </div>

          <div className="mt-5 rounded-md border border-slate-200 bg-slate-50 p-4 text-sm leading-6 text-slate-600">
            <p>{pricing?.note ?? "Ceny se načítají z backend konfigurace."}</p>
            {pricing?.reference_url && (
              <a className="mt-2 inline-flex text-mint underline-offset-4 hover:underline" href={pricing.reference_url} target="_blank" rel="noreferrer">
                OpenAI pricing
              </a>
            )}
          </div>
        </section>

        {/* Cache stats */}
        <section className="panel">
          <div className="mb-4 flex items-center justify-between">
            <PanelTitle icon={<TrendingUp size={19} />} title="Cache & časté dotazy" subtitle="Odpovědi servírované z cache" />
            <button className="icon-button" onClick={onRefreshCache} title="Obnovit">
              <RefreshCw size={16} />
            </button>
          </div>

          <div className="mb-4 grid gap-3 md:grid-cols-2">
            <Metric label="Aktivní záznamy" value={cacheStats ? `${cacheStats.active_entries}` : "–"} />
            <Metric label="Sledované dotazy" value={cacheStats ? `${cacheStats.total_tracked_queries}` : "–"} />
          </div>

          <div>
            <div className="mb-2 text-xs font-medium uppercase text-slate-400">Nejčastější dotazy</div>
            {!cacheStats || cacheStats.top_frequent.length === 0
              ? <EmptyState text="Zatím žádné opakované dotazy." />
              : (
                <div className="space-y-2">
                  {cacheStats.top_frequent.map((item, i) => {
                    const maxCount = cacheStats.top_frequent[0].count;
                    return (
                      <div key={i}>
                        <div className="mb-1 flex justify-between gap-3 text-xs text-slate-600">
                          <span className="truncate">{item.query}</span>
                          <span className="shrink-0 font-medium">{item.count}×</span>
                        </div>
                        <div className="h-1.5 rounded-full bg-slate-100">
                          <div
                            className="h-1.5 rounded-full bg-mint"
                            style={{ width: `${Math.max(6, Math.round((item.count / maxCount) * 100))}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
          </div>
        </section>
      </div>

      <aside className="space-y-5">
        <section className="panel">
          <PanelTitle icon={<Activity size={19} />} title="Poslední odpověď" subtitle="Tokeny a náklad" />
          {!usage && <EmptyState text="Pošlete dotaz a zobrazí se odhad." />}
          {usage && (
            <div className="space-y-4">
              <Metric label="Celkem" value={formatUsd(usage.total_estimated_cost_usd)} />
              <CostBar label="Chat" value={chatCost} width={(chatCost / maxCost) * 100} />
              <CostBar label="Embedding" value={embeddingCost} width={(embeddingCost / maxCost) * 100} />
              <div className="grid grid-cols-3 gap-2 text-center text-xs text-slate-500">
                <div className="rounded-md bg-slate-50 p-2">
                  <strong className="block text-sm text-ink">{usage.estimated_chat_input_tokens}</strong>
                  input
                </div>
                <div className="rounded-md bg-slate-50 p-2">
                  <strong className="block text-sm text-ink">{usage.estimated_chat_output_tokens}</strong>
                  output
                </div>
                <div className="rounded-md bg-slate-50 p-2">
                  <strong className="block text-sm text-ink">{usage.estimated_embedding_tokens}</strong>
                  embed
                </div>
              </div>
            </div>
          )}
        </section>

        <section className="panel">
          <PanelTitle icon={<Gauge size={19} />} title="Relace" subtitle="Součet historie" />
          <div className="space-y-3">
            {messages.filter((item) => item.response?.usage).length === 0 && <EmptyState text="Zatím bez nákladů." />}
            {messages
              .filter((item) => item.response?.usage)
              .slice(-6)
              .map((item) => (
                <CostBar
                  key={item.id}
                  label={item.response?.topic_label ?? "odpověď"}
                  value={item.response?.usage?.total_estimated_cost_usd ?? 0}
                  width={Math.max(8, ((item.response?.usage?.total_estimated_cost_usd ?? 0) / Math.max(sessionCost, 0.000001)) * 100)}
                />
              ))}
          </div>
        </section>
      </aside>
    </main>
  );
}

// ── small shared components ────────────────────────────────────────────────

function CostBar({ label, value, width }: { label: string; value: number; width: number }) {
  return (
    <div>
      <div className="mb-1 flex justify-between gap-3 text-xs text-slate-500">
        <span className="truncate">{label}</span>
        <span>{formatUsd(value)}</span>
      </div>
      <div className="h-2 rounded-full bg-slate-100">
        <div className="h-2 rounded-full bg-mint" style={{ width: `${Math.max(4, Math.min(100, width))}%` }} />
      </div>
    </div>
  );
}

function Input({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium uppercase text-slate-400">{label}</span>
      <input className="field-input h-10" value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function Metric({ label, value, bar }: { label: string; value: string; bar?: number }) {
  return (
    <div className="metric-card">
      <div className="text-xs font-medium uppercase text-slate-400">{label}</div>
      <div className="mt-1 truncate text-sm font-semibold text-ink">{value}</div>
      {bar !== undefined && (
        <div className="mt-2 h-1.5 rounded-full bg-slate-100">
          <div className="h-1.5 rounded-full bg-mint" style={{ width: `${Math.max(4, Math.min(100, bar))}%` }} />
        </div>
      )}
    </div>
  );
}

function PriceCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-4">
      <div className="text-xs font-medium uppercase text-slate-400">{label}</div>
      <div className="mt-2 text-lg font-semibold">{value}</div>
      <div className="mt-1 text-xs text-slate-500">za 1M tokenů</div>
    </div>
  );
}

function StatusPill({ icon, label, ok }: { icon: ReactNode; label: string; ok: boolean }) {
  return (
    <div className={`status-pill ${ok ? "border-mint/30 bg-mint/10 text-mint" : "border-amber/30 bg-amber/10 text-amber"}`}>
      {icon}
      {label}
    </div>
  );
}

function TabButton({ active, icon, label, onClick }: { active: boolean; icon: ReactNode; label: string; onClick: () => void }) {
  return (
    <button className={`tab-button ${active ? "tab-button-active" : ""}`} onClick={onClick}>
      {icon}
      {label}
    </button>
  );
}

function PanelTitle({ icon, title, subtitle }: { icon: ReactNode; title: string; subtitle: string }) {
  return (
    <div className="mb-4">
      <div className="flex items-center gap-2">
        <span className="text-mint">{icon}</span>
        <h2 className="text-base font-semibold">{title}</h2>
      </div>
      <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
    </div>
  );
}

function StepItem({ step, index, active }: { step: AgentStep; index: number; active: boolean }) {
  const style =
    step.status === "done"
      ? "border-mint/30 bg-mint/10"
      : step.status === "warning"
        ? "border-amber/30 bg-amber/10"
        : step.status === "error"
          ? "border-coral/30 bg-coral/10"
          : active
            ? "border-sky-300 bg-sky-50"
            : "border-slate-200 bg-white";

  return (
    <div className={`rounded-md border p-3 ${style}`}>
      <div className="flex items-start gap-3">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-white text-sm font-semibold shadow-sm">
          {active ? <Loader2 className="animate-spin" size={15} /> : index + 1}
        </div>
        <div className="min-w-0">
          <div className="font-medium">{step.label}</div>
          <div className="mt-1 text-sm text-slate-600">{step.detail}</div>
        </div>
      </div>
    </div>
  );
}

function SourceCard({ source, index, maxScore }: { source: Source; index: number; maxScore: number }) {
  const width = maxScore > 0 ? Math.max(8, Math.round((source.score / maxScore) * 100)) : 0;

  return (
    <article className="rounded-md border border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <span className="flex h-6 w-6 items-center justify-center rounded-md bg-mint/10 text-mint">{index + 1}</span>
            <span className="truncate">{source.source}</span>
          </div>
          <div className="mt-2 h-1.5 rounded-full bg-slate-100">
            <div className="h-1.5 rounded-full bg-mint" style={{ width: `${width}%` }} />
          </div>
        </div>
        <span className="rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-600">{source.score}</span>
      </div>
      {source.resolution && <p className="mb-2 text-sm font-medium text-ink">{source.resolution}</p>}
      <p className="text-sm leading-6 text-slate-600">{source.excerpt}</p>
    </article>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 p-4 text-sm text-slate-500">
      {text}
    </div>
  );
}

// ── utils ──────────────────────────────────────────────────────────────────

function compactUser(user: UserInfo): UserInfo | null {
  const compacted: UserInfo = {
    name: trimOrUndefined(user.name),
    clinic: trimOrUndefined(user.clinic),
    role: trimOrUndefined(user.role),
    software_version: trimOrUndefined(user.software_version),
    contact: trimOrUndefined(user.contact)
  };
  return Object.values(compacted).some(Boolean) ? compacted : null;
}

function trimOrUndefined(value?: string | null): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function loadJson<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function saveJson(key: string, value: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // localStorage may be unavailable in private/sandboxed contexts.
  }
}

function loadOrCreateSessionId(): string {
  try {
    const existing = window.localStorage.getItem(STORAGE_SESSION);
    if (existing) return existing;
    const created = createId();
    window.localStorage.setItem(STORAGE_SESSION, created);
    return created;
  } catch {
    return createId();
  }
}

function createId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function formatTime(value: string): string {
  return new Date(value).toLocaleTimeString("cs-CZ", { hour: "2-digit", minute: "2-digit" });
}

function formatUsd(value: number): string {
  if (value < 0.0001) return `$${value.toFixed(8)}`;
  return `$${value.toFixed(4)}`;
}

function formatRate(value: number, currency: string): string {
  return value > 0 ? `${value} ${currency}` : "doplnit";
}

function toleranceName(value: RetrievalTolerance): string {
  return toleranceOptions.find((option) => option.value === value)?.label ?? value;
}

export default App;
