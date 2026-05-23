import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Database,
  Loader2,
  MessageSquare,
  Mic,
  MicOff,
  RefreshCw,
  Send,
  Trash2,
  Volume2,
  VolumeX,
  X,
  Zap
} from "lucide-react";
import { type FormEvent, type KeyboardEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchCacheStats, fetchStats, ingestData, sendChatStream } from "./api";
import type { AgentStep, CacheStats, ChatMessage, RetrievalTolerance, Source, StatsResponse } from "./types";

const STORAGE_MESSAGES = "xdent.chat.messages";
const STORAGE_SESSION = "xdent.chat.session";
const STORAGE_VOICE_OUTPUT = "xdent.chat.voiceOutput";

const demoQuestions = [
  "Nejde mi odeslat ePoukaz, system pise chybu s uhradou. Co mam zkontrolovat?",
  "Po instalaci certifikatu se uzivatel nemuze prihlasit do XDENTu.",
  "Dokument se netiskne spravne a potrebuji upravit sablonu tisku.",
  "Kde v kalendari zmenim termin objednaneho pacienta?"
];

const emptySteps: AgentStep[] = [
  { id: "classify", label: "Tema", status: "queued", detail: "Cekam na dotaz." },
  { id: "retrieve", label: "Retrieval", status: "queued", detail: "Pripraveno vyhledat podobne hovory." },
  { id: "validate", label: "Jistota", status: "queued", detail: "Overim, jestli zdroje staci." },
  { id: "answer", label: "Odpoved", status: "queued", detail: "Vysledek bude kratky a se zdrojem." }
];

function App() {
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState(demoQuestions[0]);
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadJson<ChatMessage[]>(STORAGE_MESSAGES, []));
  const [sessionId] = useState(() => loadOrCreateSessionId());
  const [loading, setLoading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [liveSteps, setLiveSteps] = useState<AgentStep[]>(emptySteps);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState(() => loadJson<boolean>(STORAGE_VOICE_OUTPUT, false));
  const [ttsVoices, setTtsVoices] = useState<SpeechSynthesisVoice[]>([]);

  const abortRef = useRef<AbortController | null>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const voiceInputSupported = typeof window !== "undefined" && !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  const voiceOutputSupported = typeof window !== "undefined" && !!window.speechSynthesis;
  const isIndexed = Boolean(stats?.points_count && stats.points_count > 0);
  const savedResponse = useMemo(() => [...messages].reverse().find((item) => item.response)?.response ?? null, [messages]);
  const visibleSteps = loading ? liveSteps : savedResponse?.steps ?? liveSteps;
  const selectedVoice = selectCzechVoice(ttsVoices);
  const topFrequent = cacheStats?.top_frequent?.[0]?.query;

  useEffect(() => {
    void refreshStats();
    void refreshCacheStats();
  }, []);

  useEffect(() => { saveJson(STORAGE_MESSAGES, messages); }, [messages]);
  useEffect(() => { saveJson(STORAGE_VOICE_OUTPUT, voiceOutputEnabled); }, [voiceOutputEnabled]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, open]);

  useEffect(() => {
    if (!voiceOutputSupported) return;
    const load = () => setTtsVoices(window.speechSynthesis.getVoices());
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", load);
  }, [voiceOutputSupported]);

  useEffect(() => {
    if (!voiceOutputEnabled && isSpeaking) {
      window.speechSynthesis?.cancel();
      setIsSpeaking(false);
    }
  }, [voiceOutputEnabled, isSpeaking]);

  async function refreshStats() {
    try { setStats(await fetchStats()); } catch { setStats(null); }
  }

  async function refreshCacheStats() {
    try { setCacheStats(await fetchCacheStats()); } catch { setCacheStats(null); }
  }

  function startListening() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return;

    recognitionRef.current?.abort();
    const recognition = new SR();
    recognition.lang = "cs-CZ";
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = Array.from(event.results).map((result) => result[0].transcript).join("");
      setMessage(transcript);
      if (event.results[event.results.length - 1].isFinal) setIsListening(false);
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

  const speak = useCallback((text: string) => {
    if (!voiceOutputEnabled || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const cleaned = text.replace(/Zdroj:.*$/m, "").replace(/\[.*?\]/g, "").trim();
    const utterance = new SpeechSynthesisUtterance(cleaned);
    utterance.lang = "cs-CZ";
    utterance.rate = 1;
    utterance.pitch = 1;
    if (selectedVoice) utterance.voice = selectedVoice;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  }, [voiceOutputEnabled, selectedVoice]);

  function stopSpeaking() {
    window.speechSynthesis?.cancel();
    setIsSpeaking(false);
  }

  async function handleSend(event?: FormEvent) {
    event?.preventDefault();
    const text = message.trim();
    if (!text || !isIndexed || loading) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

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
    setLiveSteps(emptySteps);

    try {
      const result = await sendChatStream(
        {
          message: text,
          strict_mode: false,
          top_k: 6,
          retrieval_tolerance: "balanced" as RetrievalTolerance,
          session_id: sessionId,
          user: null
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
      setMessages((current) => [...current, assistantMessage]);
      if (voiceOutputEnabled) speak(result.answer);
      void refreshCacheStats();
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
      const errorText = err instanceof Error ? err.message : "Odpoved se nepodarilo sestavit.";
      setError(errorText);
      setMessages((current) => [...current, { id: createId(), role: "assistant", content: errorText, created_at: new Date().toISOString() }]);
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
      setError(err instanceof Error ? err.message : "Indexace se nepodarila.");
    } finally {
      setIndexing(false);
    }
  }

  function clearHistory() {
    setMessages([]);
    setLiveSteps(emptySteps);
  }

  async function copyHistory() {
    const transcript = messages.map((item) => `${item.role === "user" ? "Uzivatel" : "Asistent"}: ${item.content}`).join("\n\n");
    await navigator.clipboard?.writeText(transcript);
  }

  return (
    <div className="xdent-widget-page">
      <div className="xdent-widget-background">
        <XdentLogo />
        <div>
          <h1>XDENT chat asistent</h1>
          <p>Minimalni podpora nad transkripcemi, pripravena jako vlozitelny chat widget.</p>
        </div>
      </div>

      {open && (
        <section className="xdent-chat-popup" aria-label="XDENT chat">
          <header className="xdent-chat-header">
            <div className="flex min-w-0 items-center gap-3">
              <XdentLogo small />
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold">XDENT asistent</div>
                <div className="truncate text-xs text-white/75">{isIndexed ? `${stats?.points_count ?? 0} chunku v indexu` : "Index neni pripraveny"}</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button className="xdent-header-button" onClick={() => { void refreshStats(); void refreshCacheStats(); }} title="Obnovit stav">
                <RefreshCw size={15} />
              </button>
              <button className="xdent-header-button" onClick={() => setOpen(false)} title="Zavrit chat">
                <X size={16} />
              </button>
            </div>
          </header>

          <div className="xdent-chat-status">
            <StatusChip ok={isIndexed} icon={<Database size={14} />} label={isIndexed ? "Index pripraven" : "Index prazdny"} />
            {topFrequent && <span className="truncate text-xs text-slate-500">Casto: {topFrequent}</span>}
          </div>

          {!isIndexed && (
            <div className="mx-4 mt-3 rounded-md border border-amber/30 bg-amber/10 p-3 text-sm text-amber">
              Nejdrive vytvorte index transkripci.
              <button className="primary-button mt-3 w-full" onClick={handleIngest} disabled={indexing || !stats?.api_ready}>
                {indexing ? <Loader2 className="animate-spin" size={16} /> : <Zap size={16} />}
                Indexovat
              </button>
            </div>
          )}

          <div className="xdent-chat-messages thin-scroll">
            {messages.length === 0 && <WelcomeBlock voiceOutputEnabled={voiceOutputEnabled} />}
            {messages.map((item) => <ChatBubble key={item.id} message={item} />)}
            {loading && <PendingBubble steps={visibleSteps} />}
            <div ref={messagesEndRef} />
          </div>

          {error && (
            <div className="mx-4 mb-3 flex gap-2 rounded-md border border-coral/30 bg-coral/10 p-3 text-sm text-coral">
              <AlertTriangle size={17} />
              <span>{error}</span>
            </div>
          )}

          <div className="xdent-chat-samples">
            {demoQuestions.slice(0, 3).map((question) => (
              <button key={question} className="sample-chip" onClick={() => setMessage(question)}>
                {question}
              </button>
            ))}
          </div>

          <form className="xdent-chat-input" onSubmit={handleSend}>
            <textarea
              className="field-input min-h-20 resize-none"
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              onKeyDown={(event: KeyboardEvent<HTMLTextAreaElement>) => {
                if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                  event.preventDefault();
                  void handleSend();
                }
              }}
              placeholder="Napiste dotaz k XDENTu..."
              maxLength={4000}
            />
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                {voiceInputSupported && (
                  <button type="button" className={`icon-button ${isListening ? "text-coral" : ""}`} onClick={isListening ? stopListening : startListening} title="Mikrofon pouze cesky (cs-CZ)">
                    {isListening ? <MicOff size={17} /> : <Mic size={17} />}
                  </button>
                )}
                {voiceOutputSupported && (
                  <button type="button" className="icon-button" onClick={() => voiceOutputEnabled ? setVoiceOutputEnabled(false) : setVoiceOutputEnabled(true)} title="Cist odpovedi cesky">
                    {voiceOutputEnabled ? <Volume2 size={17} /> : <VolumeX size={17} />}
                  </button>
                )}
                <button type="button" className="icon-button" onClick={copyHistory} title="Kopirovat historii" disabled={messages.length === 0}>
                  <Copy size={17} />
                </button>
                <button type="button" className="icon-button" onClick={clearHistory} title="Vymazat historii" disabled={messages.length === 0}>
                  <Trash2 size={17} />
                </button>
              </div>
              <button className="primary-button" type="submit" disabled={loading || !isIndexed || !message.trim()}>
                {loading ? <Loader2 className="animate-spin" size={17} /> : <Send size={17} />}
                Odeslat
              </button>
            </div>
          </form>
        </section>
      )}

      <button className="xdent-chat-launcher" onClick={() => setOpen((value) => !value)} aria-label={open ? "Zavrit chat" : "Otevrit chat"}>
        {open ? <X size={26} /> : <MessageSquare size={27} />}
      </button>
    </div>
  );
}

function XdentLogo({ small = false }: { small?: boolean }) {
  return (
    <div className={`logo-mark ${small ? "h-10 w-10" : ""}`} aria-label="XDENT">
      <span className={small ? "relative z-10 text-xl font-black leading-4" : "logo-x"}>X</span>
      <span className={small ? "relative z-10 mt-0.5 text-[8px] font-black leading-none tracking-wide" : "logo-dent"}>DENT</span>
    </div>
  );
}

function WelcomeBlock({ voiceOutputEnabled }: { voiceOutputEnabled: boolean }) {
  return (
    <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 p-4">
      <div className="mb-2 flex items-center gap-2 font-semibold">
        <CheckCircle2 size={18} />
        Pripraveno na dotaz
      </div>
      <p className="text-sm leading-6 text-slate-600">
        Asistent odpovida kratce podle zdroju z transkripci. V odpovedi uvidite i silu zdroje.
        {voiceOutputEnabled && " Hlasovy vystup je zapnuty a cte cesky."}
      </p>
    </div>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const sourceStrength = sourceStrengthPercent(message.response?.sources ?? []);

  async function copyContent() {
    await navigator.clipboard?.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  }

  return (
    <article className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`chat-bubble ${isUser ? "chat-bubble-user" : "chat-bubble-assistant"} group relative`}>
        <div className="mb-2 flex items-center justify-between gap-3 text-xs">
          <span className="font-semibold">{isUser ? "Vy" : message.response?.topic_label ?? "XDENT asistent"}</span>
          <div className="flex items-center gap-2">
            <span className={isUser ? "text-white/70" : "text-slate-400"}>{formatTime(message.created_at)}</span>
            <button className={`opacity-0 transition-opacity group-hover:opacity-100 ${isUser ? "text-white/70 hover:text-white" : "text-slate-400 hover:text-slate-600"}`} onClick={copyContent} title="Kopirovat zpravu">
              {copied ? <CheckCircle2 size={13} /> : <Copy size={13} />}
            </button>
          </div>
        </div>
        <p className="whitespace-pre-wrap text-sm leading-6">{message.content}</p>
        {!isUser && message.response && (
          <div className="mt-3 grid gap-2 text-xs">
            <div className="rounded-md bg-white/70 p-2 text-slate-600">
              <div className="mb-1 flex items-center justify-between">
                <span>Sila zdroje</span>
                <span className="font-semibold text-ink">{sourceStrength}%</span>
              </div>
              <div className="h-1.5 rounded-full bg-slate-200">
                <div className="h-1.5 rounded-full bg-mint" style={{ width: `${sourceStrength}%` }} />
              </div>
            </div>
            <SourceDisclosure sources={message.response.sources} />
          </div>
        )}
      </div>
    </article>
  );
}

function SourceDisclosure({ sources }: { sources: Source[] }) {
  if (sources.length === 0) {
    return <span className="rounded-md bg-white/70 px-2 py-1 text-slate-600">0 zdroju</span>;
  }

  return (
    <details className="rounded-md bg-white/70 px-2 py-1 text-slate-600">
      <summary className="cursor-pointer list-none select-none">
        {sources.length} zdroju
      </summary>
      <div className="mt-2 grid gap-2">
        {sources.slice(0, 3).map((source, index) => (
          <div key={`${source.source}-${index}`} className="rounded-md border border-slate-200 bg-white p-2">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="truncate font-semibold">{index + 1}. {source.source}</span>
              <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5">{source.score}</span>
            </div>
            {source.resolution && <div className="mb-1 leading-5 text-ink">{source.resolution}</div>}
            <div className="line-clamp-3 leading-5 text-slate-500">{source.excerpt}</div>
          </div>
        ))}
      </div>
    </details>
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

function StatusChip({ icon, label, ok }: { icon: ReactNode; label: string; ok: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-semibold ${ok ? "bg-mint/10 text-mint" : "bg-amber/10 text-amber"}`}>
      {icon}
      {label}
    </span>
  );
}

function mergeStep(current: AgentStep[], next: AgentStep): AgentStep[] {
  const exists = current.some((step) => step.id === next.id);
  if (!exists) return [...current, next];
  return current.map((step) => (step.id === next.id ? { ...step, ...next } : step));
}

function sourceStrengthPercent(sources: Source[]): number {
  const best = Math.max(0, ...sources.map((source) => source.score));
  if (!best) return 0;
  return Math.max(1, Math.min(100, Math.round(best * 100)));
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

function selectCzechVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null {
  const czech = voices.filter((voice) => voice.lang.toLowerCase().startsWith("cs"));
  return czech.find((voice) => voice.localService) ?? czech[0] ?? voices[0] ?? null;
}

export default App;
