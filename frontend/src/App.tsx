import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Loader2,
  Mic,
  MicOff,
  RefreshCw,
  Send,
  SlidersHorizontal,
  Trash2,
  Volume2,
  VolumeX,
  X,
  Zap
} from "lucide-react";
import { type FormEvent, type KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchCacheStats, fetchStats, ingestData, sendChatStream } from "./api";
import type { AgentStep, CacheStats, ChatMessage, ChatResponse, RetrievalTolerance, Source, StatsResponse } from "./types";

const STORAGE_MESSAGES = "xdent.chat.messages";
const STORAGE_SESSION = "xdent.chat.session";
const STORAGE_VOICE_OUTPUT = "xdent.chat.voiceOutput";
const STORAGE_FONT_SCALE = "xdent.chat.fontScale";

type FontScale = "normal" | "large" | "xlarge";

const sampleQuestions = [
  "Nejde mi odeslat ePoukaz, system pise chybu s uhradou. Co mam zkontrolovat?",
  "Po instalaci certifikatu se uzivatel nemuze prihlasit do XDENTu.",
  "Dokument se netiskne spravne a potrebuji upravit sablonu tisku.",
  "Nejde mi odeslat eRecept, co mam zkontrolovat?",
  "Kde v XDENTu nastavim sablonu dokumentu?"
];

const emptySteps: AgentStep[] = [
  { id: "agent", label: "XDent AI Asistent", status: "queued", detail: "Pripravuji odpoved podle dostupnych podkladu." },
  { id: "classify", label: "Tema", status: "queued", detail: "Cekam na dotaz." },
  { id: "retrieve", label: "Retrieval", status: "queued", detail: "Pripraveno vyhledat podobne hovory." },
  { id: "validate", label: "Jistota", status: "queued", detail: "Overim, jestli zdroje staci." },
  { id: "answer", label: "Odpoved", status: "queued", detail: "Vysledek bude kratky a overeny podle podkladu." }
];

function App() {
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState(sampleQuestions[0]);
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
  const [fontScale, setFontScale] = useState<FontScale>(() => loadJson<FontScale>(STORAGE_FONT_SCALE, "normal"));
  const [toolsOpen, setToolsOpen] = useState(false);
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
  const frequentQuestions = useMemo(() => mergeFrequentQuestions(cacheStats?.top_frequent.map((item) => item.query) ?? [], sampleQuestions), [cacheStats]);
  const latestEscalation = savedResponse?.escalation_packet ?? null;

  useEffect(() => {
    void refreshStats();
    void refreshCacheStats();
  }, []);

  useEffect(() => { saveJson(STORAGE_MESSAGES, messages); }, [messages]);
  useEffect(() => { saveJson(STORAGE_VOICE_OUTPUT, voiceOutputEnabled); }, [voiceOutputEnabled]);
  useEffect(() => { saveJson(STORAGE_FONT_SCALE, fontScale); }, [fontScale]);

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
    const cleaned = stripSourceLines(text).replace(/\[.*?\]/g, "").trim();
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
          session_id: sessionId
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
    if (indexing) return;
    setIndexing(true);
    setError(null);
    try {
      await ingestData();
      await refreshStats();
      await refreshCacheStats();
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
    const transcript = messages
      .map((item) => `${item.role === "user" ? "Uzivatel" : "Asistent"}: ${item.role === "assistant" ? stripSourceLines(item.content) : item.content}`)
      .join("\n\n");
    await navigator.clipboard?.writeText(transcript);
  }

  return (
    <div className="xdent-widget-page">
      <div className="xdent-widget-background">
        <XdentLogo />
        <div>
          <h1>XDent AI Asistent</h1>
          <p>Jednoduchy chatovy pomocnik pro XDENT. Soustredi se na odpoved na dotaz, drzi se dostupnych zdroju a pri nejistote doporuci predani podpore.</p>
        </div>
      </div>

      {open && (
        <section className={`xdent-chat-popup ${fontScaleClass(fontScale)}`} aria-label="XDent AI Asistent chat">
          <header className="xdent-chat-header">
            <div className="flex min-w-0 items-center gap-3">
              <XdentLogo small />
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold">XDent AI Asistent</div>
                <div className="truncate text-xs text-white/75">Strucne odpovedi podle zdroju</div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="chunk-count-chip" title={isIndexed ? "Pocet pripravenych chunku v indexu" : "Index zatim neni pripraveny"}>
                {isIndexed ? `${stats?.points_count ?? 0} chunku` : "0 chunku"}
              </span>
              <button className="xdent-header-button" onClick={() => { void refreshStats(); void refreshCacheStats(); }} title="Obnovit stav">
                <RefreshCw size={15} />
              </button>
              <button className="xdent-header-button" onClick={() => setOpen(false)} title="Zavrit chat">
                <X size={16} />
              </button>
            </div>
          </header>

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

          <FrequentQuestionsStrip
            questions={frequentQuestions}
            onPickQuestion={setMessage}
          />

          <form className="xdent-chat-input" onSubmit={handleSend}>
            {toolsOpen && (
              <ChatToolsPanel
                fontScale={fontScale}
                escalationPacket={latestEscalation}
                indexing={indexing}
                isIndexed={isIndexed}
                pointsCount={stats?.points_count ?? 0}
                apiReady={Boolean(stats?.api_ready)}
                onCycleFont={() => setFontScale((current) => nextFontScale(current))}
                onClose={() => setToolsOpen(false)}
                onIngest={handleIngest}
              />
            )}
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
                <button
                  type="button"
                  className={`icon-button ${toolsOpen ? "icon-button-active" : ""}`}
                  onClick={() => setToolsOpen((value) => !value)}
                  title="Nastroje"
                  aria-expanded={toolsOpen}
                >
                  <SlidersHorizontal size={17} />
                </button>
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
        {open ? <X size={26} /> : <XdentLauncherIcon />}
      </button>
    </div>
  );
}

function XdentLauncherIcon() {
  return (
    <span className="launcher-xdent-icon" aria-hidden="true">
      <span>XD</span>
      <small>AI</small>
    </span>
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
        Napiste, co potrebujete vyresit
      </div>
      <p className="text-sm leading-6 text-slate-600">
        Napiste otazku k XDENTu. Asistent odpovi strucne podle dostupnych transkripci.
        U odpovedi uvidite jistotu, silu podkladu a rozklikatelne zdroje.
        {voiceOutputEnabled && " Hlasovy vystup je zapnuty a cte cesky."}
      </p>
    </div>
  );
}

function FrequentQuestionsStrip({
  questions,
  onPickQuestion,
}: {
  questions: string[];
  onPickQuestion: (question: string) => void;
}) {
  return (
    <details className="quick-strip" aria-label="Nejcastejsi dotazy">
      <summary>
        <span>Nejcastejsi dotazy</span>
        <span>{questions.length} dotazu</span>
      </summary>

      <div className="quick-strip-content">
        <div className="quick-question-rail thin-scroll" aria-label="Caste dotazy">
          {questions.map((question) => (
            <button key={question} className="sample-chip quick-question-chip" type="button" onClick={() => onPickQuestion(question)}>
              {question}
            </button>
          ))}
        </div>
      </div>
    </details>
  );
}

function ChatToolsPanel({
  fontScale,
  escalationPacket,
  indexing,
  isIndexed,
  pointsCount,
  apiReady,
  onCycleFont,
  onClose,
  onIngest,
}: {
  fontScale: FontScale;
  escalationPacket: string | null;
  indexing: boolean;
  isIndexed: boolean;
  pointsCount: number;
  apiReady: boolean;
  onCycleFont: () => void;
  onClose: () => void;
  onIngest: () => void;
}) {
  const [copiedEscalation, setCopiedEscalation] = useState(false);

  useEffect(() => {
    setCopiedEscalation(false);
  }, [escalationPacket]);

  async function copyEscalation() {
    if (!escalationPacket) return;
    await navigator.clipboard?.writeText(escalationPacket);
    setCopiedEscalation(true);
    setTimeout(() => setCopiedEscalation(false), 1400);
  }

  return (
    <section className="chat-tools-panel" aria-label="Nastroje chatu">
      <div className="chat-tools-header">
        <div>
          <div className="text-xs font-semibold text-ink">Nastaveni chatu</div>
          <div className="text-[11px] text-slate-500">Pismo, index a eskalace.</div>
        </div>
        <button className="panel-close-button" type="button" onClick={onClose} title="Zavrit nastaveni" aria-label="Zavrit nastaveni">
          <X size={15} />
        </button>
      </div>

      <div className="chat-tools-scroll thin-scroll">
        <div className="tool-row">
          <div className="min-w-0">
            <div className="text-xs font-semibold text-ink">Znalostni index</div>
            <div className="truncate text-xs text-slate-500">
              {isIndexed ? `${pointsCount} chunku pripraveno` : "Index zatim neni pripraveny"}
            </div>
          </div>
          <button
            className="quick-action inline-flex items-center gap-1"
            type="button"
            onClick={() => { void onIngest(); }}
            disabled={indexing || !apiReady}
            title={apiReady ? "Spustit indexaci transkripci" : "Chybi API klic pro embeddingy"}
          >
            {indexing ? <Loader2 className="animate-spin" size={13} /> : <RefreshCw size={13} />}
            {isIndexed ? "Reindexovat" : "Indexovat"}
          </button>
        </div>

        <div className="tool-row">
          <span className="text-xs text-slate-500">Velikost textu: {fontScaleLabel(fontScale)}</span>
          <button
            className="font-size-button font-size-button-a"
            type="button"
            title="Zvetseni pisma"
            aria-label="Zvetseni pisma"
            onClick={onCycleFont}
          >
            A
          </button>
        </div>

        <div className="tool-row">
          <div className="min-w-0">
            <div className="text-xs font-semibold text-ink">Eskalace</div>
            <div className="truncate text-xs text-slate-500">{escalationPacket ? "balicek je pripraveny" : "zatim neni potreba"}</div>
          </div>
          <button
            className="quick-action"
            type="button"
            onClick={() => { void copyEscalation(); }}
            disabled={!escalationPacket}
          >
            {copiedEscalation ? "Zkopirovano" : "Kopirovat eskalaci"}
          </button>
        </div>
      </div>
    </section>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const visibleContent = isUser ? message.content : stripSourceLines(message.content);
  const answerConfidence = answerConfidencePercent(message.response);
  const sourceStrength = sourceStrengthPercent(message.response?.sources ?? []);
  const assistantTitle = message.response
    ? `XDent AI Asistent - ${message.response.topic_label ?? "odpoved"}`
    : "XDent AI Asistent";

  async function copyContent() {
    await navigator.clipboard?.writeText(visibleContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  }

  return (
    <article className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`chat-bubble ${isUser ? "chat-bubble-user" : "chat-bubble-assistant"} group relative`}>
        <div className="mb-2 flex items-center justify-between gap-3 text-xs">
          <span className="font-semibold">{isUser ? "Vy" : assistantTitle}</span>
          <div className="flex items-center gap-2">
            <span className={isUser ? "text-white/70" : "text-slate-400"}>{formatTime(message.created_at)}</span>
            <button className={`opacity-0 transition-opacity group-hover:opacity-100 ${isUser ? "text-white/70 hover:text-white" : "text-slate-400 hover:text-slate-600"}`} onClick={copyContent} title="Kopirovat zpravu">
              {copied ? <CheckCircle2 size={13} /> : <Copy size={13} />}
            </button>
          </div>
        </div>
        <p className="whitespace-pre-wrap text-sm leading-6">{visibleContent}</p>
        {!isUser && message.response && (
          <div className="mt-3 grid gap-2 text-xs">
            <MetricBar label="Jistota odpovedi" value={answerConfidence} tone="confidence" />
            <MetricBar label="Sila podkladu" value={sourceStrength} tone="source" />
            <ChunkUsage response={message.response} />
            <SourceDisclosure sources={message.response.sources} />
            {message.response.escalation_packet && <EscalationDisclosure packet={message.response.escalation_packet} />}
          </div>
        )}
      </div>
    </article>
  );
}

function ChunkUsage({ response }: { response: ChatResponse }) {
  return (
    <div className="metric-card flex items-center justify-between gap-3">
      <span>Chunky</span>
      <span className="font-semibold text-ink">
        {response.chunks_used} pouzito / {response.chunks_considered} projito
      </span>
    </div>
  );
}

function MetricBar({ label, value, tone }: { label: string; value: number; tone: "confidence" | "source" }) {
  const safeValue = Math.max(0, Math.min(100, value));
  return (
    <div className="metric-card">
      <div className="mb-1.5 flex items-center justify-between gap-3">
        <span>{label}</span>
        <span className="font-semibold text-ink">{safeValue}%</span>
      </div>
      <div
        className="metric-track"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={safeValue}
      >
        <div
          className={`metric-fill ${tone === "confidence" ? "metric-fill-confidence" : "metric-fill-source"}`}
          style={{ width: `${safeValue}%` }}
        />
      </div>
    </div>
  );
}

function EscalationDisclosure({ packet }: { packet: string }) {
  const [copied, setCopied] = useState(false);

  async function copyPacket() {
    await navigator.clipboard?.writeText(packet);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  }

  return (
    <details className="rounded-md border border-amber/30 bg-amber/10 px-2 py-1 text-amber">
      <summary className="cursor-pointer list-none select-none">Eskalace pripravena</summary>
      <pre className="thin-scroll mt-2 max-h-36 overflow-auto whitespace-pre-wrap rounded bg-white/80 p-2 text-[11px] leading-5 text-slate-700">{packet}</pre>
      <button className="mt-2 inline-flex items-center gap-1 text-xs font-semibold" type="button" onClick={copyPacket}>
        {copied ? <CheckCircle2 size={13} /> : <Copy size={13} />}
        {copied ? "Zkopirovano" : "Kopirovat eskalaci"}
      </button>
    </details>
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
              <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5">
                {source.source_type === "qa_generated" ? "nauceno" : source.source_type === "qa_seed" ? "Q&A" : source.score}
              </span>
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

function answerConfidencePercent(response?: ChatResponse): number {
  if (!response) return 0;
  if (typeof response.answer_confidence === "number") {
    return Math.max(0, Math.min(100, Math.round(response.answer_confidence * 100)));
  }
  const topicConfidence = Math.max(0, Math.min(1, response.confidence ?? 0));
  const sourceStrength = Math.max(0, Math.min(1, Math.max(0, ...response.sources.map((source) => source.score))));
  const combined = response.sources.length > 0
    ? (topicConfidence * 0.45) + (sourceStrength * 0.55)
    : topicConfidence * 0.6;
  return Math.max(1, Math.min(100, Math.round(combined * 100)));
}

function stripSourceLines(text: string): string {
  return text
    .split("\n")
    .filter((line) => !line.trim().toLowerCase().startsWith("zdroj:"))
    .join("\n")
    .trim();
}

function mergeFrequentQuestions(frequent: string[], samples: string[]): string[] {
  const merged: string[] = [];
  for (const question of [...frequent, ...samples]) {
    const cleaned = question.trim();
    if (!cleaned || merged.includes(cleaned)) continue;
    merged.push(cleaned);
    if (merged.length >= 10) break;
  }
  return merged;
}

function nextFontScale(current: FontScale): FontScale {
  if (current === "normal") return "large";
  if (current === "large") return "xlarge";
  return "normal";
}

function fontScaleClass(value: FontScale): string {
  if (value === "large") return "font-large";
  if (value === "xlarge") return "font-xlarge";
  return "";
}

function fontScaleLabel(value: FontScale): string {
  if (value === "large") return "vetsi";
  if (value === "xlarge") return "nejvetsi";
  return "normalni";
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
