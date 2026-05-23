import {
  Activity,
  AlertTriangle,
  Bot,
  Calendar,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  Copy,
  HeartPulse,
  LifeBuoy,
  Loader2,
  MessageSquare,
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
import { type FormEvent, type KeyboardEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchCacheStats, fetchStats, forgetMemory, ingestData, sendChatStream } from "./api";
import type { AgentMode, AgentStep, CacheStats, ChatMessage, ChatResponse, RetrievalTolerance, Source, StatsResponse, UserInfo } from "./types";

const STORAGE_MESSAGES = "xdent.chat.messages";
const STORAGE_SESSION = "xdent.chat.session";
const STORAGE_VOICE_OUTPUT = "xdent.chat.voiceOutput";
const STORAGE_AGENT_MODE = "xdent.chat.agentMode";
const STORAGE_PATIENT_MEMORY = "xdent.chat.patientMemory";
const STORAGE_FONT_SCALE = "xdent.chat.fontScale";

type FontScale = "normal" | "large" | "xlarge";

const agentOptions: Array<{ id: AgentMode; label: string; hint: string; icon: ReactNode }> = [
  { id: "auto", label: "Nechat AI vybrat", hint: "nejlepsi pomocnik podle dotazu", icon: <Zap size={15} /> },
  { id: "support", label: "Problem s XDENTem", hint: "program, prihlaseni, tisk, eRecept", icon: <Bot size={15} /> },
  { id: "patient", label: "Pruvodce pacienta", hint: "ulozi kontakt, mesto a problem", icon: <HeartPulse size={15} /> },
  { id: "triage", label: "Bolest nebo akutni stav", hint: "odhadi nalehavost dalsiho kroku", icon: <Activity size={15} /> },
  { id: "scheduler", label: "Najit nejdrivejsi termin", hint: "vybere ordinaci a cas", icon: <Calendar size={15} /> },
  { id: "handoff", label: "Predat cloveku", hint: "pripravi shrnuti pro recepci/podporu", icon: <LifeBuoy size={15} /> }
];

const demoQuestions: Record<AgentMode, string[]> = {
  auto: [
    "Boli me zub a potrebuji poradit, jestli je to akutni. Jsem z Prahy.",
    "Mam otok, horecku a silnou bolest. Jsem v Brne, telefon 777 123 456.",
    "Nejde mi prihlaseni do XDENTu a nevim, jestli je problem v certifikatu."
  ],
  support: [
    "Nejde mi odeslat ePoukaz, system pise chybu s uhradou. Co mam zkontrolovat?",
    "Po instalaci certifikatu se uzivatel nemuze prihlasit do XDENTu.",
    "Dokument se netiskne spravne a potrebuji upravit sablonu tisku."
  ],
  patient: [
    "Jmenuji se Jana Novakova, jsem z Prahy, boli me zub a potrebuji nejdrivejsi termin. Telefon 777 123 456.",
    "Mam otok a silnou bolest, jsem v Brne. Co mam delat?",
    "Chci objednat dentalni hygienu v Praze, muj e-mail je pacient@example.cz."
  ],
  triage: [
    "Mam otok, horecku a silnou bolest zubu. Jsem v Brne.",
    "Boli me zub uz treti den, ale nemam otok. Jakou to ma prioritu?",
    "Chci jen preventivni kontrolu a nevim, jestli je to urgentni."
  ],
  scheduler: [
    "Najdi nejdrivejsi termin v Praze pro bolest zubu. Telefon 777 123 456.",
    "Potrebuji dentalni hygienu v Brne co nejdrive.",
    "Mam ulozeny kontakt, najdi nejrychlejsi dostupny termin."
  ],
  handoff: [
    "Priprav eskalaci pro podporu, eRecept nejde odeslat a mam screenshot chyby.",
    "Pacient je z Ostravy, ma akutni bolest a chybi nam potvrzeny kontakt.",
    "Predat 2. urovni: po instalaci certifikatu nejde prihlaseni."
  ]
};

const emptySteps: AgentStep[] = [
  { id: "agent", label: "AI agent", status: "queued", detail: "Cekam na predani spravnemu specialistovi." },
  { id: "classify", label: "Tema", status: "queued", detail: "Cekam na dotaz." },
  { id: "retrieve", label: "Retrieval", status: "queued", detail: "Pripraveno vyhledat podobne hovory." },
  { id: "validate", label: "Jistota", status: "queued", detail: "Overim, jestli zdroje staci." },
  { id: "answer", label: "Odpoved", status: "queued", detail: "Vysledek bude kratky a overeny podle podkladu." }
];

function App() {
  const [open, setOpen] = useState(false);
  const [agentMode, setAgentMode] = useState<AgentMode>(() => loadJson<AgentMode>(STORAGE_AGENT_MODE, "auto"));
  const [message, setMessage] = useState(demoQuestions.auto[0]);
  const [messages, setMessages] = useState<ChatMessage[]>(() => loadJson<ChatMessage[]>(STORAGE_MESSAGES, []));
  const [patientMemory, setPatientMemory] = useState<UserInfo | null>(() => loadJson<UserInfo | null>(STORAGE_PATIENT_MEMORY, null));
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
  const samples = demoQuestions[agentMode];
  const frequentQuestions = useMemo(() => mergeFrequentQuestions(cacheStats?.top_frequent.map((item) => item.query) ?? [], samples), [cacheStats, samples]);
  const latestActions = savedResponse?.next_actions ?? [];
  const latestEscalation = savedResponse?.escalation_packet ?? null;
  const visibleActions = useMemo(() => latestActions.filter((action) => !isCopyEscalationAction(action)), [latestActions]);

  useEffect(() => {
    void refreshStats();
    void refreshCacheStats();
  }, []);

  useEffect(() => { saveJson(STORAGE_MESSAGES, messages); }, [messages]);
  useEffect(() => { saveJson(STORAGE_VOICE_OUTPUT, voiceOutputEnabled); }, [voiceOutputEnabled]);
  useEffect(() => { saveJson(STORAGE_AGENT_MODE, agentMode); }, [agentMode]);
  useEffect(() => { saveJson(STORAGE_PATIENT_MEMORY, patientMemory); }, [patientMemory]);
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

  function chooseAgent(next: AgentMode) {
    if (next === agentMode) return;
    const agent = agentOptions.find((option) => option.id === next);
    setAgentMode(next);
    setMessages((current) => [
      ...current,
      {
        id: createId(),
        role: "assistant",
        content: handoffMessage(next, agent?.label ?? "AI agent"),
        created_at: new Date().toISOString()
      }
    ]);
    if (!message.trim() || Object.values(demoQuestions).flat().includes(message)) {
      setMessage(demoQuestions[next][0]);
    }
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
          agent_mode: agentMode,
          strict_mode: false,
          top_k: 6,
          retrieval_tolerance: "balanced" as RetrievalTolerance,
          session_id: sessionId,
          user: patientMemory
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
      if (result.user) setPatientMemory(result.user);
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

  async function clearPatientMemory() {
    setPatientMemory(null);
    await forgetMemory(sessionId).catch(() => undefined);
  }

  async function handleNextAction(action: string) {
    if (action.toLowerCase().includes("kopirovat") && latestEscalation) {
      await navigator.clipboard?.writeText(latestEscalation);
      return;
    }
    setMessage(promptForAction(action, patientMemory));
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
          <h1>XDENT chat asistent</h1>
          <p>Chatovy pomocnik pro pacienty i ordinaci. Vybere spravneho AI agenta, odpovi kratce a opre se o dostupne podklady.</p>
        </div>
      </div>

      {open && (
        <section className={`xdent-chat-popup ${fontScaleClass(fontScale)}`} aria-label="XDENT chat">
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

          <AgentSwitcher selected={agentMode} onSelect={chooseAgent} />

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

          {visibleActions.length > 0 && (
            <div className="xdent-chat-actions">
              {visibleActions.map((action) => (
                <button key={action} className="quick-action" type="button" onClick={() => { void handleNextAction(action); }}>
                  {action}
                </button>
              ))}
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
                memory={patientMemory}
                memoryUpdates={savedResponse?.memory_updates ?? []}
                escalationPacket={latestEscalation}
                onCycleFont={() => setFontScale((current) => nextFontScale(current))}
                onForgetMemory={clearPatientMemory}
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
              placeholder={placeholderForAgent(agentMode)}
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
        Napiste, co potrebujete vyresit
      </div>
      <p className="text-sm leading-6 text-slate-600">
        AI sama pozna, jestli jde o problem v XDENTu, bolest, objednani nebo predani cloveku.
        U odpovedi uvidite jistotu a rozklikatelne podklady.
        {voiceOutputEnabled && " Hlasovy vystup je zapnuty a cte cesky."}
      </p>
    </div>
  );
}

function AgentSwitcher({ selected, onSelect }: { selected: AgentMode; onSelect: (mode: AgentMode) => void }) {
  const [expanded, setExpanded] = useState(false);
  const active = agentOptions.find((agent) => agent.id === selected) ?? agentOptions[0];

  return (
    <div className="agent-handoff-wrap">
      <button
        className="agent-handoff-compact"
        type="button"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
      >
        <span className="agent-handoff-icon">{active.icon}</span>
        <span className="min-w-0 flex-1">
          <span className="agent-kicker">AI pomocnik</span>
          <span className="agent-title">{active.label}</span>
          <span className="agent-subtitle">
            {selected === "auto" ? "AI sama vybere nejvhodnejsi postup" : active.hint}
          </span>
        </span>
        <span className="agent-handoff-toggle">
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </span>
      </button>
      {expanded && (
        <div className="agent-handoff-expanded">
          <p className="agent-handoff-summary">
            Prepnete jen tehdy, kdyz chcete AI nasmerovat na konkretni typ pomoci.
          </p>
          <div className="agent-preset-grid">
            {agentOptions.map((agent) => (
              <button
                key={agent.id}
                className={`agent-preset ${agent.id === selected ? "agent-preset-active" : ""}`}
                type="button"
                onClick={() => {
                  onSelect(agent.id);
                  setExpanded(false);
                }}
              >
                <span>{agent.icon}</span>
                <span className="min-w-0">
                  <span className="block truncate font-semibold">{agent.label}</span>
                  <span className="block truncate text-[11px] font-medium text-slate-400">{agent.hint}</span>
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
      <div className="sr-only">
        <label className="agent-select-label">
          Zmenit pomocnika
          <select className="agent-select" value={selected} onChange={(event) => onSelect(event.target.value as AgentMode)}>
            {agentOptions.map((agent) => (
              <option key={agent.id} value={agent.id}>{agent.label}: {agent.hint}</option>
            ))}
          </select>
        </label>
      </div>
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
  memory,
  memoryUpdates,
  escalationPacket,
  onCycleFont,
  onForgetMemory,
}: {
  fontScale: FontScale;
  memory: UserInfo | null;
  memoryUpdates: string[];
  escalationPacket: string | null;
  onCycleFont: () => void;
  onForgetMemory: () => void;
}) {
  const [copiedEscalation, setCopiedEscalation] = useState(false);
  const memoryItems = memoryChips(memory);
  const memorySummary = memoryUpdates.length
    ? `ulozeno: ${memoryUpdates.join(", ")}`
    : memoryItems.length
      ? `${memoryItems.length} udaje`
      : "zatim prazdna";

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
      <div className="tool-row">
        <span className="text-xs text-slate-500">Velikost textu: {fontScaleLabel(fontScale)}</span>
        <button
          className="font-size-button font-size-button-a"
          type="button"
          title="Zvětšení písma"
          aria-label="Zvětšení písma"
          onClick={onCycleFont}
        >
          A
        </button>
      </div>

      <div className="tool-row">
        <div className="min-w-0">
          <div className="text-xs font-semibold text-ink">Pamet pacienta</div>
          <div className="truncate text-xs text-slate-500">{memorySummary}</div>
        </div>
        {memoryItems.length > 0 && (
          <button className="memory-forget" type="button" onClick={onForgetMemory}>
            Zapomenout
          </button>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        {memoryItems.length > 0
          ? memoryItems.map((chip) => <span key={chip} className="memory-chip">{chip}</span>)
          : <span className="text-xs text-slate-500">Agent si ulozi jen udaje, ktere pacient sam napise do chatu.</span>}
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
    </section>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);
  const visibleContent = isUser ? message.content : stripSourceLines(message.content);
  const answerConfidence = answerConfidencePercent(message.response);
  const sourceStrength = sourceStrengthPercent(message.response?.sources ?? []);
  const routedByOrchestrator = message.response?.requested_agent_mode === "auto";
  const assistantTitle = message.response
    ? `${message.response.agent_label ?? "XDENT asistent"} - ${message.response.topic_label ?? "odpoved"}`
    : "XDENT asistent";

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
            {routedByOrchestrator && (
              <div className="rounded-md border border-mint/20 bg-mint/10 p-2 text-mint">
                <span className="font-semibold">Proc odpovida tento pomocnik:</span> {message.response.agent_route_reason ?? "AI vybrala nejvhodnejsiho pomocnika."}
              </div>
            )}
            <div className="rounded-md bg-white/70 p-2 text-slate-600">
              <div className="mb-1 flex items-center justify-between">
                <span>Jistota odpovedi</span>
                <span className="font-semibold text-ink">{answerConfidence}%</span>
              </div>
              <div className="h-1.5 rounded-full bg-slate-200">
                <div className="h-1.5 rounded-full bg-ocean" style={{ width: `${answerConfidence}%` }} />
              </div>
            </div>
            <div className="rounded-md bg-white/70 p-2 text-slate-600">
              <div className="mb-1 flex items-center justify-between">
                <span>Sila podkladu</span>
                <span className="font-semibold text-ink">{sourceStrength}%</span>
              </div>
              <div className="h-1.5 rounded-full bg-slate-200">
                <div className="h-1.5 rounded-full bg-mint" style={{ width: `${sourceStrength}%` }} />
              </div>
            </div>
            <SourceDisclosure sources={message.response.sources} />
            {message.response.escalation_packet && <EscalationDisclosure packet={message.response.escalation_packet} />}
          </div>
        )}
      </div>
    </article>
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

function memoryChips(memory: UserInfo | null): string[] {
  if (!memory) return [];
  const chips: string[] = [];
  if (memory.patient_name) chips.push(`Pacient: ${memory.patient_name}`);
  if (memory.patient_phone) chips.push(`Tel.: ${memory.patient_phone}`);
  if (memory.patient_email) chips.push(`E-mail: ${memory.patient_email}`);
  if (memory.patient_city) chips.push(`Mesto: ${memory.patient_city}`);
  if (memory.urgency) chips.push(`Urgence: ${urgencyLabel(memory.urgency)}`);
  if (memory.problem_summary) chips.push("Problem ulozen");
  return chips;
}

function urgencyLabel(value: UserInfo["urgency"]): string {
  return {
    low: "nizka",
    normal: "bezna",
    high: "vysoka",
    critical: "kriticka"
  }[value ?? "normal"];
}

function handoffMessage(mode: AgentMode, label: string): string {
  if (mode === "auto") {
    return "Zapinam chytry vyber. Napiste dotaz a AI sama rozhodne, ktery pomocnik ma odpovedet.";
  }
  return `Predavam konverzaci na: ${label}. Odpovi kratce a bude se drzet dostupnych podkladu.`;
}

function placeholderForAgent(agentMode: AgentMode): string {
  if (agentMode === "auto") return "Napiste svuj dotaz. AI sama vybere, kdo vam nejlepe pomuze...";
  if (agentMode === "patient") return "Napiste jmeno, mesto, kontakt a co pacienta trapi...";
  if (agentMode === "triage") return "Popiste bolest, otok, horecku, krvaceni a jak dlouho to trva...";
  if (agentMode === "scheduler") return "Napiste mesto, typ osetreni a telefon/e-mail pro potvrzeni...";
  if (agentMode === "handoff") return "Napiste, co ma recepce nebo podpora prevzit...";
  return "Napiste dotaz k programu XDENT...";
}

function promptForAction(action: string, memory: UserInfo | null): string {
  const lowered = action.toLowerCase();
  if (lowered.includes("telefon") || lowered.includes("kontakt")) return "Telefon/e-mail pacienta je: ";
  if (lowered.includes("mesto")) return "Pacient je z mesta: ";
  if (lowered.includes("popis") || lowered.includes("trapi")) return "Pacienta trapi: ";
  if (lowered.includes("nejdrivejsi")) {
    const city = memory?.patient_city ? ` v ${memory.patient_city}` : "";
    return `Najdi nejdrivejsi termin${city}.`;
  }
  if (lowered.includes("screenshot")) return "Mam screenshot chyby. Chci pripravit eskalaci pro podporu.";
  if (lowered.includes("2. urovni")) return "Predat 2. urovni podpory: ";
  if (lowered.includes("zavolat")) return "Chci kontakt na nejvhodnejsi ordinaci pro akutni pripad.";
  return action;
}

function isCopyEscalationAction(action: string): boolean {
  return action.toLowerCase().includes("kopirovat") && action.toLowerCase().includes("eskalaci");
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
