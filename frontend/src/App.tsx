import {
  Activity,
  AlertTriangle,
  Bot,
  CheckCircle2,
  Database,
  FileSearch,
  Gauge,
  Loader2,
  MessageSquare,
  RefreshCw,
  Send,
  ShieldCheck,
  Sparkles,
  Zap
} from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { fetchStats, ingestData, sendChatStream } from "./api";
import type { AgentStep, ChatResponse, Source, StatsResponse } from "./types";

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
  { id: "answer", label: "Odpověď", status: "queued", detail: "Výsledek bude stručný a se zdrojem." }
];

function App() {
  const [message, setMessage] = useState(demoQuestions[0]);
  const [strictMode, setStrictMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [liveSteps, setLiveSteps] = useState<AgentStep[]>(emptySteps);

  const isIndexed = Boolean(stats?.points_count && stats.points_count > 0);
  const steps = response?.steps ?? liveSteps;
  const sources = response?.sources ?? [];
  const topScore = Math.max(0, ...sources.map((source) => source.score));

  const topicCoverage = useMemo(() => {
    const counts = stats?.topics.map((topic) => topic.chunks) ?? [];
    const max = Math.max(1, ...counts);
    return (stats?.topics ?? []).slice(0, 8).map((topic) => ({
      ...topic,
      width: Math.max(6, Math.round((topic.chunks / max) * 100))
    }));
  }, [stats]);

  useEffect(() => {
    refreshStats();
  }, []);

  async function refreshStats() {
    try {
      setStats(await fetchStats());
    } catch {
      setStats(null);
    }
  }

  async function handleSend() {
    if (!message.trim() || !isIndexed) return;
    setLoading(true);
    setError(null);
    setResponse(null);
    setLiveSteps(emptySteps);
    try {
      const result = await sendChatStream(message, strictMode, (step) => {
        setLiveSteps((current) => mergeStep(current, step));
      });
      setResponse(result);
      await refreshStats();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Něco se nepodařilo.");
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
              <p className="text-sm text-slate-500">První úroveň podpory nad reálnými transkripcemi</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill icon={<Database size={16} />} label={isIndexed ? `${stats?.points_count} chunků` : "Index prázdný"} ok={isIndexed} />
            <StatusPill icon={<Sparkles size={16} />} label={stats?.api_ready ? "OpenAI připraveno" : "Chybí API klíč"} ok={Boolean(stats?.api_ready)} />
            <button className="icon-button" onClick={refreshStats} title="Obnovit stav">
              <RefreshCw size={18} />
            </button>
            <button className="primary-button" onClick={handleIngest} disabled={indexing || !stats?.api_ready}>
              {indexing ? <Loader2 className="animate-spin" size={17} /> : <Zap size={17} />}
              {isIndexed ? "Přeindexovat" : "Indexovat"}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-5 px-5 py-5 xl:grid-cols-[minmax(0,1.45fr)_minmax(340px,0.75fr)]">
        <section className="space-y-5">
          <IndexBanner stats={stats} isIndexed={isIndexed} indexing={indexing} />

          <div className="panel">
            <PanelTitle icon={<MessageSquare size={19} />} title="Dotaz zákazníka" subtitle="Vyber ukázku nebo napiš vlastní problém." />
            <div className="mb-4 flex flex-wrap gap-2">
              {demoQuestions.map((question) => (
                <button key={question} className="sample-chip" onClick={() => setMessage(question)}>
                  {question}
                </button>
              ))}
            </div>
            <textarea
              className="min-h-28 w-full resize-none rounded-md border border-slate-200 bg-white p-3 text-sm leading-6 outline-none transition focus:border-mint focus:ring-2 focus:ring-mint/20"
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              placeholder="Napište dotaz zákazníka..."
            />
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
              <label className="flex items-center gap-2 text-sm text-slate-600">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-slate-300 text-mint focus:ring-mint"
                  checked={strictMode}
                  onChange={(event) => setStrictMode(event.target.checked)}
                />
                Strict mode: odpovědět jen při vyšší jistotě
              </label>
              <button className="primary-button" onClick={handleSend} disabled={loading || !isIndexed}>
                {loading ? <Loader2 className="animate-spin" size={17} /> : <Send size={17} />}
                Zeptat se asistenta
              </button>
            </div>
            {error && (
              <div className="mt-4 flex gap-2 rounded-md border border-coral/30 bg-coral/10 p-3 text-sm text-coral">
                <AlertTriangle size={18} />
                <span>{error}</span>
              </div>
            )}
          </div>

          <div className="panel">
            <PanelTitle icon={<Bot size={19} />} title="Odpověď" subtitle="Krátce, přímo k věci a se zdrojem." />
            <SignalBar response={response} loading={loading} topScore={topScore} isIndexed={isIndexed} />
            <div className="answer-box mt-4">
              {loading ? (
                <div className="flex items-center gap-2 text-slate-500">
                  <Loader2 className="animate-spin" size={18} />
                  Agent hledá podobné hovory a ověřuje zdroje...
                </div>
              ) : response ? (
                response.answer
              ) : (
                "Pošlete dotaz. Asistent nejdřív najde podobné hovory, ukáže zdroje a potom odpoví."
              )}
            </div>
          </div>

          <div className="panel">
            <PanelTitle icon={<FileSearch size={19} />} title="Zdroje z transkripcí" subtitle="Evidence, ze které odpověď vychází." />
            {sources.length === 0 && <EmptyState text="Zdroje se zobrazí po odpovědi asistenta." />}
            <div className="grid gap-3">
              {sources.map((source, index) => (
                <SourceCard key={`${source.source}-${index}`} source={source} index={index} maxScore={topScore} />
              ))}
            </div>
          </div>
        </section>

        <aside className="space-y-5">
          <section className="panel">
            <PanelTitle icon={<Activity size={19} />} title="Průběh agenta" subtitle="Co systém právě udělal." />
            <div className="space-y-3">
              {steps.map((step, index) => (
                <StepItem key={step.id} step={step} index={index} active={loading && step.status === "running"} />
              ))}
            </div>
          </section>

          <section className="panel">
            <PanelTitle icon={<Gauge size={19} />} title="Pokrytí znalostí" subtitle="Rozložení indexu podle témat." />
            <div className="space-y-3">
              {topicCoverage.length === 0 && <EmptyState text="Po indexaci se zde ukáže přehled témat." />}
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

          <section className="panel">
            <PanelTitle icon={<ShieldCheck size={19} />} title="Eskalace" subtitle="Vznikne pouze při nízké jistotě." />
            <pre className="whitespace-pre-wrap rounded-md border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-700">
              {response?.escalation_packet ?? "Když v transkripcích není dostatečný podklad, agent nevymýšlí odpověď a připraví balíček pro 2. úroveň podpory."}
            </pre>
          </section>
        </aside>
      </main>
    </div>
  );
}

function mergeStep(current: AgentStep[], next: AgentStep): AgentStep[] {
  const exists = current.some((step) => step.id === next.id);
  if (!exists) return [...current, next];
  return current.map((step) => (step.id === next.id ? { ...step, ...next } : step));
}

function IndexBanner({ stats, isIndexed, indexing }: { stats: StatsResponse | null; isIndexed: boolean; indexing: boolean }) {
  if (isIndexed) {
    return (
      <div className="success-banner">
        <CheckCircle2 size={19} />
        Index je připravený. Můžete pokládat dotazy.
      </div>
    );
  }

  return (
    <div className="warning-banner">
      <AlertTriangle size={19} />
      {indexing
        ? "Probíhá indexace. U větší sady transkripcí to může chvíli trvat."
        : stats?.api_ready
          ? "Index je zatím prázdný. Klikněte na Indexovat."
          : "Doplňte OPENAI_API_KEY v .env, restartujte Docker a potom spusťte indexaci."}
    </div>
  );
}

function SignalBar({ response, loading, topScore, isIndexed }: { response: ChatResponse | null; loading: boolean; topScore: number; isIndexed: boolean }) {
  const confidence = Math.round((response?.confidence ?? 0) * 100);
  const scoreWidth = Math.min(100, Math.round(topScore * 100));

  return (
    <div className="grid gap-3 md:grid-cols-3">
      <Metric label="Téma" value={response?.topic_label ?? (isIndexed ? "čeká na dotaz" : "nejdřív index")} />
      <Metric label="Jistota tématu" value={loading ? "počítám" : `${confidence} %`} bar={confidence} />
      <Metric label="Síla zdroje" value={topScore ? `${topScore}` : "bez zdroje"} bar={scoreWidth} />
    </div>
  );
}

function Metric({ label, value, bar }: { label: string; value: string; bar?: number }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-3">
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

function StatusPill({ icon, label, ok }: { icon: ReactNode; label: string; ok: boolean }) {
  return (
    <div className={`status-pill ${ok ? "border-mint/30 bg-mint/10 text-mint" : "border-amber/30 bg-amber/10 text-amber"}`}>
      {icon}
      {label}
    </div>
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
        : active
          ? "border-sky-300 bg-sky-50"
          : "border-slate-200 bg-white";

  return (
    <div className={`rounded-md border p-3 ${style}`}>
      <div className="flex items-start gap-3">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-white text-sm font-semibold shadow-sm">
          {active ? <Loader2 className="animate-spin" size={15} /> : index + 1}
        </div>
        <div>
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

export default App;
