import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Bot,
  CheckCircle2,
  Database,
  FileSearch,
  Gauge,
  Loader2,
  MessageSquare,
  Radar,
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
  "Kde v kalendáři změním termín objednaného pacienta?",
  "Při vykazování na pojišťovnu se mi vrací kontrolní chyba VZP."
];

const emptySteps: AgentStep[] = [
  { id: "classify", label: "Rozpoznání tématu", status: "queued", detail: "Čekám na dotaz." },
  { id: "retrieve", label: "Vyhledání znalostí", status: "queued", detail: "Připraveno pro Qdrant." },
  { id: "validate", label: "Kontrola jistoty", status: "queued", detail: "Ověřím, jestli zdroje stačí." },
  { id: "answer", label: "Sestavení odpovědi", status: "queued", detail: "Odpověď bude stručná a věcná." }
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

  const steps = response?.steps ?? liveSteps;
  const sources = response?.sources ?? [];

  const topicCoverage = useMemo(() => {
    const counts = stats?.topics.map((topic) => topic.chunks) ?? [];
    const max = Math.max(1, ...counts);
    return (stats?.topics ?? []).slice(0, 7).map((topic) => ({
      ...topic,
      width: Math.max(8, Math.round((topic.chunks / max) * 100))
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
    if (!message.trim()) return;
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
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-4 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-md bg-mint text-white">
              <Bot size={24} />
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-normal">XDENT AI Support Console</h1>
              <p className="text-sm text-slate-500">RAG demonstrátor pro 1. úroveň zákaznické podpory</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill icon={<Database size={16} />} label={`${stats?.points_count ?? 0} chunků`} ok={Boolean(stats?.qdrant_ready)} />
            <StatusPill icon={<Sparkles size={16} />} label={stats?.api_ready ? "OpenAI ready" : "API klíč chybí"} ok={Boolean(stats?.api_ready)} />
            <button className="icon-button" onClick={refreshStats} title="Obnovit statistiky">
              <RefreshCw size={18} />
            </button>
            <button className="primary-button" onClick={handleIngest} disabled={indexing}>
              {indexing ? <Loader2 className="animate-spin" size={17} /> : <Zap size={17} />}
              Indexovat data
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-5 px-5 py-5 xl:grid-cols-[1.05fr_1.2fr_0.9fr]">
        <section className="panel flex min-h-[640px] flex-col">
          <PanelTitle icon={<MessageSquare size={19} />} title="Chat" subtitle="Dotaz zákazníka v přirozeném jazyce" />
          <div className="mb-4 grid gap-2">
            {demoQuestions.map((question) => (
              <button
                key={question}
                className="sample-button"
                onClick={() => setMessage(question)}
              >
                {question}
              </button>
            ))}
          </div>
          <textarea
            className="min-h-36 w-full resize-none rounded-md border border-slate-200 bg-white p-3 text-sm leading-6 outline-none transition focus:border-mint focus:ring-2 focus:ring-mint/20"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            placeholder="Napište dotaz zákazníka..."
          />
          <div className="mt-4 flex items-center justify-between gap-3">
            <label className="flex items-center gap-2 text-sm text-slate-600">
              <input
                type="checkbox"
                className="h-4 w-4 rounded border-slate-300 text-mint focus:ring-mint"
                checked={strictMode}
                onChange={(event) => setStrictMode(event.target.checked)}
              />
              Strict mode
            </label>
            <button className="primary-button" onClick={handleSend} disabled={loading}>
              {loading ? <Loader2 className="animate-spin" size={17} /> : <Send size={17} />}
              Odpovědět
            </button>
          </div>
          {error && (
            <div className="mt-4 flex gap-2 rounded-md border border-coral/30 bg-coral/10 p-3 text-sm text-coral">
              <AlertTriangle size={18} />
              <span>{error}</span>
            </div>
          )}
          <div className="mt-auto pt-5">
            <PanelTitle icon={<Gauge size={18} />} title="Coverage témat" subtitle="Kolik znalostních chunků je v indexu" compact />
            <div className="space-y-3">
              {topicCoverage.length === 0 && <p className="text-sm text-slate-500">Zatím nejsou načtené statistiky.</p>}
              {topicCoverage.map((topic) => (
                <div key={topic.topic}>
                  <div className="mb-1 flex justify-between text-xs text-slate-500">
                    <span>{topic.label}</span>
                    <span>{topic.chunks}</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-100">
                    <div className="h-2 rounded-full bg-mint" style={{ width: `${topic.width}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="panel min-h-[640px]">
          <PanelTitle icon={<Bot size={19} />} title="Odpověď asistenta" subtitle="Stručná odpověď, zdroje a fallback pravidla" />
          <div className="rounded-md border border-slate-200 bg-white p-4">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <Badge icon={<Radar size={15} />} text={response?.topic_label ?? "Téma zatím neurčeno"} />
              <Badge icon={<ShieldCheck size={15} />} text={`Jistota ${Math.round((response?.confidence ?? 0) * 100)} %`} />
              <Badge icon={<Sparkles size={15} />} text={response?.used_llm ? "LLM odpověď" : "Bezpečný fallback"} />
            </div>
            <div className="answer-box">
              {loading ? (
                <div className="flex items-center gap-2 text-slate-500">
                  <Loader2 className="animate-spin" size={18} />
                  Agent vyhledává relevantní hovory...
                </div>
              ) : response ? (
                response.answer
              ) : (
                "Zde se zobrazí odpověď asistenta. Odpověď je vždy opřená o nalezené transkripce nebo bezpečně eskaluje."
              )}
            </div>
          </div>

          <div className="mt-5 grid gap-4">
            <PanelTitle icon={<FileSearch size={18} />} title="Relevantní zdroje" subtitle="Evidence cards z transkripcí" compact />
            {sources.length === 0 && <EmptyState text="Po odeslání dotazu se zde ukážou nalezené části hovorů." />}
            {sources.map((source, index) => (
              <SourceCard key={`${source.source}-${index}`} source={source} index={index} />
            ))}
          </div>
        </section>

        <aside className="space-y-5">
          <section className="panel">
            <PanelTitle icon={<Activity size={19} />} title="Agent timeline" subtitle="Realtime pohled na rozhodování" />
            <div className="space-y-3">
              {steps.map((step, index) => (
                <StepItem key={step.id} step={step} index={index} active={loading && index === 1} />
              ))}
            </div>
          </section>

          <section className="panel">
            <PanelTitle icon={<ShieldCheck size={19} />} title="Eskalace" subtitle="Balíček pro 2. úroveň podpory" />
            <pre className="whitespace-pre-wrap rounded-md border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-700">
              {response?.escalation_packet ?? "Pokud agent nemá jistotu, automaticky připraví stručný balíček pro podporu: téma, dotaz, důvod a co doplnit."}
            </pre>
          </section>

          <section className="panel">
            <PanelTitle icon={<Sparkles size={19} />} title="Demo highlights" subtitle="Co ukázat porotě" />
            <ul className="space-y-2 text-sm text-slate-600">
              <li className="highlight-item"><CheckCircle2 size={16} /> hybridní retrieval přes Qdrant</li>
              <li className="highlight-item"><CheckCircle2 size={16} /> OpenAI embeddingy + chat model</li>
              <li className="highlight-item"><CheckCircle2 size={16} /> kontrola jistoty a fallback</li>
              <li className="highlight-item"><CheckCircle2 size={16} /> evidence cards se zdroji</li>
              <li className="highlight-item"><CheckCircle2 size={16} /> logování interakcí pro evaluaci</li>
            </ul>
          </section>
        </aside>
      </main>
    </div>
  );
}

function mergeStep(current: AgentStep[], next: AgentStep): AgentStep[] {
  const exists = current.some((step) => step.id === next.id);
  if (!exists) {
    return [...current, next];
  }
  return current.map((step) => (step.id === next.id ? { ...step, ...next } : step));
}

function StatusPill({ icon, label, ok }: { icon: ReactNode; label: string; ok: boolean }) {
  return (
    <div className={`status-pill ${ok ? "border-mint/30 bg-mint/10 text-mint" : "border-amber/30 bg-amber/10 text-amber"}`}>
      {icon}
      {label}
    </div>
  );
}

function PanelTitle({ icon, title, subtitle, compact = false }: { icon: ReactNode; title: string; subtitle: string; compact?: boolean }) {
  return (
    <div className={compact ? "mb-3" : "mb-5"}>
      <div className="flex items-center gap-2">
        <span className="text-mint">{icon}</span>
        <h2 className="text-base font-semibold">{title}</h2>
      </div>
      <p className="mt-1 text-sm text-slate-500">{subtitle}</p>
    </div>
  );
}

function Badge({ icon, text }: { icon: ReactNode; text: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600">
      {icon}
      {text}
    </span>
  );
}

function StepItem({ step, index, active }: { step: AgentStep; index: number; active: boolean }) {
  const style =
    step.status === "done"
      ? "border-mint/30 bg-mint/10"
      : step.status === "warning"
        ? "border-amber/30 bg-amber/10"
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

function SourceCard({ source, index }: { source: Source; index: number }) {
  return (
    <article className="rounded-md border border-slate-200 bg-white p-4">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-mint/10 text-mint">{index + 1}</span>
          <span className="truncate">{source.source}</span>
        </div>
        <span className="rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-600">score {source.score}</span>
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
