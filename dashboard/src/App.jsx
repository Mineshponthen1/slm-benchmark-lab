import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  Legend, ResponsiveContainer, Cell, LabelList,
} from "recharts";

const WINNER = "llama3.2 q4";
const BLUE = "#4C72B0";
const ORANGE = "#DD8452";
const GREY = "#9AA5B1";

function ModelChart({ title, note, data, dataKey, unit }) {
  return (
    <div className="card">
      <h3>{title}</h3>
      <p className="note">{note}</p>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} margin={{ top: 20, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="short" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip formatter={(value) => `${value} ${unit}`} />
          <Bar dataKey={dataKey} radius={[4, 4, 0, 0]}>
            <LabelList dataKey={dataKey} position="top" fontSize={12} />
            {data.map((m) => (
              <Cell key={m.short} fill={m.short === WINNER ? ORANGE : BLUE} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function CompareChart({ title, note, rows, keys }) {
  return (
    <div className="card">
      <h3>{title}</h3>
      <p className="note">{note}</p>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={rows} margin={{ top: 20, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="measure" tick={{ fontSize: 12 }} />
          <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} unit="%" />
          <Tooltip formatter={(value) => `${value}%`} />
          <Legend />
          {keys.map((k) => (
            <Bar key={k.key} dataKey={k.key} name={k.label} fill={k.color} radius={[4, 4, 0, 0]}>
              <LabelList dataKey={k.key} position="top" fontSize={11} formatter={(v) => `${v}%`} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function Kpi({ label, value }) {
  return (
    <div className="card kpi">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
    </div>
  );
}

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("/data.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <main>
        <p>Could not load data.json ({error}). Run <code>python export_dashboard_data.py</code> first.</p>
      </main>
    );
  }
  if (!data) return <main><p>Loading...</p></main>;

  const winner = data.models.find((m) => m.short === WINNER);
  const t0 = data.validity["0.0"];
  const t07 = data.validity["0.7"];
  const validityRows = Object.keys(t0).map((m) => ({ measure: m, t0: t0[m], t07: t07[m] }));
  const groundingRows = data.grounding
    ? ["Correct", "Silently wrong"].map((m) => ({
        measure: m, before: data.grounding.before[m], after: data.grounding.after[m],
      }))
    : null;

  return (
    <main>
      <header>
        <h1>SLM Benchmark Dashboard</h1>
        <p>{data.hardware}</p>
      </header>

      <section className="summary">
        <Kpi label="Recommended model" value={WINNER} />
        <Kpi label="Its speed" value={`${winner.tokens_per_s} tokens/s`} />
        <Kpi label="Its time to first token" value={`${winner.ttft_s} s`} />
        <Kpi label="Silently wrong forms (temp 0)" value={`${t0["Silently wrong"]}%`} />
      </section>

      <h2>1. Model comparison (same laptop, same questions)</h2>
      <div className="grid">
        <ModelChart title="Speed" note="Tokens per second. Higher is better."
          data={data.models} dataKey="tokens_per_s" unit="tokens/s" />
        <ModelChart title="Time to first token" note="Seconds until the first word. Lower is better."
          data={data.models} dataKey="ttft_s" unit="s" />
        <ModelChart title="Memory" note="GB of RAM while loaded. Lower is better."
          data={data.models} dataKey="memory_gb" unit="GB" />
        <ModelChart title="Quality" note="% correct on the 30-question exam. Higher is better."
          data={data.models} dataKey="quality_pct" unit="%" />
      </div>

      <h2>2. Structured-output validity (leave-request extraction, {data.extraction_model})</h2>
      <div className="grid">
        <CompareChart
          title="Temperature 0 vs 0.7"
          note="15 leave messages x 5 runs each. Silently wrong: lower is better."
          rows={validityRows}
          keys={[
            { key: "t0", label: "Temperature 0", color: BLUE },
            { key: "t07", label: "Temperature 0.7", color: ORANGE },
          ]}
        />
        {groundingRows && (
          <CompareChart
            title="Effect of leave-type grounding (temperature 0)"
            note="Before vs after the rule that the leave type must appear in the message."
            rows={groundingRows}
            keys={[
              { key: "before", label: "Before", color: GREY },
              { key: "after", label: "After", color: BLUE },
            ]}
          />
        )}
      </div>

      <footer>Data generated on {data.generated} by export_dashboard_data.py</footer>
    </main>
  );
}