/* =====================================================================
   Metrics — research dashboard. Reads StoreAPI.getMetrics().
   Charts are plain SVG (data viz), no external libraries.
   ===================================================================== */
const { useState: useStateM, useEffect: useEffectM } = React;

function MetricsPage() {
  const [m, setM] = useStateM(null);
  useEffectM(() => { let on = true; window.StoreAPI.getMetrics().then((d) => { if (on) setM(d); }); return () => { on = false; }; }, []);

  if (!m) return <div className="wrap" style={{ paddingTop: 30 }}><div className="skel" style={{ height: 120, borderRadius: 16, marginBottom: 16 }}></div><div className="skel" style={{ height: 300, borderRadius: 16 }}></div></div>;

  return (
    <div className="wrap">
      <div className="metrics-head">
        <div>
          <div className="eyebrow">Evaluation dashboard</div>
          <h1>Recommendation quality</h1>
        </div>
        <div className="metrics-stamp"><span className="dot"></span> offline eval · build {m.stats.index_build}</div>
      </div>

      {/* stat cards */}
      <div className="stat-grid">
        <StatCard label="Catalog" value={window.fmtInt(m.catalog.products)} sub="products" />
        <StatCard label="Coverage" value={(m.coverage * 100).toFixed(2) + "%"} sub={`${window.fmtInt(m.catalog.with_reco)} with recos`} accent />
        <StatCard label="Brands" value={window.fmtInt(m.catalog.brands)} sub="distinct" />
        <StatCard label="TVR" value={((m.stats.tvr || 0) * 100).toFixed(2) + "%"} sub="verified rate" />
        <StatCard label="HSV" value={(m.stats.hsv || 0).toFixed(3)} sub="hybrid validity" />
        <StatCard label="SLD" value={(m.stats.sld || 0).toFixed(3)} sub="semantic-logical div." />
      </div>

      <div className="chart-row">
        <div className="chart-card">
          <div className="chart-title"><h3>Precision / Recall / NDCG @ K</h3><span className="muted">offline, held-out set</span></div>
          <BarChartPK data={m.pAtK} />
          <div className="legend">
            <span><i style={{ background: "#2f6bed" }}></i> Precision</span>
            <span><i style={{ background: "#1f9d57" }}></i> Recall</span>
            <span><i style={{ background: "#d4622a" }}></i> NDCG</span>
          </div>
        </div>

        <div className="chart-card">
          <div className="chart-title"><h3>Alpha curve</h3><span className="muted">hybrid weight · semantic ↔ compatibility</span></div>
          <AlphaChart data={m.alpha} best={m.best_alpha} />
          <div className="chart-foot">Best quality at <b>alpha = {m.best_alpha}</b>: a balanced mix of semantic and compatibility signals.</div>
        </div>
      </div>

      <div className="chart-row">
        <div className="chart-card">
          <div className="chart-title"><h3>context_code distribution</h3><span className="muted">share of recommended items</span></div>
          <ContextDist data={m.contextDist} />
        </div>
        <div className="chart-card coverage-card">
          <div className="chart-title"><h3>Catalog coverage</h3></div>
          <Donut value={m.coverage} />
          <div className="chart-foot">{window.fmtInt(m.catalog.with_reco)} of {window.fmtInt(m.catalog.products)} products receive at least one recommendation.</div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, sub, accent }) {
  return (
    <div className={"stat-card" + (accent ? " accent" : "")}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-sub">{sub}</div>
    </div>
  );
}

/* — Grouped bar chart (P/R/NDCG over K) — */
function BarChartPK({ data }) {
  const W = 560, H = 250, padL = 38, padR = 14, padT = 14, padB = 38;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const measures = [["precision", "#2f6bed"], ["recall", "#1f9d57"], ["ndcg", "#d4622a"]];
  const gw = plotW / data.length;
  const bw = Math.min(26, (gw - 24) / 3);
  const y = (v) => padT + (1 - v) * plotH;
  return (
    <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={padL} y1={y(g)} x2={W - padR} y2={y(g)} stroke="var(--line)" strokeWidth="1" />
          <text x={padL - 8} y={y(g) + 4} textAnchor="end" className="ax-txt">{g.toFixed(2)}</text>
        </g>
      ))}
      {data.map((d, gi) => {
        const gx = padL + gi * gw;
        return (
          <g key={d.k}>
            {measures.map(([key, color], mi) => {
              const x = gx + (gw - bw * 3 - 8) / 2 + mi * (bw + 4);
              const h = (1 - 0) * (plotH) * d[key];
              return (
                <g key={key}>
                  <rect x={x} y={y(d[key])} width={bw} height={plotH * d[key]} rx="3" fill={color} />
                  <text x={x + bw / 2} y={y(d[key]) - 5} textAnchor="middle" className="ax-val">{d[key].toFixed(2)}</text>
                </g>
              );
            })}
            <text x={gx + gw / 2} y={H - 14} textAnchor="middle" className="ax-txt">K = {d.k}</text>
          </g>
        );
      })}
    </svg>
  );
}

/* — Alpha curve line chart — */
function AlphaChart({ data, best }) {
  const W = 560, H = 250, padL = 38, padR = 16, padT = 14, padB = 38;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const x = (a) => padL + a * plotW;
  const y = (q) => padT + (1 - q) * plotH;
  const pts = data.map((d) => `${x(d.alpha)},${y(d.quality)}`).join(" ");
  const area = `${padL},${y(0)} ` + pts + ` ${x(1)},${y(0)}`;
  const bestPt = data.reduce((a, b) => (b.quality > a.quality ? b : a), data[0]);
  return (
    <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={padL} y1={y(g)} x2={W - padR} y2={y(g)} stroke="var(--line)" strokeWidth="1" />
          <text x={padL - 8} y={y(g) + 4} textAnchor="end" className="ax-txt">{g.toFixed(2)}</text>
        </g>
      ))}
      {[0, 0.25, 0.5, 0.75, 1].map((a) => (
        <text key={a} x={x(a)} y={H - 14} textAnchor="middle" className="ax-txt">{a.toFixed(2)}</text>
      ))}
      <polygon points={area} fill="var(--accent)" opacity="0.10" />
      <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
      <line x1={x(best)} y1={padT} x2={x(best)} y2={y(0)} stroke="var(--accent)" strokeWidth="1.2" strokeDasharray="4 4" opacity="0.5" />
      <circle cx={x(bestPt.alpha)} cy={y(bestPt.quality)} r="5" fill="var(--accent)" stroke="#fff" strokeWidth="2" />
      <text x={x(best)} y={padT - 2} textAnchor="middle" className="ax-val" fill="var(--accent)">α={best}</text>
    </svg>
  );
}

/* — context_code distribution (horizontal bars) — */
function ContextDist({ data }) {
  const max = Math.max(...data.map((d) => d.pct));
  return (
    <div className="dist">
      {data.map((d) => {
        const c = (window.CONTEXT[d.code]) || { color: "#5a6472" };
        return (
          <div className="dist-row" key={d.code}>
            <div className="dist-label"><ContextBadge code={d.code} size="sm" /></div>
            <div className="dist-bar"><span style={{ width: (d.pct / max * 100) + "%", background: c.color }}></span></div>
            <div className="dist-pct">{(d.pct * 100).toFixed(2)}%</div>
          </div>
        );
      })}
    </div>
  );
}

/* — Coverage donut (conic-gradient) — */
function Donut({ value }) {
  const pct = value * 100;
  return (
    <div className="donut-wrap">
      <div className="donut" style={{ background: `conic-gradient(var(--accent) ${pct}%, var(--bg-2) 0)` }}>
        <div className="donut-hole"><span className="donut-val">{pct.toFixed(2)}%</span><span className="donut-cap">coverage</span></div>
      </div>
    </div>
  );
}

Object.assign(window, { MetricsPage });
