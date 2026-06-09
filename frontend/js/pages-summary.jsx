/* =====================================================================
   Summary — run totals ("итоги"). Reads StoreAPI.getSummary().
   High-level outcome of the compatibility run: coverage, verdict split,
   average scores and the context_code breakdown. Plain CSS, no charts libs.
   ===================================================================== */
const { useState: useStateS, useEffect: useEffectS } = React;

function SummaryStat({ label, value, sub, accent }) {
  return (
    <div className={"stat-card" + (accent ? " accent" : "")}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-sub">{sub}</div>
    </div>
  );
}

function SummaryPage() {
  const [s, setS] = useStateS(null);
  useEffectS(() => {
    let on = true;
    window.StoreAPI.getSummary().then((d) => { if (on) setS(d); }).catch(() => {});
    return () => { on = false; };
  }, []);

  if (!s) {
    return (
      <div className="wrap" style={{ paddingTop: 30 }}>
        <div className="skel" style={{ height: 120, borderRadius: 16, marginBottom: 16 }}></div>
        <div className="skel" style={{ height: 240, borderRadius: 16 }}></div>
      </div>
    );
  }

  const pct = (x) => ((x || 0) * 100).toFixed(2) + "%";
  const v1 = s.verdict1_pairs || 0;
  const v0 = s.verdict0_pairs || 0;
  const ctxMax = Math.max(1, ...s.by_context.map((c) => c.count));

  return (
    <div className="wrap">
      <div className="metrics-head">
        <div>
          <div className="eyebrow">Run summary</div>
          <h1>Compatibility run totals</h1>
        </div>
        <div className="metrics-stamp"><span className="dot"></span> model · {s.model}</div>
      </div>

      <div className="stat-grid">
        <SummaryStat label="Catalog" value={window.fmtInt(s.catalog)} sub="active products" />
        <SummaryStat label="Evaluated" value={window.fmtInt(s.evaluated)} sub="products processed" />
        <SummaryStat label="Coverage" value={pct(s.coverage)} sub={`${window.fmtInt(s.with_reco)} with companions`} accent />
        <SummaryStat label="Compatible pairs" value={window.fmtInt(v1)} sub={`of ${window.fmtInt(s.total_pairs)} evaluated`} />
        <SummaryStat label="Compatible share" value={pct(s.verdict1_share)} sub="verdict = 1" />
        <SummaryStat label="Avg hybrid" value={(s.avg_hybrid || 0).toFixed(3)} sub="α·S + (1-α)·L" />
      </div>

      <div className="chart-row">
        <div className="chart-card">
          <div className="chart-title"><h3>Verdict split</h3><span className="muted">all evaluated pairs</span></div>
          <div className="dist">
            <div className="dist-row">
              <div className="dist-label"><span className="ctx-badge" style={{ color: "#1f9d57", background: "#e7f6ee" }}>Compatible</span></div>
              <div className="dist-bar"><span style={{ width: pct(s.total_pairs ? v1 / s.total_pairs : 0), background: "#1f9d57" }}></span></div>
              <div className="dist-pct">{window.fmtInt(v1)}</div>
            </div>
            <div className="dist-row">
              <div className="dist-label"><span className="ctx-badge" style={{ color: "#b1374c", background: "#fbe9ec" }}>Rejected</span></div>
              <div className="dist-bar"><span style={{ width: pct(s.total_pairs ? v0 / s.total_pairs : 0), background: "#b1374c" }}></span></div>
              <div className="dist-pct">{window.fmtInt(v0)}</div>
            </div>
          </div>
          <div className="chart-foot">
            The LLM rejected {pct(s.total_pairs ? v0 / s.total_pairs : 0)} of candidate companions as not
            technically compatible — being in a complementary category is not the same as fitting.
          </div>
        </div>

        <div className="chart-card">
          <div className="chart-title"><h3>Average scores</h3><span className="muted">across all pairs</span></div>
          <div className="dist">
            {[["Semantic S", s.avg_semantic, "#2f6bd6"], ["Logical L", s.avg_logical, "#7257d6"], ["Hybrid", s.avg_hybrid, "#1f9d57"]].map(([lab, val, col]) => (
              <div className="dist-row" key={lab}>
                <div className="dist-label" style={{ fontWeight: 600 }}>{lab}</div>
                <div className="dist-bar"><span style={{ width: pct(val), background: col }}></span></div>
                <div className="dist-pct">{(val || 0).toFixed(3)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="chart-row">
        <div className="chart-card">
          <div className="chart-title"><h3>context_code breakdown</h3><span className="muted">verified companions by relation</span></div>
          <div className="dist">
            {s.by_context.map((c) => (
              <div className="dist-row" key={c.code}>
                <div className="dist-label"><ContextBadge code={c.code} size="sm" /></div>
                <div className="dist-bar"><span style={{ width: (c.count / ctxMax * 100) + "%", background: (window.CONTEXT[c.code] || {}).color || "#5a6472" }}></span></div>
                <div className="dist-pct">{window.fmtInt(c.count)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { SummaryPage });
