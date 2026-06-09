/* =====================================================================
   Compare — side-by-side metrics of the retrieval strategies (cat_priors /
   wide / semantic). Reads StoreAPI.getCompare(). Shows coverage, verdict rate,
   HSV, SLD and the IR metrics (P/R/F1/NDCG/MAP/MRR @10) with a delta column.
   ===================================================================== */
const { useState: useStateCmp, useEffect: useEffectCmp } = React;

const CMP_ROWS = [
  { key: "coverage", label: "Coverage", kind: "pct", better: "up" },
  { key: "verdict1_share", label: "Verified rate (TVR)", kind: "pct", better: "up" },
  { key: "with_reco", label: "Products with companions", kind: "int", better: "up" },
  { key: "total_pairs", label: "Pairs evaluated", kind: "int", better: "flat" },
  { key: "hsv", label: "HSV (hybrid validity)", kind: "f3", better: "up" },
  { key: "sld", label: "SLD (semantic-logical divergence)", kind: "f3", better: "down" },
  { key: "precision_at_10", label: "Precision@10", kind: "f3", better: "up" },
  { key: "recall_at_10", label: "Recall@10", kind: "f3", better: "up" },
  { key: "f1_at_10", label: "F1@10", kind: "f3", better: "up" },
  { key: "ndcg_at_10", label: "NDCG@10", kind: "f3", better: "up" },
  { key: "map_at_10", label: "MAP@10", kind: "f3", better: "up" },
  { key: "mrr", label: "MRR", kind: "f3", better: "up" },
  { key: "alpha_optimal", label: "Optimal α", kind: "f2", better: "flat" },
  { key: "sample_size", label: "Labeled sample size", kind: "int", better: "flat" },
  { key: "cohens_kappa", label: "Cohen's κ (verify vs judge)", kind: "f3", better: "up" },
];

function fmtVal(v, kind) {
  if (v == null) return "—";
  if (kind === "pct") return (v * 100).toFixed(2) + "%";
  if (kind === "int") return window.fmtInt(v);
  if (kind === "f2") return Number(v).toFixed(2);
  return Number(v).toFixed(3);
}

function ComparePage() {
  const [data, setData] = useStateCmp(null);
  useEffectCmp(() => {
    let on = true;
    window.StoreAPI.getCompare().then((d) => { if (on) setData(d); }).catch(() => {});
    return () => { on = false; };
  }, []);

  if (!data) return (
    <div className="wrap" style={{ paddingTop: 30 }}>
      <div className="skel" style={{ height: 120, borderRadius: 16, marginBottom: 16 }}></div>
      <div className="skel" style={{ height: 320, borderRadius: 16 }}></div>
    </div>
  );

  const exps = data.experiments;
  const labelOf = (id) => (window.EXPERIMENTS.find((e) => e.id === id) || {}).label || id;
  const twoWithData = exps.length === 2 && exps[0].has_data && exps[1].has_data;

  return (
    <div className="wrap">
      <div className="metrics-head">
        <div>
          <div className="eyebrow">Experiment comparison</div>
          <h1>Retrieval strategy comparison</h1>
        </div>
        <div className="metrics-stamp"><span className="dot"></span> catalog {window.fmtInt(data.catalog)} products</div>
      </div>

      <div className="chart-card" style={{ overflowX: "auto" }}>
        <table className="cmp-table">
          <thead>
            <tr>
              <th>Metric</th>
              {exps.map((e) => (
                <th key={e.experiment_id}>
                  {labelOf(e.experiment_id)}
                  {e.has_data ? null : <div className="cmp-empty">not run yet</div>}
                </th>
              ))}
              {twoWithData ? <th>Δ</th> : null}
            </tr>
          </thead>
          <tbody>
            {CMP_ROWS.map((row) => {
              const vals = exps.map((e) => e[row.key]);
              let delta = null, deltaCls = "";
              if (twoWithData && vals[0] != null && vals[1] != null) {
                const d = vals[1] - vals[0];
                const good = row.better === "up" ? d > 0 : row.better === "down" ? d < 0 : null;
                deltaCls = good == null ? "" : good ? "cmp-up" : "cmp-down";
                const shown = row.kind === "pct" ? ((d * 100).toFixed(2) + " pp")
                  : row.kind === "int" ? window.fmtInt(d)
                  : d.toFixed(3);
                delta = (d > 0 ? "+" : "") + shown;
              }
              return (
                <tr key={row.key}>
                  <td className="cmp-metric">{row.label}</td>
                  {vals.map((v, i) => <td key={i}>{fmtVal(v, row.kind)}</td>)}
                  {twoWithData ? <td className={deltaCls}>{delta || "—"}</td> : null}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="chart-foot" style={{ marginTop: 16 }}>
        Label-based metrics (P/R/NDCG/MAP/MRR, HSV, κ) require the ground-truth set; run
        <code> python -m app.eval label</code> then <code>compute</code> per experiment. Coverage, TVR and
        SLD are label-free. Δ is Experiment 2 minus Experiment 1 (green = improvement).
      </div>
    </div>
  );
}

Object.assign(window, { ComparePage });
