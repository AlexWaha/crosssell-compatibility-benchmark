/* =====================================================================
   Product page — full card per the data contract + the centrepiece:
   the recommendation block (technically-compatible companions with
   context_code badges and a hybrid_score).
   ===================================================================== */
const { useState: useStateP, useEffect: useEffectP, useMemo: useMemoP } = React;

function ProductPage({ productId }) {
  const [p, setP] = useStateP(null);
  const [notFound, setNotFound] = useStateP(false);
  const [reco, setReco] = useStateP(null);
  const [recoLoading, setRecoLoading] = useStateP(true);
  const [ctxFilter, setCtxFilter] = useStateP("all");
  const [variant, setVariant] = useStateP(window.DEFAULT_EXPERIMENT);

  useEffectP(() => {
    let on = true;
    setP(null); setNotFound(false);
    window.StoreAPI.getProduct(productId).then((d) => { if (on) { if (d) setP(d); else setNotFound(true); } });
    return () => { on = false; };
  }, [productId]);

  useEffectP(() => {
    let on = true;
    setReco(null); setRecoLoading(true); setCtxFilter("all");
    window.StoreAPI.getRecommendations(productId, variant).then((r) => { if (on) { setReco(r); setRecoLoading(false); } });
    return () => { on = false; };
  }, [productId, variant]);

  const ctxCounts = useMemoP(() => {
    const m = {};
    (reco && reco.items || []).forEach((i) => { m[i.context_code] = (m[i.context_code] || 0) + 1; });
    return m;
  }, [reco]);

  if (notFound) return (
    <div className="wrap"><div className="empty-state"><div className="em-ic"><Icon name="search" size={34} /></div>
      <h2>Product not found</h2><button className="btn btn-primary" onClick={() => go("/")}>Back to home</button></div></div>
  );

  if (!p) return (
    <div className="wrap" style={{ paddingTop: 24 }}>
      <div className="pdp">
        <div className="skel" style={{ aspectRatio: "1/1", borderRadius: 16 }}></div>
        <div>
          <div className="skel" style={{ height: 26, width: "80%", marginBottom: 14 }}></div>
          <div className="skel" style={{ height: 14, width: "50%", marginBottom: 24 }}></div>
          <div className="skel" style={{ height: 14, width: "100%", marginBottom: 10 }}></div>
          <div className="skel" style={{ height: 14, width: "90%" }}></div>
        </div>
      </div>
    </div>
  );

  const attrs = Object.entries(p.attributes || {});
  const recoItems = (reco && reco.items || []).filter((i) => ctxFilter === "all" || i.context_code === ctxFilter);

  return (
    <React.Fragment>
      <Crumbs trail={[{ label: "Home", to: "/" },
        ...(p.category_path[0] ? [{ label: p.category_path[0].name, to: `/category/${p.category_path[0].slug}` }] : []),
        ...(p.category_path[1] ? [{ label: p.category_path[1].name }] : []),
        { label: p.name }]} />

      <div className="wrap">
        <div className="pdp">
          {/* single image */}
          <div className="pdp-gallery">
            <div className="pdp-stage"><ProductImage product={p} iconSize={130} /></div>
            <div className="pdp-imgnote"><Icon name="camera" size={14} /> single image per product · <code>{p.image}</code></div>
          </div>

          {/* info */}
          <div className="pdp-info">
            <div className="pdp-type">{p.product_type}</div>
            <h1>{p.name}</h1>
            <div className="pdp-brand">by {p.brand}</div>
            <div className="pdp-price">{window.fmtPrice(p.price)} <span className="cur">{p.currency}</span></div>

            {p.description ? <p className="pdp-desc">{p.description}</p> : null}

            {(p.compatibility_tags || []).length ? (
              <div className="pdp-block">
                <h3>Compatibility</h3>
                <div className="compat-row">
                  {p.compatibility_tags.map((t) => <CompatTag key={t}>{t}</CompatTag>)}
                </div>
                <div className="compat-note">These interface / standard tags drive the engine's compatibility matching.</div>
              </div>
            ) : null}

            {attrs.length ? (
              <div className="pdp-block">
                <h3>Specifications</h3>
                <div className="spec-list">
                  {attrs.map(([k, v]) => (
                    <div className="spec-row" key={k}><span className="k">{k}</span><span className="v">{v}</span></div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </div>

        {/* ============ RECOMMENDATION BLOCK (centrepiece) ============ */}
        <section className="reco-block">
          <div className="reco-head">
            <div>
              <h2>Compatible companions</h2>
              <p>Selected by the cross-sell engine: products that are technically compatible with this item. Each one is tagged by <b>connection type</b>, and the small number is the engine's <b>hybrid match score</b> (0 to 1).</p>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10, alignItems: "flex-end" }}>
              <div className="variant-toggle" role="group" aria-label="Experiment variant">
                {window.EXPERIMENTS.map((e) => (
                  <button key={e.id} className={"variant-btn" + (variant === e.id ? " is-active" : "")}
                          onClick={() => setVariant(e.id)} title={e.label}>{e.short}</button>
                ))}
              </div>
              <div className="reco-engine-pill"><Icon name="link2" size={15} /> recommendation engine · via API</div>
            </div>
          </div>

          {recoLoading ? (
            <SkeletonGrid n={5} />
          ) : (reco.items.length === 0) ? (
            <div className="reco-empty">
              <Icon name="link2" size={28} />
              <div>
                <b>No compatible companions found</b>
                <span>This product exposes no interface tags the engine can match against.</span>
              </div>
            </div>
          ) : (
            <React.Fragment>
              <div className="ctx-filter">
                <button className={"ctx-chip" + (ctxFilter === "all" ? " is-active" : "")} onClick={() => setCtxFilter("all")}>
                  All <span className="cnt">{reco.items.length}</span>
                </button>
                {Object.keys(ctxCounts).map((code) => (
                  <button key={code} className={"ctx-chip" + (ctxFilter === code ? " is-active" : "")} onClick={() => setCtxFilter(code)}>
                    <ContextBadge code={code} size="sm" /> <span className="cnt">{ctxCounts[code]}</span>
                  </button>
                ))}
              </div>
              <Carousel>
                {recoItems.map((it) => <RecoCard key={it.id} product={it} />)}
              </Carousel>
            </React.Fragment>
          )}
        </section>
      </div>
    </React.Fragment>
  );
}

Object.assign(window, { ProductPage });
