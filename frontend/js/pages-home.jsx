/* =====================================================================
   Home — research stand landing. Explains the experiment (problem / method /
   this stand) and lets you jump into the catalog or a sample product.
   ===================================================================== */
const { useState: useStateH, useEffect: useEffectH } = React;

function HomePage({ categories }) {
  const catN = (window.MockData.metrics && window.MockData.metrics.catalog) || {};
  const [top, setTop] = useStateH([]);
  useEffectH(() => {
    let on = true;
    window.StoreAPI.getTopProducts(20).then((d) => { if (on) setTop(d || []); }).catch(() => {});
    return () => { on = false; };
  }, []);
  return (
    <div className="wrap">
      <section className="home-hero">
        <div className="home-hero-text">
          <div className="eyebrow">AVTC · automated technical-compatibility verification</div>
          <h1>Technically compatible cross-sell, without sales history</h1>
          <p>
            Open any product and the engine suggests companions that actually fit it: for a
            laptop, a bag and an SSD; for a camera, a tripod and memory cards. It works for
            brand-new products that have <b>no clicks or purchase history</b>. Relevance comes
            from a category-complement prior plus an LLM compatibility check, not from user
            behavior.
          </p>
          <div className="home-hero-actions">
            <button className="btn btn-primary btn-lg" onClick={() => go("/product/artline-x37v62")}>Open a sample product</button>
            <button className="btn btn-ghost btn-lg" onClick={() => go("/metrics")}>View metrics</button>
          </div>
          <div className="home-hero-meta">
            {window.fmtInt(catN.products || 0)} products · {window.fmtInt(catN.categories || 0)} categories ·
            OpenAI models (text-embedding-3-small + gpt-5-nano) · Typesense vectors
          </div>
        </div>

        <div className="home-hero-side">
          <div className="hint-card">
            <div className="hint-card-title">How a recommendation is made</div>
            <div className="hint-row"><span className="hint-num">1</span> A category-complement prior retrieves candidates from complementary categories (accessories), not look-alikes</div>
            <div className="hint-row"><span className="hint-num">2</span> An LLM (gpt-5-nano) verifies technical compatibility for each candidate</div>
            <div className="hint-row"><span className="hint-num">3</span> Hybrid score <code>α·S + (1-α)·L</code> ranks the verified companions</div>
          </div>
        </div>
      </section>

      {/* — the experiment, in three cards — */}
      <section className="section">
        <div className="method-grid">
          <div className="method-card">
            <div className="method-tag">The problem</div>
            <h3>Cold start &amp; the semantic gap</h3>
            <p>Classic recommenders rely on behavioral data, which new products do not have.
               And semantic similarity is not technical compatibility: two TVs are similar
               yet not complementary, while a TV actually needs a cable or a mount.</p>
          </div>
          <div className="method-card">
            <div className="method-tag">The method</div>
            <h3>Complement prior + LLM verification</h3>
            <p>Candidates come from a category-complement prior (a curated map of which
               categories accompany which), so accessories are surfaced instead of look-alikes.
               An LLM (gpt-5-nano) then verifies technical compatibility per candidate and the
               incompatible ones are dropped.</p>
          </div>
          <div className="method-card">
            <div className="method-tag">This stand</div>
            <h3>Validation on {window.fmtInt(catN.products || 3000)} products</h3>
            <p>A curated catalog scored against an independent LLM judge (gpt-5-mini) that
               labels every candidate pair. Embeddings: OpenAI text-embedding-3-small in
               Typesense. Open <a href="#/metrics" onClick={(e) => { e.preventDefault(); go("/metrics"); }}>Metrics</a>
               for precision/recall, the α-curve and coverage.</p>
          </div>
        </div>
      </section>

      {top.length ? (
        <section className="section">
          <div className="section-head">
            <h2>Most recommended products</h2>
            <span className="muted">top {top.length} by number of verified companions</span>
          </div>
          <div className="grid-products cols-5">
            {top.map((p) => (
              <div key={p.id} style={{ position: "relative" }}>
                <span style={{
                  position: "absolute", top: 8, left: 8, zIndex: 2,
                  background: "var(--accent)", color: "#fff", fontSize: 11, fontWeight: 700,
                  padding: "3px 8px", borderRadius: 999, lineHeight: 1.4,
                }}>{window.fmtInt(p.reco_count)} recos</span>
                <ProductCard product={p} />
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <section className="section">
        <div className="section-head"><h2>All categories</h2><span className="muted">{categories.length} top-level</span></div>
        <div className="home-cat-grid">
          {categories.map((c) => (
            <Link key={c.id} to={`/category/${c.slug}`} className="home-cat">
              <span className="ic" style={{ background: c.tint, color: c.fg }}><Icon name={c.icon} size={26} /></span>
              <span className="home-cat-name">{c.name}</span>
              <span className="home-cat-count">{window.fmtInt(c.product_count)} items</span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}

Object.assign(window, { HomePage });
