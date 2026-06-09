/* =====================================================================
   Grid product card (Home/Category/Search) + recommendation card
   (adds a context_code badge and a small hybrid_score). No cart, no
   ratings, no badges — only contract fields.
   ===================================================================== */
const { useState, useEffect, useRef, useCallback } = React;

function ProductCard({ product }) {
  return (
    <article className="pcard" onClick={() => go(`/product/${product.slug || product.id}`)} role="link" tabIndex={0}
             onKeyDown={(e) => { if (e.key === "Enter") go(`/product/${product.slug || product.id}`); }}>
      <div className="pcard-media"><ProductImage product={product} iconSize={54} /></div>
      <div className="pcard-type">{product.product_type}</div>
      <h3 className="pcard-title">{product.name}</h3>
      <div className="pcard-foot">
        <span className="pcard-brand">{product.brand}</span>
        <Price price={product.price} />
      </div>
    </article>
  );
}

/* — Recommendation card: same card + connection-type badge + score — */
function RecoCard({ product, showScore = true }) {
  return (
    <article className="pcard reco" onClick={() => go(`/product/${product.slug || product.id}`)} role="link" tabIndex={0}
             onKeyDown={(e) => { if (e.key === "Enter") go(`/product/${product.slug || product.id}`); }}>
      <div className="pcard-media">
        <div className="reco-badge"><ContextBadge code={product.context_code} /></div>
        <ProductImage product={product} iconSize={54} />
      </div>
      <div className="pcard-type">{product.product_type}</div>
      <h3 className="pcard-title">{product.name}</h3>
      <div className="pcard-foot">
        <Price price={product.price} />
        {showScore ? <span className="reco-score" title="Hybrid match score (0-1)">{product.hybrid_score.toFixed(2)}</span> : null}
      </div>
    </article>
  );
}

/* — Horizontal carousel with arrows (used by the recommendation block) — */
function Carousel({ children }) {
  const trackRef = useRef(null);
  const [edge, setEdge] = useState({ left: true, right: false });
  // A handful of cards do not need scrolling; keep them at natural width, no nav arrows.
  const sparse = React.Children.count(children) < 3;
  const update = useCallback(() => {
    const el = trackRef.current; if (!el) return;
    setEdge({ left: el.scrollLeft < 8, right: el.scrollLeft + el.clientWidth >= el.scrollWidth - 8 });
  }, []);
  useEffect(() => { update(); }, [children, update]);
  const scroll = (dir) => {
    const el = trackRef.current; if (!el) return;
    el.scrollBy({ left: dir * Math.round(el.clientWidth * 0.85), behavior: "smooth" });
    setTimeout(update, 360);
  };
  return (
    <div className={"carousel" + (sparse ? " is-sparse" : "")}>
      {sparse ? null : (
        <button className="carousel-nav prev" disabled={edge.left} onClick={() => scroll(-1)} aria-label="Previous">
          <Icon name="chevron" size={20} style={{ transform: "rotate(180deg)" }} />
        </button>
      )}
      <div className="carousel-track" ref={trackRef} onScroll={update}>{children}</div>
      {sparse ? null : (
        <button className="carousel-nav next" disabled={edge.right} onClick={() => scroll(1)} aria-label="Next">
          <Icon name="chevron" size={20} />
        </button>
      )}
    </div>
  );
}

function ProductGrid({ products, cols }) {
  return (
    <div className={"grid-products" + (cols ? " cols-" + cols : "")}>
      {products.map((p) => <ProductCard key={p.id} product={p} />)}
    </div>
  );
}

Object.assign(window, { ProductCard, RecoCard, Carousel, ProductGrid });
