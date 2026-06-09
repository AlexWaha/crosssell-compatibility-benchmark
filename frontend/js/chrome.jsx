/* =====================================================================
   Chrome: header, search + dropdown (mock Typesense), catalog mega-menu,
   breadcrumbs, footer. No cart, accounts, or wishlist — this is a demo stand.
   ===================================================================== */
const { useState: useStateC, useEffect: useEffectC, useRef: useRefC } = React;

function highlight(text, query) {
  const nq = (query || "").trim();
  if (!nq) return text;
  try {
    const re = new RegExp("(" + nq.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").split(/\s+/).join("|") + ")", "ig");
    return String(text).split(re).map((p, i) => (re.test(p) ? <mark key={i}>{p}</mark> : <span key={i}>{p}</span>));
  } catch (e) { return text; }
}

/* ----------------------- Search with dropdown ----------------------- */
function SearchBox() {
  const [q, setQ] = useStateC("");
  const [open, setOpen] = useStateC(false);
  const [res, setRes] = useStateC(null);
  const [loading, setLoading] = useStateC(false);
  const [active, setActive] = useStateC(-1);
  const boxRef = useRefC(null);
  const reqId = useRefC(0);

  useEffectC(() => {
    if (!q.trim()) { setRes(null); setLoading(false); return; }
    setLoading(true);
    const myId = ++reqId.current;
    const t = setTimeout(async () => {
      const r = await window.StoreAPI.suggest(q);
      if (myId === reqId.current) { setRes(r); setLoading(false); }
    }, 170);
    return () => clearTimeout(t);
  }, [q]);

  useEffectC(() => {
    const h = (e) => { if (boxRef.current && !boxRef.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  const submit = (query) => {
    const v = (query != null ? query : q).trim();
    if (!v) return;
    setOpen(false);
    go(`/search?q=${encodeURIComponent(v)}`);
  };
  const items = res ? res.items : [];
  const onKey = (e) => {
    if (!open) return;
    if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(items.length - 1, a + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(-1, a - 1)); }
    else if (e.key === "Enter") { if (active >= 0 && items[active]) { setOpen(false); go(`/product/${items[active].slug || items[active].id}`); } else submit(); }
    else if (e.key === "Escape") setOpen(false);
  };

  return (
    <div className="search" ref={boxRef}>
      <div className={"search-field" + (open ? " is-focus" : "")}>
        <Icon name="search" size={20} style={{ color: "var(--ink-3)", flex: "none", marginRight: 10 }} />
        <input value={q} placeholder="Search the catalog, e.g. 4K TV, USB-C charger, camera"
               onChange={(e) => { setQ(e.target.value); setOpen(true); setActive(-1); }}
               onFocus={() => setOpen(true)} onKeyDown={onKey} aria-label="Search products" />
        {q ? <button className="search-clear" onClick={() => { setQ(""); setRes(null); }} aria-label="Clear"><Icon name="x" size={16} /></button> : null}
        <button className="search-go" onClick={() => submit()}>Search</button>
      </div>

      {open ? (
        <div className="sd">
          {!q.trim() ? (
            <React.Fragment>
              <div className="sd-section-label">Popular searches</div>
              <div className="sd-recent">
                {window.MockData.popularQueries.map((p) => (
                  <button key={p} className="chip" onClick={() => { setQ(p); submit(p); }}><Icon name="search" size={14} /> {p}</button>
                ))}
              </div>
            </React.Fragment>
          ) : loading && !res ? (
            <div style={{ padding: 8 }}>
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="sd-item">
                  <div className="skel" style={{ width: 46, height: 46, borderRadius: 8 }}></div>
                  <div style={{ flex: 1 }}>
                    <div className="skel" style={{ height: 12, width: "70%", marginBottom: 7 }}></div>
                    <div className="skel" style={{ height: 10, width: "40%" }}></div>
                  </div>
                </div>
              ))}
            </div>
          ) : items.length === 0 ? (
            <div className="sd-empty">No matches for “{q}”. Try a different term.</div>
          ) : (
            <React.Fragment>
              <div className="sd-section-label">Products · {items.length} of {res.total}</div>
              {items.map((p, i) => (
                <a key={p.id} href={`/product/${p.slug || p.id}`} className={"sd-item" + (i === active ? " is-active" : "")}
                   onClick={(e) => { e.preventDefault(); setOpen(false); go(`/product/${p.slug || p.id}`); }} onMouseEnter={() => setActive(i)}>
                  <div className="sd-thumb"><ProductImage product={p} caption={false} iconSize={22} /></div>
                  <div className="sd-item-main">
                    <div className="sd-item-title">{highlight(p.name, q)}</div>
                    <div className="sd-item-cat">{p.product_type} · {p.brand}</div>
                  </div>
                  <div className="sd-item-price">{window.fmtPrice(p.price)}</div>
                </a>
              ))}
              <div className="sd-footer">
                <a href={`/search?q=${encodeURIComponent(q)}`} className="sd-all" onClick={(e) => { e.preventDefault(); submit(); }}>
                  <span>See all results for “{q}”</span><Icon name="arrow" size={18} />
                </a>
              </div>
            </React.Fragment>
          )}
        </div>
      ) : null}
    </div>
  );
}

/* ----------------------- Catalog mega-menu ----------------------- */
function MegaMenu({ categories, onClose }) {
  const [active, setActive] = useStateC(categories[0]);
  if (!categories || !categories.length) return null;
  const cur = active || categories[0];
  const subs = cur.subs || [];
  return (
    <React.Fragment>
      <div className="mega-overlay" onClick={onClose}></div>
      <div className="mega" onMouseLeave={onClose}>
        <button className="mega-close" onClick={onClose} aria-label="Close catalog"><Icon name="x" size={20} /></button>
        <div className="wrap" style={{ padding: 0 }}>
          <div className="mega-grid">
            <div className="mega-rail">
              {categories.map((c) => (
                <div key={c.id} className={"mega-rail-item" + (cur.id === c.id ? " is-active" : "")}
                     onMouseEnter={() => setActive(c)} onClick={() => { onClose(); go(`/category/${c.slug}`); }}>
                  <span style={{ width: 28, height: 28, borderRadius: 8, background: c.tint, color: c.fg, display: "grid", placeItems: "center" }}>
                    <Icon name={c.icon} size={17} />
                  </span>
                  <span style={{ flex: 1 }}>{c.name}</span>
                  <Icon name="chevron" size={15} className="ar" />
                </div>
              ))}
            </div>
            <div className="mega-panel">
              <div className="mega-panel-head">
                <h3>{cur.name}</h3>
                <a href={`/category/${cur.slug}`} onClick={(e) => { e.preventDefault(); onClose(); go(`/category/${cur.slug}`); }}>
                  Browse all {window.fmtInt(cur.product_count)} →
                </a>
              </div>
              <div className="mega-subs">
                {subs.map((s) => (
                  <a key={s.id} href={`/category/${s.slug}`} onClick={(e) => { e.preventDefault(); onClose(); go(`/category/${s.slug}`); }} className="mega-sub">
                    <span>{s.name}</span>
                    <span className="cnt">{window.fmtInt(s.product_count)}</span>
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </React.Fragment>
  );
}

/* ----------------------- Header ----------------------- */
function Header({ categories, tweaks, route }) {
  const [mega, setMega] = useStateC(false);
  const navCats = categories.slice(0, 8);
  return (
    <header className={"hdr" + (tweaks.darkHeader ? " is-dark" : "")}>
      <div className="wrap">
        <div className="hdr-top">
          <Link to="/" className="logo">
            <span className="logo-mark"><Icon name="link2" size={22} style={{ color: "#fff" }} /></span>
            <span style={{ display: "flex", flexDirection: "column", lineHeight: 1 }}>
              <span className="logo-name">Storefront</span>
              <span className="logo-sub">cross-sell demo</span>
            </span>
          </Link>

          <button className="catalog-btn" onClick={() => setMega((v) => !v)}>
            <span className="bars"><i></i><i></i><i></i></span>
            <span className="lbl">Catalog</span>
          </button>

          <SearchBox />

          <div className="hdr-actions">
            <Link to="/metrics" className={"hdr-link" + (route && route.name === "metrics" ? " is-active" : "")}>
              <Icon name="bars" size={18} /> Metrics
            </Link>
            <Link to="/summary" className={"hdr-link" + (route && route.name === "summary" ? " is-active" : "")}>
              <Icon name="doc" size={18} /> Summary
            </Link>
            <Link to="/compare" className={"hdr-link" + (route && route.name === "compare" ? " is-active" : "")}>
              <Icon name="percent" size={18} /> Compare
            </Link>
          </div>
        </div>
      </div>

      <div className="hdr-bottom">
        <div className="wrap">
          <nav className="nav">
            <a href="/" onClick={(e) => { e.preventDefault(); setMega(true); }} style={{ fontWeight: 600 }}>All categories</a>
            {navCats.map((c) => <Link key={c.id} to={`/category/${c.slug}`}>{c.name}</Link>)}
          </nav>
        </div>
      </div>

      {mega ? <MegaMenu categories={categories} onClose={() => setMega(false)} /> : null}
    </header>
  );
}

/* ----------------------- Breadcrumbs ----------------------- */
function Crumbs({ trail }) {
  return (
    <div className="wrap">
      <nav className="crumbs">
        {trail.map((c, i) => (
          <React.Fragment key={i}>
            {i > 0 ? <Icon name="chevron" size={13} className="sep" /> : null}
            {c.to ? <Link to={c.to}>{c.label}</Link> : <span style={{ color: "var(--ink-2)" }}>{c.label}</span>}
          </React.Fragment>
        ))}
      </nav>
    </div>
  );
}

/* ----------------------- Footer ----------------------- */
function Footer({ categories }) {
  return (
    <footer className="ftr">
      <div className="wrap">
        <div className="ftr-top">
          <div className="ftr-brand">
            <div className="logo">
              <span className="logo-mark"><Icon name="link2" size={20} style={{ color: "#fff" }} /></span>
              <span className="logo-name">Storefront</span>
            </div>
            <p className="ftr-desc">A research storefront for automatic cross-sell. It demonstrates how the engine picks technically compatible companion products for any item.</p>
            <div className="ftr-stack">React · FastAPI · Typesense · OpenAI (gpt-5-nano verify · gpt-5-mini judge)</div>
          </div>
          <div className="ftr-cols">
            <h5>Top categories</h5>
            <div className="ftr-cat-grid">
              {categories.slice(0, 10).map((c) => (
                <a key={c.id} href={`/category/${c.slug}`} onClick={(e) => { e.preventDefault(); go(`/category/${c.slug}`); }}>{c.name}</a>
              ))}
            </div>
          </div>
        </div>
        <div className="ftr-bottom">
          <span>Cross-sell research stand · catalog {window.fmtInt(window.MockData.metrics.catalog.products)} items / {window.fmtInt(window.MockData.metrics.catalog.categories)} categories</span>
          <span className="ftr-note"><span className="dot"></span> Live data · MySQL + Typesense · local recommendation engine</span>
        </div>
      </div>
    </footer>
  );
}

Object.assign(window, { SearchBox, MegaMenu, Header, Crumbs, Footer });
