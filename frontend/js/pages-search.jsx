/* =====================================================================
   Search — query from URL, results grid (hybrid semantic on backend).
   No facet filters at launch (space reserved for them later).
   ===================================================================== */
const { useState: useStateS, useEffect: useEffectS } = React;

function SearchPage({ query }) {
  const [data, setData] = useStateS(null);
  const [loading, setLoading] = useStateS(true);
  const [sort, setSort] = useStateS("relevance");

  useEffectS(() => {
    let on = true; setLoading(true);
    window.StoreAPI.search(query, { sort }).then((d) => { if (on) { setData(d); setLoading(false); } });
    return () => { on = false; };
  }, [query, sort]);

  return (
    <React.Fragment>
      <Crumbs trail={[{ label: "Home", to: "/" }, { label: "Search" }]} />
      <div className="wrap">
        <div className="search-summary">
          <h1>Results for “<b>{query}</b>”</h1>
          <div className="muted">{loading ? "Searching…" : `${window.fmtInt(data.total)} products found`}</div>
        </div>

        {!loading && data.total > 0 ? (
          <div className="cat-toolbar">
            <span className="muted" style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              <Icon name="bolt" size={15} style={{ color: "var(--accent)" }} /> hybrid semantic search
            </span>
            <div className="sort">
              <span className="muted" style={{ fontSize: 14 }}>Sort:</span>
              <select value={sort} onChange={(e) => setSort(e.target.value)}>
                <option value="relevance">Relevance</option>
                <option value="cheap">Price: low to high</option>
                <option value="expensive">Price: high to low</option>
                <option value="name">Name A-Z</option>
              </select>
            </div>
          </div>
        ) : null}

        {loading ? (
          <div style={{ marginTop: 18 }}><SkeletonGrid n={10} /></div>
        ) : data.total === 0 ? (
          <div className="empty-state">
            <div className="em-ic"><Icon name="search" size={34} /></div>
            <h2>Nothing found</h2>
            <p>No products match “{query}”. Try a shorter or different term.</p>
            <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap" }}>
              {window.MockData.popularQueries.slice(0, 4).map((q) => (
                <button key={q} className="chip" onClick={() => go(`/search?q=${encodeURIComponent(q)}`)}>{q}</button>
              ))}
            </div>
          </div>
        ) : (
          <div style={{ marginTop: 8 }}><ProductGrid products={data.items} /></div>
        )}
      </div>
    </React.Fragment>
  );
}

Object.assign(window, { SearchPage });
