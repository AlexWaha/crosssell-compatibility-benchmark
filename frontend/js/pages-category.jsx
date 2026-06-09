/* =====================================================================
   Category — product grid with pagination + subcategory chips.
   (No facet filters yet — space is reserved for them per the spec.)
   ===================================================================== */
const { useState: useStateCat, useEffect: useEffectCat } = React;

function CategoryPage({ catId }) {
  const [data, setData] = useStateCat(null);
  const [loading, setLoading] = useStateCat(true);
  const [sort, setSort] = useStateCat("default");
  const [page, setPage] = useStateCat(1);
  const PER = 24;

  useEffectCat(() => { setPage(1); }, [catId, sort]);

  useEffectCat(() => {
    let on = true; setLoading(true);
    window.StoreAPI.listProducts({ category: catId, sort, page, perPage: PER }).then((d) => {
      if (on) { setData(d); setLoading(false); }
    });
    return () => { on = false; };
  }, [catId, sort, page]);

  const cat = data && data.category;
  const title = cat
    ? (data.activeSub ? ((cat.subs.find((s) => s.slug === data.activeSub) || {}).name || cat.name) : cat.name)
    : "Category";

  const trail = [{ label: "Home", to: "/" }];
  if (cat) {
    trail.push({ label: cat.name, to: data.activeSub ? `/category/${cat.slug}` : null });
    if (data.activeSub) trail.push({ label: title });
  }

  return (
    <React.Fragment>
      <Crumbs trail={trail} />
      <div className="wrap">
        <div className="cat-head">
          <h1>{title}</h1>
          {data ? <span className="cat-count">{window.fmtInt(data.total)} products</span> : null}
        </div>

        {cat ? (
          <div className="chips-row">
            <button className={"chip" + (!data.activeSub ? " is-active" : "")} onClick={() => go(`/category/${cat.slug}`)}>All</button>
            {cat.subs.map((s) => (
              <button key={s.id} className={"chip" + (data.activeSub === s.slug ? " is-active" : "")} onClick={() => go(`/category/${s.slug}`)}>{s.name}</button>
            ))}
          </div>
        ) : null}

        <div className="cat-toolbar">
          <span className="muted">{data && !loading ? `Page ${data.page} of ${data.pages}` : "Loading…"}</span>
          <div className="sort">
            <span className="muted" style={{ fontSize: 14 }}>Sort:</span>
            <select value={sort} onChange={(e) => setSort(e.target.value)}>
              <option value="default">Featured</option>
              <option value="cheap">Price: low to high</option>
              <option value="expensive">Price: high to low</option>
              <option value="name">Name A-Z</option>
            </select>
          </div>
        </div>

        {loading ? (
          <SkeletonGrid n={12} />
        ) : data.items.length === 0 ? (
          <div className="empty-state"><div className="em-ic"><Icon name="search" size={34} /></div><h2>No products here yet</h2></div>
        ) : (
          <React.Fragment>
            <ProductGrid products={data.items} />
            {data.pages > 1 ? <Pagination page={data.page} pages={data.pages} onGo={setPage} /> : null}
          </React.Fragment>
        )}
      </div>
    </React.Fragment>
  );
}

/* — Pagination control — */
function Pagination({ page, pages, onGo }) {
  const nums = [];
  const win = 2;
  for (let i = 1; i <= pages; i++) {
    if (i === 1 || i === pages || (i >= page - win && i <= page + win)) nums.push(i);
    else if (nums[nums.length - 1] !== "…") nums.push("…");
  }
  const jump = (p) => { onGo(p); window.scrollTo({ top: 0 }); };
  return (
    <div className="pager">
      <button className="pager-btn" disabled={page <= 1} onClick={() => jump(page - 1)}><Icon name="chevron" size={16} style={{ transform: "rotate(180deg)" }} /> Prev</button>
      <div className="pager-nums">
        {nums.map((n, i) => n === "…"
          ? <span key={"e" + i} className="pager-dots">…</span>
          : <button key={n} className={"pager-num" + (n === page ? " is-active" : "")} onClick={() => jump(n)}>{n}</button>)}
      </div>
      <button className="pager-btn" disabled={page >= pages} onClick={() => jump(page + 1)}>Next <Icon name="chevron" size={16} /></button>
    </div>
  );
}

Object.assign(window, { CategoryPage, Pagination });
