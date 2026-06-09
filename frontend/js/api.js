/* =====================================================================
   StoreAPI — real backend data-access layer (no mock).
   Everything on the pages comes from the backend REST API (MySQL/OpenCart
   schema + avtc_metrics) and Typesense (search). Same-origin: the SPA is
   served by the FastAPI app, so API base is "". URLs are slug-based (ЧПУ).
   ===================================================================== */
(function () {
  const API = "";

  const ICON_BY_NAME = {
    "Mobile Phones & Gadgets": "phone", "Computing": "laptop", "Office & Stationery": "doc",
    "Photo & Video": "camera", "TVs & Video": "tv", "Sound & Hi-Fi": "speaker",
    "Appliances": "fridge", "Small Appliances": "mug", "Climate, Heating & Water Supply": "fan",
    "Kids & Toys": "kite", "Automotive": "car", "Tools": "wrench", "Camping & Fishing": "tent",
    "Sports & Outdoor": "dumbbell", "Home & Renovation": "home",
    "Watches, Jewelry & Accessories": "watch", "Health & Beauty": "spark",
    "Pet Supplies": "paw", "Security & Surveillance": "shield", "Energy & Power": "bolt",
  };
  const TINTS = [
    ["#e8eefc", "#3b6fd6"], ["#e7f6ee", "#1f9d57"], ["#fdeaf3", "#d4458a"],
    ["#fff1e3", "#e07b2a"], ["#f0eefb", "#7257d6"], ["#eaf3fd", "#3b8fd6"],
    ["#eef0f3", "#5a6472"], ["#fbf0e8", "#c2752f"],
  ];

  let TREE = null;          // enriched top-level categories
  let BY_SLUG = {};         // slug -> node (top or sub) for lookups
  const norm = (s) => (s || "").toLowerCase().trim();
  const imgUrl = (p) => (p ? API + "/api/images/" + p : "");

  // Experiment variants (must match experiment_id values written by the pipeline).
  window.EXPERIMENTS = [
    { id: "cat_priors_v1", label: "cat_priors · oneshot verify", short: "cat_priors" },
    { id: "cat_priors_wide_v1", label: "cat_priors · wide retrieval", short: "wide" },
    { id: "semantic_v1", label: "semantic baseline", short: "semantic" },
  ];
  window.DEFAULT_EXPERIMENT = "cat_priors_v1";

  window.MockData = {
    categories: [],
    metrics: { catalog: { products: 0, categories: 0, with_reco: 0, brands: 0 } },
    popularQueries: [
      "hdmi cable", "nvme ssd", "wall mount", "power bank",
      "graphics card", "usb-c charger", "card reader", "mechanical keyboard",
    ],
  };

  async function getJSON(path, opts) {
    const r = await fetch(API + path, opts);
    if (!r.ok) throw new Error(path + " -> " + r.status);
    return r.json();
  }

  function buildTree(flat) {
    const tops = flat.filter((c) => !c.parent_id);
    const childrenOf = (id) => flat.filter((c) => c.parent_id === id);
    TREE = tops.map((t, i) => {
      const [bg, fg] = TINTS[i % TINTS.length];
      const subs = childrenOf(t.id).map((s) => ({
        id: s.id, slug: s.slug, name: s.name, parent_id: t.id, product_count: s.product_count,
      }));
      const count = t.product_count + subs.reduce((a, s) => a + s.product_count, 0);
      return {
        id: t.id, slug: t.slug, name: t.name, parent_id: null,
        icon: ICON_BY_NAME[t.name] || "grid", tint: bg, fg, product_count: count, subs,
      };
    });
    BY_SLUG = {};
    for (const t of TREE) {
      BY_SLUG[t.slug] = t;
      for (const s of t.subs) {
        BY_SLUG[s.slug] = Object.assign({}, s, { tint: t.tint, fg: t.fg, icon: t.icon, top: t });
      }
    }
    window.MockData = window.MockData || {};
    window.MockData.categories = TREE;
    return TREE;
  }

  const SORTERS = {
    cheap: (a, b) => a.price - b.price,
    expensive: (a, b) => b.price - a.price,
    name: (a, b) => (a.name || "").localeCompare(b.name || ""),
  };

  const StoreAPI = {
    async getCategories() {
      const [cats, metrics] = await Promise.all([
        getJSON("/api/categories"),
        getJSON("/api/metrics").catch(() => null),
      ]);
      const tree = buildTree(cats.items || []);
      window.MockData = window.MockData || {};
      window.MockData.metrics = metrics || { catalog: { products: 0, categories: tree.length } };
      return tree;
    },

    async getProduct(slugOrId) {
      let p;
      try {
        p = await getJSON("/api/products/" + encodeURIComponent(slugOrId));
      } catch {
        return null;
      }
      const topId = (p.category_path && p.category_path[0] && p.category_path[0].id) || null;
      // category_path kept as {name, slug} objects for slug-based breadcrumb links
      return Object.assign({}, p, { image: imgUrl(p.image), cat: topId });
    },

    async listProducts(params = {}) {
      const { category, sort = "default", page = 1, perPage = 24 } = params;
      const node = BY_SLUG[category];
      const isSub = node && node.parent_id;
      const top = isSub ? node.top : node;
      const data = await getJSON(
        `/api/categories/${encodeURIComponent(category)}/products?page=${page}&page_size=${perPage}`
      );
      let items = (data.items || []).map((c) =>
        Object.assign({}, c, { image: imgUrl(c.image), cat: top ? top.id : null })
      );
      if (SORTERS[sort]) items = items.slice().sort(SORTERS[sort]);
      const pages = Math.max(1, Math.ceil((data.total || 0) / perPage));
      return {
        items, total: data.total || 0, page: data.page || page, pages, perPage,
        category: top || null, activeSub: isSub ? category : null,
      };
    },

    async suggest(query) {
      if (!norm(query)) return { items: [], total: 0, query };
      const d = await getJSON("/api/search", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 10 }),
      });
      const items = (d.items || []).map((c) => Object.assign({}, c, { image: imgUrl(c.image) }));
      return { items, total: items.length, query };
    },

    async search(query, params = {}) {
      const { sort = "relevance" } = params;
      const d = await getJSON("/api/search", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, limit: 48 }),
      });
      let items = (d.items || []).map((c) => Object.assign({}, c, { image: imgUrl(c.image) }));
      if (sort !== "relevance" && SORTERS[sort]) items = items.slice().sort(SORTERS[sort]);
      return { items, total: items.length, query };
    },

    async getRecommendations(productId, experiment = window.DEFAULT_EXPERIMENT) {
      const d = await getJSON(
        `/api/products/${encodeURIComponent(productId)}/recommendations?limit=24&experiment=${encodeURIComponent(experiment)}`
      );
      const items = (d.items || []).map((it) => Object.assign({}, it, { image: imgUrl(it.image) }));
      return { slot: "cross_sell", productId, items };
    },

    async getMetrics(experiment = window.DEFAULT_EXPERIMENT) {
      return getJSON(`/api/metrics?experiment=${encodeURIComponent(experiment)}`);
    },

    async getTopProducts(limit = 20, experiment = window.DEFAULT_EXPERIMENT) {
      const d = await getJSON(`/api/top-products?limit=${limit}&experiment=${encodeURIComponent(experiment)}`);
      return (d.items || []).map((it) => Object.assign({}, it, { image: imgUrl(it.image) }));
    },

    async getSummary(experiment = window.DEFAULT_EXPERIMENT) {
      return getJSON(`/api/summary?experiment=${encodeURIComponent(experiment)}`);
    },

    async getCompare(experimentIds) {
      const ids = (experimentIds || window.EXPERIMENTS.map((e) => e.id)).join(",");
      return getJSON(`/api/compare?experiments=${encodeURIComponent(ids)}`);
    },
  };

  window.fmtPrice = (n) =>
    n == null ? "—" : "$" + new Intl.NumberFormat("en-US").format(Math.round(n * 100) / 100);
  window.fmtInt = (n) => new Intl.NumberFormat("en-US").format(n || 0);
  window.StoreAPI = StoreAPI;
})();
