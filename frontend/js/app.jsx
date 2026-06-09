/* =====================================================================
   App — hash router, theme tweaks, mount. No cart/account state.
   ===================================================================== */
const { useState: useStateA, useEffect: useEffectA } = React;

function parseRoute() {
  const seg = window.location.pathname.split("/").filter(Boolean);
  const params = new URLSearchParams(window.location.search);
  if (!seg.length) return { name: "home" };
  if (seg[0] === "category" && seg[1]) return { name: "category", slug: decodeURIComponent(seg[1]) };
  if (seg[0] === "product" && seg[1]) return { name: "product", slug: decodeURIComponent(seg[1]) };
  if (seg[0] === "search") return { name: "search", q: params.get("q") || "" };
  if (seg[0] === "metrics") return { name: "metrics" };
  if (seg[0] === "summary") return { name: "summary" };
  if (seg[0] === "compare") return { name: "compare" };
  return { name: "home" };
}

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "#15171c",
  "font": "Inter",
  "radius": "sharp",
  "density": "compact",
  "darkHeader": false
}/*EDITMODE-END*/;

const FONT_STACKS = {
  "Inter": '"Inter", -apple-system, "Segoe UI", Helvetica, Arial, sans-serif',
  "Manrope": '"Manrope", "Inter", -apple-system, sans-serif',
  "Helvetica / system": '-apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif',
};
const RADIUS_SCALE = { sharp: 0.4, regular: 1, rounded: 1.45 };
const DENSITY_GAP = { compact: 12, regular: 16, roomy: 22 };

function applyTweaks(t) {
  const r = document.documentElement.style;
  r.setProperty("--accent", t.accent);
  r.setProperty("--accent-press", `color-mix(in srgb, ${t.accent} 80%, #000)`);
  r.setProperty("--accent-soft", `color-mix(in srgb, ${t.accent} 12%, #fff)`);
  r.setProperty("--font", FONT_STACKS[t.font] || FONT_STACKS.Inter);
  const k = RADIUS_SCALE[t.radius] != null ? RADIUS_SCALE[t.radius] : 1;
  r.setProperty("--r-sm", (8 * k) + "px");
  r.setProperty("--r-md", (12 * k) + "px");
  r.setProperty("--r-lg", (16 * k) + "px");
  r.setProperty("--r-xl", (22 * k) + "px");
  r.setProperty("--gap", (DENSITY_GAP[t.density] || 16) + "px");
}

function App() {
  const [route, setRoute] = useStateA(parseRoute());
  const [categories, setCategories] = useStateA([]);
  const t = TWEAK_DEFAULTS;

  useEffectA(() => { applyTweaks(t); }, [t]);
  useEffectA(() => {
    const onNav = () => setRoute(parseRoute());
    window.addEventListener("popstate", onNav);
    return () => window.removeEventListener("popstate", onNav);
  }, []);
  useEffectA(() => { window.StoreAPI.getCategories().then(setCategories); }, []);

  let page = null;
  if (route.name === "metrics") page = <MetricsPage />;
  else if (route.name === "summary") page = <SummaryPage />;
  else if (route.name === "compare") page = <ComparePage />;
  else if (categories.length) {
    if (route.name === "home") page = <HomePage categories={categories} />;
    else if (route.name === "category") page = <CategoryPage catId={route.slug} />;
    else if (route.name === "product") page = <ProductPage productId={route.slug} />;
    else if (route.name === "search") page = <SearchPage query={route.q} />;
  }

  return (
    <div className="app-root">
      <Header categories={categories} tweaks={{ darkHeader: t.darkHeader }} route={route} />
      <main className="app-main">{page}</main>
      <Footer categories={categories} />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
