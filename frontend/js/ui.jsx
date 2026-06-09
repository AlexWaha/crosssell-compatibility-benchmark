/* =====================================================================
   UI-примитивы: иконки, цена, рейтинг, плейсхолдер фото, навигация.
   Экспортируются в window для использования в других babel-файлах.
   ===================================================================== */
const { useState, useEffect, useRef, useCallback, useMemo } = React;

/* — Простые линейные иконки (UI-аффордансы) — */
const ICONS = {
  search: "M11 4a7 7 0 105.2 11.7l4 4M11 4a7 7 0 014.2 12.7",
  cart: "M3 4h2l2.4 11.2a1 1 0 001 .8h8.2a1 1 0 001-.8L20 8H6.5M9 20a1 1 0 100-2 1 1 0 000 2zM17 20a1 1 0 100-2 1 1 0 000 2z",
  heart: "M12 20s-7-4.4-9.2-8.5C1.3 8.6 2.6 5 6 5c2 0 3.2 1.3 4 2.5C10.8 6.3 12 5 14 5c3.4 0 4.7 3.6 3.2 6.5C19 15.6 12 20 12 20z",
  user: "M12 12a4 4 0 100-8 4 4 0 000 8zM4 21a8 8 0 0116 0",
  chevron: "M9 6l6 6-6 6",
  chevronDown: "M6 9l6 6 6-6",
  arrow: "M5 12h14M13 6l6 6-6 6",
  arrowLeft: "M19 12H5M11 18l-6-6 6-6",
  check: "M20 6L9 17l-5-5",
  plus: "M12 5v14M5 12h14",
  minus: "M5 12h14",
  x: "M6 6l12 12M18 6L6 18",
  star: "M12 3l2.6 5.5 6 .8-4.4 4.2 1.1 6L12 16.8 6.7 19.5l1.1-6L3.4 9.3l6-.8z",
  truck: "M3 7h11v8H3zM14 10h4l3 3v2h-7M7 19a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM18 19a1.5 1.5 0 100-3 1.5 1.5 0 000 3z",
  shield: "M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z",
  shieldCheck: "M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6zM9 11.5l2 2 4-4",
  ret: "M3 9l3-3 3 3M6 6v7a4 4 0 004 4h8",
  percent: "M5 19L19 5M7.5 9a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM16.5 18a1.5 1.5 0 100-3 1.5 1.5 0 000 3z",
  bolt: "M13 3L4 14h6l-1 7 9-11h-6z",
  grid: "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
  rows: "M4 5h16M4 12h16M4 19h16",
  filter: "M3 5h18l-7 8v5l-4 2v-7z",
  pin: "M12 21s7-5.6 7-11a7 7 0 10-14 0c0 5.4 7 11 7 11zM12 12a2 2 0 100-4 2 2 0 000 4z",
  phone: "M5 4h4l2 5-3 2a12 12 0 005 5l2-3 5 2v4a2 2 0 01-2 2A16 16 0 013 6a2 2 0 012-2z",
  // категории
  tv: "M3 5h18v11H3zM8 20h8M12 16v4",
  home: "M4 11l8-7 8 7M6 10v9h12v-9",
  spark: "M12 3l1.8 5.4L19 10l-5.2 1.6L12 17l-1.8-5.4L5 10l5.2-1.6z",
  dumbbell: "M4 8v8M7 6v12M17 6v12M20 8v8M7 12h10",
  kite: "M12 3l7 7-7 11-7-11zM5 10h14M12 3v18",
  shirt: "M8 4l4 3 4-3 4 3-2 3-2-1v11H8V9L6 10 4 7z",
  car: "M5 13l1.5-5h11L19 13M4 17h16v-4H4zM7 20a1.5 1.5 0 100-3 1.5 1.5 0 000 3zM17 20a1.5 1.5 0 100-3 1.5 1.5 0 000 3z",
  plug: "M9 3v5M15 3v5M7 8h10v3a5 5 0 01-10 0zM12 16v5",
  menu: "M4 7h16M4 12h16M4 17h16",
  laptop: "M4 6h16v10H4zM2 19h20M9 19h6",
  doc: "M7 3h7l4 4v14H7zM14 3v4h4",
  camera: "M3 8h3l2-2h8l2 2h3v11H3zM12 16.5a3.2 3.2 0 100-6.4 3.2 3.2 0 000 6.4z",
  speaker: "M7 3h10v18H7zM12 16a3 3 0 100-6 3 3 0 000 6zM12 7h.01",
  fridge: "M6 3h12v18H6zM6 10h12M9 6v2M9 13v3",
  mug: "M4 8h12v7a4 4 0 01-4 4H8a4 4 0 01-4-4zM16 10h2.5a2 2 0 010 4H16",
  fan: "M12 2v20M2 12h20M5 5l14 14M19 5L5 19",
  wrench: "M15 6a4 4 0 01-5.2 5.2L5 16l3 3 4.8-4.8A4 4 0 0118 9l-2.3 2.3-1.5-1.5z",
  tent: "M12 4L3 20h18zM12 4v16M12 20l-4-7M12 20l4-7",
  watch: "M8 7V4h8v3M8 17v3h8v-3M7 12a5 5 0 1110 0 5 5 0 01-10 0M12 10v2l1.5 1",
  paw: "M6 13a1.8 1.8 0 100-3.6 1.8 1.8 0 000 3.6zM18 13a1.8 1.8 0 100-3.6 1.8 1.8 0 000 3.6zM9.5 8a1.8 1.8 0 100-3.6 1.8 1.8 0 000 3.6zM14.5 8a1.8 1.8 0 100-3.6 1.8 1.8 0 000 3.6zM12 13c-2.8 0-4.5 1.8-4.5 3.6A1.9 1.9 0 009.4 18.5h5.2a1.9 1.9 0 001.9-1.9C16.5 14.8 14.8 13 12 13z",
  battery: "M4 8h14v8H4zM18 11h2v2h-2M7 10.5v3M10 10.5v3",
  link2: "M9.5 14.5l5-5M10.5 6.5l1-1a4 4 0 015.7 5.7l-1 1M13.5 17.5l-1 1a4 4 0 01-5.7-5.7l1-1",
};

/* — Connection-type (context_code) badge meta for recommendations — */
const CONTEXT = {
  cable:     { label: "Cable",     icon: "link2",  color: "#2f6bd6", bg: "#e8eefc" },
  mount:     { label: "Mount",     icon: "wrench", color: "#7257d6", bg: "#f0eefb" },
  charger:   { label: "Charger",   icon: "bolt",   color: "#d07a1f", bg: "#fff1e3" },
  adapter:   { label: "Adapter",   icon: "plug",   color: "#1f9d57", bg: "#e7f6ee" },
  case:      { label: "Case",      icon: "shield", color: "#5a6472", bg: "#eef0f3" },
  battery:   { label: "Battery",   icon: "battery",color: "#b9772b", bg: "#fbf0e8" },
  accessory: { label: "Accessory", icon: "spark",  color: "#c43f86", bg: "#fdeaf3" },
};
function ContextBadge({ code, size = "md" }) {
  const c = CONTEXT[code] || { label: code, icon: "spark", color: "#5a6472", bg: "#eef0f3" };
  return (
    <span className={"ctx-badge" + (size === "sm" ? " sm" : "")} style={{ color: c.color, background: c.bg }}>
      <Icon name={c.icon} size={size === "sm" ? 12 : 14} /> {c.label}
    </span>
  );
}

/* — Compatibility tag chip (interfaces / standards / form-factors) — */
function CompatTag({ children }) {
  return <span className="compat-tag">{children}</span>;
}

/* — Map the prototype's icon vocabulary to Lucide icon names (window.lucide). — */
const LUCIDE_MAP = {
  search: "Search", cart: "ShoppingCart", heart: "Heart", user: "User",
  chevron: "ChevronRight", chevronDown: "ChevronDown", arrow: "ArrowRight",
  arrowLeft: "ArrowLeft", check: "Check", plus: "Plus", minus: "Minus", x: "X",
  star: "Star", truck: "Truck", shield: "Shield", shieldCheck: "ShieldCheck",
  ret: "RotateCcw", percent: "Percent", bolt: "Zap", grid: "LayoutGrid",
  rows: "Rows3", filter: "Filter", pin: "MapPin", phone: "Smartphone", tv: "Tv",
  home: "House", spark: "Sparkles", dumbbell: "Dumbbell", kite: "Blocks",
  shirt: "Shirt", car: "Car", plug: "Plug", menu: "Menu", laptop: "Laptop",
  doc: "FileText", camera: "Camera", speaker: "Speaker", fridge: "Refrigerator",
  mug: "Coffee", fan: "Fan", wrench: "Wrench", tent: "Tent", watch: "Watch",
  paw: "PawPrint", battery: "BatteryCharging", link2: "Link2", bars: "BarChart3",
};

function Icon({ name, size = 20, stroke = 2, className = "", style }) {
  const L = window.lucide || {};
  const lname = LUCIDE_MAP[name] || name;
  const node = L[lname] || L.Box || [];
  const children = Array.isArray(node) ? node : [];
  return (
    <svg className={"svg-ico " + className} style={style} width={size} height={size}
         viewBox="0 0 24 24" fill="none" stroke="currentColor"
         strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {children.map((c, i) => React.createElement(c[0], { ...c[1], key: i }))}
    </svg>
  );
}

/* — Навигация по hash-роутеру — */
function go(path) {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
  window.scrollTo({ top: 0 });
}
function Link({ to, children, className, ...rest }) {
  return (
    <a href={to} className={className}
       onClick={(e) => { e.preventDefault(); go(to); }} {...rest}>
      {children}
    </a>
  );
}

/* — Price — */
function Price({ price, big }) {
  return (
    <div className="pcard-price">
      <span className="price-now" style={big ? { fontSize: 30 } : null}>{window.fmtPrice(price)}</span>
    </div>
  );
}

/* — Детерминированный плейсхолдер фото товара —
   Имитирует место, куда бэкенд подставит реальное изображение.
   Тинт берётся из категории, иконка — из категории, подпись — мелким моно. */
function ProductImage({ product, caption = true, iconSize = 54 }) {
  const cats = (window.MockData && window.MockData.categories) || [];
  const cat = cats.find((c) => c.id === product.cat) || {};
  const bg = cat.tint || "#eef1f6";
  const fg = cat.fg || "#9aa3b2";
  const [err, setErr] = useState(false);
  const hasImg = product.image && !err;
  const style = hasImg
    ? { "--ph-bg": bg, "--ph-fg": fg, background: "#fff" }
    : { "--ph-bg": bg, "--ph-fg": fg };
  return (
    <div className="ph" style={style}>
      {hasImg ? (
        <img className="ph-img" src={product.image} alt={product.name || ""} loading="lazy"
             onError={() => setErr(true)}
             style={{ width: "100%", height: "100%", objectFit: "contain", padding: "8px", boxSizing: "border-box" }} />
      ) : (
        <div className="ph-inner">
          <Icon name={cat.icon || "grid"} size={iconSize} stroke={1.4} style={{ opacity: .9 }} />
          {caption ? <span className="ph-cap">{(product.product_type || "product").toLowerCase()}</span> : null}
        </div>
      )}
    </div>
  );
}

/* — Quantity stepper — */
function QtyStepper({ value, onChange, size = "md" }) {
  const cls = size === "mini" ? "mini-stepper" : "qty-stepper";
  return (
    <div className={cls} onClick={(e) => e.stopPropagation()}>
      <button onClick={() => onChange(Math.max(1, value - 1))} aria-label="Меньше">
        <Icon name="minus" size={size === "mini" ? 14 : 18} />
      </button>
      <span>{value}</span>
      <button onClick={() => onChange(value + 1)} aria-label="Больше">
        <Icon name="plus" size={size === "mini" ? 14 : 18} />
      </button>
    </div>
  );
}

/* — Скелетон карточки (загрузка) — */
function SkeletonCard() {
  return (
    <div className="skel-card">
      <div className="skel media"></div>
      <div className="skel line" style={{ width: "55%" }}></div>
      <div className="skel line" style={{ width: "90%" }}></div>
      <div className="skel line" style={{ width: "70%" }}></div>
      <div className="skel line" style={{ width: "100%", height: 40, marginTop: 12 }}></div>
    </div>
  );
}
function SkeletonGrid({ n = 5, cols }) {
  return (
    <div className={"grid-products" + (cols ? " cols-" + cols : "")}>
      {Array.from({ length: n }).map((_, i) => <SkeletonCard key={i} />)}
    </div>
  );
}

Object.assign(window, {
  Icon, Link, go, Price, ProductImage, SkeletonCard, SkeletonGrid, ContextBadge, CompatTag, CONTEXT,
});
