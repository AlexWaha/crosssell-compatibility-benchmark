"""Build curated AVTC benchmark dataset (3000 products) with category-level ground truth.

Source: project/dataset/json/*  ->  dataset_v2/*
Produces a controlled cross-sell benchmark:
  - anchor products (devices that have upsell)
  - accessory products (genuinely compatible with anchors, by category graph)
  - noise/distractor products (must never be recommended -> test precision)
Ground truth pairs are derived from a hand-built anchor_cat -> accessory_cats graph.
"""

from __future__ import annotations

import collections
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "project" / "dataset"
SRC_JSON = SRC / "json"
SRC_IMAGES = SRC / "images"
OUT = ROOT
OUT_IMAGES = OUT / "images"

SEED = 42
TARGET_TOTAL = 3000
KEEP_FIELDS = ("id", "slug", "name", "description", "brand", "category_ids", "specs", "image_path", "price")

# anchor_cat_id -> [accessory_cat_id, ...]
# Built by hand from the catalog category tree. Each leaf category holds ~30 products.
GRAPH: dict[int, list[int]] = {
    # --- MOBILE ---
    623: [773, 497, 583, 1479, 106],          # Mobile Phones -> Cases, Chargers, Power banks, Headsets, Memory Cards
    246: [628, 497, 583, 1479, 106],          # Tablets -> Tablet Cases, Chargers, Power banks, Headsets, Memory Cards
    # --- PHOTO ---
    1730: [1620, 1251, 783, 140, 1727, 106, 650],  # Digital Cameras -> Lenses, Filters, Tripods, Bags, Batteries, MemCards, Flashes
    1390: [783, 140, 1727, 106],              # Camcorders -> Tripods, Bags, Batteries, Memory Cards
    373: [783, 140, 106],                     # Action Cameras -> Tripods, Bags, Memory Cards
    # --- COMPUTING ---
    152: [23, 49, 31, 29, 98, 104, 106],      # Laptops -> Bags, Mice, Keyboards, Mouse Pads, RAM, SSD, Memory Cards
    15: [49, 31, 29, 98, 104, 832, 830],      # Desktop PCs -> Mice, Keyboards, Pads, RAM, SSD, GPU, Motherboards
    161: [49, 31, 29],                        # Monitors -> Mice, Keyboards, Mouse Pads
    # --- CYCLING ---
    749: [942, 834, 150, 134, 944, 1261],     # Bikes -> Lights, Cycle Computers, Bags&Cages, Locks, Bike Tyres, Water Bottles
    # --- SELECTIVE AUTO / TOOLS ---
    258: [2293, 456, 3517],                   # Drills & Screwdrivers -> Bits/Sockets, Tool Kits, Tool boxes
    2176: [643],                              # Welders -> Welding Helmets
    882: [166, 976],                          # Electric Motorbikes -> Motorcycle Helmets, Motorcycle Tyres
}


def main() -> None:
    random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    products = json.loads((SRC_JSON / "products.json").read_text(encoding="utf-8"))
    categories = json.loads((SRC_JSON / "categories.json").read_text(encoding="utf-8"))
    attributes = json.loads((SRC_JSON / "attributes.json").read_text(encoding="utf-8"))

    cid2cat = {c["id"]: c for c in categories}
    bycat: dict[int, list[int]] = collections.defaultdict(list)
    pcats: dict[int, set[int]] = {}
    for p in products:
        cs = set(p["category_ids"])
        pcats[p["id"]] = cs
        for c in cs:
            bycat[c].append(p["id"])

    anchor_cats = set(GRAPH)
    accessory_cats: set[int] = set()
    for accs in GRAPH.values():
        accessory_cats.update(accs)
    core_cats = anchor_cats | accessory_cats

    # role assignment: anchor wins over accessory if a product sits in both kinds of cat
    role: dict[int, str] = {}
    for pid, cs in pcats.items():
        if cs & anchor_cats:
            role[pid] = "anchor"
        elif cs & accessory_cats:
            role[pid] = "accessory"
    core_ids = set(role)

    # noise: products with NO overlap with any core category
    noise_pool = [p["id"] for p in products if not (pcats[p["id"]] & core_cats)]
    random.shuffle(noise_pool)
    need_noise = TARGET_TOTAL - len(core_ids)
    if need_noise < 0:
        raise SystemExit(f"core ({len(core_ids)}) exceeds target {TARGET_TOTAL}")
    noise_ids = set(noise_pool[:need_noise])
    for pid in noise_ids:
        role[pid] = "noise"

    keep_ids = core_ids | noise_ids

    # --- expanded ground-truth pairs (anchor_id, accessory_id) ---
    # a pair is positive iff accessory product sits in an accessory cat mapped to one of the
    # anchor product's anchor cats. Many-to-many is fine (a memory card pairs with phone, camera, laptop).
    acc_cat_to_products: dict[int, list[int]] = {c: bycat[c] for c in accessory_cats}
    pairs: list[dict] = []
    pair_seen: set[tuple[int, int]] = set()
    for pid in core_ids:
        if role[pid] != "anchor":
            continue
        # union of accessory cats reachable from this product's anchor cats
        reachable: set[int] = set()
        for c in pcats[pid] & anchor_cats:
            reachable.update(GRAPH[c])
        for acc_cat in reachable:
            for acc_pid in acc_cat_to_products[acc_cat]:
                if acc_pid == pid:
                    continue
                key = (pid, acc_pid)
                if key in pair_seen:
                    continue
                pair_seen.add(key)
                pairs.append({"anchor_id": pid, "accessory_id": acc_pid, "via_category": acc_cat})

    # --- filtered products (stripped fields) ---
    out_products = []
    for p in products:
        if p["id"] not in keep_ids:
            continue
        out_products.append({k: p[k] for k in KEEP_FIELDS if k in p})
    out_products.sort(key=lambda x: x["id"])

    # --- filtered categories: used cats + all ancestors ---
    used_cats: set[int] = set()
    for p in out_products:
        used_cats.update(p["category_ids"])
    closure: set[int] = set()
    for c in used_cats:
        x = c
        seen: set[int] = set()
        while x and x in cid2cat and x not in seen:
            seen.add(x)
            closure.add(x)
            x = cid2cat[x]["parent_id"]
    out_categories = [cid2cat[c] for c in sorted(closure)]

    # --- roles file ---
    roles_out = {str(pid): role[pid] for pid in sorted(keep_ids)}

    # --- graph (named, for readability) ---
    graph_named = {
        str(a): {
            "anchor": cid2cat[a]["name"],
            "accessories": [{"id": ac, "name": cid2cat[ac]["name"]} for ac in accs],
        }
        for a, accs in GRAPH.items()
    }

    # --- write outputs ---
    (OUT / "products.json").write_text(
        json.dumps(out_products, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "categories.json").write_text(
        json.dumps(out_categories, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "attributes.json").write_text(
        json.dumps(attributes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "ground_truth_pairs.json").write_text(
        json.dumps(pairs, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "ground_truth_graph.json").write_text(
        json.dumps(graph_named, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "roles.json").write_text(
        json.dumps(roles_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- copy images ---
    copied = 0
    missing = []
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    for p in out_products:
        rel = p.get("image_path")
        if not rel:
            continue
        # image_path = catalog/product/<NNNN>/<FILE> ; source mirrors it under SRC_IMAGES/<NNNN>/<FILE>
        subpath = Path(rel).relative_to("catalog/product") if rel.startswith("catalog/product") else Path(rel)
        src = SRC_IMAGES / subpath
        if not src.exists():
            missing.append((p["id"], rel))
            continue
        dst = OUT_IMAGES / subpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1

    # --- counts ---
    role_counts = collections.Counter(role[pid] for pid in keep_ids)
    counts = {
        "total_products": len(out_products),
        "anchors": role_counts["anchor"],
        "accessories": role_counts["accessory"],
        "noise": role_counts["noise"],
        "anchor_categories": len(anchor_cats),
        "accessory_categories": len(accessory_cats),
        "core_categories": len(core_cats),
        "categories_kept": len(out_categories),
        "ground_truth_pairs": len(pairs),
        "images_copied": copied,
        "images_missing": len(missing),
    }
    (OUT / "counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    if missing[:5]:
        print("MISSING sample:", missing[:5])


if __name__ == "__main__":
    main()
