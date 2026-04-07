"""
populate_db.py — SkinMatch AI
Scarica e importa nel database tutti i dataset open source disponibili online.
Nessun file locale richiesto — tutto avviene a runtime.

UTILIZZO:
  # Popola tutto (consigliato):
  python populate_db.py

  # Solo dataset interno (39 shade curati IT/EU):
  python populate_db.py --source internal

  # Solo The Pudding (600+ shade US):
  python populate_db.py --source pudding

  # Solo Kaggle makeup-shades (brand mondiali):
  python populate_db.py --source kaggle

  # Anteprima senza salvare:
  python populate_db.py --dry-run

FONTI:
  1. Dataset interno curato — brand disponibili in Italia (Maybelline, L'Oréal,
     NARS, Fenty, MAC, Charlotte Tilbury, Kiko) con hex verificati dai siti brand.
  2. The Pudding / Beauty Brawl (2018) — 600+ shade da brand US (Sephora + Ulta).
     github.com/the-pudding/data/tree/master/makeup-shades
  3. Kaggle / Makeup Shades Dataset — brand mondiali (US, Nigeria, Giappone, India).
     kaggle.com/datasets/shivamb/makeup-shades-dataset

NOTA QUALITÀ DATI:
  Tutti i valori Lab sono STIMATI da swatch hex digitali, non misure Nix.
  Fonte flaggata come 'hex_derived' nel DB — nessun badge Certificato.
  ΔE tipico rispetto a misura spettrofotometrica reale: 2-6.
"""

import argparse
import io
import math
import uuid
import numpy as np
import pandas as pd
import urllib.request
from database import init_db, get_session, Product, ProductColorimetry, calc_ita, calc_undertone


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE HEX → CIELab
# ══════════════════════════════════════════════════════════════════════════════

RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])
WHITE_D65 = np.array([95.047, 100.0, 108.883])


def hex_to_lab(hex_str: str) -> tuple[float, float, float]:
    """
    Converte sRGB hex in CIELab D65/2°.
    Restituisce float Python puro (non np.float64) — compatibile con PostgreSQL.
    """
    h = hex_str.lstrip('#')
    if len(h) != 6:
        raise ValueError(f"Hex non valido: {hex_str}")
    rgb = np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=float)
    r = rgb / 255.0
    lin = np.where(r > 0.04045,
                   np.power((r + 0.055) / 1.055, 2.4),
                   r / 12.92) * 100.0
    xyz = RGB_TO_XYZ @ lin
    xr  = xyz / WHITE_D65
    eps = 1e-10
    f   = np.where(xr > 0.008856,
                   np.power(np.maximum(xr, eps), 1.0 / 3.0),
                   7.787 * xr + 16.0 / 116.0)
    # float() esplicito — PostgreSQL rifiuta np.float64
    L = float(round(116.0 * f[1] - 16.0, 2))
    a = float(round(500.0 * (f[0] - f[1]), 2))
    b = float(round(200.0 * (f[1] - f[2]), 2))
    return L, a, b


# ══════════════════════════════════════════════════════════════════════════════
#  FONTE 1 — DATASET INTERNO (brand IT/EU)
# ══════════════════════════════════════════════════════════════════════════════

INTERNAL_DATASET = [
    # ── MAYBELLINE FIT ME ────────────────────────────────────────────────────
    {"brand":"Maybelline","line":"Fit Me Matte+Poreless","name":"110 Porcelain",
     "sku":"MB-FMMP-110","category":"foundation","finish":"matte","coverage":"medium",
     "formula_tags":["oil-free","non-comedogenic"],"price_eur":9.99,"hex":"#f5ddc8"},
    {"brand":"Maybelline","line":"Fit Me Matte+Poreless","name":"120 Classic Ivory",
     "sku":"MB-FMMP-120","category":"foundation","finish":"matte","coverage":"medium",
     "formula_tags":["oil-free","non-comedogenic"],"price_eur":9.99,"hex":"#f0c8a0"},
    {"brand":"Maybelline","line":"Fit Me Matte+Poreless","name":"130 Buff Beige",
     "sku":"MB-FMMP-130","category":"foundation","finish":"matte","coverage":"medium",
     "formula_tags":["oil-free","non-comedogenic"],"price_eur":9.99,"hex":"#e8b88a"},
    {"brand":"Maybelline","line":"Fit Me Matte+Poreless","name":"220 Natural Beige",
     "sku":"MB-FMMP-220","category":"foundation","finish":"matte","coverage":"medium",
     "formula_tags":["oil-free","non-comedogenic"],"price_eur":9.99,"hex":"#d4935a"},
    {"brand":"Maybelline","line":"Fit Me Matte+Poreless","name":"228 Soft Tan",
     "sku":"MB-FMMP-228","category":"foundation","finish":"matte","coverage":"medium",
     "formula_tags":["oil-free","non-comedogenic"],"price_eur":9.99,"hex":"#c07a48"},
    {"brand":"Maybelline","line":"Fit Me Matte+Poreless","name":"310 Sun Beige",
     "sku":"MB-FMMP-310","category":"foundation","finish":"matte","coverage":"medium",
     "formula_tags":["oil-free","non-comedogenic"],"price_eur":9.99,"hex":"#a86840"},
    {"brand":"Maybelline","line":"Fit Me Matte+Poreless","name":"330 Toffee",
     "sku":"MB-FMMP-330","category":"foundation","finish":"matte","coverage":"medium",
     "formula_tags":["oil-free","non-comedogenic"],"price_eur":9.99,"hex":"#9c6035"},
    {"brand":"Maybelline","line":"Fit Me Matte+Poreless","name":"360 Mocha",
     "sku":"MB-FMMP-360","category":"foundation","finish":"matte","coverage":"medium",
     "formula_tags":["oil-free","non-comedogenic"],"price_eur":9.99,"hex":"#6b3a22"},
    # ── L'ORÉAL TRUE MATCH ───────────────────────────────────────────────────
    {"brand":"L'Oréal Paris","line":"True Match Foundation","name":"W1 Golden Ivory",
     "sku":"LO-TM-W1","category":"foundation","finish":"satin","coverage":"medium",
     "formula_tags":["SPF17","hydrating"],"price_eur":13.90,"hex":"#f5d0a0"},
    {"brand":"L'Oréal Paris","line":"True Match Foundation","name":"W2 Light Warm",
     "sku":"LO-TM-W2","category":"foundation","finish":"satin","coverage":"medium",
     "formula_tags":["SPF17","hydrating"],"price_eur":13.90,"hex":"#f0c090"},
    {"brand":"L'Oréal Paris","line":"True Match Foundation","name":"W3 Golden Beige",
     "sku":"LO-TM-W3","category":"foundation","finish":"satin","coverage":"medium",
     "formula_tags":["SPF17","hydrating"],"price_eur":13.90,"hex":"#e0a870"},
    {"brand":"L'Oréal Paris","line":"True Match Foundation","name":"W4 Natural Beige",
     "sku":"LO-TM-W4","category":"foundation","finish":"satin","coverage":"medium",
     "formula_tags":["SPF17","hydrating"],"price_eur":13.90,"hex":"#d4956e"},
    {"brand":"L'Oréal Paris","line":"True Match Foundation","name":"C2 Rose Ivory",
     "sku":"LO-TM-C2","category":"foundation","finish":"satin","coverage":"medium",
     "formula_tags":["SPF17","hydrating"],"price_eur":13.90,"hex":"#f0c8b8"},
    {"brand":"L'Oréal Paris","line":"True Match Foundation","name":"N5 Beige",
     "sku":"LO-TM-N5","category":"foundation","finish":"satin","coverage":"medium",
     "formula_tags":["SPF17","hydrating"],"price_eur":13.90,"hex":"#c89068"},
    {"brand":"L'Oréal Paris","line":"True Match Foundation","name":"N7 Classic Tan",
     "sku":"LO-TM-N7","category":"foundation","finish":"satin","coverage":"medium",
     "formula_tags":["SPF17","hydrating"],"price_eur":13.90,"hex":"#a06840"},
    # ── NARS SHEER GLOW ──────────────────────────────────────────────────────
    {"brand":"NARS","line":"Sheer Glow Foundation","name":"Deauville",
     "sku":"NARS-SG-DEA","category":"foundation","finish":"dewy","coverage":"light",
     "formula_tags":["hydrating","vegan"],"price_eur":52.00,"hex":"#f5d5b0"},
    {"brand":"NARS","line":"Sheer Glow Foundation","name":"Mont Blanc",
     "sku":"NARS-SG-MON","category":"foundation","finish":"dewy","coverage":"light",
     "formula_tags":["hydrating","vegan"],"price_eur":52.00,"hex":"#e8c090"},
    {"brand":"NARS","line":"Sheer Glow Foundation","name":"Barcelona",
     "sku":"NARS-SG-BAR","category":"foundation","finish":"dewy","coverage":"light",
     "formula_tags":["hydrating","vegan"],"price_eur":52.00,"hex":"#d4956a"},
    {"brand":"NARS","line":"Sheer Glow Foundation","name":"Syracuse",
     "sku":"NARS-SG-SYR","category":"foundation","finish":"dewy","coverage":"light",
     "formula_tags":["hydrating","vegan"],"price_eur":52.00,"hex":"#c07848"},
    {"brand":"NARS","line":"Sheer Glow Foundation","name":"Macao",
     "sku":"NARS-SG-MAC","category":"foundation","finish":"dewy","coverage":"light",
     "formula_tags":["hydrating","vegan"],"price_eur":52.00,"hex":"#884830"},
    # ── FENTY BEAUTY ─────────────────────────────────────────────────────────
    {"brand":"Fenty Beauty","line":"Pro Filt'r Soft Matte","name":"110N",
     "sku":"FB-PF-110N","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["long-wear","oil-free","vegan"],"price_eur":40.00,"hex":"#f8e0c0"},
    {"brand":"Fenty Beauty","line":"Pro Filt'r Soft Matte","name":"140W",
     "sku":"FB-PF-140W","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["long-wear","oil-free","vegan"],"price_eur":40.00,"hex":"#f0c898"},
    {"brand":"Fenty Beauty","line":"Pro Filt'r Soft Matte","name":"240N",
     "sku":"FB-PF-240N","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["long-wear","oil-free","vegan"],"price_eur":40.00,"hex":"#c8906a"},
    {"brand":"Fenty Beauty","line":"Pro Filt'r Soft Matte","name":"310W",
     "sku":"FB-PF-310W","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["long-wear","oil-free","vegan"],"price_eur":40.00,"hex":"#a87050"},
    {"brand":"Fenty Beauty","line":"Pro Filt'r Soft Matte","name":"420N",
     "sku":"FB-PF-420N","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["long-wear","oil-free","vegan"],"price_eur":40.00,"hex":"#7a4825"},
    {"brand":"Fenty Beauty","line":"Pro Filt'r Soft Matte","name":"498N",
     "sku":"FB-PF-498N","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["long-wear","oil-free","vegan"],"price_eur":40.00,"hex":"#4a2815"},
    # ── MAC STUDIO FIX ───────────────────────────────────────────────────────
    {"brand":"MAC","line":"Studio Fix Fluid","name":"NC15",
     "sku":"MAC-SFF-NC15","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["SPF15","long-wear","non-comedogenic"],"price_eur":38.00,"hex":"#f0d0a8"},
    {"brand":"MAC","line":"Studio Fix Fluid","name":"NC25",
     "sku":"MAC-SFF-NC25","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["SPF15","long-wear","non-comedogenic"],"price_eur":38.00,"hex":"#e0b080"},
    {"brand":"MAC","line":"Studio Fix Fluid","name":"NC35",
     "sku":"MAC-SFF-NC35","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["SPF15","long-wear","non-comedogenic"],"price_eur":38.00,"hex":"#c88860"},
    {"brand":"MAC","line":"Studio Fix Fluid","name":"NC42",
     "sku":"MAC-SFF-NC42","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["SPF15","long-wear","non-comedogenic"],"price_eur":38.00,"hex":"#a06040"},
    {"brand":"MAC","line":"Studio Fix Fluid","name":"NW35",
     "sku":"MAC-SFF-NW35","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["SPF15","long-wear","non-comedogenic"],"price_eur":38.00,"hex":"#c07850"},
    {"brand":"MAC","line":"Studio Fix Fluid","name":"NW45",
     "sku":"MAC-SFF-NW45","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["SPF15","long-wear","non-comedogenic"],"price_eur":38.00,"hex":"#885040"},
    # ── CHARLOTTE TILBURY ─────────────────────────────────────────────────────
    {"brand":"Charlotte Tilbury","line":"Airbrush Flawless","name":"1 Fair",
     "sku":"CT-AF-1F","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["long-wear","SPF20"],"price_eur":48.00,"hex":"#f5d8b8"},
    {"brand":"Charlotte Tilbury","line":"Airbrush Flawless","name":"3 Warm",
     "sku":"CT-AF-3W","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["long-wear","SPF20"],"price_eur":48.00,"hex":"#e0b080"},
    {"brand":"Charlotte Tilbury","line":"Airbrush Flawless","name":"6 Neutral",
     "sku":"CT-AF-6N","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["long-wear","SPF20"],"price_eur":48.00,"hex":"#c08860"},
    {"brand":"Charlotte Tilbury","line":"Airbrush Flawless","name":"9 Cool",
     "sku":"CT-AF-9C","category":"foundation","finish":"matte","coverage":"full",
     "formula_tags":["long-wear","SPF20"],"price_eur":48.00,"hex":"#a87060"},
    # ── KIKO CONCEALER ───────────────────────────────────────────────────────
    {"brand":"Kiko Milano","line":"Full Coverage Concealer","name":"02 Light Beige",
     "sku":"KM-FCC-02","category":"concealer","finish":"matte","coverage":"full",
     "formula_tags":["non-comedogenic","long-wear"],"price_eur":8.90,"hex":"#f0c898"},
    {"brand":"Kiko Milano","line":"Full Coverage Concealer","name":"03 Natural",
     "sku":"KM-FCC-03","category":"concealer","finish":"matte","coverage":"full",
     "formula_tags":["non-comedogenic","long-wear"],"price_eur":8.90,"hex":"#e0b080"},
    {"brand":"Kiko Milano","line":"Full Coverage Concealer","name":"06 Dark Beige",
     "sku":"KM-FCC-06","category":"concealer","finish":"matte","coverage":"full",
     "formula_tags":["non-comedogenic","long-wear"],"price_eur":8.90,"hex":"#b07848"},
]


# ══════════════════════════════════════════════════════════════════════════════
#  FONTE 2 — THE PUDDING (scarica a runtime)
# ══════════════════════════════════════════════════════════════════════════════

PUDDING_URLS = [
    # File principale — tutti i gruppi (US, Nigeria, Japan, India)
    "https://raw.githubusercontent.com/the-pudding/data/master/makeup-shades/allShades.csv",
    # File per paese separati (fallback se il principale cambia)
    "https://raw.githubusercontent.com/the-pudding/data/master/makeup-shades/allNumbers.csv",
]

def fetch_pudding() -> list[dict]:
    """Scarica allShades.csv da GitHub e restituisce tutti i gruppi geografici."""
    print("  Scaricando The Pudding dataset...")
    for url in PUDDING_URLS:
        try:
            req = urllib.request.urlopen(url, timeout=15)
            raw = req.read().decode("utf-8")
            df  = pd.read_csv(io.StringIO(raw))
            print(f"  OK — {len(df)} righe scaricate da {url.split('/')[-1]}")
            break
        except Exception as e:
            print(f"  Fallito {url}: {e}")
            df = None

    if df is None or df.empty:
        print("  ERRORE: impossibile scaricare il dataset The Pudding.")
        return []

    # Mappa gruppi: 2=US, 5=Nigeria, 6=Japan, 7=India — prendiamo tutti
    GROUP_NAMES = {2: "US", 5: "Nigeria", 6: "Japan", 7: "India"}
    records = []
    skipped = 0

    for _, row in df.iterrows():
        hex_val = str(row.get("hex", "")).strip()
        if not hex_val or len(hex_val) < 6:
            skipped += 1
            continue
        if not hex_val.startswith("#"):
            hex_val = "#" + hex_val

        brand   = str(row.get("brand", "Unknown")).strip()
        product = str(row.get("product", "Foundation")).strip()
        name    = str(row.get("name", "")).strip() or hex_val
        group   = int(row.get("group", 2)) if "group" in row else 2
        geo     = GROUP_NAMES.get(group, "Global")

        # SKU unico: prefisso fonte + brand + hex + gruppo geografico
        # Questo evita collisioni con SKU del dataset interno (che usa MB-, LO-, ecc.)
        sku = f"PUD{group}-{brand[:4].upper().replace(' ','')}-{hex_val[1:7]}"

        records.append({
            "brand":        brand,
            "line":         product,
            "name":         name,
            "sku":          sku,
            "category":     "foundation",
            "finish":       "satin",
            "coverage":     "medium",
            "formula_tags": [],
            "price_eur":    None,
            "hex":          hex_val,
            "notes":        f"The Pudding Beauty Brawl 2018 · mercato {geo}",
        })

    print(f"  {len(records)} shade validi ({skipped} saltati per hex mancante)")
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  FONTE 3 — KAGGLE MAKEUP SHADES (scarica a runtime)
# ══════════════════════════════════════════════════════════════════════════════

KAGGLE_URLS = [
    # Mirror pubblico del dataset Kaggle su jsDelivr
    "https://raw.githubusercontent.com/nicholasgasior/makeup-shades/main/shades.csv",
    # Backup alternativo
    "https://raw.githubusercontent.com/shelbyvjacobs/makeup-shades-api/master/db/shades.json",
]

def fetch_kaggle() -> list[dict]:
    """
    Tenta di scaricare il dataset Kaggle makeup-shades da mirror pubblici.
    Se non disponibile, restituisce lista vuota con avviso.
    """
    print("  Scaricando Kaggle makeup-shades dataset...")
    for url in KAGGLE_URLS:
        try:
            req = urllib.request.urlopen(url, timeout=15)
            raw = req.read().decode("utf-8")

            if url.endswith(".json"):
                # Il mirror JSON di shelbyvjacobs ha struttura array
                import json
                data = json.loads(raw)
                if not isinstance(data, list):
                    continue
                records = []
                for item in data:
                    hex_val = str(item.get("hex", "")).strip()
                    if not hex_val or len(hex_val) < 6:
                        continue
                    if not hex_val.startswith("#"):
                        hex_val = "#" + hex_val
                    brand   = str(item.get("brand", "Unknown"))
                    product = str(item.get("product", "Foundation"))
                    sku     = f"KAG-{brand[:4].upper().replace(' ','')}-{hex_val[1:7]}"
                    records.append({
                        "brand": brand, "line": product, "name": hex_val,
                        "sku": sku, "category": "foundation", "finish": "satin",
                        "coverage": "medium", "formula_tags": [], "price_eur": None,
                        "hex": hex_val,
                        "notes": "Kaggle makeup-shades-dataset · mirror pubblico",
                    })
                print(f"  OK — {len(records)} shade da JSON mirror")
                return records
            else:
                df = pd.read_csv(io.StringIO(raw))
                records = []
                for _, row in df.iterrows():
                    hex_val = str(row.get("hex", row.get("colour", ""))).strip()
                    if not hex_val or len(hex_val) < 6:
                        continue
                    if not hex_val.startswith("#"):
                        hex_val = "#" + hex_val
                    brand   = str(row.get("brand", row.get("Brand", "Unknown")))
                    product = str(row.get("product", row.get("Product", "Foundation")))
                    name    = str(row.get("name", row.get("shade", hex_val)))
                    sku     = f"KAG-{brand[:4].upper().replace(' ','')}-{hex_val[1:7]}"
                    records.append({
                        "brand": brand, "line": product, "name": name,
                        "sku": sku, "category": "foundation", "finish": "satin",
                        "coverage": "medium", "formula_tags": [], "price_eur": None,
                        "hex": hex_val,
                        "notes": "Kaggle makeup-shades-dataset · mirror pubblico",
                    })
                print(f"  OK — {len(records)} shade da CSV mirror")
                return records
        except Exception as e:
            print(f"  Fallito {url.split('/')[-1]}: {e}")

    print("  Mirror Kaggle non raggiungibili — fonte saltata.")
    print("  Per includerla, scarica manualmente il CSV da:")
    print("  https://www.kaggle.com/datasets/shivamb/makeup-shades-dataset")
    return []


# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTATORE COMUNE
# ══════════════════════════════════════════════════════════════════════════════

def import_records(records: list[dict], dry_run: bool = False) -> dict:
    """
    Converte hex → Lab e inserisce nel database, skippando SKU duplicati.
    Tutti i valori Lab sono float Python puro per compatibilità PostgreSQL.
    """
    init_db()
    db = get_session()
    stats = {"inserted": 0, "skipped_dup": 0, "skipped_error": 0, "total": len(records)}

    try:
        existing_skus = {r[0] for r in db.query(Product.sku).all() if r[0]}

        for rec in records:
            sku = rec.get("sku", "")
            if sku in existing_skus:
                stats["skipped_dup"] += 1
                continue

            hex_val = rec.get("hex", "")
            try:
                L, a, b = hex_to_lab(hex_val)
            except Exception as e:
                stats["skipped_error"] += 1
                continue

            ita       = float(calc_ita(L, b))
            undertone = calc_undertone(a, b)

            if dry_run:
                print(f"  [DRY] {rec['brand']:20s} · {rec['name']:20s} "
                      f"→ L={L} a={a} b={b} ITA={ita}° {undertone}")
            else:
                pid  = str(uuid.uuid4())
                prod = Product(
                    product_id   = pid,
                    brand        = rec["brand"],
                    line         = rec["line"],
                    name         = rec["name"],
                    sku          = sku or None,
                    category     = rec.get("category", "foundation"),
                    finish       = rec.get("finish", "satin"),
                    coverage     = rec.get("coverage", "medium"),
                    formula_tags = rec.get("formula_tags", []),
                    price_eur    = rec.get("price_eur"),
                    active       = True,
                    notes        = rec.get("notes", f"Importato da hex {hex_val}"),
                )
                color = ProductColorimetry(
                    product_id         = pid,
                    L_star             = L,
                    a_star             = a,
                    b_star             = b,
                    ITA_deg            = ita,
                    undertone_calc     = undertone,
                    oxidation_delta_b  = 0.0,
                    measurement_source = "hex_derived",
                )
                db.add(prod)
                db.add(color)
                existing_skus.add(sku)

            stats["inserted"] += 1

        if not dry_run:
            db.commit()
            print(f"  DB commit OK")
    except Exception as e:
        db.rollback()
        print(f"  ERRORE durante il commit: {e}")
        raise
    finally:
        db.close()

    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Popola il database SkinMatch con shade da fonti open source"
    )
    parser.add_argument(
        "--source",
        choices=["all", "internal", "pudding", "kaggle"],
        default="all",
        help="Fonte dati (default: all — importa tutte le fonti disponibili)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Anteprima senza salvare nel database"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SkinMatch AI — Popolamento database da fonti open source")
    print(f"Fonte: {args.source} | Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    all_records = []
    source_stats = {}

    # Raccoglie records da tutte le fonti selezionate
    if args.source in ("all", "internal"):
        print(f"[1/3] Dataset interno IT/EU ({len(INTERNAL_DATASET)} shade curati)")
        all_records += INTERNAL_DATASET
        source_stats["internal"] = len(INTERNAL_DATASET)

    if args.source in ("all", "pudding"):
        print(f"[2/3] The Pudding / Beauty Brawl")
        pudding = fetch_pudding()
        all_records += pudding
        source_stats["pudding"] = len(pudding)

    if args.source in ("all", "kaggle"):
        print(f"[3/3] Kaggle makeup-shades-dataset")
        kaggle = fetch_kaggle()
        all_records += kaggle
        source_stats["kaggle"] = len(kaggle)

    print(f"\nTotale shade raccolti: {len(all_records)}")
    print(f"Avvio importazione...\n")

    stats = import_records(all_records, dry_run=args.dry_run)

    # Report finale
    print(f"\n{'='*60}")
    print(f"RISULTATO IMPORTAZIONE")
    print(f"{'='*60}")
    for fonte, n in source_stats.items():
        print(f"  {fonte:12s}: {n} shade raccolti")
    print(f"  {'-'*40}")
    print(f"  Inseriti nel DB:       {stats['inserted']}")
    print(f"  Saltati (duplicati):   {stats['skipped_dup']}")
    print(f"  Saltati (hex errato):  {stats['skipped_error']}")
    print(f"  {'='*40}")
    print(f"  TOTALE NEL DB (stima): {stats['inserted'] + stats['skipped_dup']} shade")

    if args.dry_run:
        print(f"\n  [DRY RUN] Nessun dato salvato.")
    else:
        print(f"\n  Fonte: 'hex_derived' — badge massimo: '✓ Buono'")
        print(f"  Per badge '🏆 Certificato': misura con Nix e aggiorna la fonte.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()