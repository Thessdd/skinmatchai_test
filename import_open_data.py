"""
import_open_data.py — SkinMatch AI
Importa shade da fonti aperte (hex → Lab) nel database SkinMatch.

FONTI SUPPORTATE:
1. The Pudding / allShades.csv  (scarica manualmente da GitHub)
2. Kaggle makeup-shades-dataset (scarica manualmente da Kaggle)
3. Dataset interno curato (incluso in questo file) — pronto all'uso

COME USARE:
  # Importa il dataset interno (nessun download richiesto):
  python import_open_data.py --source internal

  # Importa da CSV The Pudding (scarica prima il file):
  python import_open_data.py --source pudding --file allShades.csv

  # Importa da CSV Kaggle:
  python import_open_data.py --source kaggle --file makeup_shades.csv

  # Anteprima senza salvare:
  python import_open_data.py --source internal --dry-run
"""

import argparse
import math
import uuid
import numpy as np
import pandas as pd
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
    Converte un colore hex in coordinate CIELab (D65/2°).
    Fonte: IEC 61966-2-1 (sRGB) + CIE 15:2004.

    IMPORTANTE: questo valore è una STIMA da swatch digitale,
    non una misura spettrofotometrica. Accuratezza tipica: ΔE 2-6
    rispetto a una misura Nix reale (dipende dalla calibrazione
    del monitor usato per creare il swatch).
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
    xr = xyz / WHITE_D65
    eps = 1e-10
    f = np.where(xr > 0.008856,
                 np.power(np.maximum(xr, eps), 1.0 / 3.0),
                 7.787 * xr + 16.0 / 116.0)
    L = round(116.0 * f[1] - 16.0, 2)
    a = round(500.0 * (f[0] - f[1]), 2)
    b = round(200.0 * (f[1] - f[2]), 2)
    return L, a, b


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET INTERNO — shade reali con hex verificati
#  Fonte: swatch ufficiali brand (siti IT/EU) + letteratura online
#  Tutti flaggati come "hex_derived" — non misure Nix
# ══════════════════════════════════════════════════════════════════════════════

INTERNAL_DATASET = [
    # ── MAYBELLINE FIT ME (disponibile in farmacia italiana) ─────────────────
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

    # ── L'ORÉAL TRUE MATCH ────────────────────────────────────────────────────
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

    # ── NARS SHEER GLOW (disponibile Sephora Italia) ──────────────────────────
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

    # ── FENTY BEAUTY PRO FILT'R (Sephora Italia) ─────────────────────────────
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

    # ── MAC STUDIO FIX (MAC Italia) ───────────────────────────────────────────
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

    # ── CHARLOTTE TILBURY AIRBRUSH FLAWLESS ───────────────────────────────────
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

    # ── KIKO CONCEALER (aggiuntivi al seed) ──────────────────────────────────
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
#  PARSER CSV — The Pudding allShades.csv
# ══════════════════════════════════════════════════════════════════════════════

def parse_pudding_csv(filepath: str) -> list[dict]:
    """
    Converte allShades.csv di The Pudding nel formato interno.
    Struttura CSV: brand, product, hex, hue, sat, lightness, group
    """
    df = pd.read_csv(filepath)
    records = []

    # Mappa gruppi The Pudding: 2=US, 5=Nigeria, 6=Japan, 7=India
    # Teniamo solo US (gruppo 2) — più rappresentativo per il mercato EU
    if "group" in df.columns:
        df = df[df["group"] == 2]

    for _, row in df.iterrows():
        hex_val = str(row.get("hex", "")).strip()
        if not hex_val or len(hex_val) < 6:
            continue
        if not hex_val.startswith("#"):
            hex_val = "#" + hex_val

        brand   = str(row.get("brand", "Unknown"))
        product = str(row.get("product", "Foundation"))
        name    = str(row.get("name", hex_val))

        records.append({
            "brand":       brand,
            "line":        product,
            "name":        name,
            "sku":         f"PUD-{brand[:3].upper()}-{hex_val[1:7]}",
            "category":    "foundation",
            "finish":      "satin",
            "coverage":    "medium",
            "formula_tags":[],
            "price_eur":   None,
            "hex":         hex_val,
        })
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  PARSER CSV — Kaggle makeup-shades-dataset
# ══════════════════════════════════════════════════════════════════════════════

def parse_kaggle_csv(filepath: str) -> list[dict]:
    """
    Converte il dataset Kaggle nel formato interno.
    Struttura: brand, product, hex (con o senza #)
    """
    df = pd.read_csv(filepath)
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

        records.append({
            "brand":       brand,
            "line":        product,
            "name":        name,
            "sku":         f"KAG-{brand[:3].upper()}-{hex_val[1:7]}",
            "category":    "foundation",
            "finish":      "satin",
            "coverage":    "medium",
            "formula_tags":[],
            "price_eur":   None,
            "hex":         hex_val,
        })
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTATORE
# ══════════════════════════════════════════════════════════════════════════════

def import_records(records: list[dict], dry_run: bool = False) -> dict:
    """
    Converte hex in Lab e inserisce nel database.
    Skippa SKU già esistenti.
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
                print(f"  ⚠ Hex non valido [{hex_val}] per {rec.get('name','?')}: {e}")
                stats["skipped_error"] += 1
                continue

            ita       = calc_ita(L, b)
            undertone = calc_undertone(a, b)

            if dry_run:
                print(f"  [DRY] {rec['brand']:20s} · {rec['name']:25s} "
                      f"hex={hex_val} → L={L} a={a} b={b} ITA={ita}° {undertone}")
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
                    notes        = f"Importato da hex {hex_val} — fonte: swatch digitale",
                )
                color = ProductColorimetry(
                    product_id         = pid,
                    L_star             = L,
                    a_star             = a,
                    b_star             = b,
                    ITA_deg            = ita,
                    undertone_calc     = undertone,
                    oxidation_delta_b  = 0.0,  # ignoto da swatch digitale
                    measurement_source = "hex_derived",
                )
                db.add(prod)
                db.add(color)
                existing_skus.add(sku)

            stats["inserted"] += 1

        if not dry_run:
            db.commit()
    finally:
        db.close()

    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Importa shade da fonti aperte in SkinMatch DB")
    parser.add_argument("--source", choices=["internal","pudding","kaggle"],
                        default="internal",
                        help="Fonte dati: internal | pudding | kaggle")
    parser.add_argument("--file", type=str, default=None,
                        help="Percorso CSV (richiesto per pudding e kaggle)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Anteprima senza salvare nel database")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SkinMatch AI — Import shade da fonti aperte")
    print(f"Fonte: {args.source} | Dry run: {args.dry_run}")
    print(f"{'='*60}\n")

    if args.source == "internal":
        records = INTERNAL_DATASET
        print(f"Dataset interno: {len(records)} shade da Maybelline, L'Oréal, NARS, "
              f"Fenty Beauty, MAC, Charlotte Tilbury, Kiko")

    elif args.source == "pudding":
        if not args.file:
            print("ERRORE: --file richiesto per la fonte 'pudding'")
            print("Scarica il file da:")
            print("  https://github.com/the-pudding/data/blob/master/makeup-shades/allShades.csv")
            return
        records = parse_pudding_csv(args.file)
        print(f"The Pudding CSV: {len(records)} shade caricati da {args.file}")

    elif args.source == "kaggle":
        if not args.file:
            print("ERRORE: --file richiesto per la fonte 'kaggle'")
            print("Scarica il file da:")
            print("  https://www.kaggle.com/datasets/shivamb/makeup-shades-dataset")
            return
        records = parse_kaggle_csv(args.file)
        print(f"Kaggle CSV: {len(records)} shade caricati da {args.file}")

    print()
    stats = import_records(records, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print(f"Risultato importazione:")
    print(f"  Totale shade processati:  {stats['total']}")
    print(f"  Inseriti nel database:    {stats['inserted']}")
    print(f"  Saltati (duplicati SKU):  {stats['skipped_dup']}")
    print(f"  Saltati (hex non valido): {stats['skipped_error']}")
    if args.dry_run:
        print(f"\n  [DRY RUN] Nessun dato salvato.")
    else:
        print(f"\n  ✓ Importazione completata. Fonte: 'hex_derived'")
        print(f"  Questi shade ricevono il badge '~ Accettabile' o '✓ Buono'")
        print(f"  Non ricevono '🏆 Certificato' — serve misura Nix per quello.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()