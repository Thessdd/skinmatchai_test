"""
SkinMatch AI — app_v7.py
Pipeline completa: analisi pelle + database prodotti + matching colorimetrico.
"""

import copy
import json
import math
import streamlit as st
import numpy as np
from PIL import Image
import pillow_heif
import io as _io
import re
import uuid as _uuid
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
from database import (
    init_db, seed_demo_data, get_session,
    Product, ProductColorimetry, SkinProfile, Match, SavedClientSkin,
    list_saved_client_skins,
    find_matches, calc_ita, calc_undertone,
    MATCH_WEIGHTS
)

pillow_heif.register_heif_opener()
init_db()

# Seed al primo avvio
_db_init = get_session()
seed_demo_data(_db_init)
_db_init.close()


# ══════════════════════════════════════════════════════════════════════════════
#  ENGINE (identico a v6)
# ══════════════════════════════════════════════════════════════════════════════

class ColorCheckerCalibrator:
    REFERENCE_PATCHES = {
        "Patch bianca (CC)":   {"lab": (96.5, -0.2,  1.4)},
        "Patch grigia 65% (CC)":{"lab": (65.3, -0.1, -0.1)},
        "Patch grigia 50% (CC)":{"lab": (49.4, -0.1, -0.2)},
        "Patch dark skin (CC)": {"lab": (38.4, 13.6, 14.4)},
    }
    LOWCOST_PATCHES = {
        "Foglio carta A4":     {"lab": (94.0, 0.5, 3.0)},
        "Cartoncino grigio 18%":{"lab": (49.0, 0.0, 0.5)},
        "Scheda fotografica":  {"lab": (50.0, 0.0, 0.0)},
    }
    def __init__(self):
        self.deltas = None
        self.patch_name = None

    def calibrate(self, patch_image, patch_name, engine, is_lowcost=False):
        img = np.array(Image.open(patch_image).convert("RGB"))
        h, w = img.shape[:2]
        roi = img[int(h*.3):int(h*.7), int(w*.3):int(w*.7)]
        med = np.median(roi.reshape(-1, 3), axis=0)
        Lm, am, bm = engine.srgb_to_lab(med)
        ref = (self.LOWCOST_PATCHES if is_lowcost else self.REFERENCE_PATCHES)[patch_name]["lab"]
        self.deltas = (ref[0]-Lm, ref[1]-am, ref[2]-bm)
        self.patch_name = patch_name
        de = math.sqrt(sum(d**2 for d in self.deltas))
        if de < 3:   lvl, col = "Ottima",    "success"
        elif de < 8: lvl, col = "Buona",     "success"
        elif de < 15:lvl, col = "Necessaria","warning"
        else:        lvl, col = "Critica",   "error"
        return {"livello": lvl, "color": col, "delta_e": round(de,2),
                "misurato": {"L*":round(Lm,2),"a*":round(am,2),"b*":round(bm,2)},
                "riferimento": {"L*":ref[0],"a*":ref[1],"b*":ref[2]}}

    def apply(self, L, a, b):
        if not self.deltas: return L, a, b
        return round(L+self.deltas[0],2), round(a+self.deltas[1],2), round(b+self.deltas[2],2)

    @property
    def is_active(self): return self.deltas is not None


class SkinIDEngine:
    RGB_TO_XYZ = np.array([[0.4124564,0.3575761,0.1804375],
                            [0.2126729,0.7151522,0.0721750],
                            [0.0193339,0.1191920,0.9503041]])
    WHITE_D65  = np.array([95.047, 100.0, 108.883])
    ZONE_WEIGHTS = {"Collo": 0.50, "Guancia": 0.35, "Fronte": 0.15}

    def srgb_to_lab(self, rgb):
        r = np.asarray(rgb, dtype=float)/255.0
        lin = np.where(r>0.04045, np.power((r+0.055)/1.055,2.4), r/12.92)*100.0
        xyz = self.RGB_TO_XYZ @ lin
        xr  = xyz/self.WHITE_D65
        eps = 1e-10
        f   = np.where(xr>0.008856, np.power(np.maximum(xr,eps),1/3), 7.787*xr+16/116)
        return round(116*f[1]-16,2), round(500*(f[0]-f[1]),2), round(200*(f[1]-f[2]),2)

    def calculate_ita(self, L, b):
        return round(math.atan2(L-50.0, b if abs(b)>0.001 else 0.001)*(180/math.pi), 2)

    def get_ita_category(self, ita):
        if ita>55: return "VERY-LIGHT"
        if ita>41: return "LIGHT"
        if ita>28: return "INTERMEDIATE"
        if ita>10: return "TAN"
        if ita>-30:return "BROWN"
        return "DARK"

    def get_undertone(self, a, b):
        if b>18 and a<12:  return "WARM"
        if b>18 and a>=12: return "WARM-PEACH"
        if b<13:           return "COOL"
        if a<6 and b<18:   return "OLIVE"
        return "NEUTRAL"

    def get_reactivity_index(self, b, skin_type):
        k = {"Oleosa":1.15,"Mista":1.05,"Secca / Normale":0.95}.get(skin_type,1.05)
        ri = round(b*k,2)
        return {"valore":ri, "livello":"ALTO" if ri>21 else "MEDIO" if ri>17 else "BASSO"}

    def build_skin_id(self, cat, L, undertone, a, ri):
        return f"{cat}-{int(L)}-{undertone}-a{int(a)}-RI-{ri}"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS UI
# ══════════════════════════════════════════════════════════════════════════════

BADGE_COLOR = {
    "🏆 Certificato": "success",
    "⭐ Eccellente":  "success",
    "✓ Buono":        "info",
    "~ Accettabile":  "warning",
    "✗ Non consigliato": "error",
}

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def lab_to_srgb(L: float, a: float, b: float) -> tuple[int, int, int]:
    """
    CIE Lab (D65/2°) -> sRGB 8-bit (clamped).
    """
    fy = (L + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    def f_inv(t: float) -> float:
        t3 = t ** 3
        return t3 if t3 > 0.008856 else (t - 16.0 / 116.0) / 7.787

    xr = f_inv(fx)
    yr = f_inv(fy)
    zr = f_inv(fz)

    Xn, Yn, Zn = SkinIDEngine.WHITE_D65 / 100.0
    X = xr * Xn
    Y = yr * Yn
    Z = zr * Zn

    # XYZ -> linear RGB (sRGB D65)
    r_lin =  3.2404542 * X + (-1.5371385) * Y + (-0.4985314) * Z
    g_lin = (-0.9692660) * X +  1.8760108 * Y +  0.0415560 * Z
    b_lin =  0.0556434 * X + (-0.2040259) * Y +  1.0572252 * Z

    def gamma(u: float) -> float:
        u = _clamp01(u)
        return 12.92 * u if u <= 0.0031308 else 1.055 * (u ** (1.0 / 2.4)) - 0.055

    r = int(round(gamma(r_lin) * 255.0))
    g = int(round(gamma(g_lin) * 255.0))
    bb = int(round(gamma(b_lin) * 255.0))
    r = 0 if r < 0 else 255 if r > 255 else r
    g = 0 if g < 0 else 255 if g > 255 else g
    bb = 0 if bb < 0 else 255 if bb > 255 else bb
    return r, g, bb

def lab_to_hex(L: float, a: float, b: float) -> str:
    r, g, bb = lab_to_srgb(L, a, b)
    return f"#{r:02x}{g:02x}{bb:02x}"

def render_color_swatch(hex_color: str, size_px: int = 22, shape: str = "circle") -> str:
    radius = "999px" if shape == "circle" else "6px"
    return (
        f"<div style='width:{size_px}px;height:{size_px}px;"
        f"border-radius:{radius};background:{hex_color};"
        f"border:1px solid rgba(255,255,255,0.35);"
        f"box-shadow:0 0 0 1px rgba(0,0,0,0.12) inset;"
        f"display:inline-block;vertical-align:middle'></div>"
    )

def render_palette(zones: dict, weights: dict) -> None:
    items = []
    for z in ["Fronte", "Guancia", "Collo"]:
        if z not in zones:
            continue
        v = zones[z]
        hx = lab_to_hex(float(v["L*"]), float(v["a*"]), float(v["b*"]))
        items.append((z, hx))

    tw = sum(weights.get(z, 0.0) for z in zones)
    if tw <= 0:
        return
    avg_L = sum(float(zones[z]["L*"]) * weights.get(z, 0.0) for z in zones) / tw
    avg_a = sum(float(zones[z]["a*"]) * weights.get(z, 0.0) for z in zones) / tw
    avg_b = sum(float(zones[z]["b*"]) * weights.get(z, 0.0) for z in zones) / tw
    avg_hex = lab_to_hex(avg_L, avg_a, avg_b)

    st.subheader("Palette visiva (zone + media pesata)")
    cols = st.columns(len(items) + 1)
    for i, (label, hx) in enumerate(items):
        cols[i].markdown(render_color_swatch(hx, size_px=56, shape="circle"), unsafe_allow_html=True)
        cols[i].caption(f"**{label}**  \n`{hx}`")
    cols[len(items)].markdown(render_color_swatch(avg_hex, size_px=56, shape="circle"), unsafe_allow_html=True)
    cols[len(items)].caption("**Media pesata**  \n" + f"`{avg_hex}`")

def show_match_results(matches: list):
    if not matches:
        st.info("Nessun prodotto trovato con i filtri selezionati.")
        return

    st.subheader(f"Top {len(matches)} prodotti abbinati")

    for i, m in enumerate(matches):
        p  = m["product"]
        c  = m["colorimetry"]
        bc = BADGE_COLOR.get(m["badge"], "info")

        with st.container(border=True):
            sw_hex = lab_to_hex(float(c.L_star), float(c.a_star), float(c.b_star))
            cols = st.columns([0.45, 0.6, 3, 1.2, 1.2, 1.4])
            cols[0].markdown(f"**#{i+1}**")
            cols[1].markdown(render_color_swatch(sw_hex, size_px=24, shape="square"), unsafe_allow_html=True)
            cols[2].markdown(f"**{p.brand}**  \n{p.line} · *{p.name}*  \n"
                             f"`{p.finish}` · `{p.coverage}` · €{p.price_eur or '—'}")
            cols[3].metric("Score", f"{m['score']:.3f}")
            cols[4].metric("ΔE", f"{m['delta_e']:.2f}")
            getattr(cols[5], bc)(m["badge"])

            with st.expander("Dettaglio colorimetrico"):
                d1, d2, d3 = st.columns(3)
                d1.metric("L* prodotto", c.L_star)
                d2.metric("a* prodotto", c.a_star)
                d3.metric("b* prodotto", c.b_star)
                d1.metric("ΔL (pelle-prod)", f"{m['dL']:+.2f}")
                d2.metric("Δa", f"{m['da']:+.2f}")
                d3.metric("Δb", f"{m['db']:+.2f}")
                if m["oxid_penalty"] > 0:
                    st.warning(f"Penalità ossidazione applicata: +{m['oxid_penalty']} ΔE "
                               "(pelle con RI alto + prodotto che ossida)")
                tags = p.formula_tags or []
                if tags:
                    st.caption("Tag formula: " + " · ".join(f"`{t}`" for t in tags))
                src = c.measurement_source or "?"
                st.caption(f"Fonte dati colorimetrici: `{src}`"
                           + (" ⚠️ stimato — misura con Nix per certificare" if src=="manual_estimated" else " ✓"))
            st.divider()


def mostra_skinid_header(skin_id, avg_ita, categoria, undertone, avg_L, ri_data,
                          skin_type, source_label):
    st.header(f"SkinID™: `{skin_id}`")
    st.caption(f"Sorgente: {source_label} · Pelle: {skin_type.lower()}")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Categoria ITA", categoria)
    c2.metric("ITA°", f"{avg_ita:.1f}°")
    c3.metric("Undertone", undertone)
    c4.metric("L* globale", f"{avg_L:.1f}")
    c5.metric("Reactivity", ri_data["livello"])


def skin_data_from_saved_row(row: SavedClientSkin) -> dict:
    zones = row.zones_json
    if isinstance(zones, str):
        zones = json.loads(zones)
    return {
        "zones": copy.deepcopy(zones),
        "skin_type": row.skin_type,
        "source": row.source,
        "saved_client_id": row.saved_id,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="SkinMatch AI", page_icon="🧬", layout="wide")

# Global CSS (beauty-tech theme)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root{
  --sm-bg: #FAF7F2;         /* crema */
  --sm-surface: #FFFFFF;
  --sm-beige: #F5E6D3;      /* beige caldo */
  --sm-terracotta: #C4845A; /* terracotta */
  --sm-text: #2C2C2C;       /* antracite */
  --sm-muted: rgba(44,44,44,0.72);
  --sm-border: rgba(44,44,44,0.10);
  --sm-shadow: 0 10px 30px rgba(44,44,44,0.10);
}

html, body, [class*="css"]  { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; color: var(--sm-text); }
.stApp { background: var(--sm-bg); }

h1, h2, h3, h4, h5 { font-family: 'Cormorant Garamond', Georgia, serif; letter-spacing: 0.2px; }
h1 { font-weight: 700; }
h2, h3 { font-weight: 600; }

/* Header */
.sm-header{
  background: linear-gradient(90deg, var(--sm-beige), rgba(250,247,242,0.0));
  border: 1px solid var(--sm-border);
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
  box-shadow: var(--sm-shadow);
  margin: 6px 0 18px 0;
}
.sm-header__top{
  display:flex;
  gap:12px;
  align-items: baseline;
  justify-content: space-between;
  flex-wrap: wrap;
}
.sm-brand{
  font-family: 'Cormorant Garamond', Georgia, serif;
  font-size: 40px;
  font-weight: 700;
  line-height: 1.05;
  margin: 0;
}
.sm-claim{
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 14px;
  color: var(--sm-muted);
  margin-top: 6px;
}
.sm-pill{
  font-family: 'Inter', system-ui, sans-serif;
  font-size: 12px;
  color: var(--sm-text);
  background: rgba(245,230,211,0.65);
  border: 1px solid var(--sm-border);
  border-radius: 999px;
  padding: 6px 10px;
}

/* Buttons */
.stButton>button, .stDownloadButton>button{
  border-radius: 12px !important;
  border: 1px solid var(--sm-border) !important;
}
.stButton>button[kind="primary"], .stButton>button[data-testid="baseButton-primary"]{
  background: var(--sm-terracotta) !important;
  border-color: rgba(196,132,90,0.55) !important;
}

/* Cards: use Streamlit bordered containers as ecommerce cards */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background: var(--sm-surface);
  border: 1px solid var(--sm-border) !important;
  border-radius: 16px !important;
  box-shadow: 0 10px 24px rgba(44,44,44,0.08);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div{
  padding: 14px 14px 10px 14px;
}

/* Metrics */
div[data-testid="stMetric"]{
  background: rgba(250,247,242,0.65);
  border: 1px solid var(--sm-border);
  border-radius: 14px;
  padding: 10px 12px;
}

/* Sidebar cleanup */
section[data-testid="stSidebar"]{
  background: #fff;
  border-right: 1px solid var(--sm-border);
}
</style>
""",
    unsafe_allow_html=True,
)

# Session state
for k,v in [("engine",SkinIDEngine()),("calibrator",ColorCheckerCalibrator()),
            ("calib_result",None),("skin_data",None),
            ("uploaded_files_bytes",{}),("zone_map",{}),("show_result",False),
            ("flow_step", 1), ("admin_page", "flow"), ("last_match_results", None)]:
    if k not in st.session_state: st.session_state[k] = v

engine     = st.session_state.engine
calibrator = st.session_state.calibrator

# Top header (investor-first)
st.markdown(
    """
<div class="sm-header">
  <div class="sm-header__top">
    <div>
      <div class="sm-brand">SkinMatch AI</div>
      <div class="sm-claim">Beauty-tech colorimetry: SkinID™ in Lab → match immediato con gli shade.</div>
    </div>
    <div class="sm-pill">D65/2° · CIE Lab · ΔE</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parametri globali")
    skin_type = st.radio("Tipo di pelle", ["Oleosa","Mista","Secca / Normale"], index=1)
    st.divider()

    st.markdown("**Onboarding**")
    st.caption("Step 1 Carica → Step 2 Zone → Step 3 Risultato → Step 4 Match")
    c1, c2 = st.columns(2)
    if c1.button("Vai al flow", use_container_width=True):
        st.session_state.admin_page = "flow"
        st.rerun()
    if c2.button("Riparti", use_container_width=True):
        st.session_state.flow_step = 1
        st.session_state.skin_data = None
        st.session_state.uploaded_files_bytes = {}
        st.session_state.zone_map = {}
        st.session_state.last_match_results = None
        st.session_state.admin_page = "flow"
        st.rerun()

    with st.expander("⚙️ Gestione (admin)", expanded=False):
        a1, a2 = st.columns(2)
        if a1.button("Database", use_container_width=True):
            st.session_state.admin_page = "db"
            st.rerun()
        if a2.button("Aggiungi", use_container_width=True):
            st.session_state.admin_page = "add"
            st.rerun()

    st.divider()

    if st.button("▶ Avvia demo", help="Carica un profilo pelle sintetico per esplorare l'app senza foto"):
        st.session_state.skin_data = {
            "zones": {
                "Fronte":    {"L*": 72.4, "a*": 7.8,  "b*": 16.2},
                "Guancia":   {"L*": 68.1, "a*": 9.2,  "b*": 18.9},
                "Collo":     {"L*": 64.3, "a*": 10.5, "b*": 21.4},
            },
            "skin_type": "Mista",
            "source":    "Demo sintetico · TAN-WARM",
        }
        st.session_state.uploaded_files_bytes = {}
        st.session_state.zone_map = {}
        st.session_state.flow_step = 3
        st.session_state.admin_page = "flow"
        st.rerun()

    st.caption("CIE 15:2004 · Del Bino et al. (2006)")


def show_flow_header(step: int) -> None:
    steps = ["Carica foto", "Seleziona zone", "Risultato", "Trova match"]
    total = len(steps)
    step = 1 if step < 1 else total if step > total else step
    progress = 0.0 if total <= 1 else (step - 1) / (total - 1)
    st.progress(progress)
    st.caption(
        " → ".join(
            [
                f"**Step {i+1}** {name}" if (i + 1) == step else f"Step {i+1} {name}"
                for i, name in enumerate(steps)
            ]
        )
    )


# Router (flow vs admin pages)
admin_page = st.session_state.get("admin_page", "flow")
flow_step = int(st.session_state.get("flow_step", 1))

if admin_page == "db":
    nav = "📦 Database Prodotti"
elif admin_page == "add":
    nav = "➕ Aggiungi Prodotto"
else:
    nav = "🎯 Matching Prodotti" if flow_step >= 4 else "🔬 Analisi Pelle"
    show_flow_header(flow_step)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — ANALISI PELLE
# ══════════════════════════════════════════════════════════════════════════════

if nav == "🔬 Analisi Pelle":
    st.title("🔬 Analisi Pelle")

    # Calibrazione
    with st.expander("🎯 Calibrazione ColorChecker — " +
                     ("ATTIVA ✓" if calibrator.is_active else "non attiva"),
                     expanded=False):
        il, ir = st.columns(2)
        with il:
            cm = st.radio("Tipo patch",
                          ["ColorChecker Classic","Alternativa low-cost"], key="cm")
        with ir:
            is_lc = cm != "ColorChecker Classic"
            opts  = list((ColorCheckerCalibrator.LOWCOST_PATCHES
                          if is_lc else ColorCheckerCalibrator.REFERENCE_PATCHES).keys())
            sp    = st.selectbox("Patch", opts, key="sp")
        pf = st.file_uploader("Foto patch", type=["jpg","jpeg","png","heic"],
                              key="patch_up")
        cb, cr = st.columns([2,1])
        with cb:
            if pf and st.button("CALIBRA"):
                r = calibrator.calibrate(pf, sp, engine, is_lc)
                st.session_state.calib_result = r
                getattr(st, r["color"])(
                    f"**{r['livello']}** — ΔE={r['delta_e']}")
        with cr:
            if calibrator.is_active and st.button("Reset"):
                st.session_state.calibrator = ColorCheckerCalibrator()
                st.rerun()

    st.divider()

    # Modalità input
    modo = st.radio("Modalità input",
                    ["📷 Foto","🔬 Nix manuale"],
                    horizontal=True, key="input_mode")

    # ── Foto ──────────────────────────────────────────────────────────────────
    if modo == "📷 Foto":
        st.info("Luce naturale diffusa · viso 80% del frame · no make up · no filtri")

        files = st.file_uploader("3 immagini (JPG/PNG/HEIC)",
                                 type=["jpg","jpeg","png","heic"],
                                 accept_multiple_files=True, key="skin_up")

        # Salva bytes in session_state non appena arrivano 3 file nuovi.
        # IMPORTANTE: non toccare uploaded_files_bytes se files è vuoto —
        # dopo il click ANALIZZA Streamlit fa rerun con files=[] e svuoterebbe i dati.
        if files and len(files) == 3:
            new_names = sorted(f.name for f in files)
            old_names = sorted(st.session_state.get("uploaded_files_bytes", {}).keys())
            if new_names != old_names:
                resized = {}
                MAX_SAVE = 800
                for f in files:
                    raw = f.read()
                    try:
                        img = Image.open(_io.BytesIO(raw)).convert("RGB")
                        wo, ho = img.size
                        if max(wo, ho) > MAX_SAVE:
                            ratio = MAX_SAVE / max(wo, ho)
                            img = img.resize((int(wo*ratio), int(ho*ratio)), Image.LANCZOS)
                        buf = _io.BytesIO()
                        img.save(buf, format="JPEG", quality=92)
                        resized[f.name] = buf.getvalue()
                        del img, buf
                    except Exception:
                        resized[f.name] = raw
                st.session_state.uploaded_files_bytes = resized
                st.session_state.zone_map = {}
        # Se files è vuoto ma abbiamo già i bytes salvati: non fare nulla.
        # Il rerun post-ANALIZZA non deve cancellare i dati in session_state.

        saved = st.session_state.get("uploaded_files_bytes", {})

        if len(saved) == 3:
            names = list(saved.keys())
            if "zone_map" not in st.session_state:
                st.session_state.zone_map = {}

            if flow_step <= 1:
                st.subheader("Step 1 — Foto caricate")
                st.caption("Quando sei pronto, passa allo Step 2 per assegnare le zone e vedere il colore live.")
                cols_prev = st.columns(3)
                for i, fname in enumerate(names):
                    with cols_prev[i]:
                        st.image(_io.BytesIO(saved[fname]), use_container_width=True)
                if st.button("Continua → Step 2 (seleziona zone)", type="primary"):
                    st.session_state.flow_step = 2
                    st.rerun()
                st.stop()

            st.subheader("Step 2 — Seleziona zone (preview colore live)")
            cols = st.columns(3)
            opz  = ["Seleziona...","Fronte","Guancia","Collo"]
            for i, fname in enumerate(names):
                with cols[i]:
                        st.image(_io.BytesIO(saved[fname]), use_container_width=True)
                        # Recupera la selezione precedente se esiste
                        prev = st.session_state.zone_map.get(fname, "Seleziona...")
                        idx  = opz.index(prev) if prev in opz else 0
                        sel  = st.selectbox(f"Zona {i+1}", opz, index=idx, key=f"z{i}")
                        st.session_state.zone_map[fname] = sel
                        if sel != "Seleziona...":
                            try:
                                pil_img_prev = Image.open(_io.BytesIO(saved[fname])).convert("RGB")
                                img_prev = np.array(pil_img_prev)
                                del pil_img_prev
                                h_prev, w_prev = img_prev.shape[:2]
                                roi_coords_prev = {
                                    "Fronte":  (int(h_prev*.15), int(h_prev*.40), int(w_prev*.30), int(w_prev*.70)),
                                    "Guancia": (int(h_prev*.35), int(h_prev*.65), int(w_prev*.20), int(w_prev*.80)),
                                    "Collo":   (int(h_prev*.65), int(h_prev*.90), int(w_prev*.25), int(w_prev*.75)),
                                }
                                y1,y2,x1,x2 = roi_coords_prev.get(sel,(int(h_prev*.35),int(h_prev*.65),int(w_prev*.35),int(w_prev*.65)))
                                roi_prev = img_prev[y1:y2, x1:x2]
                                if roi_prev.size == 0:
                                    roi_prev = img_prev[int(h_prev*.35):int(h_prev*.65), int(w_prev*.35):int(w_prev*.65)]
                                rgb_prev = np.median(roi_prev.reshape(-1, 3), axis=0)
                                Lp, ap, bp = engine.srgb_to_lab(rgb_prev)
                                Lp, ap, bp = calibrator.apply(Lp, ap, bp)
                                hx = lab_to_hex(float(Lp), float(ap), float(bp))
                                st.markdown(render_color_swatch(hx, size_px=22, shape="circle"), unsafe_allow_html=True)
                                st.caption(f"Skin tone preview live: `{hx}`")
                            except Exception:
                                pass

            zone_map  = st.session_state.zone_map
            zone_vals = [v for v in zone_map.values() if v != "Seleziona..."]

            if len(set(zone_vals)) < 3:
                st.warning("Seleziona 3 zone distinte (Fronte, Guancia, Collo).")
            else:
                st.success("Configurazione valida — pronto per l'analisi.")
                # Form + submit: st.button qui dentro non è affidabile al 2° click (widget annidati).
                with st.form("form_analizza_foto", clear_on_submit=False):
                    do_analizza = st.form_submit_button("ANALIZZA", type="primary")

                if do_analizza:
                    zm = dict(st.session_state.zone_map)
                    zv = [v for v in zm.values() if v != "Seleziona..."]
                    if len(set(zv)) < 3:
                        st.error("Seleziona 3 zone distinte (Fronte, Guancia, Collo) prima di analizzare.")
                    else:
                        res = {}
                        with st.spinner("Elaborazione..."):
                            for fname, zona in zm.items():
                                if zona == "Seleziona...":
                                    continue
                                try:
                                    pil_img = Image.open(_io.BytesIO(saved[fname])).convert("RGB")
                                    MAX_DIM = 800
                                    w_orig, h_orig = pil_img.size
                                    if max(w_orig, h_orig) > MAX_DIM:
                                        ratio   = MAX_DIM / max(w_orig, h_orig)
                                        pil_img = pil_img.resize((int(w_orig*ratio), int(h_orig*ratio)), Image.LANCZOS)
                                    img_array = np.array(pil_img)
                                    del pil_img
                                    h, w_img  = img_array.shape[:2]
                                    roi_coords = {
                                        "Fronte":    (int(h*.15),int(h*.40),int(w_img*.30),int(w_img*.70)),
                                        "Guancia":   (int(h*.35),int(h*.65),int(w_img*.20),int(w_img*.80)),
                                        "Collo": (int(h*.65),int(h*.90),int(w_img*.25),int(w_img*.75)),
                                    }
                                    y1,y2,x1,x2 = roi_coords.get(zona,(int(h*.35),int(h*.65),int(w_img*.35),int(w_img*.65)))
                                    roi       = img_array[y1:y2,x1:x2]
                                    if roi.size == 0:
                                        roi = img_array[int(h*.35):int(h*.65),int(w_img*.35):int(w_img*.65)]
                                    rgb       = np.median(roi.reshape(-1, 3), axis=0)
                                    L,a,b     = engine.srgb_to_lab(rgb)
                                    L,a,b     = calibrator.apply(L,a,b)
                                    res[zona] = {
                                        "L*": float(L), "a*": float(a), "b*": float(b),
                                    }
                                except Exception as e:
                                    st.error(f"Errore su {zona}: {e}")
                                    continue

                        if not res:
                            st.error("Nessun risultato prodotto — verifica che le zone siano selezionate correttamente.")
                        else:
                            st.session_state.skin_data = {
                                "zones":     res,
                                "skin_type": skin_type,
                                "source":    "Foto · " + ("calibrato" if calibrator.is_active else "non calibrato"),
                            }
                            st.session_state.flow_step = 3
                            st.rerun()

    # ── Nix manuale ───────────────────────────────────────────────────────────
    else:
        st.info("Usa illuminante **D65** e osservatore **2°** nell'app Nix.")
        ZONE_LIST = ["Fronte","Guancia","Collo"]
        PESI = {"Fronte":"15%","Guancia":"35%","Collo":"50%"}
        res  = {}
        for z in ZONE_LIST:
            with st.expander(f"📍 {z} · peso {PESI[z]}", expanded=True):
                c1,c2,c3,c4 = st.columns([1,1,1,0.6])
                L = c1.number_input(f"L* {z}", 0.0, 100.0, 60.0, 0.01, key=f"L{z}")
                a = c2.number_input(f"a* {z}",-50.0, 50.0, 10.0, 0.01, key=f"a{z}")
                b = c3.number_input(f"b* {z}",-50.0, 50.0, 18.0, 0.01, key=f"b{z}")
                ita = round(math.atan2(L-50, b if abs(b)>0.001 else 0.001)*(180/math.pi),1)
                hx = lab_to_hex(float(L), float(a), float(b))
                c4.markdown(render_color_swatch(hx, size_px=22, shape="circle"), unsafe_allow_html=True)
                st.caption(f"ITA° live: **{ita}°** → {engine.get_ita_category(ita)} · Hex: `{hx}`")
                res[z] = {"L*":L,"a*":a,"b*":b}

        recap = st.columns(3)
        for i,z in enumerate(ZONE_LIST):
            v = res[z]
            recap[i].metric(z, f"L={v['L*']} a={v['a*']} b={v['b*']}")

        with st.form("form_calc_nix", clear_on_submit=False):
            calc_nix = st.form_submit_button("CALCOLA SKINID™", type="primary")
        if calc_nix:
            st.session_state.skin_data = {
                "zones": {
                    z: {"L*": float(v["L*"]), "a*": float(v["a*"]), "b*": float(v["b*"])}
                    for z, v in res.items()
                },
                "skin_type": skin_type,
                "source": "Nix Spectro L · D65/2° · manuale",
            }
            st.session_state.flow_step = 3
            st.rerun()

    # ── Risultato analisi ─────────────────────────────────────────────────────
    if st.session_state.skin_data:
        sd        = st.session_state.skin_data
        zones     = sd["zones"]
        s_type    = sd["skin_type"]
        s_source  = sd["source"]
        w         = engine.ZONE_WEIGHTS
        tw        = sum(w[z] for z in zones)
        avg_L     = sum(zones[z]["L*"]*w[z] for z in zones)/tw
        avg_a     = sum(zones[z]["a*"]*w[z] for z in zones)/tw
        avg_b     = sum(zones[z]["b*"]*w[z] for z in zones)/tw
        avg_ita   = engine.calculate_ita(avg_L, avg_b)
        categoria = engine.get_ita_category(avg_ita)
        undertone = engine.get_undertone(avg_a, avg_b)
        ri_data   = engine.get_reactivity_index(avg_b, s_type)
        skin_id   = engine.build_skin_id(categoria, avg_L, undertone, avg_a, ri_data["livello"])

        st.divider()
        mostra_skinid_header(skin_id, avg_ita, categoria, undertone,
                             avg_L, ri_data, s_type, s_source)

        render_palette(zones, w)

        st.subheader("Dettaglio per zona")
        st.table({z: {"L*":v["L*"],"a*":v["a*"],"b*":v["b*"],
                      "ITA°":engine.calculate_ita(v["L*"],v["b*"]),
                      "Peso":f"{w.get(z,0)*100:.0f}%"}
                  for z,v in zones.items()})

        st.divider()
        st.subheader("Salvataggio profilo cliente")
        st.caption(
            "I dati sono memorizzati nel database locale dell’applicazione. "
            "Usa questa funzione solo in conformità alla normativa sulla privacy (es. GDPR)."
        )
        save_prompt = st.radio(
            "Vuoi salvare questo SkinID™ con nome, cognome, email e telefono?",
            ("No", "Sì"),
            horizontal=True,
            key="save_skin_prompt",
        )
        want_save = save_prompt == "Sì"
        if want_save:
            # Niente st.columns dentro st.form: con layout annidato i campi non compaiono.
            st.markdown("**Dati da memorizzare nel database**")
            fn = st.text_input("Nome *", key="client_save_fn")
            ln = st.text_input("Cognome *", key="client_save_ln")
            em = st.text_input("Email *", key="client_save_em")
            ph = st.text_input("Telefono *", key="client_save_ph")
            if st.button("Salva nel database", type="primary", key="btn_save_client"):
                fn_t, ln_t = (fn or "").strip(), (ln or "").strip()
                em_t, ph_t = (em or "").strip(), (ph or "").strip()
                if not fn_t or not ln_t or not em_t or not ph_t:
                    st.error("Compila tutti i campi obbligatori.")
                elif not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', em_t):
                    st.error("Inserisci un indirizzo email valido (es. nome@dominio.it).")
                else:
                    db_sv = get_session()
                    try:
                        row = SavedClientSkin(
                            first_name=fn_t,
                            last_name=ln_t,
                            email=em_t.lower(),
                            phone=ph_t,
                            zones_json=json.loads(json.dumps(zones)),
                            skin_type=s_type,
                            source=s_source,
                            skin_id_code=skin_id,
                        )
                        db_sv.add(row)
                        db_sv.flush()
                        new_sid = row.saved_id
                        db_sv.commit()
                        st.session_state.skin_data["saved_client_id"] = new_sid
                        st.success(
                            f"Profilo salvato per **{fn_t} {ln_t}**. "
                            "Puoi richiamarlo da **Matching Prodotti** → SkinID salvati. "
                            "I risultati **TROVA MATCH** saranno collegati a questo cliente."
                        )
                    except Exception as e:
                        db_sv.rollback()
                        st.error(f"Errore durante il salvataggio: {e}")
                    finally:
                        db_sv.close()
        else:
            st.info(
                "Puoi andare su **Matching Prodotti** con il profilo attuale in sessione "
                "(nessun salvataggio anagrafico)."
            )

        col_ok, col_mid, col_reset = st.columns([2.2, 1.2, 1])
        col_ok.success(
            "SkinID™ pronto — passa allo **Step 4** per trovare i match. "
            "Per richiamare questo profilo in un secondo momento, salvalo qui sopra."
        )
        if col_mid.button("Step 4 →", type="primary"):
            st.session_state.flow_step = 4
            st.rerun()
        if col_reset.button("🔄 Nuova analisi"):
            st.session_state.skin_data = None
            st.session_state.uploaded_files_bytes = {}
            st.session_state.zone_map = {}
            st.session_state.flow_step = 1
            # Reset anche del radio salvataggio cliente
            for k in ["save_skin_prompt", "client_save_fn", "client_save_ln",
                      "client_save_em", "client_save_ph"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MATCHING PRODOTTI
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "🎯 Matching Prodotti":
    st.title("🎯 Matching Prodotti")

    db_list = get_session()
    try:
        saved_catalog = list_saved_client_skins(db_list)
    finally:
        db_list.close()

    with st.expander(
        "📂 SkinID™ salvati — richiama un profilo",
        expanded=not bool(st.session_state.skin_data),
    ):
        if not saved_catalog:
            st.caption(
                "Nessun profilo salvato. Dopo **Analisi Pelle**, scegli «Sì» al salvataggio "
                "e compila nome, cognome, email e telefono."
            )
        else:
            st.caption(f"{len(saved_catalog)} profilo/i nel database.")
            for rec in saved_catalog:
                ca, cb, cd = st.columns([4, 1, 1])
                created = rec.created_at.strftime("%d/%m/%Y %H:%M") if rec.created_at else "—"
                ca.markdown(
                    f"**{rec.first_name} {rec.last_name}** · {rec.email} · {rec.phone}  \n"
                    f"`{rec.skin_id_code}` · {created} · {rec.source}"
                )
                if cb.button("Carica", key=f"match_load_{rec.saved_id}", type="primary"):
                    st.session_state.skin_data = skin_data_from_saved_row(rec)
                    st.rerun()
                if cd.button("Elimina", key=f"match_del_{rec.saved_id}"):
                    dbd = get_session()
                    try:
                        dbd.query(SkinProfile).filter(
                            SkinProfile.saved_client_skin_id == rec.saved_id
                        ).update(
                            {SkinProfile.saved_client_skin_id: None},
                            synchronize_session=False,
                        )
                        ent = dbd.query(SavedClientSkin).filter(
                            SavedClientSkin.saved_id == rec.saved_id
                        ).first()
                        if ent:
                            dbd.delete(ent)
                        dbd.commit()
                    finally:
                        dbd.close()
                    cur = st.session_state.get("skin_data")
                    if cur and cur.get("saved_client_id") == rec.saved_id:
                        cur.pop("saved_client_id", None)
                    st.rerun()
            st.caption("**Elimina** rimuove l’anagrafica e lo SkinID salvato; i match già registrati restano storici senza collegamento al cliente.")
            rows_preview = [
                {
                    "Data": rec.created_at.strftime("%Y-%m-%d %H:%M") if rec.created_at else "",
                    "Nome": rec.first_name,
                    "Cognome": rec.last_name,
                    "Email": rec.email,
                    "Telefono": rec.phone,
                    "SkinID™": rec.skin_id_code,
                }
                for rec in saved_catalog
            ]
            st.dataframe(rows_preview, use_container_width=True, hide_index=True)

    if not st.session_state.skin_data:
        st.warning(
            "Nessun profilo pelle attivo. Carica uno **SkinID salvato** sopra oppure esegui "
            "l’analisi nella sezione **Analisi Pelle**."
        )
        st.stop()

    sd        = st.session_state.skin_data
    zones     = sd["zones"]
    s_type    = sd["skin_type"]
    w         = engine.ZONE_WEIGHTS
    tw        = sum(w[z] for z in zones)
    avg_L     = sum(zones[z]["L*"]*w[z] for z in zones)/tw
    avg_a     = sum(zones[z]["a*"]*w[z] for z in zones)/tw
    avg_b     = sum(zones[z]["b*"]*w[z] for z in zones)/tw
    avg_ita   = engine.calculate_ita(avg_L, avg_b)
    categoria = engine.get_ita_category(avg_ita)
    undertone = engine.get_undertone(avg_a, avg_b)
    ri_data   = engine.get_reactivity_index(avg_b, s_type)
    skin_id   = engine.build_skin_id(categoria, avg_L, undertone, avg_a, ri_data["livello"])

    # Recap SkinID attivo
    with st.expander("SkinID™ attivo", expanded=True):
        mostra_skinid_header(skin_id, avg_ita, categoria, undertone,
                             avg_L, ri_data, s_type, sd["source"])
        render_palette(zones, w)
        if sd.get("saved_client_id"):
            st.caption(
                "Profilo collegato a un **cliente salvato**: ogni ricerca **TROVA MATCH** "
                "memorizza i match nel database assieme a questo riferimento anagrafico."
            )

    # Storico trend — mostra solo se il profilo è collegato a un cliente salvato
    if sd.get("saved_client_id"):
        db_hist = get_session()
        try:
            history = db_hist.query(SkinProfile)\
                .filter(SkinProfile.saved_client_skin_id == sd["saved_client_id"])\
                .order_by(SkinProfile.created_at.asc())\
                .all()
        finally:
            db_hist.close()

        if len(history) >= 2:
            with st.expander(f"📈 Storico analisi — {len(history)} misurazioni", expanded=False):
                import pandas as pd
                hist_data = [{
                    "Data":      h.created_at.strftime("%d/%m/%Y") if h.created_at else "—",
                    "L* (luce)": round(h.L_weighted, 1),
                    "a* (ross)": round(h.a_weighted, 1),
                    "b* (cald)": round(h.b_weighted, 1),
                    "ITA°":      round(h.ITA_deg, 1) if h.ITA_deg else "—",
                    "Categoria": h.category or "—",
                } for h in history]
                st.dataframe(hist_data, use_container_width=True, hide_index=True)

                l_vals = [h.L_weighted for h in history]
                b_vals = [h.b_weighted for h in history]
                dates  = [h.created_at.strftime("%d/%m") if h.created_at else str(i) for i,h in enumerate(history)]

                col_l, col_b = st.columns(2)
                with col_l:
                    st.caption("L* nel tempo (luminosità — abbronzatura)")
                    st.line_chart({"L*": l_vals}, height=150)
                with col_b:
                    st.caption("b* nel tempo (calore undertone — stagionalità)")
                    st.line_chart({"b*": b_vals}, height=150)

                delta_l = l_vals[-1] - l_vals[0]
                delta_b = b_vals[-1] - b_vals[0]
                st.caption(
                    f"Variazione totale: L* {delta_l:+.1f} · b* {delta_b:+.1f}  "
                    f"({'più abbronzata/o' if delta_l < -2 else 'più chiara/o' if delta_l > 2 else 'stabile'})"
                )

    st.divider()

    # Filtri matching
    st.subheader("Filtri di ricerca")
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        cat_filter = st.selectbox("Categoria",
            ["foundation","concealer","powder","blush","bronzer"], index=0)
    with f2:
        finish_opts = ["Tutti","matte","satin","dewy","luminous"]
        finish_sel  = st.selectbox("Finish", finish_opts)
        finish_filter = None if finish_sel=="Tutti" else finish_sel
    with f3:
        top_n = st.slider("Risultati", 3, 15, 8)
    with f4:
        tags_available = ["non-comedogenic","vegan","SPF10","SPF15",
                          "long-wear","hydrating","oil-free"]
        tags_req = st.multiselect("Tag formula richiesti", tags_available)

    if st.button("TROVA MATCH", type="primary"):
        skin_dict = {"L": avg_L, "a": avg_a, "b": avg_b}
        db = get_session()
        try:
            results = find_matches(
                db, skin_dict, ri_data["livello"],
                category=cat_filter,
                finish_filter=finish_filter,
                tags_required=tags_req if tags_req else None,
                top_n=top_n
            )
        finally:
            db.close()

        st.divider()
        show_match_results(results)

        # Export PDF
        if PDF_AVAILABLE and results:
            st.divider()
            if st.button("📄 Scarica report PDF", key="btn_pdf"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 16)
                pdf.cell(0, 10, "SkinMatch AI — Report Analisi Pelle", ln=True)
                pdf.set_font("Helvetica", "", 11)
                pdf.cell(0, 8, f"SkinID: {skin_id}", ln=True)
                pdf.cell(0, 8, f"Categoria ITA: {categoria}  |  Undertone: {undertone}  |  ITA: {avg_ita:.1f}", ln=True)
                pdf.cell(0, 8, f"Reactivity Index: {ri_data['livello']}  |  Sorgente: {sd['source']}", ln=True)
                pdf.ln(4)
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Top Match Fondotinta", ln=True)
                pdf.set_font("Helvetica", "", 10)
                for i, m in enumerate(results[:5], 1):
                    p = m["product"]
                    pdf.cell(0, 7,
                        f"{i}. {p.brand} — {p.name}  |  Score: {m['score']:.3f}  |  DE: {m['delta_e']:.2f}  |  {m['badge']}",
                        ln=True)
                pdf.ln(4)
                pdf.set_font("Helvetica", "I", 8)
                pdf.cell(0, 6, "Generato da SkinMatch AI — skinmatch.ai", ln=True)
                pdf_bytes = bytes(pdf.output())
                st.download_button(
                    label="⬇ Download PDF",
                    data=pdf_bytes,
                    file_name=f"SkinMatch_{skin_id}.pdf",
                    mime="application/pdf",
                    key="dl_pdf"
                )
        elif not PDF_AVAILABLE:
            st.caption("_Per abilitare l'export PDF aggiungi `fpdf2` a requirements.txt_")

        # Salva su DB
        db2 = get_session()
        try:
            profile = SkinProfile(
                L_weighted=round(avg_L,2), a_weighted=round(avg_a,2),
                b_weighted=round(avg_b,2), ITA_deg=avg_ita,
                category=categoria, undertone=undertone,
                skin_type=s_type, reactivity_index=ri_data["livello"],
                skin_id_code=skin_id, source=sd["source"],
                saved_client_skin_id=sd.get("saved_client_id"),
            )
            db2.add(profile)
            db2.flush()
            for rank, m in enumerate(results, 1):
                db2.add(Match(
                    profile_id=profile.profile_id,
                    product_id=m["product"].product_id,
                    delta_e_total=m["delta_e"],
                    score=m["score"], rank=rank,
                ))
            db2.commit()
        except Exception as e:
            db2.rollback()
            st.warning(f"Match trovati ma errore nel salvataggio storico: {e}")
        finally:
            db2.close()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DATABASE PRODOTTI
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "📦 Database Prodotti":
    st.title("📦 Database Prodotti")

    db = get_session()
    try:
        products = db.query(Product, ProductColorimetry)\
                     .join(ProductColorimetry)\
                     .order_by(Product.brand, Product.name)\
                     .all()
    finally:
        db.close()

    # Filtri
    f1, f2, f3 = st.columns(3)
    brand_list  = sorted(set(p.brand for p,_ in products))
    brand_filter= f1.selectbox("Brand", ["Tutti"] + brand_list)
    cat_list    = sorted(set(p.category for p,_ in products))
    cat_filter  = f2.selectbox("Categoria", ["Tutti"] + cat_list)
    src_list    = sorted(set(c.measurement_source for _,c in products))
    src_filter  = f3.selectbox("Fonte dati", ["Tutti"] + src_list)

    filtered = [
        (p,c) for p,c in products
        if (brand_filter=="Tutti" or p.brand==brand_filter)
        and (cat_filter=="Tutti" or p.category==cat_filter)
        and (src_filter=="Tutti" or c.measurement_source==src_filter)
    ]

    st.metric("Prodotti nel database", len(filtered))
    st.divider()

    for p, c in filtered:
        src_badge = "✓ Nix" if "nix" in (c.measurement_source or "") else "⚠️ stimato"
        with st.expander(f"{p.brand} · {p.name}  [{p.category}] {src_badge}"):
            cols = st.columns([2, 1, 1, 1, 1, 1])
            cols[0].markdown(f"**{p.line}**  \nSKU: `{p.sku or '—'}`  \n"
                             f"Finish: `{p.finish}` · Coverage: `{p.coverage}`")
            cols[1].metric("L*", c.L_star)
            cols[2].metric("a*", c.a_star)
            cols[3].metric("b*", c.b_star)
            cols[4].metric("ITA°", round(c.ITA_deg or 0, 1))
            cols[5].metric("ΔOssid b*", f"+{c.oxidation_delta_b or 0}")
            st.caption(
                f"Undertone: `{c.undertone_calc}` · "
                f"Fonte: `{c.measurement_source}` · "
                f"Tags: {', '.join(p.formula_tags or []) or '—'}"
            )

            # Bottone elimina
            if st.button("🗑 Elimina", key=f"del_{p.product_id}"):
                db2 = get_session()
                try:
                    pr = db2.query(Product).get(p.product_id)
                    if pr:
                        db2.delete(pr)
                        db2.commit()
                        st.success(f"{p.name} eliminato.")
                        st.rerun()
                finally:
                    db2.close()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — AGGIUNGI PRODOTTO
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "➕ Aggiungi Prodotto":
    st.title("➕ Aggiungi Prodotto")
    st.info(
        "Inserisci un nuovo shade con i valori Lab. "
        "Se hai già misurato con il Nix Spectro L, seleziona `nix_spectro_l` come fonte — "
        "questo abilita il badge 🏆 Certificato nei risultati."
    )

    with st.form("add_product"):
        st.subheader("Anagrafica")
        c1, c2 = st.columns(2)
        brand    = c1.text_input("Brand *", placeholder="es. Kiko Milano")
        line     = c2.text_input("Linea *", placeholder="es. Skin Tone Foundation")
        name     = c1.text_input("Nome shade *", placeholder="es. N30 Light Medium")
        sku      = c2.text_input("SKU", placeholder="es. KM-STF-N30")
        category = c1.selectbox("Categoria *",
                                ["foundation","concealer","powder","blush","bronzer"])
        finish   = c2.selectbox("Finish", ["satin","matte","dewy","luminous"])
        coverage = c1.selectbox("Coverage", ["medium","light","full","buildable"])
        price    = c2.number_input("Prezzo (€)", 0.0, 500.0, 0.0, 0.1)
        url      = st.text_input("URL prodotto / affiliate")
        notes    = st.text_area("Note", height=60)

        st.subheader("Dati colorimetrici")
        st.caption("Inserisci i valori Lab dello shade. Fonte: app Nix (D65/2°) o stima visiva.")
        lc1, lc2, lc3 = st.columns(3)
        L_val = lc1.number_input("L* *", 0.0, 100.0, 65.0, 0.01)
        a_val = lc2.number_input("a*",  -50.0, 50.0, 8.0,  0.01)
        b_val = lc3.number_input("b*",  -50.0, 50.0, 17.0, 0.01)
        oxid  = st.number_input("Ossidazione Δb* (quanto b* aumenta dopo 3h)",
                                -5.0, 10.0, 0.0, 0.1,
                                help="Misura il prodotto fresco e dopo 3h di usura sulla pelle.")
        source = st.selectbox("Fonte dati *",
                              ["manual_estimated","nix_spectro_l","nix_spectro_2","xrite_capsure"])

        st.subheader("Tag formula")
        tag_opts = ["non-comedogenic","vegan","SPF10","SPF15","long-wear","hydrating","oil-free"]
        tags_sel = st.multiselect("Seleziona i tag applicabili", tag_opts)

        # Preview ITA in tempo reale
        ita_prev  = calc_ita(L_val, b_val)
        ut_prev   = calc_undertone(a_val, b_val)
        st.info(f"Preview: ITA° = **{ita_prev}°** · Undertone calcolato: **{ut_prev}**")

        submitted = st.form_submit_button("SALVA PRODOTTO", type="primary")

    if submitted:
        if not brand or not line or not name:
            st.error("Compila i campi obbligatori: Brand, Linea, Nome shade.")
        else:
            pid = str(_uuid.uuid4())
            db  = get_session()
            try:
                prod = Product(
                    product_id=pid, brand=brand, line=line, name=name,
                    sku=sku or None, category=category, finish=finish,
                    coverage=coverage, formula_tags=tags_sel,
                    price_eur=price or None, url=url or None,
                    notes=notes or None, active=True,
                )
                color = ProductColorimetry(
                    product_id=pid,
                    L_star=L_val, a_star=a_val, b_star=b_val,
                    ITA_deg=ita_prev, undertone_calc=ut_prev,
                    oxidation_delta_b=oxid,
                    measurement_source=source,
                )
                db.add(prod)
                db.add(color)
                db.commit()
                st.success(f"✓ **{brand} · {name}** salvato nel database! "
                           f"ITA°={ita_prev} · Undertone={ut_prev}")
                if source == "manual_estimated":
                    st.warning(
                        "Fonte impostata come **stimata** — i match di questo prodotto "
                        "non riceveranno il badge 🏆 Certificato. "
                        "Misura con il Nix Spectro L e aggiorna la fonte per certificarlo."
                    )
            except Exception as e:
                db.rollback()
                st.error(f"Errore salvataggio: {e}")
            finally:
                db.close()