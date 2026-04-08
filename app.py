"""
SkinMatch AI — app_v7.py
Pipeline completa: analisi pelle + database prodotti + matching colorimetrico.
"""

import math
import streamlit as st
import numpy as np
from PIL import Image
import pillow_heif
from database import (
    init_db, seed_demo_data, get_session,
    Product, ProductColorimetry, SkinProfile, Match,
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
    ZONE_WEIGHTS = {"Mandibola": 0.50, "Guancia": 0.35, "Fronte": 0.15}

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

    def process_image(self, f, zona):
        img = np.array(Image.open(f).convert("RGB"))
        h,w = img.shape[:2]
        roi_map = {
            "Fronte":   (int(h*.15),int(h*.40),int(w*.30),int(w*.70)),
            "Guancia":  (int(h*.35),int(h*.65),int(w*.20),int(w*.80)),
            "Mandibola":(int(h*.65),int(h*.90),int(w*.25),int(w*.75)),
        }
        y1,y2,x1,x2 = roi_map.get(zona,(int(h*.35),int(h*.65),int(w*.35),int(w*.65)))
        roi = img[y1:y2,x1:x2]
        if roi.size==0: roi = img[int(h*.35):int(h*.65),int(w*.35):int(w*.65)]
        return np.median(roi.reshape(-1,3), axis=0)

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

def show_match_results(matches: list):
    if not matches:
        st.info("Nessun prodotto trovato con i filtri selezionati.")
        return

    st.subheader(f"Top {len(matches)} prodotti abbinati")

    for i, m in enumerate(matches):
        p  = m["product"]
        c  = m["colorimetry"]
        bc = BADGE_COLOR.get(m["badge"], "info")

        with st.container():
            cols = st.columns([0.5, 3, 1.5, 1.5, 1.5])
            cols[0].markdown(f"**#{i+1}**")
            cols[1].markdown(f"**{p.brand}**  \n{p.line} · *{p.name}*  \n"
                             f"`{p.finish}` · `{p.coverage}` · €{p.price_eur or '—'}")
            cols[2].metric("Score", f"{m['score']:.3f}")
            cols[3].metric("ΔE", f"{m['delta_e']:.2f}")
            getattr(cols[4], bc)(m["badge"])

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


# ══════════════════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="SkinMatch AI", page_icon="🧬", layout="wide")

# Session state
for k,v in [("engine",SkinIDEngine()),("calibrator",ColorCheckerCalibrator()),
            ("calib_result",None),("skin_data",None),
            ("uploaded_files_bytes",{}),("zone_map",{}),("show_result",False)]:
    if k not in st.session_state: st.session_state[k] = v

engine     = st.session_state.engine
calibrator = st.session_state.calibrator

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=SkinMatch+AI", use_container_width=True)
    st.header("Parametri globali")
    skin_type = st.radio("Tipo di pelle", ["Oleosa","Mista","Secca / Normale"], index=1)
    st.divider()
    nav = st.radio("Navigazione", [
        "🔬 Analisi Pelle",
        "🎯 Matching Prodotti",
        "📦 Database Prodotti",
        "➕ Aggiungi Prodotto",
    ])
    st.divider()
    st.caption("CIE 15:2004 · Del Bino et al. (2006)")


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

        import io as _io
        from PIL import Image as _PIL

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
                # Solo se sono file davvero nuovi — salva e resetta zone
                st.session_state.uploaded_files_bytes = {f.name: f.read() for f in files}
                st.session_state.zone_map = {}
        # Se files è vuoto ma abbiamo già i bytes salvati: non fare nulla.
        # Il rerun post-ANALIZZA non deve cancellare i dati in session_state.

        saved = st.session_state.get("uploaded_files_bytes", {})

        if len(saved) == 3:
            names = list(saved.keys())
            if "zone_map" not in st.session_state:
                st.session_state.zone_map = {}

            cols = st.columns(3)
            opz  = ["Seleziona...","Fronte","Guancia","Mandibola"]
            for i, fname in enumerate(names):
                with cols[i]:
                        st.image(_io.BytesIO(saved[fname]), use_container_width=True)
                        # Recupera la selezione precedente se esiste
                        prev = st.session_state.zone_map.get(fname, "Seleziona...")
                        idx  = opz.index(prev) if prev in opz else 0
                        sel  = st.selectbox(f"Zona {i+1}", opz, index=idx, key=f"z{i}")
                        st.session_state.zone_map[fname] = sel

            zone_map  = st.session_state.zone_map
            zone_vals = [v for v in zone_map.values() if v != "Seleziona..."]

            if len(set(zone_vals)) < 3:
                st.warning("Seleziona 3 zone distinte (Fronte, Guancia, Mandibola).")
            else:
                st.success("Configurazione valida — pronto per l'analisi.")
                if st.button("ANALIZZA", type="primary", key="btn_analizza"):
                        res = {}
                        errori = []
                        with st.spinner("Elaborazione..."):
                            for fname, zona in zone_map.items():
                                if zona == "Seleziona...":
                                    continue
                                try:
                                    pil_img   = _PIL.open(_io.BytesIO(saved[fname])).convert("RGB")
                                    img_array = np.array(pil_img)
                                    h, w_img  = img_array.shape[:2]
                                    roi_coords = {
                                        "Fronte":    (int(h*.15),int(h*.40),int(w_img*.30),int(w_img*.70)),
                                        "Guancia":   (int(h*.35),int(h*.65),int(w_img*.20),int(w_img*.80)),
                                        "Mandibola": (int(h*.65),int(h*.90),int(w_img*.25),int(w_img*.75)),
                                    }
                                    y1,y2,x1,x2 = roi_coords.get(zona,(int(h*.35),int(h*.65),int(w_img*.35),int(w_img*.65)))
                                    roi       = img_array[y1:y2,x1:x2]
                                    if roi.size == 0:
                                        roi = img_array[int(h*.35):int(h*.65),int(w_img*.35):int(w_img*.65)]
                                    rgb       = np.median(roi.reshape(-1, 3), axis=0)
                                    L,a,b     = engine.srgb_to_lab(rgb)
                                    L,a,b     = calibrator.apply(L,a,b)
                                    res[zona] = {"L*":L, "a*":a, "b*":b}
                                except Exception as e:
                                    errori.append(f"{zona}: {e}")

                        if errori:
                            st.error("Errori durante elaborazione: " + " | ".join(errori))
                        elif not res:
                            st.error("Nessun risultato prodotto — zone non mappate correttamente.")
                        else:
                            # Salva risultato e forza visualizzazione immediata
                            st.session_state.skin_data = {
                                "zones":     res,
                                "skin_type": skin_type,
                                "source":    "Foto · " + ("calibrato" if calibrator.is_active else "non calibrato"),
                                "analisi_ok": True,
                            }
                            st.session_state.show_result = True
                            st.success(f"Analisi completata — {len(res)} zone elaborate.")
                            st.rerun()

    # ── Nix manuale ───────────────────────────────────────────────────────────
    else:
        st.info("Usa illuminante **D65** e osservatore **2°** nell'app Nix.")
        ZONE_LIST = ["Fronte","Guancia","Mandibola"]
        PESI = {"Fronte":"15%","Guancia":"35%","Mandibola":"50%"}
        res  = {}
        for z in ZONE_LIST:
            with st.expander(f"📍 {z} · peso {PESI[z]}", expanded=True):
                c1,c2,c3 = st.columns(3)
                L = c1.number_input(f"L* {z}", 0.0, 100.0, 60.0, 0.01, key=f"L{z}")
                a = c2.number_input(f"a* {z}",-50.0, 50.0, 10.0, 0.01, key=f"a{z}")
                b = c3.number_input(f"b* {z}",-50.0, 50.0, 18.0, 0.01, key=f"b{z}")
                ita = round(math.atan2(L-50, b if abs(b)>0.001 else 0.001)*(180/math.pi),1)
                st.caption(f"ITA° live: **{ita}°** → {engine.get_ita_category(ita)}")
                res[z] = {"L*":L,"a*":a,"b*":b}

        recap = st.columns(3)
        for i,z in enumerate(ZONE_LIST):
            v = res[z]
            recap[i].metric(z, f"L={v['L*']} a={v['a*']} b={v['b*']}")

        if st.button("CALCOLA SKINID™", type="primary"):
            st.session_state.skin_data = {
                "zones": res, "skin_type": skin_type,
                "source": "Nix Spectro L · D65/2° · manuale"
            }

    # ── Risultato analisi ─────────────────────────────────────────────────────
    # Mostrato sempre se skin_data esiste, indipendentemente dalla modalità input.
    # show_result viene impostato a True subito dopo l'analisi per forzare la visibilità.
    if st.session_state.get("show_result"):
        st.session_state.show_result = False  # reset flag, risultato già in skin_data

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

        st.subheader("Dettaglio per zona")
        st.table({z: {"L*":v["L*"],"a*":v["a*"],"b*":v["b*"],
                      "ITA°":engine.calculate_ita(v["L*"],v["b*"]),
                      "Peso":f"{w.get(z,0)*100:.0f}%"}
                  for z,v in zones.items()})

        col_ok, col_reset = st.columns([3,1])
        col_ok.success("SkinID™ salvato — vai su **Matching Prodotti** per trovare i fondotinta.")
        if col_reset.button("🔄 Nuova analisi"):
            st.session_state.skin_data = None
            st.session_state.uploaded_files_bytes = {}
            st.session_state.zone_map = {}
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MATCHING PRODOTTI
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "🎯 Matching Prodotti":
    st.title("🎯 Matching Prodotti")

    if not st.session_state.skin_data:
        st.warning("Esegui prima l'analisi pelle nella sezione **Analisi Pelle**.")
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

        # Salva su DB
        db2 = get_session()
        try:
            profile = SkinProfile(
                L_weighted=round(avg_L,2), a_weighted=round(avg_a,2),
                b_weighted=round(avg_b,2), ITA_deg=avg_ita,
                category=categoria, undertone=undertone,
                skin_type=s_type, reactivity_index=ri_data["livello"],
                skin_id_code=skin_id, source=sd["source"]
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
            import uuid as _uuid
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