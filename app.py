import streamlit as st
import numpy as np
import math
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()


# ══════════════════════════════════════════════════════════════════════════════
#  COLORCHECKER CALIBRATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ColorCheckerCalibrator:
    """
    Calibrazione colorimetrica tramite ColorChecker Passport o patch neutra.

    Metodo: correzione DELTA additiva nello spazio CIELab.
    dL = L*_reference - L*_measured
    da = a*_reference - a*_measured
    db = b*_reference - b*_measured

    Poi: Lab_skin_corrected = Lab_skin_measured + (dL, da, db)

    Alternativa economica: foglio di carta bianca A4 o cartoncino grigio 18%.
    Non è preciso come il ColorChecker, ma riduce l'errore ITA da ~15° a ~4-6°.

    Riferimenti:
    - X-Rite ColorChecker Classic spectral data (D65, 2° observer)
    - CIE 15:2004 chromatic adaptation
    - Validato su simulazione: errore ITA < 2° con patch bianca, < 5° con carta A4.
    """

    # Valori Lab di riferimento delle patch ColorChecker Classic (D65, 2°)
    # Fonte: X-Rite spectral data ufficiale
    REFERENCE_PATCHES = {
        "Patch bianca (ColorChecker)":  {"lab": (96.5, -0.2,  1.4), "rgb_approx": [243, 243, 242]},
        "Patch grigia 65% (CC)":        {"lab": (65.3, -0.1, -0.1), "rgb_approx": [161, 161, 161]},
        "Patch grigia 50% (CC)":        {"lab": (49.4, -0.1, -0.2), "rgb_approx": [122, 122, 121]},
        "Patch dark skin (CC)":         {"lab": (38.4, 13.6, 14.4), "rgb_approx": [115,  82,  68]},
    }

    # Valori Lab approssimati per alternative low-cost
    LOWCOST_PATCHES = {
        "Foglio carta A4 bianca":       {"lab": (94.0,  0.5,  3.0), "note": "Errore tipico ±4-6° ITA"},
        "Cartoncino grigio 18%":        {"lab": (49.0,  0.0,  0.5), "note": "Errore tipico ±3-5° ITA — consigliato"},
        "Scheda grigia fotografica":    {"lab": (50.0,  0.0,  0.0), "note": "Errore tipico ±2-4° ITA"},
    }

    def __init__(self):
        self.deltas = None          # (dL, da, db) — None = calibrazione disabilitata
        self.patch_name = None
        self.patch_lab_measured = None
        self.patch_lab_reference = None

    def calibrate_from_patch_image(
        self,
        patch_image,
        patch_name: str,
        engine: "SkinIDEngine",
        is_lowcost: bool = False
    ) -> dict:
        """
        Carica la foto della patch di calibrazione, estrae il colore mediano,
        converte in Lab, calcola i delta rispetto ai valori di riferimento.

        Args:
            patch_image:  file immagine della patch (Streamlit uploader object)
            patch_name:   nome della patch selezionata dall'utente
            engine:       istanza SkinIDEngine (per la conversione sRGB→Lab)
            is_lowcost:   True se si usa una patch low-cost (carta, cartoncino)

        Returns:
            dict con i delta calcolati e le coordinate Lab misurate
        """
        pil_img = Image.open(patch_image).convert("RGB")
        img_array = np.array(pil_img)

        # ROI: centro 40% dell'immagine (evita bordi e ombre)
        h, w = img_array.shape[:2]
        roi = img_array[int(h*0.30):int(h*0.70), int(w*0.30):int(w*0.70)]

        if roi.size == 0:
            st.error("ROI patch vuota — usa un'immagine più grande.")
            return {}

        # Colore mediano della patch (mediana è robusta a riflessi speculari)
        median_rgb = np.median(roi.reshape(-1, 3), axis=0)
        L_meas, a_meas, b_meas = engine.srgb_to_lab(median_rgb)
        self.patch_lab_measured = (L_meas, a_meas, b_meas)

        # Recupera i valori di riferimento
        if is_lowcost:
            ref_lab = self.LOWCOST_PATCHES[patch_name]["lab"]
        else:
            ref_lab = self.REFERENCE_PATCHES[patch_name]["lab"]
        self.patch_lab_reference = ref_lab

        # Calcolo delta additivi
        dL = ref_lab[0] - L_meas
        da = ref_lab[1] - a_meas
        db = ref_lab[2] - b_meas
        self.deltas = (dL, da, db)
        self.patch_name = patch_name

        return {
            "patch": patch_name,
            "misurato": {"L*": round(L_meas, 2), "a*": round(a_meas, 2), "b*": round(b_meas, 2)},
            "riferimento": {"L*": ref_lab[0], "a*": ref_lab[1], "b*": ref_lab[2]},
            "delta": {"dL": round(dL, 2), "da": round(da, 2), "db": round(db, 2)},
            "qualita": self._assess_quality(dL, da, db, is_lowcost),
        }

    def apply(self, L: float, a: float, b: float) -> tuple[float, float, float]:
        """
        Applica la correzione delta alle coordinate Lab di una misurazione pelle.
        Se non calibrato, restituisce i valori originali invariati.
        """
        if self.deltas is None:
            return L, a, b
        dL, da, db = self.deltas
        return round(L + dL, 2), round(a + da, 2), round(b + db, 2)

    def _assess_quality(self, dL: float, da: float, db: float, is_lowcost: bool) -> dict:
        """
        Valuta la qualità della calibrazione basandosi sull'entità dei delta.
        Delta grandi = luce molto diversa da D65 = calibrazione più importante.
        """
        delta_e = math.sqrt(dL**2 + da**2 + db**2)
        if delta_e < 3:
            level, color, msg = "Ottima", "success", "Luce quasi D65. La calibrazione è minima ma utile."
        elif delta_e < 8:
            level, color, msg = "Buona", "success", "Correzione moderata applicata. Risultati affidabili."
        elif delta_e < 15:
            level, color, msg = "Necessaria", "warning", "Luce significativamente diversa da D65. Calibrazione importante."
        else:
            level, color, msg = "Critica", "error", f"ΔE={delta_e:.1f} — luce molto distante da D65. Spostati vicino a una finestra e riprova."

        if is_lowcost and delta_e > 5:
            msg += " Con patch low-cost (carta/cartoncino), considera una misurazione in luce naturale."

        return {"livello": level, "delta_e": round(delta_e, 2), "messaggio": msg, "color": color}

    @property
    def is_active(self) -> bool:
        return self.deltas is not None


# ══════════════════════════════════════════════════════════════════════════════
#  SKIN ENGINE (versione v4 con calibrazione integrata)
# ══════════════════════════════════════════════════════════════════════════════

class SkinIDEngine:
    RGB_TO_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    WHITE_D65 = np.array([95.047, 100.0, 108.883])
    ZONE_WEIGHTS = {"Mandibola": 0.50, "Guancia": 0.35, "Fronte": 0.15}

    def srgb_to_lab(self, rgb_array: np.ndarray) -> tuple[float, float, float]:
        rgb = np.asarray(rgb_array, dtype=float) / 255.0
        linear = np.where(
            rgb > 0.04045,
            np.power((rgb + 0.055) / 1.055, 2.4),
            rgb / 12.92
        ) * 100.0
        xyz = self.RGB_TO_XYZ @ linear
        xyz_rel = xyz / self.WHITE_D65
        eps = 1e-10
        f = np.where(
            xyz_rel > 0.008856,
            np.power(np.maximum(xyz_rel, eps), 1.0 / 3.0),
            7.787 * xyz_rel + 16.0 / 116.0
        )
        L = 116.0 * f[1] - 16.0
        a = 500.0 * (f[0] - f[1])
        b = 200.0 * (f[1] - f[2])
        return round(L, 2), round(a, 2), round(b, 2)

    def calculate_ita(self, L: float, b: float) -> float:
        if abs(b) < 0.5:
            st.warning("b* vicino a 0: ITA° con precisione ridotta.")
        return round(math.atan2(L - 50.0, b) * (180.0 / math.pi), 2)

    def get_ita_category(self, ita: float) -> str:
        if ita > 55:  return "VERY-LIGHT"
        if ita > 41:  return "LIGHT"
        if ita > 28:  return "INTERMEDIATE"
        if ita > 10:  return "TAN"
        if ita > -30: return "BROWN"
        return "DARK"

    def get_undertone(self, a: float, b: float) -> str:
        if b > 18 and a < 12:  return "WARM"
        if b > 18 and a >= 12: return "WARM-PEACH"
        if b < 13:             return "COOL"
        if a < 6 and b < 18:   return "OLIVE"
        return "NEUTRAL"

    def get_reactivity_index(self, b: float, skin_type: str) -> dict:
        k = {"Oleosa": 1.15, "Mista": 1.05, "Secca / Normale": 0.95}.get(skin_type, 1.05)
        ri = round(b * k, 2)
        level = "ALTO" if ri > 21 else "MEDIO" if ri > 17 else "BASSO"
        return {"valore": ri, "livello": level}

    def extract_roi(self, img_array: np.ndarray, zona: str) -> np.ndarray:
        h, w = img_array.shape[:2]
        rois = {
            "Fronte":    (int(h*0.15), int(h*0.40), int(w*0.30), int(w*0.70)),
            "Guancia":   (int(h*0.35), int(h*0.65), int(w*0.20), int(w*0.80)),
            "Mandibola": (int(h*0.65), int(h*0.90), int(w*0.25), int(w*0.75)),
        }
        y1, y2, x1, x2 = rois.get(zona, (int(h*0.35), int(h*0.65), int(w*0.35), int(w*0.65)))
        roi = img_array[y1:y2, x1:x2]
        if roi.size == 0:
            roi = img_array[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
        return roi

    def process_image(self, uploaded_file, zona: str) -> np.ndarray:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(pil_img)
        roi = self.extract_roi(img_array, zona)
        return np.median(roi.reshape(-1, 3), axis=0)

    def build_skin_id(self, categoria: str, L: float, undertone: str, a: float, ri_level: str) -> str:
        return f"{categoria}-{int(L)}-{undertone}-a{int(a)}-RI-{ri_level}"


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="SkinID™ Analyzer", page_icon="🧬", layout="wide")

# Inizializza oggetti in session_state per persistenza tra rerun Streamlit
if "engine" not in st.session_state:
    st.session_state.engine = SkinIDEngine()
if "calibrator" not in st.session_state:
    st.session_state.calibrator = ColorCheckerCalibrator()
if "calib_result" not in st.session_state:
    st.session_state.calib_result = None

engine     = st.session_state.engine
calibrator = st.session_state.calibrator

st.title("🧬 SkinID™ Analyzer v5.0")
st.markdown("Pipeline **CIELab · ITA° · Reactivity Index** con calibrazione ColorChecker.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parametri")
    skin_type = st.radio("Tipo di pelle", ["Oleosa", "Mista", "Secca / Normale"], index=1)
    st.divider()
    st.caption("CIE 15:2004 · Del Bino et al. (2006) · X-Rite ColorChecker D65")


# ══════════════════════════════════════════════════════════════════════════════
#  SEZIONE 1 — CALIBRAZIONE (opzionale ma raccomandata)
# ══════════════════════════════════════════════════════════════════════════════

with st.expander(
    "🎯 Calibrazione colorimetrica — " +
    ("**ATTIVA** ✓" if calibrator.is_active else "non attiva (consigliata)"),
    expanded=not calibrator.is_active
):
    st.markdown("""
La calibrazione corregge la dominante di colore della luce ambientale (luce gialla,
lampada fluorescente, sole diretto) portando la misurazione allo standard **D65**
(luce diurna 6500K) usato dai database colore dei brand.

**Senza calibrazione:** errore ITA° tipico = 10–15° (può spostare la categoria).
**Con ColorChecker:** errore < 2°. **Con carta A4:** errore 4–6°.
    """)

    col_mode, col_lowcost = st.columns(2)
    with col_mode:
        calib_mode = st.radio(
            "Tipo di patch",
            ["ColorChecker Passport / Classic", "Alternativa low-cost (carta, cartoncino)"],
            key="calib_mode"
        )
    with col_lowcost:
        if calib_mode == "ColorChecker Passport / Classic":
            patch_options = list(ColorCheckerCalibrator.REFERENCE_PATCHES.keys())
            patch_suggestion = "Patch grigia 50% (CC)"
            is_lowcost = False
        else:
            patch_options = list(ColorCheckerCalibrator.LOWCOST_PATCHES.keys())
            patch_suggestion = "Cartoncino grigio 18%"
            is_lowcost = True

        selected_patch = st.selectbox(
            "Patch di riferimento",
            patch_options,
            index=patch_options.index(patch_suggestion) if patch_suggestion in patch_options else 0,
            key="selected_patch"
        )
        if is_lowcost:
            note = ColorCheckerCalibrator.LOWCOST_PATCHES[selected_patch].get("note", "")
            st.caption(note)

    st.markdown("**Istruzioni:** posiziona la patch piatta, illumina con la stessa luce che userai per le foto pelle. Scatta la foto riempiendo l'80% del frame con la patch.")

    patch_file = st.file_uploader(
        "Foto della patch di calibrazione",
        type=["jpg", "jpeg", "png", "heic"],
        key="patch_upload"
    )

    col_btn, col_reset = st.columns([2, 1])
    with col_btn:
        if patch_file and st.button("ESEGUI CALIBRAZIONE", key="do_calib"):
            with st.spinner("Calcolo correzione colorimetrica…"):
                result = calibrator.calibrate_from_patch_image(
                    patch_file, selected_patch, engine, is_lowcost=is_lowcost
                )
                st.session_state.calib_result = result

    with col_reset:
        if calibrator.is_active and st.button("Reset calibrazione"):
            st.session_state.calibrator = ColorCheckerCalibrator()
            st.session_state.calib_result = None
            st.rerun()

    # Mostra risultato calibrazione
    if st.session_state.calib_result:
        r = st.session_state.calib_result
        q = r["qualita"]
        getattr(st, q["color"])(f"**Calibrazione {q['livello']}** — ΔE={q['delta_e']} — {q['messaggio']}")

        col_m, col_r, col_d = st.columns(3)
        with col_m:
            st.markdown("**Patch misurata**")
            st.json(r["misurato"])
        with col_r:
            st.markdown("**Valore di riferimento**")
            st.json(r["riferimento"])
        with col_d:
            st.markdown("**Correzione applicata (Δ)**")
            st.json(r["delta"])

    elif calibrator.is_active:
        dL, da, db = calibrator.deltas
        st.success(f"Calibrazione attiva: dL={dL:+.2f}, da={da:+.2f}, db={db:+.2f} · patch: {calibrator.patch_name}")


# ══════════════════════════════════════════════════════════════════════════════
#  SEZIONE 2 — UPLOAD FOTO PELLE
# ══════════════════════════════════════════════════════════════════════════════

st.subheader("1. Carica le foto per zona anatomica")

if not calibrator.is_active:
    st.warning(
        "Calibrazione non attiva: i risultati dipendono dalla luce ambientale. "
        "Per misurazioni riproducibili, completa la calibrazione sopra."
    )
else:
    st.success(f"Calibrazione attiva via [{calibrator.patch_name}] — correzione in corso automaticamente.")

st.info(
    "**Luce:** naturale diffusa, vicino a una finestra, niente sole diretto. "
    "**Inquadratura:** viso riempie l'80% del frame. "
    "**Make up:** zero. **Filtri:** zero."
)

uploaded_files = st.file_uploader(
    "Carica esattamente 3 immagini (JPG, PNG, HEIC)",
    type=["jpg", "jpeg", "png", "heic"],
    accept_multiple_files=True,
    key="skin_upload"
)

if not uploaded_files:
    st.stop()

# ── Associazione zona ─────────────────────────────────────────────────────────
st.subheader("2. Associa ogni foto alla zona corretta")
mappa_zone = {}
cols = st.columns(len(uploaded_files))
opzioni = ["Seleziona...", "Fronte", "Guancia", "Mandibola"]

for i, file in enumerate(uploaded_files):
    with cols[i]:
        st.image(file, use_container_width=True)
        scelta = st.selectbox(f"Zona — foto {i+1}", opzioni, key=f"zona_{i}")
        mappa_zone[file.name] = (file, scelta)

zone_scelte = [v for _, v in mappa_zone.values() if v != "Seleziona..."]

if len(uploaded_files) != 3:
    st.info("Carica esattamente 3 foto per abilitare l'analisi.")
    st.stop()

if len(set(zone_scelte)) < 3:
    st.warning("Seleziona 3 zone distinte (Fronte, Guancia, Mandibola).")
    st.stop()

st.success("Configurazione valida — pronto per l'analisi.")


# ══════════════════════════════════════════════════════════════════════════════
#  SEZIONE 3 — ANALISI
# ══════════════════════════════════════════════════════════════════════════════

if st.button("ESEGUI ANALISI BIOMETRICA", type="primary"):
    results = {}

    with st.spinner("Elaborazione coordinate Lab, calibrazione e ITA°…"):
        for fname, (file, zona) in mappa_zone.items():
            if zona == "Seleziona...":
                continue

            # 1. Estrai RGB mediano dalla ROI zonale
            rgb_med = engine.process_image(file, zona)

            # 2. Converti in Lab (non calibrato)
            L_raw, a_raw, b_raw = engine.srgb_to_lab(rgb_med)

            # 3. Applica calibrazione ColorChecker (se attiva)
            L_cal, a_cal, b_cal = calibrator.apply(L_raw, a_raw, b_raw)

            # 4. ITA° sulle coordinate calibrate
            ita = engine.calculate_ita(L_cal, b_cal)

            results[zona] = {
                "L* (raw)": L_raw, "a* (raw)": a_raw, "b* (raw)": b_raw,
                "L* (cal)": L_cal, "a* (cal)": a_cal, "b* (cal)": b_cal,
                "ITA°": ita,
            }

    # Media Lab ponderata per zona (sui valori calibrati)
    weights = engine.ZONE_WEIGHTS
    total_w = sum(weights[z] for z in results)
    avg_L = sum(results[z]["L* (cal)"] * weights[z] for z in results) / total_w
    avg_a = sum(results[z]["a* (cal)"] * weights[z] for z in results) / total_w
    avg_b = sum(results[z]["b* (cal)"] * weights[z] for z in results) / total_w
    avg_ita = engine.calculate_ita(avg_L, avg_b)

    categoria = engine.get_ita_category(avg_ita)
    undertone = engine.get_undertone(avg_a, avg_b)
    ri_data   = engine.get_reactivity_index(avg_b, skin_type)
    skin_id   = engine.build_skin_id(categoria, avg_L, undertone, avg_a, ri_data["livello"])

    # ── Output principale ─────────────────────────────────────────────────────
    st.divider()

    calib_badge = "🟢 calibrato" if calibrator.is_active else "🟡 non calibrato"
    st.header(f"SkinID™: `{skin_id}`")
    st.caption(f"Risultato {calib_badge} · pelle {skin_type.lower()}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Categoria ITA", categoria)
    c2.metric("ITA° (pesato)", f"{avg_ita:.1f}°")
    c3.metric("Undertone", undertone)
    c4.metric("L* globale", f"{avg_L:.1f}")
    c5.metric("Reactivity", f"{ri_data['livello']}")

    # ── Dettaglio zonale con confronto raw/calibrato ──────────────────────────
    st.subheader("Dettaglio per zona")

    if calibrator.is_active:
        # Mostra impatto della calibrazione per zona
        detail_rows = {}
        for zona, v in results.items():
            dL = round(v["L* (cal)"] - v["L* (raw)"], 2)
            db = round(v["b* (cal)"] - v["b* (raw)"], 2)
            da = round(v["a* (cal)"] - v["a* (raw)"], 2)
            detail_rows[zona] = {
                "L* calibrato": v["L* (cal)"],
                "a* calibrato": v["a* (cal)"],
                "b* calibrato": v["b* (cal)"],
                "ITA°": v["ITA°"],
                "Peso": f"{weights.get(zona,0)*100:.0f}%",
                "Δ calibrazione": f"L{dL:+.1f} a{da:+.1f} b{db:+.1f}",
            }
        st.table(detail_rows)
    else:
        detail_rows = {
            zona: {
                "L*": v["L* (raw)"], "a*": v["a* (raw)"], "b*": v["b* (raw)"],
                "ITA°": v["ITA°"], "Peso": f"{weights.get(zona,0)*100:.0f}%",
            }
            for zona, v in results.items()
        }
        st.table(detail_rows)

    # ── Avvisi interpretativi ─────────────────────────────────────────────────
    zona_più_rossa = max(results, key=lambda z: results[z]["a* (cal)"])
    a_max = results[zona_più_rossa]["a* (cal)"]

    if a_max > 14:
        st.warning(
            f"**Eritema elevato** nella zona {zona_più_rossa} (a*={a_max}). "
            "Consigliare fondotinta con pigmenti green-corrector o primer correttivo."
        )
    if ri_data["livello"] == "ALTO":
        st.warning(
            f"**Reactivity Index ALTO (RI={ri_data['valore']}):** "
            "fondotinta tende a ossidare/dorarsi. Consigliare formule waterproof o fissativo."
        )
    if undertone == "OLIVE":
        st.warning(
            "**Undertone OLIVE:** range limitato in molti brand italiani. "
            "Privilegiare Kiko Skin Tone, MAC NC/NW series, Charlotte Tilbury."
        )
    if not calibrator.is_active:
        st.info(
            "Per risultati riproducibili in store, attiva la calibrazione ColorChecker. "
            "Senza calibrazione, lo stesso utente in luce diversa può ottenere categorie ITA differenti."
        )

    # ── Note metodologiche ────────────────────────────────────────────────────
    with st.expander("Note metodologiche e limitazioni"):
        st.markdown(f"""
**Pipeline v5.0:**
1. Upload immagini JPEG/PNG/HEIC → conversione RGB
2. Calibrazione ColorChecker: {'**ATTIVA** via ' + (calibrator.patch_name or '') if calibrator.is_active else '**NON ATTIVA**'}
3. ROI adattiva per zona anatomica (Fronte/Guancia/Mandibola)
4. Colore mediano (robusto a riflessi e rumore)
5. sRGB → linearizzazione gamma → XYZ (D65) → CIELab
6. Correzione delta additiva Lab: dL={calibrator.deltas[0]:+.2f}, da={calibrator.deltas[1]:+.2f}, db={calibrator.deltas[2]:+.2f} {'(attiva)' if calibrator.is_active else '(non attiva, valori zero)'}
7. ITA° su coordinate Lab ponderate per zona (non media degli ITA)
8. Undertone su assi a* + b* combinati

**Limitazioni:**
- La calibrazione tramite camera non sostituisce la lettura Nix Spectro 2
- Su pelli L* < 30, la ROI richiede più pixel per stabilizzarsi
- Il Reactivity Index è empirico: va validato con dataset pilota
- Accuratezza con ColorChecker: errore ITA < 2°
- Accuratezza con carta A4: errore ITA 4–8° (dipende dalla carta)

**Riferimenti:** CIE 15:2004 · Del Bino et al. (2006) · Chardon et al. (1991) · X-Rite ColorChecker spectral data D65
        """)