import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import pillow_heif

# Registra il supporto per i file HEIC di Apple
pillow_heif.register_heif_opener()

class SkinIDEngine:
    def __init__(self):
        # Matrice di conversione sRGB -> XYZ (Illuminante D65)
        self.rgb_to_xyz_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        # Punto bianco standard D65
        self.white_point = np.array([95.047, 100.0, 108.883])

    def _srgb_to_lab(self, rgb_v):
        """Converte RGB lineare in spazio CIELab."""
        rgb = np.array(rgb_v) / 255.0
        mask = rgb > 0.04045
        rgb[mask] = np.power((rgb[mask] + 0.055) / 1.055, 2.4)
        rgb[~mask] = rgb[~mask] / 12.92
        rgb *= 100

        xyz = np.dot(self.rgb_to_xyz_matrix, rgb)
        xyz_rel = xyz / self.white_point
        mask = xyz_rel > 0.008856
        f_xyz = np.where(mask, np.power(xyz_rel, 1/3), (7.787 * xyz_rel) + (16/116))
        
        L = (116 * f_xyz[1]) - 16
        a = 500 * (f_xyz[0] - f_xyz[1])
        b = 200 * (f_xyz[1] - f_xyz[2])
        return round(L, 2), round(a, 2), round(b, 2)

    def calculate_ita(self, L, b):
        """Calcola l'Individual Typology Angle (ITA°)."""
        ita_rad = math.atan2((L - 50), b)
        return round(ita_rad * (180 / math.pi), 2)

    def get_ita_category(self, ita):
        """Classificazione basata su Del Bino et al."""
        if ita > 55: return "VERY-LIGHT"
        if 41 < ita <= 55: return "LIGHT"
        if 28 < ita <= 41: return "INTERMEDIATE"
        if 10 < ita <= 28: return "TAN"
        if -30 < ita <= 10: return "BROWN"
        return "DARK"

    def process_image_stream(self, uploaded_file):
        """Trasforma il file caricato in colore mediano Lab."""
        pil_img = Image.open(uploaded_file).convert('RGB')
        img_rgb = np.array(pil_img)
        
        # Estrazione ROI centrale (30% dell'immagine)
        h, w, _ = img_rgb.shape
        roi = img_rgb[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
        
        # Mediana per eliminare peli, riflessi o rumore
        median_color = np.median(roi.reshape(-1, 3), axis=0)
        return median_color

# --- INTERFACCIA STREAMLIT ---
st.set_page_config(page_title="SkinID™ Mapping", page_icon="🧬", layout="wide")

st.title("🧬 SkinID™ Analyzer v3.0")
st.markdown("Analisi colorimetrica avanzata basata su spazio colore **CIELab** e angolo **ITA°**.")

uploaded_files = st.file_uploader(
    "Carica 3 immagini (JPG, PNG o HEIC)", 
    type=['jpg', 'jpeg', 'png', 'heic'], 
    accept_multiple_files=True
)

if uploaded_files:
    mappa_zone = {}
    st.write("### 1. Associa ogni foto alla zona corretta")
    
    # Griglia per le anteprime e i menu a tendina
    cols = st.columns(len(uploaded_files))
    opzioni_zone = ["Seleziona...", "Fronte", "Guancia", "Mandibola"]
    
    for i, file in enumerate(uploaded_files):
        with cols[i]:
            st.image(file, use_container_width=True)
            scelta = st.selectbox(f"Zona per Foto {i+1}", opzioni_zone, key=f"sel_{i}")
            mappa_zone[file.name] = scelta

    # Validazione delle scelte
    zone_selezionate = [v for v in mappa_zone.values() if v != "Seleziona..."]
    
    if len(uploaded_files) == 3:
        if len(set(zone_selezionate)) == 3:
            st.success("Configurazione valida! Procedi con l'analisi.")
            
            if st.button("ESEGUI ANALISI BIOMETRICA"):
                engine = SkinIDEngine()
                results = {}
                
                with st.spinner('Calcolo delle coordinate Lab e ITA...'):
                    for file in uploaded_files:
                        zona = mappa_zone[file.name]
                        rgb_mediano = engine.process_image_stream(file)
                        L, a, b = engine._srgb_to_lab(rgb_mediano)
                        ita = engine.calculate_ita(L, b)
                        
                        results[zona] = {
                            "L* (Luminosità)": L,
                            "a* (Rosso/Verde)": a,
                            "b* (Giallo/Blu)": b,
                            "ITA°": ita
                        }
                    
                    # Calcolo Medie Globali
                    avg_ita = np.mean([r["ITA°"] for r in results.values()])
                    avg_L = np.mean([r["L* (Luminosità)"] for r in results.values()])
                    avg_b = np.mean([r["b* (Giallo/Blu)"] for r in results.values()])
                    
                    categoria = engine.get_ita_category(avg_ita)
                    sottotono = "WARM" if avg_b > 17 else "COOL" if avg_b < 13 else "NEUTRAL"
                    
                    st.divider()
                    st.header(f"Risultato SkinID™: {categoria}")
                    
                    # Mostra tabella risultati zonali
                    st.write("### Dettaglio per Zona")
                    st.table(results)
                    
                    # Indicatori finali
                    c1, c2, c3 = st.columns(3)
                    c1.metric("ITA° Medio", f"{round(avg_ita, 1)}°")
                    c2.metric("Sottotono", sottotono)
                    c3.metric("SkinID Code", f"{categoria[:2]}-{int(avg_L)}")
                    
                    st.info(f"L'analisi mostra che la zona più luminosa è la **{max(results, key=lambda x: results[x]['L* (Luminosità)'])}**.")

        else:
            st.warning("⚠️ Seleziona 3 zone distinte: una foto per la Fronte, una per la Guancia e una per la Mandibola.")
    else:
        st.info("ℹ️ Carica esattamente 3 foto per abilitare l'analisi completa.")