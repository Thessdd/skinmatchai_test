import cv2
import numpy as np
import math
import os  # Necessario per la gestione dei percorsi

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
        """Converte un vettore RGB [R, G, B] in coordinate CIELab."""
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
        """Calcola l'angolo ITA° (Individual Typology Angle)."""
        ita_rad = math.atan2((L - 50), b)
        ita_deg = ita_rad * (180 / math.pi)
        return round(ita_deg, 2)

    def get_ita_category(self, ita):
        """Classificazione scientifica basata sull'angolo ITA°."""
        if ita > 55: return "VERY-LIGHT"
        if 41 < ita <= 55: return "LIGHT"
        if 28 < ita <= 41: return "INTERMEDIATE"
        if 10 < ita <= 28: return "TAN"
        if -30 < ita <= 10: return "BROWN"
        return "DARK"

    def extract_skin_color(self, image_path):
        """Carica l'immagine, estrae la ROI centrale e calcola il colore mediano."""
        img = cv2.imread(image_path)
        if img is None:
            # Errore più descrittivo per debuggare il percorso
            raise FileNotFoundError(f"Errore: Impossibile caricare '{image_path}'. Verifica che il file esista.")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, _ = img_rgb.shape
        start_h, end_h = int(h * 0.35), int(h * 0.65)
        start_w, end_w = int(w * 0.35), int(w * 0.65)
        roi = img_rgb[start_h:end_h, start_w:end_w]
        
        median_color = np.median(roi.reshape(-1, 3), axis=0)
        return median_color

    def generate_skin_id(self, rgb_input):
        """Genera il report tecnico e il codice SkinID™."""
        L, a, b = self._srgb_to_lab(rgb_input)
        ita = self.calculate_ita(L, b)
        category = self.get_ita_category(ita)
        
        undertone = "WARM" if b > 17 else "COOL" if b < 13 else "NEUTRAL"
        if a > 15: undertone += "-ROSACEA_PROBE"

        reactivity = "HIGH" if b > 20 else "LOW"
        skin_id_code = f"{category}-{int(L)}-{undertone}-{reactivity}"
        
        return {
            "code": skin_id_code,
            "metrics": {"L": L, "a": a, "b": b, "ITA": ita},
            "labels": {"cat": category, "tone": undertone, "react": reactivity}
        }

    def process_triple_analysis(self, image_paths):
        """Analizza i campioni e restituisce il SkinID™ medio."""
        samples = []
        for path in image_paths:
            print(f"Analisi in corso: {os.path.basename(path)}...")
            samples.append(self.extract_skin_color(path))
        
        avg_rgb = np.mean(samples, axis=0)
        return self.generate_skin_id(avg_rgb)

# --- ESECUZIONE MIGLIORATA ---
if __name__ == "__main__":
    engine = SkinIDEngine()
    
    # 1. Identifica la cartella dove si trova questo script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Definisce i nomi dei file
    nomi_file = ["fronte.jpg", "guancia.jpg", "mandibola.jpg"]
    
    # 3. Costruisce i percorsi assoluti unendo cartella e nome file
    mie_foto = [os.path.join(script_dir, f) for f in nomi_file]
    
    try:
        print(f"Cartella di lavoro: {script_dir}\n")
        report = engine.process_triple_analysis(mie_foto)
        
        print("\n" + "="*40)
        print(f" RISULTATO SKINID™: {report['code']}")
        print("="*40)
        print(f"Valore ITA°:          {report['metrics']['ITA']}°")
        print(f"Luminosità (L*):      {report['metrics']['L']}")
        print(f"Sottotono rilevato:   {report['labels']['tone']}")
        print(f"Indice Reattività:    {report['labels']['react']}")
        print("="*40)
        
    except Exception as e:
        print(f"\n[ERRORE DI SISTEMA]: {e}")
        print("\nSuggerimento: Verifica che le immagini siano nella cartella:")
        print(f"--> {script_dir}")