"""
database.py — SkinMatch AI · Schema SQLite + seed data
SQLAlchemy ORM, migrabile su PostgreSQL senza modifiche al codice.
"""

import math
import uuid
from datetime import date
from sqlalchemy import (
    create_engine, Column, String, Float, Boolean, Integer,
    Date, DateTime, Enum, ForeignKey, JSON, func
)
from sqlalchemy.orm import declarative_base, relationship, Session
import enum as pyenum

# ── Engine ────────────────────────────────────────────────────────────────────
# In locale: usa SQLite (nessuna configurazione richiesta)
# In cloud:  legge DATABASE_URL dall'environment (Supabase / Railway / Render)
import os
_db_url = os.environ.get("DATABASE_URL", "sqlite:///skinmatch.db")
if _db_url.startswith("postgres://"):
    _db_url = _db_url.replace("postgres://", "postgresql://", 1)
ENGINE = create_engine(_db_url, echo=False)
Base   = declarative_base()


# ══════════════════════════════════════════════════════════════════════════════
#  ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class CategoryEnum(str, pyenum.Enum):
    foundation = "foundation"
    concealer  = "concealer"
    powder     = "powder"
    blush      = "blush"
    bronzer    = "bronzer"

class FinishEnum(str, pyenum.Enum):
    matte  = "matte"
    satin  = "satin"
    dewy   = "dewy"
    luminous = "luminous"

class CoverageEnum(str, pyenum.Enum):
    light  = "light"
    medium = "medium"
    full   = "full"
    buildable = "buildable"

class MeasurementSourceEnum(str, pyenum.Enum):
    nix_spectro_l   = "nix_spectro_l"
    nix_spectro_2   = "nix_spectro_2"
    xrite_capsure   = "xrite_capsure"
    manual_estimated= "manual_estimated"

class InputSourceEnum(str, pyenum.Enum):
    photo         = "photo"
    nix_manual    = "nix_manual"
    nix_bluetooth = "nix_bluetooth"


# ══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════════════════

class Product(Base):
    __tablename__ = "products"

    product_id   = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    brand        = Column(String, nullable=False)
    line         = Column(String, nullable=False)
    name         = Column(String, nullable=False)   # nome shade
    sku          = Column(String, unique=True)
    category     = Column(String, nullable=False)   # foundation / concealer / ...
    finish       = Column(String)                   # matte / satin / dewy
    coverage     = Column(String)                   # light / medium / full
    formula_tags = Column(JSON, default=list)       # ["non-comedogenic", "vegan"]
    price_eur    = Column(Float)
    url          = Column(String)
    image_url    = Column(String)
    active       = Column(Boolean, default=True)
    notes        = Column(String)

    colorimetry  = relationship("ProductColorimetry", back_populates="product",
                                cascade="all, delete-orphan")
    matches      = relationship("Match", back_populates="product",
                                cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Product {self.brand} · {self.name}>"


class ProductColorimetry(Base):
    __tablename__ = "product_colorimetry"

    colorimetry_id     = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id         = Column(String, ForeignKey("products.product_id"), nullable=False)
    L_star             = Column(Float, nullable=False)
    a_star             = Column(Float, nullable=False)
    b_star             = Column(Float, nullable=False)
    ITA_deg            = Column(Float)
    undertone_calc     = Column(String)
    oxidation_delta_b  = Column(Float, default=0.0)  # Δb* dopo 3h (positivo = scurisce)
    measurement_source = Column(String, default="manual_estimated")
    measured_at        = Column(Date, default=date.today)

    product = relationship("Product", back_populates="colorimetry")

    def __repr__(self):
        return f"<Colorimetry L={self.L_star} a={self.a_star} b={self.b_star}>"


class SkinProfile(Base):
    __tablename__ = "skin_profiles"

    profile_id       = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id       = Column(String)
    L_weighted       = Column(Float, nullable=False)
    a_weighted       = Column(Float, nullable=False)
    b_weighted       = Column(Float, nullable=False)
    ITA_deg          = Column(Float)
    category         = Column(String)   # VERY-LIGHT / LIGHT / INTERMEDIATE / TAN / BROWN / DARK
    undertone        = Column(String)   # WARM / COOL / NEUTRAL / OLIVE / WARM-PEACH
    skin_type        = Column(String)   # Oleosa / Mista / Secca
    reactivity_index = Column(String)   # BASSO / MEDIO / ALTO
    skin_id_code     = Column(String)
    source           = Column(String)   # photo / nix_manual / nix_bluetooth
    created_at       = Column(DateTime, default=func.now())

    matches = relationship("Match", back_populates="profile",
                           cascade="all, delete-orphan")


class Match(Base):
    __tablename__ = "matches"

    match_id      = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    profile_id    = Column(String, ForeignKey("skin_profiles.profile_id"), nullable=False)
    product_id    = Column(String, ForeignKey("products.product_id"), nullable=False)
    delta_e_total = Column(Float)
    score         = Column(Float)    # 0.0 – 1.0
    rank          = Column(Integer)
    finish_match  = Column(Boolean, default=True)
    formula_ok    = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=func.now())

    profile = relationship("SkinProfile", back_populates="matches")
    product = relationship("Product",     back_populates="matches")


# ══════════════════════════════════════════════════════════════════════════════
#  MATCHING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Pesi assi Lab per categoria prodotto
# Foundation: b* (warm/cool) e L* (luminosità) dominano
# Concealer:  a* (eritema) diventa cruciale
# Powder:     L* domina (luminosità e finish)
MATCH_WEIGHTS = {
    "foundation": {"L": 0.40, "a": 0.20, "b": 0.40},
    "concealer":  {"L": 0.25, "a": 0.50, "b": 0.25},
    "powder":     {"L": 0.50, "a": 0.20, "b": 0.30},
    "blush":      {"L": 0.30, "a": 0.40, "b": 0.30},
    "bronzer":    {"L": 0.35, "a": 0.25, "b": 0.40},
}

# Soglie ΔE per badge qualità
DELTA_E_THRESHOLDS = {
    "certified": 2.0,   # match spettrofotometrico certificato
    "excellent": 3.5,   # match eccellente
    "good":      5.5,   # match buono
    "acceptable":8.0,   # match accettabile
}

def calc_ita(L, b):
    if abs(b) < 0.001:
        b = 0.001
    return round(math.atan2(L - 50.0, b) * (180.0 / math.pi), 2)

def calc_undertone(a, b):
    if b > 18 and a < 12:  return "WARM"
    if b > 18 and a >= 12: return "WARM-PEACH"
    if b < 13:             return "COOL"
    if a < 6 and b < 18:   return "OLIVE"
    return "NEUTRAL"

def match_score(skin: dict, prod_color: ProductColorimetry,
                category: str, skin_ri: str) -> dict:
    """
    Calcola score di matching tra un profilo pelle e un prodotto.

    Args:
        skin:      {"L": float, "a": float, "b": float}
        prod_color: ProductColorimetry ORM object
        category:  stringa categoria prodotto
        skin_ri:   Reactivity Index della pelle ("BASSO"/"MEDIO"/"ALTO")

    Returns:
        dict con delta_e, score, badge
    """
    w  = MATCH_WEIGHTS.get(category, MATCH_WEIGHTS["foundation"])
    dL = skin["L"] - prod_color.L_star
    da = skin["a"] - prod_color.a_star
    db = skin["b"] - prod_color.b_star

    # Delta E pesato (ottimizzato per skin matching, non formula CIE standard)
    delta_e = math.sqrt(w["L"]*dL**2 + w["a"]*da**2 + w["b"]*db**2)

    # Penalità ossidazione: pelle con RI alto + prodotto che ossida molto
    oxid_penalty = 0.0
    if skin_ri == "ALTO" and (prod_color.oxidation_delta_b or 0) > 2.0:
        oxid_penalty = 1.5
    elif skin_ri == "MEDIO" and (prod_color.oxidation_delta_b or 0) > 3.0:
        oxid_penalty = 0.8

    delta_e_final = delta_e + oxid_penalty

    # Score 0-1 (1 = perfetto)
    score = round(max(0.0, 1.0 - delta_e_final / 10.0), 3)

    # Badge qualità
    if delta_e_final <= DELTA_E_THRESHOLDS["certified"]:
        badge = "🏆 Certificato"
    elif delta_e_final <= DELTA_E_THRESHOLDS["excellent"]:
        badge = "⭐ Eccellente"
    elif delta_e_final <= DELTA_E_THRESHOLDS["good"]:
        badge = "✓ Buono"
    elif delta_e_final <= DELTA_E_THRESHOLDS["acceptable"]:
        badge = "~ Accettabile"
    else:
        badge = "✗ Non consigliato"

    return {
        "delta_e":      round(delta_e_final, 3),
        "delta_e_raw":  round(delta_e, 3),
        "oxid_penalty": oxid_penalty,
        "score":        score,
        "badge":        badge,
        "dL": round(dL,2), "da": round(da,2), "db": round(db,2),
    }


def find_matches(db: Session, skin: dict, skin_ri: str,
                 category: str = "foundation",
                 finish_filter: str = None,
                 tags_required: list = None,
                 top_n: int = 10) -> list:
    """
    Trova i migliori N prodotti per un dato profilo pelle.

    Args:
        db:            sessione SQLAlchemy
        skin:          {"L": float, "a": float, "b": float}
        skin_ri:       Reactivity Index ("BASSO"/"MEDIO"/"ALTO")
        category:      categoria prodotto da cercare
        finish_filter: "matte" / "satin" / "dewy" / None = tutti
        tags_required: es. ["non-comedogenic"] — filtra per tag formula
        top_n:         quanti risultati restituire

    Returns:
        lista di dict ordinata per score decrescente
    """
    # Query base: prodotti attivi della categoria selezionata
    query = db.query(Product, ProductColorimetry)\
              .join(ProductColorimetry)\
              .filter(Product.active == True)\
              .filter(Product.category == category)

    if finish_filter:
        query = query.filter(Product.finish == finish_filter)

    rows = query.all()

    results = []
    for prod, color in rows:
        # Filtro tag formula (es. non-comedogenic)
        if tags_required:
            prod_tags = prod.formula_tags or []
            if not all(t in prod_tags for t in tags_required):
                continue

        m = match_score(skin, color, category, skin_ri)
        results.append({
            "product":      prod,
            "colorimetry":  color,
            "score":        m["score"],
            "delta_e":      m["delta_e"],
            "badge":        m["badge"],
            "dL":           m["dL"],
            "da":           m["da"],
            "db":           m["db"],
            "oxid_penalty": m["oxid_penalty"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
#  INIT DB + SEED DATA
# ══════════════════════════════════════════════════════════════════════════════

def init_db():
    """Crea le tabelle se non esistono."""
    Base.metadata.create_all(ENGINE)


def seed_demo_data(db: Session):
    """
    Inserisce prodotti di esempio con valori Lab stimati.
    In produzione questi valori vengono sostituiti con misurazioni Nix reali.
    FONTE VALORI: stime da schede tecniche brand + letteratura dermocosmetica.
    """
    if db.query(Product).count() > 0:
        return  # seed già eseguito

    demo_products = [
        # ── KIKO MILANO ──────────────────────────────────────────────────────
        {
            "product": Product(
                product_id="kiko-stf-n10",
                brand="Kiko Milano", line="Skin Tone Foundation", name="N10 Fair",
                sku="KM-STF-N10", category="foundation", finish="satin",
                coverage="medium", formula_tags=["non-comedogenic"],
                price_eur=12.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="kiko-stf-n10",
                L_star=80.2, a_star=4.1, b_star=10.8,
                oxidation_delta_b=0.8, measurement_source="manual_estimated"
            ),
        },
        {
            "product": Product(
                product_id="kiko-stf-n20",
                brand="Kiko Milano", line="Skin Tone Foundation", name="N20 Light",
                sku="KM-STF-N20", category="foundation", finish="satin",
                coverage="medium", formula_tags=["non-comedogenic"],
                price_eur=12.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="kiko-stf-n20",
                L_star=72.5, a_star=7.2, b_star=16.4,
                oxidation_delta_b=1.2, measurement_source="manual_estimated"
            ),
        },
        {
            "product": Product(
                product_id="kiko-stf-n30",
                brand="Kiko Milano", line="Skin Tone Foundation", name="N30 Light Medium",
                sku="KM-STF-N30", category="foundation", finish="satin",
                coverage="medium", formula_tags=["non-comedogenic"],
                price_eur=12.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="kiko-stf-n30",
                L_star=65.8, a_star=9.4, b_star=19.2,
                oxidation_delta_b=1.8, measurement_source="manual_estimated"
            ),
        },
        {
            "product": Product(
                product_id="kiko-stf-w30",
                brand="Kiko Milano", line="Skin Tone Foundation", name="W30 Warm Light Medium",
                sku="KM-STF-W30", category="foundation", finish="satin",
                coverage="medium", formula_tags=["non-comedogenic"],
                price_eur=12.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="kiko-stf-w30",
                L_star=64.1, a_star=10.8, b_star=22.5,
                oxidation_delta_b=2.1, measurement_source="manual_estimated"
            ),
        },
        {
            "product": Product(
                product_id="kiko-stf-n40",
                brand="Kiko Milano", line="Skin Tone Foundation", name="N40 Medium",
                sku="KM-STF-N40", category="foundation", finish="satin",
                coverage="medium", formula_tags=["non-comedogenic"],
                price_eur=12.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="kiko-stf-n40",
                L_star=57.3, a_star=11.2, b_star=20.8,
                oxidation_delta_b=2.4, measurement_source="manual_estimated"
            ),
        },
        {
            "product": Product(
                product_id="kiko-stf-n50",
                brand="Kiko Milano", line="Skin Tone Foundation", name="N50 Medium Dark",
                sku="KM-STF-N50", category="foundation", finish="satin",
                coverage="medium", formula_tags=["non-comedogenic"],
                price_eur=12.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="kiko-stf-n50",
                L_star=46.2, a_star=12.8, b_star=18.9,
                oxidation_delta_b=1.9, measurement_source="manual_estimated"
            ),
        },
        # ── COLLISTAR ─────────────────────────────────────────────────────────
        {
            "product": Product(
                product_id="collistar-pf-01",
                brand="Collistar", line="Perfect Wear Foundation", name="01 Alabaster",
                sku="CS-PWF-01", category="foundation", finish="matte",
                coverage="full", formula_tags=["long-wear", "SPF10"],
                price_eur=38.50, active=True
            ),
            "color": ProductColorimetry(
                product_id="collistar-pf-01",
                L_star=82.1, a_star=3.8, b_star=9.2,
                oxidation_delta_b=0.5, measurement_source="manual_estimated"
            ),
        },
        {
            "product": Product(
                product_id="collistar-pf-03",
                brand="Collistar", line="Perfect Wear Foundation", name="03 Ivory",
                sku="CS-PWF-03", category="foundation", finish="matte",
                coverage="full", formula_tags=["long-wear", "SPF10"],
                price_eur=38.50, active=True
            ),
            "color": ProductColorimetry(
                product_id="collistar-pf-03",
                L_star=74.6, a_star=8.5, b_star=17.3,
                oxidation_delta_b=1.5, measurement_source="manual_estimated"
            ),
        },
        {
            "product": Product(
                product_id="collistar-pf-05",
                brand="Collistar", line="Perfect Wear Foundation", name="05 Sand",
                sku="CS-PWF-05", category="foundation", finish="matte",
                coverage="full", formula_tags=["long-wear", "SPF10"],
                price_eur=38.50, active=True
            ),
            "color": ProductColorimetry(
                product_id="collistar-pf-05",
                L_star=62.4, a_star=10.1, b_star=21.6,
                oxidation_delta_b=2.8, measurement_source="manual_estimated"
            ),
        },
        {
            "product": Product(
                product_id="collistar-pf-08",
                brand="Collistar", line="Perfect Wear Foundation", name="08 Caramel",
                sku="CS-PWF-08", category="foundation", finish="matte",
                coverage="full", formula_tags=["long-wear", "SPF10"],
                price_eur=38.50, active=True
            ),
            "color": ProductColorimetry(
                product_id="collistar-pf-08",
                L_star=44.8, a_star=13.4, b_star=19.2,
                oxidation_delta_b=2.2, measurement_source="manual_estimated"
            ),
        },
        # ── PUPA ─────────────────────────────────────────────────────────────
        {
            "product": Product(
                product_id="pupa-mf-010",
                brand="Pupa", line="Make Up Stories Foundation", name="010 Porcelain",
                sku="PU-MSF-010", category="foundation", finish="dewy",
                coverage="light", formula_tags=["hydrating", "vegan"],
                price_eur=24.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="pupa-mf-010",
                L_star=84.3, a_star=2.9, b_star=8.1,
                oxidation_delta_b=0.3, measurement_source="manual_estimated"
            ),
        },
        {
            "product": Product(
                product_id="pupa-mf-030",
                brand="Pupa", line="Make Up Stories Foundation", name="030 Natural Beige",
                sku="PU-MSF-030", category="foundation", finish="dewy",
                coverage="light", formula_tags=["hydrating", "vegan"],
                price_eur=24.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="pupa-mf-030",
                L_star=68.7, a_star=8.9, b_star=18.5,
                oxidation_delta_b=1.4, measurement_source="manual_estimated"
            ),
        },
        {
            "product": Product(
                product_id="pupa-mf-060",
                brand="Pupa", line="Make Up Stories Foundation", name="060 Dark Beige",
                sku="PU-MSF-060", category="foundation", finish="dewy",
                coverage="light", formula_tags=["hydrating", "vegan"],
                price_eur=24.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="pupa-mf-060",
                L_star=42.5, a_star=14.2, b_star=17.8,
                oxidation_delta_b=1.8, measurement_source="manual_estimated"
            ),
        },
        # ── WYCON ─────────────────────────────────────────────────────────────
        {
            "product": Product(
                product_id="wycon-sf-02",
                brand="Wycon", line="Skin Fit Foundation", name="02 Nude",
                sku="WY-SFF-02", category="foundation", finish="satin",
                coverage="buildable", formula_tags=["non-comedogenic", "vegan"],
                price_eur=9.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="wycon-sf-02",
                L_star=76.1, a_star=6.3, b_star=14.8,
                oxidation_delta_b=1.6, measurement_source="manual_estimated"
            ),
        },
        {
            "product": Product(
                product_id="wycon-sf-05",
                brand="Wycon", line="Skin Fit Foundation", name="05 Warm Beige",
                sku="WY-SFF-05", category="foundation", finish="satin",
                coverage="buildable", formula_tags=["non-comedogenic", "vegan"],
                price_eur=9.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="wycon-sf-05",
                L_star=61.9, a_star=11.5, b_star=23.1,
                oxidation_delta_b=2.5, measurement_source="manual_estimated"
            ),
        },
        # ── KIKO CONCEALER ────────────────────────────────────────────────────
        {
            "product": Product(
                product_id="kiko-cc-01",
                brand="Kiko Milano", line="Full Coverage Concealer", name="01 Fair",
                sku="KM-FCC-01", category="concealer", finish="matte",
                coverage="full", formula_tags=["non-comedogenic", "long-wear"],
                price_eur=8.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="kiko-cc-01",
                L_star=82.5, a_star=3.2, b_star=9.5,
                oxidation_delta_b=0.4, measurement_source="manual_estimated"
            ),
        },
        {
            "product": Product(
                product_id="kiko-cc-04",
                brand="Kiko Milano", line="Full Coverage Concealer", name="04 Medium",
                sku="KM-FCC-04", category="concealer", finish="matte",
                coverage="full", formula_tags=["non-comedogenic", "long-wear"],
                price_eur=8.90, active=True
            ),
            "color": ProductColorimetry(
                product_id="kiko-cc-04",
                L_star=61.8, a_star=10.4, b_star=18.9,
                oxidation_delta_b=1.1, measurement_source="manual_estimated"
            ),
        },
    ]

    for item in demo_products:
        prod  = item["product"]
        color = item["color"]
        # Calcola ITA e undertone automaticamente
        color.ITA_deg        = calc_ita(color.L_star, color.b_star)
        color.undertone_calc = calc_undertone(color.a_star, color.b_star)
        db.add(prod)
        db.add(color)

    db.commit()
    print(f"[seed] {len(demo_products)} prodotti inseriti.")


def get_session() -> Session:
    return Session(ENGINE)