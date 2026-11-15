"""
================================================================================
ASHES OF MICROGRAVITY — RESEARCH-GRADE COMBUSTION & SPACECRAFT FIRE SAFETY API
================================================================================
FastAPI Backend (v2.1.0) integrating:
  1. Saffire-Calibrated 1D & Quasi-2D Analytical Physics Engine (de Ris / Quintiere)
  2. NumPy-Accelerated 2D Transient Thermal-Pyrolysis PDE Solver (/simulate-2d)
  3. 41-Feature Physics-Informed Neural Surrogate Ensemble with OOD Guardrails
  4. Grounded NASA Literature RAG Engine (ChromaDB + Anti-Hallucination Filter)
  5. Strict Pydantic Request & Response Contracts + Constants Provenance (7009A-inspired)
================================================================================
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
)
logger = logging.getLogger("ashes.api")

RESEARCH_DISCLAIMER = (
    "Research and educational screening tool. Not medical advice, not a certified safety "
    "assessment, and not a substitute for NASA-STD-6001B testing. Not affiliated with NASA."
)

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
for p in (str(BACKEND_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Hard core dependency: fail fast if physics engine or material database is missing
from fire_model import FireModel
from materials import ALL_CATEGORIES, CORE_NASA_NAMES, MAT_DICT, MATERIALS_118
from toxicology import (
    HABITAT_VOLUMES_M3,
    NASA_SMAC_LIMITS_PPM,
    compute_toxic_hazard_summary,
    evaluate_crew_toxicity,
)

fire_model = FireModel(MAT_DICT)

PROVENANCE_CANDIDATES = [
    PROJECT_ROOT / "ai" / "constants_provenance.json",
    BACKEND_DIR / "ai" / "constants_provenance.json",
]


def _load_provenance() -> dict:
    for path in PROVENANCE_CANDIDATES:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse {path}: {e}")
    return {}


_surrogate = None   # None=not tried, False=failed, instance=loaded
_kb = None

def _get_surrogate():
    global _surrogate
    if _surrogate is None:
        try:
            from ai.surrogate import EnsembleSurrogate
            _surrogate = EnsembleSurrogate()
            logger.info("AI surrogate lazy-loaded (5-model 41-feature ensemble).")
        except Exception as e:
            logger.warning(f"AI surrogate unavailable: {e}")
            _surrogate = False
    return _surrogate or None

def _get_kb():
    global _kb
    if _kb is None:
        try:
            from ai.rag import NASAKnowledgeBase
            _kb = NASAKnowledgeBase()
            logger.info(f"RAG lazy-loaded: {_kb.collection.count()} chunks.")
        except Exception as e:
            logger.warning(f"RAG unavailable: {e}")
            _kb = False
    return _kb or None

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provenance = _load_provenance()
    logger.info("Startup: physics + materials ready. Surrogate/RAG lazy-load on first use.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Ashes of Microgravity API",
    description=(
        "Hybrid Physics + AI Spacecraft Fire Safety & Microgravity Flame Spread "
        "Simulation Platform (calibrated on NASA Saffire I-V & BASS-II flight data). "
        + RESEARCH_DISCLAIMER
    ),
    version="2.1.0",
    lifespan=lifespan,
)

# ------------------------------------------------------------------------------
# Lightweight in-memory rate limiting (no extra dependency).
# Protects the expensive endpoints (/simulate-2d, /rag -> Groq quota) from abuse.
# Config via env: RATE_LIMIT_PER_MIN (default 120), RATE_LIMIT_HEAVY_PER_MIN (default 20),
# RATE_LIMIT_DISABLED=1, TRUST_PROXY_HEADERS=1 (read client IP from X-Forwarded-For when
# running behind a reverse proxy / Space; otherwise every user shares the proxy's IP).
# ------------------------------------------------------------------------------
RATE_LIMIT_DISABLED = os.getenv("RATE_LIMIT_DISABLED", "0") == "1"
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))
RATE_LIMIT_HEAVY_PER_MIN = int(os.getenv("RATE_LIMIT_HEAVY_PER_MIN", "20"))
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "0") == "1"
HEAVY_PATHS = {"/simulate-2d", "/rag", "/transient", "/compare-materials", "/sensitivity"}
RATE_LIMIT_EXEMPT_PATHS = {"/", "/health"}
RATE_LIMIT_EXEMPT_PREFIXES = ("/static", "/docs", "/redoc", "/openapi.json")


class SlidingWindowLimiter:
    """Per-key sliding-window counter (in-memory; resets on restart, not shared across workers)."""

    def __init__(self, window_s: float = 60.0, max_keys: int = 10_000):
        self.window_s = window_s
        self.max_keys = max_keys
        self.hits = defaultdict(deque)

    def allow(self, key: str, limit: int, now: Optional[float] = None) -> Tuple[bool, int]:
        now = time.monotonic() if now is None else now
        q = self.hits[key]
        cutoff = now - self.window_s
        while q and q[0] <= cutoff:
            q.popleft()
        if len(q) >= limit:
            return False, max(1, int(self.window_s - (now - q[0])) + 1)
        q.append(now)
        if len(self.hits) > self.max_keys:  # prune idle keys so memory stays bounded
            for k in [k for k, v in self.hits.items() if not v or v[-1] <= cutoff]:
                del self.hits[k]
        return True, 0


_limiter = SlidingWindowLimiter()


def _client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _too_many(retry_after: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please retry shortly."},
        headers={"Retry-After": str(retry_after)},
    )


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if (RATE_LIMIT_DISABLED or request.method == "OPTIONS" or path in RATE_LIMIT_EXEMPT_PATHS
            or path.startswith(RATE_LIMIT_EXEMPT_PREFIXES)):
        return await call_next(request)
    ip = _client_ip(request)
    if ip == "testclient":  # FastAPI TestClient (pytest) is never rate limited
        return await call_next(request)
    if path in HEAVY_PATHS:
        ok, retry = _limiter.allow(f"heavy:{ip}", RATE_LIMIT_HEAVY_PER_MIN)
        if not ok:
            return _too_many(retry)
    ok, retry = _limiter.allow(f"all:{ip}", RATE_LIMIT_PER_MIN)
    if not ok:
        return _too_many(retry)
    return await call_next(request)


# CORS: defaults to wildcard (public demo, no credentials). For a real deployment set
# ALLOWED_ORIGINS="https://your-site.example,https://other.example". Added AFTER the rate
# limiter so it is the outermost layer and 429 responses still carry CORS headers.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# PYDANTIC REQUEST & RESPONSE SCHEMAS (Explicit OpenAPI & CI Contract)
# ==============================================================================

GeometryType = Literal["plate", "rod", "cylinder", "sphere"]
EdgeBoundaryType = Literal["free", "clamped", "insulated"]
ConfigType = Literal["Opposed", "Concurrent"]
IgnitionModeType = Literal["line", "point", "edge"]


class PredictRequest(BaseModel):
    material: str = Field(..., min_length=1, description="Material name from database")
    g_level: float = Field(default=0.0, ge=0.0, le=2.0, description="Gravity level [g0]")
    pressure_kPa: float = Field(default=101.3, gt=1.0, le=300.0, description="Ambient pressure [kPa]")
    oxygen_pct: float = Field(default=21.0, ge=0.0, le=100.0, description="Ambient O2 volume %")
    flow_vel_cms: float = Field(default=20.0, ge=0.0, le=300.0, description="Forced flow velocity [cm/s]")
    thickness_mm: float = Field(default=10.0, gt=0.05, le=100.0, description="Fuel thickness or diameter [mm]")
    sample_width_mm: float = Field(default=410.0, gt=1.0, le=2000.0, description="Sample transverse width [mm]")
    geometry: GeometryType = Field(default="plate", description="Specimen geometry")
    edge_boundary: EdgeBoundaryType = Field(default="free", description="Lateral edge boundary condition")
    configuration: ConfigType = Field(default="Opposed", description="Flame spread vs flow orientation")
    use_ai_surrogate: bool = Field(default=True, description="Include 5-model neural surrogate cross-check")
    cabin_volume_m3: float = Field(default=106.0, gt=2.0, le=2000.0, description="Habitable cabin volume [m^3] (106 m^3 = ISS Destiny)")
    exposure_minutes: float = Field(default=15.0, gt=0.5, le=240.0, description="Crew exposure / uncontained burn duration [min]")
    eclss_active: bool = Field(default=True, description="Whether spacecraft ECLSS trace contaminant scrubbers are active")
    crew_activity: Literal["resting", "light_work", "emergency"] = Field(default="light_work", description="Crew respiratory workload state")
    mask_don_minute: Optional[float] = Field(default=None, ge=0.0, le=240.0, description="Minute at which crew dons 100% O2 PBA mask (None = unmasked)")

class SurrogateSummary(BaseModel):
    spread_mms: float
    ensemble_disagreement_mms: float
    delta_vs_physics_mms: float
    delta_pct_vs_physics: float
    flame_temp_K: float
    status: str
    note: str


class PredictResponse(BaseModel):
    material: str
    flame_spread_mms: float
    flame_spread_uncertainty_mms: float
    v_eff_cms: float
    flame_temp_K: float
    effective_flame_temp_K: float
    standoff_mm: float
    q_conv_kWm2: float
    q_rad_kWm2: float
    q_surf_loss_kWm2: float
    q_net_kWm2: float
    thermal_regime: Optional[str] = None
    thick_blend_weight: Optional[float] = None
    status: str
    is_extinct: bool
    sample_width_mm: float
    geometry: str
    edge_boundary: str
    extinction_reason: Optional[str] = None
    extinction_detail: Optional[str] = None
    ul94: str
    note: str
    confidence_score: int
    confidence_level: str
    confidence_color: str
    confidence_issues: List[str]
    applicability_score: int
    applicability_level: str
    applicability_issues: List[str]
    prediction_source: str
    surrogate_agrees_with_physics: bool
    surrogate: Optional[SurrogateSummary] = None
    compute_time_ms: float
    toxicology: Optional[Dict[str, Any]] = None
    disclaimer: Optional[str] = None

class Baseline(BaseModel):
    g_level: float = Field(default=0.0, ge=0.0, le=2.0)
    pressure_kPa: float = Field(default=101.3, gt=1.0, le=300.0)
    oxygen_pct: float = Field(default=21.0, ge=0.0, le=100.0)
    flow_vel_cms: float = Field(default=20.0, ge=0.0, le=300.0)
    thickness_mm: float = Field(default=10.0, gt=0.05, le=100.0)
    sample_width_mm: float = Field(default=410.0, gt=1.0, le=2000.0)
    geometry: GeometryType = "plate"
    edge_boundary: EdgeBoundaryType = "free"
    configuration: ConfigType = "Opposed"


class SensitivityRequest(BaseModel):
    material: str = Field(..., min_length=1)
    baseline: Baseline
    perturbation: float = Field(default=0.10, gt=0.01, le=0.50)


class SensitivityItem(BaseModel):
    parameter: str
    impact: float
    delta_pct: float


class TransientRequest(BaseModel):
    material: str = Field(..., min_length=1)
    duration_s: float = Field(default=120.0, gt=0.5, le=600.0)
    dt: float = Field(default=1.0, gt=0.05, le=10.0)
    g_level: float = Field(default=0.0, ge=0.0, le=2.0)
    pressure_kPa: float = Field(default=101.3, gt=1.0, le=300.0)
    oxygen_pct: float = Field(default=21.0, ge=0.0, le=100.0)
    flow_vel_cms: float = Field(default=20.0, ge=0.0, le=300.0)
    thickness_mm: float = Field(default=2.0, gt=0.05, le=100.0)
    sample_width_mm: float = Field(default=410.0, gt=1.0, le=2000.0)
    geometry: GeometryType = "plate"
    edge_boundary: EdgeBoundaryType = "free"
    configuration: ConfigType = "Opposed"


class TransientPoint(BaseModel):
    t: float
    x_flame_mm: float
    spread_mms: float
    thickness_remaining: float
    regime: Optional[str] = None
    status: str
    reason: Optional[str] = None


class Simulate2DRequest(BaseModel):
    material: str = Field(..., min_length=1)
    length_mm: float = Field(default=240.0, ge=20.0, le=1500.0)
    width_mm: float = Field(default=150.0, ge=10.0, le=1000.0)
    nx: int = Field(default=120, ge=15, le=200, description="Streamwise spatial grid cells")
    ny: int = Field(default=60, ge=8, le=120, description="Transverse spatial grid cells")
    duration_s: float = Field(default=30.0, gt=0.5, le=300.0)
    dt: float = Field(default=0.25, gt=0.02, le=2.0)
    snapshot_interval_s: float = Field(default=2.0, gt=0.2, le=60.0)
    ignition_mode: IgnitionModeType = "line"
    ignition_pos: Tuple[float, float] = (0.15, 0.5)
    edge_boundary: EdgeBoundaryType = "free"
    g_level: float = Field(default=0.0, ge=0.0, le=2.0)
    pressure_kPa: float = Field(default=101.3, gt=1.0, le=300.0)
    oxygen_pct: float = Field(default=21.0, ge=0.0, le=100.0)
    flow_vel_cms: float = Field(default=20.0, ge=0.0, le=300.0)
    thickness_mm: float = Field(default=0.33, gt=0.05, le=50.0)
    configuration: ConfigType = "Opposed"

    @field_validator("ignition_pos")
    @classmethod
    def validate_ignition_pos(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        x, y = v
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("ignition_pos coordinates must be normalized within [0.0, 1.0].")
        return v


class Snapshot2D(BaseModel):
    t: float
    max_front_mm: float
    mean_front_mm: float
    burned_area_pct: float
    active_burning_pct: float
    front_contour_mm: List[float]
    temperature_K: List[List[float]]  # Shape: [nx][ny]


class Simulate2DResponse(BaseModel):
    material: str
    status: str
    reason: Optional[str] = None
    grid_shape: Optional[List[int]] = None
    dimensions_mm: Optional[List[float]] = None
    edge_boundary: Optional[str] = None
    ignition_mode: Optional[str] = None
    reference_1d_spread_mms: Optional[float] = None
    mean_2d_front_speed_mms: float = 0.0
    final_burned_area_pct: float = 0.0
    snapshots: List[Snapshot2D] = []
    compute_time_ms: float


class CompareMaterialsRequest(BaseModel):
    materials: List[str] = Field(..., min_length=1, max_length=15)
    g_level: float = Field(default=0.0, ge=0.0, le=2.0)
    pressure_kPa: float = Field(default=101.3, gt=1.0, le=300.0)
    oxygen_pct: float = Field(default=21.0, ge=0.0, le=100.0)
    flow_vel_cms: float = Field(default=20.0, ge=0.0, le=300.0)
    thickness_mm: float = Field(default=1.0, gt=0.05, le=100.0)
    sample_width_mm: float = Field(default=410.0, gt=1.0, le=2000.0)
    geometry: GeometryType = "plate"
    edge_boundary: EdgeBoundaryType = "free"
    configuration: ConfigType = "Opposed"


class MaterialRankItem(BaseModel):
    material: str
    category: Optional[str] = None
    saffire_validated: bool
    ul94: str
    loi_1g: Optional[float] = None
    flame_spread_mms: float
    flame_spread_uncertainty_mms: float
    flame_temp_K: float
    q_net_kWm2: float
    thermal_regime: str
    is_extinct: bool
    extinction_reason: Optional[str] = None
    applicability_score: int
    applicability_level: str
    toxicity: Optional[Dict[str, Any]] = None


class CompareMaterialsResponse(BaseModel):
    conditions: Dict[str, Any]
    ranking: List[MaterialRankItem]


class MaterialsListResponse(BaseModel):
    materials: List[Dict[str, Any]]
    count: int
    core: List[str]
    categories: List[str]


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    n_results: int = Field(default=5, ge=1, le=10)


class RAGResponse(BaseModel):
    answer: str
    sources: List[str]


class InsightRequest(BaseModel):
    result: Dict[str, Any]
    material: Optional[Dict[str, Any]] = None
    query_context: str = Field(default="", max_length=1000)


class InsightResponse(BaseModel):
    summary: str
    regime: str
    key_points: List[str]
    safety_implications: str
    recommendations: List[str]
    engine_source: str
    disclaimer: Optional[str] = None


class MetadataResponse(BaseModel):
    api_version: str
    validation_summary: Dict[str, Any]
    uncertainty_semantics: Dict[str, Any]
    constants_count: int
    constants: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    physics_loaded: bool
    materials_loaded: int
    surrogate_loaded: bool
    rag_loaded: bool
    rag_chunks: int
    deployment_mode: Optional[str] = None


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _ensure_material_exists(material_name: str) -> dict:
    if material_name not in MAT_DICT:
        raise HTTPException(
            status_code=404,
            detail=f"Material '{material_name}' not found in the 118-material database.",
        )
    return MAT_DICT[material_name]


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/materials", response_model=MaterialsListResponse, tags=["Data"], summary="Get all 118 thermophysical materials")
def get_materials(category: Optional[str] = Query(default=None, description="Optional category filter")):
    if category:
        cat_norm = category.lower().strip()
        filtered = [m for m in MATERIALS_118 if (m.get("category") or "").lower() == cat_norm]
        return {
            "materials": filtered,
            "count": len(filtered),
            "core": CORE_NASA_NAMES,
            "categories": ALL_CATEGORIES,
        }
    return {
        "materials": MATERIALS_118,
        "count": len(MATERIALS_118),
        "core": CORE_NASA_NAMES,
        "categories": ALL_CATEGORIES,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Simulation"], summary="1D / Quasi-2D steady-state flame spread prediction")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    mat = _ensure_material_exists(req.material)

    try:
        res = fire_model.predict(
            req.material,
            g_level=req.g_level,
            pressure_kPa=req.pressure_kPa,
            oxygen_pct=req.oxygen_pct,
            flow_vel_cms=req.flow_vel_cms,
            thickness_mm=req.thickness_mm,
            configuration=req.configuration,
            sample_width_mm=req.sample_width_mm,
            geometry=req.geometry,
            edge_boundary=req.edge_boundary,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Physics prediction error")
        raise HTTPException(status_code=400, detail="Invalid simulation parameters.")

    res["prediction_source"] = "physics"
    res["surrogate_agrees_with_physics"] = False

    surrogate = _get_surrogate() if req.use_ai_surrogate else None
    if surrogate and not res["is_extinct"]:
        try:
            ai = surrogate.predict(
                req.g_level,
                req.oxygen_pct,
                req.flow_vel_cms,
                req.thickness_mm,
                mat,
                configuration=req.configuration,
                pressure_kPa=req.pressure_kPa,
                sample_width_mm=req.sample_width_mm,
                geometry=req.geometry,
                edge_boundary=req.edge_boundary,
                )
            ai_spread = ai["spread_mms"]["mean"]
            physics_spread = res["flame_spread_mms"]
            rel_diff = abs(ai_spread - physics_spread) / physics_spread if physics_spread > 0 else 1.0
            is_reliable = physics_spread > 0 and rel_diff <= 0.30

            res["surrogate_agrees_with_physics"] = bool(is_reliable)
            res["surrogate"] = {
                "spread_mms": round(ai_spread, 3),
                "ensemble_disagreement_mms": round(ai["spread_mms"]["std"] * 1.96, 3),
                "delta_vs_physics_mms": round(ai_spread - physics_spread, 3),
                "delta_pct_vs_physics": round((ai_spread - physics_spread) / physics_spread * 100.0, 2)
                if physics_spread > 0
                else 0.0,
                "flame_temp_K": round(ai["flame_temp_K"]["mean"], 1),
                "status": "RELIABLE (Matches Physics)" if is_reliable else "OUT_OF_DISTRIBUTION (Ignored)",
                "note": "Fast emulator of the physics engine (not an independent validation); flagged OUT_OF_DISTRIBUTION when it differs from physics by more than 30%.",
            }
            if not is_reliable:
                logger.info(
                    f"Surrogate OOD for {req.material} (delta={rel_diff*100:.1f}%); physics engine authoritative."
                )
        except Exception as e:
            logger.warning(f"Surrogate inference skipped due to error: {e}")

    res["toxicology"] = evaluate_crew_toxicity(
    res,
    mat,
    g_level=req.g_level,
    pressure_kPa=req.pressure_kPa,
    oxygen_pct=req.oxygen_pct,
    thickness_mm=req.thickness_mm,
    sample_width_mm=req.sample_width_mm,
    exposure_minutes=req.exposure_minutes,
    cabin_volume_m3=req.cabin_volume_m3,
    eclss_active=req.eclss_active,
    crew_activity=req.crew_activity,
    mask_don_minute=req.mask_don_minute,
)

    res["compute_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    res["disclaimer"] = RESEARCH_DISCLAIMER
    return res


@app.post("/simulate-2d", response_model=Simulate2DResponse, tags=["Simulation"], summary="2D transient thermal-pyrolysis PDE solver")
def simulate_2d(req: Simulate2DRequest):
    t0 = time.perf_counter()
    _ensure_material_exists(req.material)

    try:
        sim = fire_model.simulate_2d(
            req.material,
            length_mm=req.length_mm,
            width_mm=req.width_mm,
            nx=req.nx,
            ny=req.ny,
            duration_s=req.duration_s,
            dt=req.dt,
            ignition_mode=req.ignition_mode,
            ignition_pos=req.ignition_pos,
            edge_boundary=req.edge_boundary,
            snapshot_interval_s=req.snapshot_interval_s,
            g_level=req.g_level,
            pressure_kPa=req.pressure_kPa,
            oxygen_pct=req.oxygen_pct,
            flow_vel_cms=req.flow_vel_cms,
            thickness_mm=req.thickness_mm,
            configuration=req.configuration,
        )
        sim["compute_time_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return sim
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("2D simulation error")
        raise HTTPException(status_code=400, detail="2D PDE solver encountered invalid parameters.")


@app.post("/compare-materials", response_model=CompareMaterialsResponse, tags=["Simulation"], summary="Compare multiple materials under identical conditions")
def compare_materials(req: CompareMaterialsRequest):
    comparisons = []
    for mat_name in req.materials:
        mat = _ensure_material_exists(mat_name)
        pred = fire_model.predict(
            mat_name,
            g_level=req.g_level,
            pressure_kPa=req.pressure_kPa,
            oxygen_pct=req.oxygen_pct,
            flow_vel_cms=req.flow_vel_cms,
            thickness_mm=req.thickness_mm,
            configuration=req.configuration,
            sample_width_mm=req.sample_width_mm,
            geometry=req.geometry,
            edge_boundary=req.edge_boundary,
        )
        comparisons.append({
            "material": mat_name,
            "category": mat.get("category"),
            "saffire_validated": mat.get("saffire_validated", False),
            "ul94": mat.get("ul94", "HB"),
            "loi_1g": mat.get("LOI"),
            "flame_spread_mms": pred["flame_spread_mms"],
            "flame_spread_uncertainty_mms": pred["flame_spread_uncertainty_mms"],
            "flame_temp_K": pred["flame_temp_K"],
            "q_net_kWm2": pred["q_net_kWm2"],
            "thermal_regime": pred.get("thermal_regime") or "N/A",
            "is_extinct": pred["is_extinct"],
            "extinction_reason": pred["extinction_reason"],
            "applicability_score": pred["applicability_score"],
            "applicability_level": pred["applicability_level"],
            "toxicity": compute_toxic_hazard_summary(
            pred, mat,
            g_level=req.g_level,
            pressure_kPa=req.pressure_kPa,
            thickness_mm=req.thickness_mm,
            sample_width_mm=req.sample_width_mm,
            ),
        })

    # Rank safest (extinct / slowest spread) to highest hazard (fastest spread)
    comparisons.sort(key=lambda x: (not x["is_extinct"], x["flame_spread_mms"]))
    return {
        "conditions": {
            "g_level": req.g_level,
            "pressure_kPa": req.pressure_kPa,
            "oxygen_pct": req.oxygen_pct,
            "flow_vel_cms": req.flow_vel_cms,
            "thickness_mm": req.thickness_mm,
            "sample_width_mm": req.sample_width_mm,
            "geometry": req.geometry,
            "edge_boundary": req.edge_boundary,
            "configuration": req.configuration,
        },
        "ranking": comparisons,
    }


@app.post("/sensitivity", response_model=List[SensitivityItem], tags=["Simulation"], summary="One-at-a-Time parameter sensitivity ranking")
def sensitivity(req: SensitivityRequest):
    _ensure_material_exists(req.material)
    try:
        return fire_model.sensitivity(
            req.material,
            req.baseline.model_dump(),
            perturbation=req.perturbation,
        )
    except Exception:
        logger.exception("Sensitivity calculation error")
        raise HTTPException(status_code=400, detail="Sensitivity calculation failed for given baseline.")


@app.post("/transient", response_model=List[TransientPoint], tags=["Simulation"], summary="1D transient burn-through simulation")
def transient(req: TransientRequest):
    _ensure_material_exists(req.material)
    try:
        return fire_model.transient_simulate(
            req.material,
            duration_s=req.duration_s,
            dt=req.dt,
            g_level=req.g_level,
            pressure_kPa=req.pressure_kPa,
            oxygen_pct=req.oxygen_pct,
            flow_vel_cms=req.flow_vel_cms,
            thickness_mm=req.thickness_mm,
            sample_width_mm=req.sample_width_mm,
            geometry=req.geometry,
            configuration=req.configuration,
            edge_boundary=req.edge_boundary,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Transient simulation error")
        raise HTTPException(status_code=400, detail="Transient simulation failed for given parameters.")


@app.post("/rag", response_model=RAGResponse, tags=["AI Knowledge"],
          summary="Grounded Q&A over NASA Saffire / BASS-II / SoFIE reports")
def rag(req: RAGRequest):
    kb = _get_kb()
    if not kb:
        raise HTTPException(status_code=503, detail="RAG not available (lite deployment).")
    try:
        return kb.query(req.query.strip(), n_results=req.n_results)
    except Exception:
        logger.exception("RAG query error")
        raise HTTPException(status_code=500, detail="Knowledge base query failed.")


@app.post("/insight", response_model=InsightResponse, tags=["AI Knowledge"], summary="Generate physics-grounded safety insights")
def insight(req: InsightRequest):
    try:
        from ai.insights import generate_insight

        mat_dict = req.material
        if not mat_dict:
            mat_name = req.result.get("material", "")
            mat_dict = MAT_DICT.get(mat_name, {"name": mat_name or "Unknown Material"})
        out = generate_insight(req.result, mat_dict, req.query_context)
        out["disclaimer"] = RESEARCH_DISCLAIMER
        return out
    except Exception:
        logger.exception("Insight generation error")
        raise HTTPException(status_code=500, detail="Safety insight generation failed.")


@app.get("/metadata", response_model=MetadataResponse, tags=["System"], summary="Model validation metrics & constants provenance")
def metadata():
    prov = getattr(app.state, "provenance", None) or _load_provenance()
    return {
        "api_version": "2.1.0",
        "validation_summary": {
            "core_in_sample_mape_pct": 2.83,
            "extended_in_sample_mape_pct": 7.83,
            "combined_in_sample_mape_pct": 5.90,
            "combined_in_sample_unrounded_mape_pct": 6.03,
            "nested_loo_mape_pct": {"seed_0": 10.67, "seed_100": 12.26, "seed_200": 10.42},
            "nested_lomo_mape_pct": {"seed_0": 19.94, "seed_100": 10.19, "seed_200": 8.07},
            "legacy_partial_loo_pct": {"core_5": 4.15, "all_13": 6.68},
            "nested_n_fitted_constants": 16,
            "validation_note": (
                "Nested LOO refits all 16 empirical constants per fold. In-sample 5.90% uses the rounded public "
                "predict() output; 6.03% uses unrounded values. Legacy LOO refit only 4 constants and is optimistic."
            ),
            "flight_benchmarks": ["Saffire-I", "Saffire-II", "Saffire-III", "Saffire-IV", "Saffire-V", "BASS", "BASS-II"],
        },
        "uncertainty_semantics": prov.get("uncertainty_semantics", {}),
        "constants_count": len(prov.get("constants", {})),
        "constants": prov.get("constants", {}),
    }


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="API and subsystem health status")
def health():
    physics_ok = fire_model is not None and len(MAT_DICT) > 0
    # Do NOT trigger lazy loads here — health probes must stay cheap.
    surrogate_ok = _surrogate is not None and _surrogate is not False
    rag_ok = _kb is not None and _kb is not False
    rag_chunks = _kb.collection.count() if rag_ok else 0
    deployment_mode = "full" if (surrogate_ok and rag_ok) else "lite"
    return {
        # Physics is the authoritative core. If it loads, the service is operational.
        # Surrogate + RAG are optional accelerators unavailable on 512MB serverless tiers.
        "status": "ok" if physics_ok else "degraded",
        "version": "2.1.0",
        "physics_loaded": physics_ok,
        "materials_loaded": len(MATERIALS_118),
        "surrogate_loaded": surrogate_ok,
        "rag_loaded": rag_ok,
        "rag_chunks": rag_chunks,
        "deployment_mode": deployment_mode,
    }


FRONTEND_DIR = PROJECT_ROOT / "frontend"
AUDIO_DIR = FRONTEND_DIR / "audio"

class ToxicologyRequest(BaseModel):
    material: str = Field(..., min_length=1)
    g_level: float = Field(default=0.0, ge=0.0, le=2.0)
    pressure_kPa: float = Field(default=101.3, gt=1.0, le=300.0)
    oxygen_pct: float = Field(default=21.0, ge=0.0, le=100.0)
    flow_vel_cms: float = Field(default=20.0, ge=0.0, le=300.0)
    thickness_mm: float = Field(default=2.0, gt=0.05, le=100.0)
    sample_width_mm: float = Field(default=150.0, gt=1.0, le=2000.0)
    geometry: GeometryType = "plate"
    edge_boundary: EdgeBoundaryType = "free"
    configuration: ConfigType = "Opposed"
    cabin_volume_m3: float = Field(default=106.0, gt=2.0, le=2000.0)
    exposure_minutes: float = Field(default=15.0, gt=0.5, le=240.0)
    eclss_active: bool = True
    crew_activity: Literal["resting", "light_work", "emergency"] = "light_work"
    mask_don_minute: Optional[float] = Field(default=None, ge=0.0, le=240.0)


@app.post("/toxicology", tags=["Crew Health"], summary="Dynamic cabin toxicology, NASA SMACs & COHb/FED simulation (illustrative)")
def toxicology_sim(req: ToxicologyRequest):
    mat = _ensure_material_exists(req.material)
    try:
        pred = fire_model.predict(
            req.material,
            g_level=req.g_level,
            pressure_kPa=req.pressure_kPa,
            oxygen_pct=req.oxygen_pct,
            flow_vel_cms=req.flow_vel_cms,
            thickness_mm=req.thickness_mm,
            configuration=req.configuration,
            sample_width_mm=req.sample_width_mm,
            geometry=req.geometry,
            edge_boundary=req.edge_boundary,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Toxicology physics prediction error")
        raise HTTPException(status_code=400, detail="Invalid simulation parameters.")
    return {
        "material": req.material,
        "flame_spread_mms": pred["flame_spread_mms"],
        "is_extinct": pred["is_extinct"],
        "habitat_presets_m3": HABITAT_VOLUMES_M3,
        "nasa_smac_limits_ppm": NASA_SMAC_LIMITS_PPM,
        "toxicology": evaluate_crew_toxicity(
            pred,
            mat,
            g_level=req.g_level,
            pressure_kPa=req.pressure_kPa,
            oxygen_pct=req.oxygen_pct,
            thickness_mm=req.thickness_mm,
            sample_width_mm=req.sample_width_mm,
            exposure_minutes=req.exposure_minutes,
            cabin_volume_m3=req.cabin_volume_m3,
            eclss_active=req.eclss_active,
            crew_activity=req.crew_activity,
            mask_don_minute=req.mask_don_minute,
        ),
        "disclaimer": RESEARCH_DISCLAIMER,
    }

if AUDIO_DIR.exists():
    # Mount only audio assets at /static/audio rather than exposing the entire frontend directory
    app.mount("/static/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
elif FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "Ashes of Microgravity API v2.1.0 running. Docs available at /docs."}