"""Unit tests for spacecraft crew toxicology, SMAC limits, and FED kinetics."""

import pytest
from backend.fire_model import FireModel
from backend.materials import MAT_DICT
from backend.toxicology import NASA_SMAC_LIMITS_PPM, evaluate_crew_toxicity

MAT_NAME = "Flexible Polyvinyl Chloride (f-PVC)"


@pytest.fixture(scope="module")
def model():
    return FireModel(MAT_DICT)


@pytest.fixture(scope="module")
def pred_fpvc(model):
    return model.predict(
        MAT_NAME, g_level=0.0, pressure_kPa=101.3, oxygen_pct=21.0,
        flow_vel_cms=20.0, thickness_mm=2.0, sample_width_mm=150.0,
    )


def test_smac_limits_positive_and_ordered():
    assert len(NASA_SMAC_LIMITS_PPM) > 0
    for gas, lim in NASA_SMAC_LIMITS_PPM.items():
        assert lim["1h"] > lim["24h"] > 0, f"{gas} SMAC ordering wrong"


def test_toxicity_monotonic_in_time(pred_fpvc):
    mat = MAT_DICT[MAT_NAME]
    tox_10 = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=10.0)
    tox_60 = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=60.0)
    assert tox_60["max_smac_1h_ratio"] >= tox_10["max_smac_1h_ratio"]
    assert tox_60["purser_fed_total"] >= tox_10["purser_fed_total"]


def test_pba_mask_reduces_cohb(pred_fpvc):
    mat = MAT_DICT[MAT_NAME]
    no_mask = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=30.0, mask_don_minute=None)
    with_mask = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=30.0, mask_don_minute=5.0)
    assert with_mask["estimated_cohb_pct"] <= no_mask["estimated_cohb_pct"]


def test_microgravity_increases_CO_yield(pred_fpvc):
    mat = MAT_DICT[MAT_NAME]
    tox_ug = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=10.0)
    tox_1g = evaluate_crew_toxicity(pred_fpvc, mat, g_level=1.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=10.0)
    assert tox_ug["gases_ppm"]["CO"]["ppm"] > tox_1g["gases_ppm"]["CO"]["ppm"]


def test_pvc_produces_HCl_but_not_HCN(pred_fpvc):
    mat = MAT_DICT[MAT_NAME]
    tox = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=10.0)
    assert tox["gases_ppm"]["HCl"]["ppm"] > 0, "PVC must release HCl"
    assert tox["gases_ppm"]["HCN"]["ppm"] == 0, "PVC has no N -> HCN must be 0"


def test_analytical_cabin_ode_steady_state(pred_fpvc):
    """Cabin ODE has closed-form solution C(t) = C_eq(1 - exp(-lambda*t)).
    At 1 hr with lambda=0.03/min: C(60)/C_eq = 1 - exp(-1.8) ≈ 0.835."""
    mat = MAT_DICT[MAT_NAME]
    
    tox_60 = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=60.0, cabin_volume_m3=106.0)
    tox_600 = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=600.0, cabin_volume_m3=106.0)
    
    # At 600 min, we should be at steady state (~99.99% of C_eq)
    # At 60 min, we should be at ~83.5% of C_eq
    ratio = tox_60["gases_ppm"]["CO"]["ppm"] / tox_600["gases_ppm"]["CO"]["ppm"]
    assert 0.80 < ratio < 0.87, f"ODE shape incorrect: 60min/600min ratio = {ratio:.3f} (expected ~0.835)"


def test_smac_values_match_nasa_jsc_20584():
    """Verify SMAC limits against NASA JSC 20584 Table 4-1."""
    assert NASA_SMAC_LIMITS_PPM["CO"]["1h"] == 55.0, "CO 1h SMAC mismatch"
    assert NASA_SMAC_LIMITS_PPM["HCN"]["1h"] == 4.5, "HCN 1h SMAC mismatch"
    assert NASA_SMAC_LIMITS_PPM["HCl"]["1h"] == 5.0, "HCl 1h SMAC mismatch"
    assert NASA_SMAC_LIMITS_PPM["HF"]["1h"] == 3.0, "HF 1h SMAC mismatch"
    assert NASA_SMAC_LIMITS_PPM["NO2"]["1h"] == 1.0, "NO2 1h SMAC mismatch"


def test_cohb_equilibrium_weaver_2009(pred_fpvc):
    """Weaver (2009, NEJM) published CFK parameters. At high CO ppm with 320min
    ambient half-life, equilibrium COHb% should be approximately 0.155 × CO_ppm (per CFK)."""
    mat = MAT_DICT[MAT_NAME]
    
    # Long exposure, no scrubbers (worst case, high CO)
    tox = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=500.0, cabin_volume_m3=106.0,
        eclss_active=False, crew_activity="resting")
    
    co_ppm = tox["gases_ppm"]["CO"]["ppm"]
    cohb_pct = tox["estimated_cohb_pct"]
    
    # CFK equilibrium: COHb% ≈ 0.155 × CO_ppm + 0.8 (baseline)
    expected_cohb = min(75.0, 0.155 * co_ppm + 0.8)
    
    # After 500 min exposure, should be >80% of equilibrium
    assert cohb_pct > 0.80 * expected_cohb, \
        f"COHb too low: {cohb_pct:.1f}% vs expected ~{expected_cohb:.1f}%"
    assert cohb_pct <= expected_cohb * 1.05, \
        f"COHb exceeds CFK equilibrium: {cohb_pct:.1f}% > {expected_cohb*1.05:.1f}%"


def test_cohb_stays_low_when_extinct_pmma():
    """Even in 'extinct' mode, the model applies a smolder off-gassing floor
    (0.012 g/s in _compute_mass_burn_rate_gs). For low-toxicity materials like
    PMMA (no Cl, no N), COHb should stay well below the 20% incapacitation
    threshold and triage should be NOMINAL or CAUTION.

    Contrast: PVC smolder produces HCl even without flame (see
    test_pvc_smolder_is_dangerous_from_hcl below) — 'extinct' does not mean
    'safe cabin' for halogenated polymers."""
    mat = MAT_DICT["PMMA (Acrylic / Plexiglas)"]  # low-toxicity baseline
    extinct_pred = {
        "is_extinct": True,
        "flame_spread_mms": 0.0,
        "q_net_kWm2": 0.0,
        "status": "EXTINCT (Test)",
    }
    tox = evaluate_crew_toxicity(extinct_pred, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=120.0, cabin_volume_m3=106.0)

    cohb = tox["estimated_cohb_pct"]
    assert 0.5 < cohb < 5.0, \
        f"PMMA extinct COHb should stay near baseline, got {cohb:.2f}%"
    assert tox["triage_severity"] in ("NOMINAL", "CAUTION"), \
        f"PMMA smolder should not trigger triage, got {tox['triage_severity']}"


def test_pvc_smolder_is_dangerous_from_hcl(pred_fpvc):
    """Extinct PVC smolder produces HCl even without flame — 'extinct' does not
    mean 'safe cabin'. This validates that the model correctly distinguishes
    low-toxicity (PMMA) from high-toxicity (PVC) smolder scenarios.

    Physics check: PVC yields 0.31 g HCl per gram burned. At the 0.012 g/s
    smolder floor, a 2-hour smolder in 106 m³ releases ~27 g HCl (~45 ppm),
    well above the 5 ppm NASA SMAC — hence CRITICAL triage is correct."""
    mat = MAT_DICT[MAT_NAME]  # PVC
    extinct_pred = {
        "is_extinct": True,
        "flame_spread_mms": 0.0,
        "q_net_kWm2": 0.0,
        "status": "EXTINCT (Test)",
    }
    tox = evaluate_crew_toxicity(extinct_pred, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=120.0, cabin_volume_m3=106.0)

    # COHb stays sub-incapacitation even while HCl is high
    assert tox["estimated_cohb_pct"] < 15.0, \
        f"PVC COHb should stay below incapacitation, got {tox['estimated_cohb_pct']:.2f}%"
    # HCl drives triage to WARNING or CRITICAL
    assert tox["triage_severity"] in ("WARNING", "CRITICAL"), \
        f"PVC smolder HCl should trigger triage, got {tox['triage_severity']}"
    # HCl dominates as limiting toxicant, not CO
    assert tox["gases_ppm"]["HCl"]["ppm"] > 5.0, \
        f"PVC HCl should exceed 5 ppm SMAC, got {tox['gases_ppm']['HCl']['ppm']:.1f} ppm"


def test_fed_scales_with_exposure_at_early_times(pred_fpvc):
    """Before CO reaches equilibrium, FED should scale roughly linearly with time."""
    mat = MAT_DICT[MAT_NAME]
    
    tox_10 = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=10.0, cabin_volume_m3=106.0)
    tox_20 = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=20.0, cabin_volume_m3=106.0)
    
    # At early times (< steady state), FED scales roughly linearly with time
    ratio = tox_20["purser_fed_total"] / max(tox_10["purser_fed_total"], 1e-6)
    assert 1.5 < ratio < 2.5, \
        f"FED should roughly double from 10→20 min, got ratio={ratio:.2f}"


def test_eclss_scrubbers_reduce_peak_concentration(pred_fpvc):
    """Active ECLSS should significantly reduce steady-state CO vs isolated cabin.
    
    Time constants: τ_active = 1/0.03 ≈ 33 min, τ_isolated = 1/0.002 = 500 min.
    At 100 min the isolated cabin is only ~18% of its C_eq, so the ratio appears
    small (2.9×). At 600 min (well past both time constants), the ratio
    approaches the steady-state value of ~0.03/0.002 = 15×."""
    mat = MAT_DICT[MAT_NAME]

    tox_active = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=600.0, cabin_volume_m3=106.0,
        eclss_active=True)
    tox_isolated = evaluate_crew_toxicity(pred_fpvc, mat, g_level=0.0, thickness_mm=2.0,
        sample_width_mm=150.0, exposure_minutes=600.0, cabin_volume_m3=106.0,
        eclss_active=False)

    # At 600 min, active is >99% of C_eq and isolated is ~70% of C_eq
    # → expect >5× reduction even before full steady state
    ratio = tox_isolated["gases_ppm"]["CO"]["ppm"] / max(tox_active["gases_ppm"]["CO"]["ppm"], 1e-6)
    assert ratio > 5.0, f"ECLSS should reduce CO by >5× at t=600min, got {ratio:.1f}×"