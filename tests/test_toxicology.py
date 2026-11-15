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