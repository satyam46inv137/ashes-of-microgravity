import pytest
try:
    from backend.fire_model import FireModel
    from backend.materials import MAT_DICT
except ModuleNotFoundError:
    from fire_model import FireModel
    from materials import MAT_DICT

MAT = "SIBAL Standard Fabric"

S2 = dict(
    g_level=0.0, pressure_kPa=101.3, oxygen_pct=21.0,
    flow_vel_cms=20.0, thickness_mm=0.33, configuration="Opposed",
)


@pytest.fixture(scope="module")
def model():
    return FireModel(MAT_DICT)


def run(model, **over):
    return model.predict(MAT, **{**S2, **over})



def test_saffire_s2_parity(model):
    assert run(model)["flame_spread_mms"] == pytest.approx(1.343, rel=0.03)


def test_concurrent_to_opposed_ratio_is_0165(model):
    opp = run(model, configuration="Opposed")["flame_spread_mms"]
    con = run(model, configuration="Concurrent")["flame_spread_mms"]
    assert con / opp == pytest.approx(0.165, rel=0.02)


def test_below_loi_is_extinct(model):
    r = run(model, oxygen_pct=5.0)
    assert r["is_extinct"] and r["flame_spread_mms"] == 0.0
    assert "LOI" in r["extinction_reason"]


def test_quiescent_quench_only_in_microgravity(model):
    r0 = run(model, g_level=0.0, flow_vel_cms=0.3)
    assert r0["is_extinct"]
    assert "Quiescent" in r0["extinction_reason"]
    r1 = run(model, g_level=1.0, flow_vel_cms=0.3)
    assert r1["extinction_reason"] is None or "Quiescent" not in r1["extinction_reason"]


def test_blowoff_at_high_flow(model):
    r = run(model, flow_vel_cms=200.0)
    assert r["is_extinct"] and "Blow-off" in r["extinction_reason"]


def test_blowoff_limit_scales_with_pressure(model):
    assert not run(model, flow_vel_cms=40.0, pressure_kPa=101.3)["is_extinct"]
    low_p = run(model, flow_vel_cms=40.0, pressure_kPa=55.0)
    assert low_p["is_extinct"] and "Blow-off" in low_p["extinction_reason"]


@pytest.mark.parametrize("o2", [15, 18, 21, 30])
@pytest.mark.parametrize("flow", [0.3, 1, 5, 20, 40, 100])
def test_spread_never_negative(model, o2, flow):
    assert run(model, oxygen_pct=o2, flow_vel_cms=flow)["flame_spread_mms"] >= 0.0


def test_more_oxygen_not_slower(model):
    assert run(model, oxygen_pct=30.0)["flame_spread_mms"] >= run(model, oxygen_pct=21.0)["flame_spread_mms"]


def test_one_g_faster_than_zero_g(model):
    assert run(model, g_level=1.0)["flame_spread_mms"] > run(model, g_level=0.0)["flame_spread_mms"]


def test_deterministic(model):
    r1, r2 = run(model), run(model)
    for key in ("flame_spread_mms", "flame_temp_K", "v_eff_cms",
                "q_net_kWm2", "is_extinct"):
        assert r1[key] == r2[key], f"Non-deterministic field: {key}"


def test_smooth_thin_to_thick_transition(model):
    s_099 = run(model, thickness_mm=0.99)["flame_spread_mms"]
    s_101 = run(model, thickness_mm=1.01)["flame_spread_mms"]
    assert abs(s_099 - s_101) / s_099 < 0.10


def test_stefan_boltzmann_surface_loss_reported(model):
    r = run(model)
    assert r["q_surf_loss_kWm2"] > 0.0
    assert r["q_net_kWm2"] == pytest.approx(r["q_conv_kWm2"] - r["q_rad_kWm2"], abs=0.05)


def test_rod_and_sphere_geometries_faster_than_plate(model):
    plate = run(model, geometry="plate")["flame_spread_mms"]
    rod = run(model, geometry="rod")["flame_spread_mms"]
    sphere = run(model, geometry="sphere")["flame_spread_mms"]
    assert rod > 1.8 * plate
    assert sphere > 1.5 * plate


def test_quasi2d_free_vs_clamped_edges(model):
    wide = run(model, sample_width_mm=410.0, edge_boundary="free")["flame_spread_mms"]
    narrow_free = run(model, sample_width_mm=80.0, edge_boundary="free")["flame_spread_mms"]
    narrow_clamped = run(model, sample_width_mm=80.0, edge_boundary="clamped")["flame_spread_mms"]
    assert narrow_free > wide
    assert narrow_clamped < narrow_free


def test_simulate_2d_line_and_point_ignition(model):
    sim_free = model.simulate_2d(
        MAT, length_mm=150.0, width_mm=80.0, nx=30, ny=16,
        duration_s=15.0, dt=0.25, ignition_mode="line", edge_boundary="free",
        thickness_mm=0.33, flow_vel_cms=20.0,
    )
    assert sim_free["status"] == "COMPLETED"
    assert len(sim_free["snapshots"]) >= 2
    assert sim_free["final_burned_area_pct"] > 0.0
    contour_free = sim_free["snapshots"][-1]["front_contour_mm"]
    assert contour_free[0] >= contour_free[len(contour_free) // 2]

    sim_clamped = model.simulate_2d(
        MAT, length_mm=150.0, width_mm=80.0, nx=30, ny=16,
        duration_s=15.0, dt=0.25, ignition_mode="line", edge_boundary="clamped",
        thickness_mm=0.33, flow_vel_cms=20.0,
    )
    contour_clamped = sim_clamped["snapshots"][-1]["front_contour_mm"]
    assert contour_clamped[0] < contour_clamped[len(contour_clamped) // 2]

    sim_point = model.simulate_2d(
        MAT, length_mm=150.0, width_mm=80.0, nx=30, ny=16,
        duration_s=10.0, dt=0.25, ignition_mode="point", edge_boundary="free",
        thickness_mm=0.33, flow_vel_cms=20.0,
    )
    assert sim_point["status"] == "COMPLETED"
    assert sim_point["final_burned_area_pct"] > 0.0


def test_unknown_kwarg_raises(model):
    with pytest.raises(TypeError):
        model.predict(MAT, oxygen=21.0)


def test_unknown_material_raises(model):
    with pytest.raises(KeyError):
        model.predict("Unobtainium")


def test_invalid_geometry_or_edge_raises(model):
    with pytest.raises(ValueError):
        run(model, geometry="cube")
    with pytest.raises(ValueError):
        run(model, edge_boundary="welded")


def test_applicability_scoring(model):
    ok = model.applicability(MAT, 0.0, 21.0, 20.0, 0.33)
    assert ok["score"] == 100 and ok["level"] == "HIGH"
    bad = model.applicability(MAT, 0.0, 40.0, 20.0, 0.33)
    assert bad["score"] < ok["score"] and bad["issues"]


def test_sensitivity_sorted_and_empty_when_extinct(model):
    out = model.sensitivity(MAT, dict(S2))
    impacts = [r["impact"] for r in out]
    assert out and impacts == sorted(impacts, reverse=True)
    assert model.sensitivity(MAT, {**S2, "flow_vel_cms": 0.3}) == []


def test_transient_monotonic(model):
    h = model.transient_simulate(MAT, duration_s=5, dt=1.0, **S2)
    xs = [p["x_flame_mm"] for p in h]
    th = [p["thickness_remaining"] for p in h]
    assert xs == sorted(xs)
    assert th == sorted(th, reverse=True)