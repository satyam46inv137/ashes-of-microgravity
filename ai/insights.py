"""
Physics-Grounded Safety & Combustion Behavior Interpreter.
Translates quantitative outputs from FireModel (1D, Quasi-2D, and 2D PDE simulations)
into actionable spacecraft fire-safety engineering insights.
"""

from typing import Any, Dict, List

try:
    from backend.fire_model import QUIESCENT_FLOW_CMS
except ImportError:
    from fire_model import QUIESCENT_FLOW_CMS

SAFETY_KEYWORDS = {"safe", "safety", "risk", "hazard", "danger", "extinguish", "prevent"}
BEHAVIOR_KEYWORDS = {"why", "how", "explain", "reason", "cause", "mechanism"}
NARROW_SHEET_MM = 200.0   # Inlined in _predict_core; keep in sync if threshold changes


def _is_safety_query(query_context):
    return bool(query_context) and any(kw in query_context.lower() for kw in SAFETY_KEYWORDS)


def _is_behavior_query(query_context):
    return bool(query_context) and any(kw in query_context.lower() for kw in BEHAVIOR_KEYWORDS)


def _build_extinction_insight(result, material, query_context):
    mat_name = material.get("name", "Unknown Material")
    reason = result.get("extinction_reason", "Quenched")
    detail = result.get("extinction_detail", "")
    edge_boundary = result.get("edge_boundary", "free")

    key_points: List[str] = []
    recommendations: List[str] = []
    reason_lower = reason.lower()

    if "blow-off" in reason_lower or "blowoff" in reason_lower:
        key_points.append(
            "Aerodynamic strain limit reached: effective flow velocity exceeds "
            "the residence time required for stable pyrolytic kinetics "
            "(Damkohler blow-off regime; Fernandez-Pello & Hirano, 1983)."
        )
        recommendations.append(
            "Dampen forced ventilation draft below the critical strain threshold, "
            "or initiate rapid inerting (N2 purge)."
        )
    elif "loi" in reason_lower or "flammability" in reason_lower:
        key_points.append(
            "Effective oxygen concentration is below the pressure- and gravity-adjusted "
            "Limiting Oxygen Index (LOI; Olson et al., 1988)."
        )
        recommendations.append(
            "The model predicts self-extinguishing behavior at this ambient O2 / pressure "
            "regime; confirm experimentally before relying on it."
        )
    elif "conductive" in reason_lower and edge_boundary == "clamped":
        key_points.append(
            "Quasi-2D lateral conductive quenching: heat sink losses into the clamped "
            "sample frame exceed convective heat input (sub-critical quenching width; Quintiere, 2006)."
        )
        recommendations.append(
            "Mounting thin panels in thermally conductive metallic frames may provide passive "
            "flame-arresting protection at this width; verify experimentally."
        )
    else:
        key_points.append(
            "Radiative quenching regime: Stefan-Boltzmann surface re-radiation and spherical "
            "gas-phase radiative losses exceed convective heat feedback (Bhattacharjee et al., 1996)."
        )
        if "quiescent" in reason_lower:
            recommendations.append(
                f"Extinction here relies on near-zero ventilation (< {QUIESCENT_FLOW_CMS} cm/s), which is a "
                "modeled limit. Do not treat low airflow as a fire-protection measure without "
                "flight-validated evidence; stagnant air also hinders smoke detection and cooling."
            )
        else:
            recommendations.append(
                "Radiative losses dominate at these conditions; confirm margin against "
                "flight data before relying on passive quenching."
            )

    if detail:
        key_points.append(f"Extinction detail: {detail}")

    return {
        "summary": (
            f"{mat_name} fails to sustain combustion under these conditions "
            f"due to {reason_lower}."
        ),
        "regime": "Extinction / Non-Flammable",
        "key_points": key_points,
        "safety_implications": (
            f"The model predicts no sustained flame spread under these conditions "
            f"(applicability: {result.get('applicability_level', result.get('confidence_level', 'UNKNOWN'))}). "
            "This is a model prediction, not a safety guarantee: ignition and transient burning are "
            "not evaluated, and unburnt toxic pyrolyzate accumulation should still be checked."
        ),
        "recommendations": recommendations,
        "engine_source": "Physics-informed diagnostic rules (de Ris / Quintiere / Saffire criteria)",
    }


def _build_active_insight(result, material, query_context):
    mat_name = material.get("name", "Unknown Material")
    cat = material.get("category", "")
    cat_lower = cat.lower()

    spread = float(result.get("flame_spread_mms", 0.0))
    uncert = float(result.get("flame_spread_uncertainty_mms", 0.0))
    temp = float(result.get("effective_flame_temp_K", result.get("flame_temp_K", 300.0)))
    q_net = float(result.get("q_net_kWm2", 0.0))
    q_surf = float(result.get("q_surf_loss_kWm2", 0.0))
    v_eff = float(result.get("v_eff_cms", 0.0))
    regime_thermal = result.get("thermal_regime", "Thick")
    edge_boundary = result.get("edge_boundary", "free")
    width_mm = float(result.get("sample_width_mm", 410.0))
    conf = result.get("applicability_level", result.get("confidence_level", "MEDIUM"))

    is_rapid = spread > 1.0
    regime = f"{'High Risk / Rapid' if is_rapid else 'Controlled / Low-Velocity'} ({regime_thermal} Fuel)"

    summary = (
        f"{mat_name} sustains a flame spread rate of {spread:.3f} ± {uncert:.3f} mm/s "
        f"({regime_thermal} thermal regime) at {temp:.1f} K "
        f"(Net Flux: {q_net:.2f} kW/m2, Surface Rad Loss: {q_surf:.2f} kW/m2)."
    )

    key_points: List[str] = []
    recommendations: List[str] = []

    if is_rapid:
        key_points.append(
            f"Rapid flame front propagation ({spread:.3f} mm/s): convective preheating "
            "efficiently warms unburnt fuel ahead of the pyrolysis zone."
        )
    else:
        key_points.append(
            f"Sluggish flame progression ({spread:.3f} mm/s): thermal inertia (k*rho*c) "
            "and surface radiative losses limit preheat zone advance."
        )

    # Quasi-2D width & boundary condition insights
    if edge_boundary == "free" and width_mm < NARROW_SHEET_MM:
        key_points.append(
            f"Quasi-2D narrow-sheet acceleration (width {width_mm:.0f} mm, free edges): "
            "lateral oxygen entrainment and two-sided corner heating accelerate edge flame spread."
        )
    elif edge_boundary == "clamped":
        key_points.append(
            f"Quasi-2D clamped edge damping (width {width_mm:.0f} mm): conductive heat sink "
            "at boundaries slows edge spread relative to the sample centerline (parabolic front)."
        )

    if any(tag in cat_lower for tag in ("commodity", "foam", "thermoplastic")):
        key_points.append(
            "Thermoplastic degradation risk: material melts with potential liquid "
            "fuel run-off. In microgravity, droplets detach and float as secondary "
            "ignition hazards."
        )
        recommendations.append(
            "Implement drip-containment barriers and flame-retardant outer sheath "
            "(Nomex 450 / Kapton wrap)."
        )
    elif any(tag in cat_lower for tag in ("thermoset", "wood", "bio")):
        key_points.append(
            "Char-forming barrier dynamic: porous carbonaceous char builds up, "
            "reducing effective thermal conductivity (k_eff) and damping further pyrolysis."
        )
        recommendations.append(
            "Monitor for deep-seated smoldering combustion behind the char layer post-extinction."
        )

    if v_eff > 25.0:
        key_points.append(
            f"High convective shear ({v_eff:.1f} cm/s): approaching Damkohler aerodynamic "
            "stretch regime, thinning boundary layer standoff distance."
        )
    elif v_eff < 2.0:
        key_points.append(
            "Near-quiescent microgravity regime: convective heat transfer suppressed, "
            "flame assumes semi-spherical profile dominated by radiative dissipation."
        )

    recommendations.append(
        "Use water-mist fine spray or clean-agent gas suppression to avoid "
        "spreading hot molten debris in reduced gravity."
    )

    if _is_safety_query(query_context):
        key_points.append(
            "Safety-focused query detected: prioritize suppression system readiness "
            "and material substitution (e.g., Nomex 450) for crewed habitats."
        )
    elif _is_behavior_query(query_context):
        key_points.append(
            "Behavior-focused query detected: cross-reference with NASA Saffire-I/II "
            "flight telemetry for microgravity flame morphology benchmarks."
        )

    return {
        "summary": summary,
        "regime": regime,
        "key_points": key_points,
        "safety_implications": (
            f"Flammability hazard evaluated at {conf} applicability against NASA "
            "Saffire/BASS-II flight telemetry."
        ),
        "recommendations": recommendations,
        "engine_source": "Physics-informed diagnostic rules (de Ris / Quintiere / Saffire criteria)",
    }


def generate_insight(result, material, query_context=""):
    if not result:
        return {
            "summary": "No simulation result provided.",
            "regime": "Unknown",
            "key_points": [],
            "safety_implications": "",
            "recommendations": [],
            "engine_source": "N/A",
        }

    insight = (
        _build_extinction_insight(result, material, query_context)
        if result.get("is_extinct", False)
        else _build_active_insight(result, material, query_context)
    )

    tox = result.get("toxicology")
    if tox:
        sev = tox.get("triage_severity", "NOMINAL")
        lim = tox.get("limiting_toxicant", "CO")
        ratio = tox.get("max_smac_1h_ratio", 0.0)
        cohb = tox.get("estimated_cohb_pct", 0.8)
        fed = tox.get("purser_fed_total", 0.0)
        insight["key_points"].append(
            f"Crew Toxicology ({sev}): Limiting gas {lim} reaches {ratio:.2f}x NASA 1-hr SMAC; "
            f"predicted COHb = {cohb:.1f}% and ISO 13571 Purser FED = {fed:.3f} ({tox.get('primary_syndrome', '')})."
        )
        if sev in ("WARNING", "CRITICAL"):
            insight["recommendations"].insert(0, tox.get("triage_action", "Don PBA 100% O2 mask immediately."))

    return insight