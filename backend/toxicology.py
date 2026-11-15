"""
================================================================================
SPACECRAFT CREW TOXICOLOGY & PHYSIOLOGICAL TRAUMA ENGINE (NASA SMACs & ISO 13571)
================================================================================
Integrates microgravity combustion physics (FireModel) with:
  1. Tewarson (SFPE Handbook Table 3-4.14) material-specific pyrolytic yields
  2. Dynamic 1D Cabin Mass-Balance ODE with ECLSS Trace Contaminant Control decay
  3. ISO 13571 / Purser Fractional Effective Dose (FED) with CO2 hyperventilation
  4. Coburn-Forster-Kane (CFK) Carboxyhemoglobin (COHb%) kinetics & 100% O2 PBA washout
  5. NASA JSC 20584 Spacecraft Maximum Allowable Concentrations (SMACs)
================================================================================
"""

import math
from typing import Any, Dict, List, Optional

# NASA JSC 20584 Spacecraft Maximum Allowable Concentrations [ppm]
NASA_SMAC_LIMITS_PPM: Dict[str, Dict[str, float]] = {
    "CO": {"1h": 55.0, "24h": 20.0, "idlh": 1200.0},
    "HCN": {"1h": 4.5, "24h": 1.5, "idlh": 50.0},
    "HCl": {"1h": 5.0, "24h": 1.0, "idlh": 50.0},
    "HF": {"1h": 3.0, "24h": 0.5, "idlh": 30.0},
    "NO2": {"1h": 1.0, "24h": 0.2, "idlh": 20.0},
}

# Standard NASA spacecraft habitable volumes [m^3]
HABITAT_VOLUMES_M3: Dict[str, float] = {
    "ISS Destiny Lab": 106.0,
    "Orion / Dragon Capsule": 9.3,
    "Lunar Gateway HALO": 60.0,
    "Mars Transit Habitat": 250.0,
}

# Crew physiological respiratory minute volume V_E [L/min] by activity state
CREW_ACTIVITY_VE_LMIN: Dict[str, float] = {
    "resting": 8.5,          # Nominal station monitoring / sleep
    "light_work": 15.0,      # Routine IVA operations
    "emergency": 32.0,       # Fire suppression / emergency egress
}

GAS_MOLAR_MASS: Dict[str, float] = {
    "CO": 28.01,
    "HCN": 27.03,
    "HCl": 36.46,
    "HF": 20.01,
    "NO2": 46.01,
    "CO2": 44.01,
}

# Exact SFPE Handbook (Tewarson, Table 3-4.14) well-ventilated yields [g/g]
# for key aerospace & spacecraft cabin materials
SFPE_MATERIAL_YIELDS: Dict[str, Dict[str, Any]] = {
    "PMMA (Acrylic / Plexiglas)": {
        "chemical_class": "Methacrylate Polymer (SFPE Calibrated)",
        "yields_gg": {"CO": 0.010, "HCN": 0.0, "HCl": 0.0, "HF": 0.0, "NO2": 0.001, "CO2": 2.12},
        "soot_yield_gg": 0.022,
        "primary_syndrome": "Carbon Monoxide Hypoxia & Micro-g Soot Deposition",
        "clinical_mechanism": "Clean-burning methacrylate yields low soot in 1g, but oxygen-starved 0g diffusion flames shift pyrolysis toward elevated CO and suspended sub-micron aerosols.",
    },
    "SIBAL Standard Fabric": {
        "chemical_class": "Cellulose-Fiberglass Blend (Saffire Flight Fuel)",
        "yields_gg": {"CO": 0.032, "HCN": 0.0, "HCl": 0.0, "HF": 0.0, "NO2": 0.002, "CO2": 1.45},
        "soot_yield_gg": 0.018,
        "primary_syndrome": "Cellulosic Smolder CO Asphyxiation & Aldehyde Irritation",
        "clinical_mechanism": "Cellulose pyrolysis releases CO, acrolein, and formaldehyde; non-settling fiberglass/char particulates pose mechanical bronchoalveolar irritation in 0g.",
    },
    "Nomex 450 (Aramid Fabric)": {
        "chemical_class": "Meta-Aramid Polymer (SFPE / NASA Flight Fuel)",
        "yields_gg": {"CO": 0.068, "HCN": 0.028, "HCl": 0.0, "HF": 0.0, "NO2": 0.019, "CO2": 1.60},
        "soot_yield_gg": 0.055,
        "primary_syndrome": "Histotoxic Cellular Hypoxia (HCN + CO Synergism)",
        "clinical_mechanism": "High thermal resistance delays ignition, but forced pyrolysis of amide linkages liberates HCN (cytochrome c oxidase inhibitor) and NO2 alongside CO.",
    },
    "Polyimide (PI / Kapton)": {
        "chemical_class": "Aromatic Polyimide Aerospace Film (SFPE)",
        "yields_gg": {"CO": 0.075, "HCN": 0.032, "HCl": 0.0, "HF": 0.0, "NO2": 0.022, "CO2": 1.55},
        "soot_yield_gg": 0.045,
        "primary_syndrome": "Avionics Wire-Arc HCN / CO Neurotoxicity",
        "clinical_mechanism": "Electrical arc pyrolysis of Kapton insulation generates synergistic HCN and CO in confined avionics bays, rapidly impairing crew executive function.",
    },
    "Flexible Polyvinyl Chloride (f-PVC)": {
        "chemical_class": "Plasticized Organochlorine Polymer (SFPE)",
        "yields_gg": {"CO": 0.063, "HCN": 0.0, "HCl": 0.310, "HF": 0.0, "NO2": 0.002, "CO2": 0.95},
        "soot_yield_gg": 0.172,
        "primary_syndrome": "Acute Corrosive Airway Edema & Laryngospasm (HCl)",
        "clinical_mechanism": "Dehydrochlorination releases ~0.31 g HCl per gram burned, forming hydrochloric acid on moist mucosa and causing rapid upper airway obstruction.",
    },
    "Rigid Polyvinyl Chloride (r-PVC)": {
        "chemical_class": "Unplasticized Organochlorine Polymer (SFPE)",
        "yields_gg": {"CO": 0.065, "HCN": 0.0, "HCl": 0.340, "HF": 0.0, "NO2": 0.002, "CO2": 0.90},
        "soot_yield_gg": 0.165,
        "primary_syndrome": "Severe Acid-Gas Chemical Pneumonitis (HCl)",
        "clinical_mechanism": "High chlorine fraction produces dense acidic smoke and HCl gas even during non-propagating thermal degradation.",
    },
    "Polytetrafluoroethylene (PTFE / Teflon)": {
        "chemical_class": "Perfluoropolymer (SFPE / NASA Offgassing)",
        "yields_gg": {"CO": 0.090, "HCN": 0.0, "HCl": 0.0, "HF": 0.180, "NO2": 0.004, "CO2": 0.55},
        "soot_yield_gg": 0.005,
        "primary_syndrome": "Polymer Fume Fever & Delayed Pulmonary Edema (HF / PFIB)",
        "clinical_mechanism": "Despite extreme LOI (95%), overheating above 700 K liberates HF and ultra-toxic Perfluoroisobutylene (PFIB), causing delayed alveolar flooding.",
    },
    "Flexible Polyurethane Foam (FPUR)": {
        "chemical_class": "Isocyanate-Polyol Foam (SFPE)",
        "yields_gg": {"CO": 0.031, "HCN": 0.019, "HCl": 0.0, "HF": 0.0, "NO2": 0.014, "CO2": 1.52},
        "soot_yield_gg": 0.131,
        "primary_syndrome": "Rapid Flashover Asphyxiation & Isocyanate Bronchospasm",
        "clinical_mechanism": "Ultra-low thermal inertia drives rapid mass loss, releasing HCN, CO, and toluene diisocyanate aerosols that trigger severe reactive airway constriction.",
    },
    "Rigid Polyurethane Foam (RPUR)": {
        "chemical_class": "Crosslinked Isocyanate Insulation (SFPE)",
        "yields_gg": {"CO": 0.048, "HCN": 0.024, "HCl": 0.0, "HF": 0.0, "NO2": 0.016, "CO2": 1.48},
        "soot_yield_gg": 0.145,
        "primary_syndrome": "Cyanide-CO Mixed Hypoxia & Dense Soot Obscuration",
        "clinical_mechanism": "High aromaticity and nitrogen content yield dense carbonaceous soot and HCN, rapidly reducing cabin visibility and cellular O2 utilization.",
    },
    "FR-4 PCB Laminate (Epoxy-Glass FR)": {
        "chemical_class": "Brominated Epoxy Electronics Substrate (SFPE)",
        "yields_gg": {"CO": 0.115, "HCN": 0.014, "HCl": 0.185, "HF": 0.0, "NO2": 0.011, "CO2": 1.15},
        "soot_yield_gg": 0.158,
        "primary_syndrome": "Avionics Rack Halogenated Acid-Gas & CO Toxicity",
        "clinical_mechanism": "Tetrabromobisphenol-A flame retardants suppress flaming by poisoning gas-phase radicals, intentionally spiking CO and corrosive hydrogen halide yields.",
    },
    "Polyamide 6 (Nylon 6)": {
        "chemical_class": "Aliphatic Polyamide (SFPE)",
        "yields_gg": {"CO": 0.038, "HCN": 0.026, "HCl": 0.0, "HF": 0.0, "NO2": 0.012, "CO2": 2.05},
        "soot_yield_gg": 0.075,
        "primary_syndrome": "HCN Histotoxic Hypoxia & Molten Polymer Thermal Burns",
        "clinical_mechanism": "Generates HCN during high-temperature pyrolysis alongside low-viscosity burning melt droplets that float in 0g and cause deep contact burns.",
    },
    "Acrylonitrile Butadiene Styrene (ABS)": {
        "chemical_class": "Styrenic Terpolymer (SFPE)",
        "yields_gg": {"CO": 0.062, "HCN": 0.021, "HCl": 0.0, "HF": 0.0, "NO2": 0.009, "CO2": 2.15},
        "soot_yield_gg": 0.185,
        "primary_syndrome": "HCN + CO Asphyxiation & Heavy Aromatic Soot",
        "clinical_mechanism": "Nitrile groups release HCN while styrene rings nucleate dense black soot aggregates that stay suspended indefinitely in microgravity.",
    },
    "General Purpose Polystyrene (GPPS)": {
        "chemical_class": "Aromatic Vinyl Polymer (SFPE)",
        "yields_gg": {"CO": 0.060, "HCN": 0.0, "HCl": 0.0, "HF": 0.0, "NO2": 0.002, "CO2": 2.33},
        "soot_yield_gg": 0.164,
        "primary_syndrome": "Dense Aromatic Soot Inhalation & CO Hypoxia",
        "clinical_mechanism": "Highest soot yield among commodity plastics; suspended polycyclic aromatic hydrocarbons (PAHs) cause acute tracheobronchitis.",
    },
    "Polyether Ether Ketone (PEEK)": {
        "chemical_class": "Polyaromatic Ketone Aerospace Polymer (SFPE)",
        "yields_gg": {"CO": 0.042, "HCN": 0.0, "HCl": 0.0, "HF": 0.0, "NO2": 0.002, "CO2": 1.85},
        "soot_yield_gg": 0.035,
        "primary_syndrome": "Low-Smoke High-Threshold CO Exposure",
        "clinical_mechanism": "High char yield and pyrolysis threshold (850 K) minimize toxic gas release, making PEEK a benchmark crew-safe structural polymer.",
    },
}


def _classify_toxic_profile(material: Dict[str, Any]) -> Dict[str, Any]:
    """Returns exact SFPE yields if available, otherwise chemical-backbone class defaults."""
    raw_name = material.get("name") or ""
    if raw_name in SFPE_MATERIAL_YIELDS:
        return SFPE_MATERIAL_YIELDS[raw_name]

    name = raw_name.lower()
    cat = (material.get("category") or "").lower()

    if any(k in name for k in ("ptfe", "teflon", "fep", "pfa", "pvdf", "fluoro")):
        return {
            "chemical_class": "Fluoropolymer",
            "yields_gg": {"CO": 0.085, "HCN": 0.0, "HCl": 0.0, "HF": 0.16, "NO2": 0.004, "CO2": 0.60},
            "soot_yield_gg": 0.01,
            "primary_syndrome": "Polymer Fume Fever & Acute Chemical Pneumonitis (HF / PFIB)",
            "clinical_mechanism": "Pyrolysis releases Hydrogen Fluoride (HF) and Perfluoroisobutylene (PFIB), causing delayed alveolar-capillary membrane breakdown.",
        }

    if any(k in name for k in ("pvc", "vinyl", "cpvc", "modacrylic", "chlor")):
        return {
            "chemical_class": "Organochlorine Polymer",
            "yields_gg": {"CO": 0.075, "HCN": 0.0, "HCl": 0.29, "HF": 0.0, "NO2": 0.002, "CO2": 0.95},
            "soot_yield_gg": 0.15,
            "primary_syndrome": "Upper Airway Corrosive Edema & Bronchospasm (HCl)",
            "clinical_mechanism": "Hydrogen Chloride (HCl) forms hydrochloric acid on moist respiratory mucosa, triggering rapid laryngeal edema.",
        }

    if any(k in name for k in (
        "polyurethane", "pur", "pir", "nylon", "polyamide", "kapton", "polyimide",
        "nomex", "kevlar", "aramid", "melamine", "urea", "abs", "nitrile", "nbr",
        "acrylic", "wool", "silk", "leather", "nitrocellulose"
    )):
        return {
            "chemical_class": "Nitrogenous Polymer / Protein Fiber",
            "yields_gg": {"CO": 0.065, "HCN": 0.025, "HCl": 0.0, "HF": 0.0, "NO2": 0.015, "CO2": 1.65},
            "soot_yield_gg": 0.085,
            "primary_syndrome": "Histotoxic Cellular Hypoxia (HCN + CO Synergism)",
            "clinical_mechanism": "Hydrogen Cyanide (HCN) inhibits mitochondrial cytochrome c oxidase, halting cellular ATP synthesis alongside CO hypoxia.",
        }

    if "fr-4" in name or "brom" in name:
        return {
            "chemical_class": "Halogenated Epoxy Composite",
            "yields_gg": {"CO": 0.11, "HCN": 0.012, "HCl": 0.17, "HF": 0.0, "NO2": 0.01, "CO2": 1.15},
            "soot_yield_gg": 0.15,
            "primary_syndrome": "Acid Gas Airway Irritation & Asphyxiant Smoke (HBr/HCl + CO)",
            "clinical_mechanism": "Halogenated flame retardants quench gas-phase radicals, spiking incomplete-combustion CO and corrosive acid halides.",
        }

    if "metal" in cat or any(k in name for k in ("lithium", "magnesium", "titanium", "zirconium")):
        return {
            "chemical_class": "Reactive / Combustible Metal",
            "yields_gg": {"CO": 0.0, "HCN": 0.0, "HCl": 0.0, "HF": 0.0, "NO2": 0.015, "CO2": 0.0},
            "soot_yield_gg": 0.18,
            "primary_syndrome": "Alkaline Metal-Oxide Pneumonitis & UV Photokeratitis",
            "clinical_mechanism": "Fine metal-oxide aerosols (Li2O, MgO) form caustic hydroxides in alveoli; intense UV radiation causes corneal burns.",
        }

    soot_heavy = any(k in name for k in ("styrene", "ps", "rubber", "sbr", "epdm", "toluene", "kerosene", "epoxy"))
    return {
        "chemical_class": "Hydrocarbon / Cellulosic Fuel",
        "yields_gg": {"CO": 0.045, "HCN": 0.0, "HCl": 0.0, "HF": 0.0, "NO2": 0.003, "CO2": 1.95},
        "soot_yield_gg": 0.14 if soot_heavy else 0.03,
        "primary_syndrome": "Carbon Monoxide Asphyxiation & Micro-g Soot Aspiration",
        "clinical_mechanism": "CO binds hemoglobin with ~210x O2 affinity, impairing peripheral O2 unloading; non-settling 0g soot deposits in distal alveoli.",
    }


def _compute_mass_burn_rate_gs(
    pred_result: Dict[str, Any],
    material: Dict[str, Any],
    g_level: float,
    thickness_mm: float,
    sample_width_mm: float,
) -> tuple[float, str]:
    """Calculates solid mass pyrolysis/burning rate m_dot [g/s]."""
    is_extinct = bool(pred_result.get("is_extinct", False))
    spread_mms = float(pred_result.get("flame_spread_mms", 0.0))
    q_net_kWm2 = float(pred_result.get("q_net_kWm2", 0.0))
    is_microg = g_level < 0.01

    if is_extinct or spread_mms <= 1e-6:
        return 0.012, "Non-Propagating / Smolder Off-Gassing"

    rho_s = float(material.get("rho_s", 1200.0))
    c_s = float(material.get("c_s", 1400.0))
    T_pyro = float(material.get("T_pyro", 600.0))
    L_v = float(material.get("L_v", 1.8e6))
    H_g = max(c_s * max(T_pyro - 298.15, 50.0) + L_v, 5e5)

    a_pyro_m2 = (sample_width_mm / 1000.0) * 0.015
    m_dot_flux = (q_net_kWm2 * 1000.0) / H_g
    v_f_ms = spread_mms / 1000.0
    m_dot_front = rho_s * (thickness_mm / 1000.0) * (sample_width_mm / 1000.0) * v_f_ms
    m_dot_gs = max(0.02, (m_dot_flux * a_pyro_m2 + 0.5 * m_dot_front) * 1000.0)
    mode = "Active Microgravity Combustion" if is_microg else "Active Buoyant Combustion"
    return m_dot_gs, mode


def evaluate_crew_toxicity(
    pred_result: Dict[str, Any],
    material: Dict[str, Any],
    g_level: float = 0.0,
    pressure_kPa: float = 101.3,
    oxygen_pct: float = 21.0,
    thickness_mm: float = 2.0,
    sample_width_mm: float = 150.0,
    exposure_minutes: float = 15.0,
    cabin_volume_m3: float = 106.0,
    eclss_active: bool = True,
    crew_activity: str = "light_work",
    mask_don_minute: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Solves the coupled 1-minute ODE cabin gas mass balance & physiological uptake:
      1. Cabin concentration ODE: dC_i/dt = (m_dot * y_i / V_cabin) - lambda_eclss * C_i
      2. ISO 13571 / Purser Fractional Effective Dose (FED) with CO2 hyperventilation
      3. Coburn-Forster-Kane (CFK) COHb% uptake & 100% O2 PBA washout after mask_don_minute
    """
    profile = _classify_toxic_profile(material)
    is_microg = g_level < 0.01
    incomplete_comb_factor = 2.1 if is_microg else (1.4 if g_level < 0.5 else 1.0)

    mass_burn_rate_gs, mode_label = _compute_mass_burn_rate_gs(
        pred_result, material, g_level, thickness_mm, sample_width_mm
    )

    # ECLSS Trace Contaminant Control + HEPA air scrubbers (~1.8 air exchanges/hr = 0.03/min)
    lambda_eclss_min = 0.030 if eclss_active else 0.002  # 0.002/min passive wall adsorption

    # Crew minute ventilation V_E [L/min]
    activity_norm = crew_activity.lower().strip()
    v_e_base = CREW_ACTIVITY_VE_LMIN.get(activity_norm, 15.0)
    vent_scale = v_e_base / 10.0  # Normalized to 10 L/min reference

    # Ideal gas molar volume [L/mol] at cabin T=298.15 K, P=pressure_kPa
    v_molar_L = (8.3145 * 298.15) / max(pressure_kPa, 10.0)

    # Precompute generation rate [ppm/min] for each species
    gen_ppm_min: Dict[str, float] = {}
    for gas, base_yield in profile["yields_gg"].items():
        eff_yield = base_yield * (incomplete_comb_factor if gas in ("CO", "HCN") else 1.0)
        mg_per_min = mass_burn_rate_gs * 60.0 * eff_yield * 1000.0
        mg_m3_per_min = mg_per_min / max(cabin_volume_m3, 1.0)
        gen_ppm_min[gas] = mg_m3_per_min * (v_molar_L / GAS_MOLAR_MASS[gas])

    soot_gen_mg_m3_min = (
        mass_burn_rate_gs * 60.0 * profile["soot_yield_gg"] * incomplete_comb_factor * 1000.0
    ) / max(cabin_volume_m3, 1.0)

    # Integrate ODE minute-by-minute across exposure_minutes (dt = 1.0 min)
    n_steps = max(1, int(math.ceil(exposure_minutes)))
    conc_ppm: Dict[str, float] = {g: 0.0 for g in GAS_MOLAR_MASS}
    conc_ppm["CO2"] = 4000.0  # Typical ISS baseline CO2 ~ 4,000 ppm (0.4%)
    soot_mg_m3 = 0.0

    cohb_pct = 0.8             # Normal non-smoker baseline COHb%
    fed_asphyxiant = 0.0       # ISO 13571 cumulative asphyxiant FED (CO + HCN)
    fec_irritant = 0.0         # ISO 13571 instantaneous irritant FEC (HCl + HF + NO2)

    timeline: List[Dict[str, Any]] = []
    time_to_incapacitation_min: Optional[float] = None
    time_to_1h_smac_min: Optional[float] = None

    for minute in range(1, n_steps + 1):
        # Update cabin concentrations via exact exponential step
        decay = math.exp(-lambda_eclss_min * 1.0)
        for gas in GAS_MOLAR_MASS:
            baseline = 4000.0 if gas == "CO2" else 0.0
            c_eq = baseline + (gen_ppm_min[gas] / max(lambda_eclss_min, 1e-6))
            conc_ppm[gas] = c_eq + (conc_ppm[gas] - c_eq) * decay

        soot_eq = soot_gen_mg_m3_min / max(lambda_eclss_min, 1e-6)
        soot_mg_m3 = soot_eq + (soot_mg_m3 - soot_eq) * decay

        # Check first breach of any 1-hour NASA SMAC limit
        if time_to_1h_smac_min is None:
            for g_check, limits in NASA_SMAC_LIMITS_PPM.items():
                if conc_ppm[g_check] >= limits["1h"]:
                    time_to_1h_smac_min = float(minute)
                    break

        # Check if crew has donned 100% O2 Portable Breathing Apparatus (PBA)
        mask_active = (mask_don_minute is not None) and (minute >= mask_don_minute)

        if mask_active:
            # 100% O2 PBA eliminates further toxic inhalation and washes out COHb (t_1/2 = 74 min)
            k_washout = math.log(2.0) / 74.0
            cohb_pct = 0.8 + (cohb_pct - 0.8) * math.exp(-k_washout * 1.0)
            fec_irritant = 0.0
        else:
            # Purser CO2 hyperventilation factor: V_CO2 = exp(0.1903 * %CO2 + 2.0004) / 7.1
            co2_pct = conc_ppm["CO2"] / 10000.0
            v_co2 = max(1.0, math.exp(0.1903 * co2_pct + 2.0004) / 7.1)
            effective_vent = vent_scale * v_co2

            # CFK COHb% uptake step (tau ~ 220 min / effective_vent)
            cohb_eq = min(75.0, 0.8 + 0.155 * conc_ppm["CO"])
            tau_cfk = 220.0 / max(effective_vent, 0.5)
            cohb_pct = cohb_eq + (cohb_pct - cohb_eq) * math.exp(-1.0 / tau_cfk)

            # ISO 13571 Asphyxiant FED increment (CO + HCN, scaled by hyperventilation)
            dfed_co = (conc_ppm["CO"] / 35000.0) * effective_vent
            dfed_hcn = ((conc_ppm["HCN"] ** 2.36) / 1.2e6) * effective_vent if conc_ppm["HCN"] > 0 else 0.0
            fed_asphyxiant += dfed_co + dfed_hcn

            # ISO 13571 Instantaneous Irritant FEC (HCl / 1000 + HF / 500 + NO2 / 250)
            fec_irritant = (
                conc_ppm["HCl"] / 1000.0
                + conc_ppm["HF"] / 500.0
                + conc_ppm["NO2"] / 250.0
            )

        fed_total = fed_asphyxiant + fec_irritant
        if time_to_incapacitation_min is None and (fed_total >= 0.30 or cohb_pct >= 20.0):
            # FED = 0.30 is ISO 13571 threshold for vulnerable/impaired performance incapacitation
            time_to_incapacitation_min = float(minute)

        timeline.append({
            "minute": minute,
            "co_ppm": round(conc_ppm["CO"], 2),
            "hcn_ppm": round(conc_ppm["HCN"], 3),
            "hcl_ppm": round(conc_ppm["HCl"], 2),
            "hf_ppm": round(conc_ppm["HF"], 3),
            "soot_mg_m3": round(soot_mg_m3, 2),
            "cohb_pct": round(cohb_pct, 2),
            "fed_total": round(fed_total, 4),
            "mask_active": mask_active,
        })

    # Final snapshot gas ratios against NASA 1h & 24h SMACs
    gases_report: Dict[str, Any] = {}
    max_smac_1h_ratio = 0.0
    limiting_gas = "CO"

    for gas, limits in NASA_SMAC_LIMITS_PPM.items():
        ppm_val = conc_ppm[gas]
        ratio_1h = ppm_val / limits["1h"]
        if ratio_1h > max_smac_1h_ratio:
            max_smac_1h_ratio = ratio_1h
            limiting_gas = gas

        gases_report[gas] = {
            "ppm": round(ppm_val, 2),
            "nasa_smac_1h_ppm": limits["1h"],
            "nasa_smac_24h_ppm": limits["24h"],
            "smac_1h_ratio": round(ratio_1h, 2),
            "exceeds_1h_smac": ppm_val >= limits["1h"],
        }

    final_fed = timeline[-1]["fed_total"]
    final_cohb = timeline[-1]["cohb_pct"]
    total_mass_burned_g = mass_burn_rate_gs * (exposure_minutes * 60.0)

    # Clinical Triage Classification
    if final_fed >= 0.30 or max_smac_1h_ratio >= 2.0 or final_cohb >= 15.0:
        triage_severity = "CRITICAL"
        triage_action = "CRITICAL — IMMEDIATE PBA 100% O2 DONNING & MODULE ISOLATION"
    elif max_smac_1h_ratio >= 1.0 or final_cohb >= 5.0 or soot_mg_m3 >= 15.0:
        triage_severity = "WARNING"
        triage_action = "WARNING — DON OXYGEN MASK (PBA) & BOOST ECLSS SCRUBBING"
    elif max_smac_1h_ratio >= 0.35 or soot_mg_m3 >= 3.0:
        triage_severity = "CAUTION"
        triage_action = "CAUTION — ACTIVATE TRACE CONTAMINANT CONTROL (TCCS)"
    else:
        triage_severity = "NOMINAL"
        triage_action = "NOMINAL — WITHIN NASA 1-HR SMAC ENVELOPE"

    clinical_pearls: List[str] = [
        (
            f"Pulse Oximetry Blindspot: Standard SpO2 sensors read Carboxyhemoglobin "
            f"(est. {final_cohb:.1f}% COHb) as Oxyhemoglobin, displaying falsely normal SpO2 (>97%) "
            f"despite cellular hypoxia. Use CO-oximetry or exhaled CO breath analysis."
        ),
        (
            f"Zero-G Aerosol Persistence: Without gravitational sedimentation, sub-micron soot "
            f"({soot_mg_m3:.1f} mg/m3) remains suspended indefinitely, driving deep alveolar "
            "deposition and corneal irritation until HEPA/ECLSS filtration clears the cabin."
        ),
    ]

    if gases_report["HCN"]["ppm"] > 1.0:
        clinical_pearls.append(
            f"Cyanide Antidote Protocol: Elevated HCN ({gases_report['HCN']['ppm']:.2f} ppm) blocks "
            "mitochondrial cytochrome c oxidase. Administer Hydroxocobalamin (Cyanokit, 5 g IV) — "
            "avoid nitrites which induce methemoglobinemia and worsen CO hypoxia."
        )
    if gases_report["HCl"]["ppm"] > 2.0 or gases_report["HF"]["ppm"] > 1.0:
        clinical_pearls.append(
            "Delayed Pulmonary Edema Watch: Corrosive acid halides (HCl/HF) cause delayed alveolar "
            "capillary leak 6–24 hours post-exposure; mandate serial SpO2/auscultation checks."
        )

    return {
        "combustion_mode": mode_label,
        "chemical_class": profile["chemical_class"],
        "sfpe_calibrated": material.get("name") in SFPE_MATERIAL_YIELDS,
        "cabin_volume_m3": cabin_volume_m3,
        "exposure_minutes": exposure_minutes,
        "eclss_active": eclss_active,
        "crew_activity": activity_norm,
        "mask_don_minute": mask_don_minute,
        "mass_burn_rate_gs": round(mass_burn_rate_gs, 4),
        "total_mass_burned_g": round(total_mass_burned_g, 2),
        "incomplete_combustion_multiplier": incomplete_comb_factor,
        "gases_ppm": gases_report,
        "suspended_soot_mg_m3": round(soot_mg_m3, 2),
        "estimated_cohb_pct": final_cohb,
        "purser_fed_total": final_fed,
        "limiting_toxicant": limiting_gas,
        "max_smac_1h_ratio": round(max_smac_1h_ratio, 2),
        "time_to_1h_smac_min": time_to_1h_smac_min,
        "time_to_incapacitation_min": time_to_incapacitation_min,
        "triage_severity": triage_severity,
        "triage_action": triage_action,
        "primary_syndrome": profile["primary_syndrome"],
        "clinical_mechanism": profile["clinical_mechanism"],
        "clinical_pearls": clinical_pearls,
        "timeline": timeline,
    }


def compute_toxic_hazard_summary(
    pred_result: Dict[str, Any],
    material: Dict[str, Any],
    g_level: float = 0.0,
    pressure_kPa: float = 101.3,
    thickness_mm: float = 1.0,
    sample_width_mm: float = 410.0,
) -> Dict[str, Any]:
    """Fast 10-minute ISS cabin toxicity summary for /compare-materials ranking."""
    tox = evaluate_crew_toxicity(
        pred_result,
        material,
        g_level=g_level,
        pressure_kPa=pressure_kPa,
        thickness_mm=thickness_mm,
        sample_width_mm=sample_width_mm,
        exposure_minutes=10.0,
        cabin_volume_m3=106.0,
        eclss_active=True,
    )
    return {
        "max_smac_1h_ratio_10m": tox["max_smac_1h_ratio"],
        "limiting_toxicant": tox["limiting_toxicant"],
        "purser_fed_10m": tox["purser_fed_total"],
        "cohb_pct_10m": tox["estimated_cohb_pct"],
        "triage_severity": tox["triage_severity"],
        "primary_syndrome": tox["primary_syndrome"],
    }