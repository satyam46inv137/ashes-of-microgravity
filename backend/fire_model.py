"""
================================================================================
PHYSICS-BASED REDUCED-ORDER & QUASI-2D/2D FLAME SPREAD MODEL (Saffire-Calibrated)
================================================================================

Scientific Basis & Primary Literature References:
--------------------------------------------------------------------------------
1. de Ris, J. N. (1969). "Spread of a laminar diffusion flame."
   Proceedings of the Combustion Institute, 12(1), 241-252.
   -> Foundational opposed-flow flame spread scaling laws for thermally thin
      and thermally thick solid fuels.

2. Fernandez-Pello, A. C., & Hirano, T. (1983). "Controlling mechanisms of
   flame spread." Combustion Science and Technology, 32(1-4), 1-31.
   -> Convective heat transfer, boundary layer thickness (delta_f ~ x/sqrt(Re)),
      and Damkohler-number / aerodynamic strain extinction limits.

3. Bhattacharjee, S., Altenkirch, R. A., & Sacksteder, K. (1996). "The effect
   of ambient pressure on flame spread over thin cellulosic fuels in a
   quiescent, microgravity environment." Journal of Heat Transfer, 118(1), 181-190.
   -> Quiescent microgravity radiative quenching and pressure scaling laws.

4. Olson, S. L., Ferkul, P. V., & T'ien, J. S. (1988). "Near-limit flame spread
   over a thin solid fuel in microgravity." Symposium (International) on
   Combustion, 22(1), 1213-1222.
   -> Low-flow microgravity flame spread regime, oxygen-transport-limited
      extinction (~0.5-1.0 cm/s ventilation threshold), and BASS-II scaling.

5. Jomaas, G., Torero, J. L., Eigenbrod, C., Niehaus, J., Olson, S. L.,
   Ferkul, P. V., Legros, G., Fernandez-Pello, A. C., Cowlard, A. J.,
   Rouvreau, S., Smirnov, N., Fujita, O., T'ien, J. S., Ruff, G. A., &
   Urban, D. L. (2015). "Fire safety in space - beyond flammability testing
   of small samples." Acta Astronautica, 109, 208-216.
   -> NASA Saffire I-VI spacecraft fire safety experiments (large-scale
      microgravity concurrent & opposed flame spread calibration data).

6. Quintiere, J. G. (2006). "Fundamentals of Fire Phenomena." John Wiley & Sons.
   -> Thermal penetration depth (delta_t = sqrt(alpha * t)), thermal inertia
      (k * rho * c), heat of gasification (H_g = c_s * dT + L_v), and
      lateral conductive heat sink / quenching width theory.

7. Williams, F. A. (1976). "Mechanisms of fire spread." Symposium
   (International) on Combustion, 16(1), 1281-1294.
   -> Fundamental energy balance at the pyrolysis front:
      rho_s * V_f * d_h = q_net * L_ph.
================================================================================
"""

import math
import numpy as np

try:
    from materials import CORE_NASA_NAMES
except ModuleNotFoundError:
    try:
        from backend.materials import CORE_NASA_NAMES
    except ModuleNotFoundError:
        CORE_NASA_NAMES = set()


# ==============================================================================
# PHYSICAL & THERMODYNAMIC CONSTANTS (SI Units)
# ==============================================================================
T_AMBIENT_K = 298.15              # Standard cabin/room ambient temperature (25 C) [K]
PRESSURE_REF_KPA = 101.3          # Standard sea-level atmospheric pressure (1 atm) [kPa]
X_O2_AIR = 0.21                   # Standard Earth air molar oxygen fraction [-]
X_REF_M = 0.01                    # Characteristic convective length scale (10 mm) [m]
                                  # Ref: Fernandez-Pello & Hirano (1983), CST 32, p. 8
T_FLAME_MAX_K = 1800.0            # Stoichiometric hydrocarbon/cellulose diffusion flame
                                  # temperature at 1 atm air [K]. Ref: Quintiere (2006), Ch. 4
STEFAN_BOLTZMANN = 5.670374419e-8 # CODATA 2018 exact Stefan-Boltzmann constant [W/(m^2*K^4)]

# ==============================================================================
# FLOW & GRAVITY REGIME THRESHOLDS (Empirical / Flight-Validated)
# ==============================================================================
MICROGRAVITY_THRESHOLD_G = 0.01   # Threshold below which buoyancy velocity V_buoy << V_forced
QUIESCENT_FLOW_CMS = 0.5          # Minimum forced ventilation to prevent radiative quench
                                  # in 0g [cm/s]. Ref: Olson et al. (1988), 22nd Symp. Combust.
FLOW_STRAIN_CRITICAL_CMS = 20.0   # Thick-fuel aerodynamic strain onset [cm/s] (Fernandez-Pello, 1983)
FLOW_STRAIN_CRITICAL_THIN_CMS = 30.0  # Thin-fuel strain onset [cm/s] (Olson & Ferkul, 2019, Combust. Flame 199)
BLOWOFF_FLOW_CMS = 50.0           # Aerodynamic blow-off extinction limit at 1 atm [cm/s],
                                  # scales with sqrt(P/P_ref). Ref: Fernandez-Pello (1983)

# ==============================================================================
# SOLID-PHASE FUEL & GAS TRANSPORT PROPERTIES
# ==============================================================================
L_PH_THIN_BASELINE_M = 0.0078     # Baseline opposed-flow preheat length for thermally thin
                                  # fuels [m] (~7.8 mm). Calibrated to Saffire-I/II & BASS-II.
L_PH_THICK_FACTOR = 0.63          # Conductive preheat length ratio (delta_th / delta_f ~ 0.63)
                                  # Ref: de Ris (1969), 12th Symp. Combust., pp. 241-252
THIN_FUEL_THRESHOLD_M = 0.001     # Nominal 1 mm half-thickness scale separating thermally
                                  # thin vs thermally thick behavior [m]. Ref: Quintiere (2006)

NU_AIR_300K = 1.8e-5              # Kinematic viscosity of air at ~300-350 K [m^2/s]
                                  # Ref: Incropera & DeWitt, Fundamentals of Heat & Mass Transfer
K_AIR_300K = 0.026                # Thermal conductivity of air at 300 K [W/(m*K)]
                                  # Ref: Incropera & DeWitt, Table A.4

# ==============================================================================
# SAFFIRE FLIGHT-CALIBRATED COEFFICIENTS (See ai/constants_provenance.json)
# ==============================================================================
F_CONCURRENT_THIN_MICROG = 0.165  # Concurrent-to-opposed L_ph ratio in 0g for thin fuels
                                  # (Saffire-I/III SIBAL fabric flight data; Jomaas et al., 2015)
F_CONCURRENT_THICK_MICROG = 0.40  # Concurrent-to-opposed L_ph ratio in 0g for thick fuels
                                  # (Saffire-IV/V PMMA flight data)
F_CONCURRENT_THIN_BUOYANT = 3.8   # Buoyant upward concurrent spread multiplier (1g), where
                                  # natural convection elongates flame plume. Ref: Quintiere (2006)
F_CONCURRENT_THICK_BUOYANT = 5.2  # Buoyant upward concurrent multiplier for thick fuels (1g)
LOI_MICROG_FACTOR_DEFAULT = 0.85  # Limiting Oxygen Index reduction in 0g due to absence of
                                  # buoyant convective cooling. Ref: Olson et al. (1988)
STRAIN_EXPONENT_THICK = 1.35      # Damkohler strain-rate roll-off exponent at high V_forced
                                  # for thermally thick fuels. Ref: Fernandez-Pello & Hirano (1983)
RAD_LOSS_COEFF_MICROG = 0.28      # Gas-phase radiative loss fraction in 0g (larger spherical
                                  # flame volume increases soot/CO2/H2O radiation).
                                  # Ref: Bhattacharjee et al. (1996), J. Heat Transfer 118

# ------------------------------------------------------------------------------
# GEOMETRY / PRESSURE / THERMAL-DEPTH CORRECTIONS (EMPIRICAL - see provenance JSON)
# These are fitted to very few flight points; treat as calibrated, not derived.
# ------------------------------------------------------------------------------
ROD_CURVATURE_BASE = 1.42         # Cylindrical-fuel speed-up at delta_f/D -> 0 [-]
ROD_CURVATURE_SLOPE = 0.35        # Growth with flame-standoff-to-diameter ratio [-]
ROD_CURVATURE_RATIO_MAX = 1.5     # Cap on delta_f / D [-]
LOW_P_RAD_FEEDBACK_EXP = 0.45     # Sub-atmospheric flame-radiation feedback scaling (P_ref/P)^n
THERMAL_DEPTH_COEFF = 3000.0      # Empirical solid thermal-depth coefficient [-]
                                  # delta_th = C * alpha_s * V_eff^-0.25 (V_eff in m/s)
THERMAL_DEPTH_V_EXP = 0.25
THIN_THICK_RATIO_LO = 0.85        # tau/delta_th below which fuel is fully thermally thin [-]
THIN_THICK_RATIO_HI = 1.60        # tau/delta_th above which fuel is fully thermally thick [-]


def _smoothstep(x, edge0, edge1):
    """
    C1-continuous cubic Hermite interpolation between 0 and 1 across [edge0, edge1].
    Formula: S(t) = 3t^2 - 2t^3, where t = clamp((x - edge0)/(edge1 - edge0), 0, 1).
    Used to avoid unphysical step discontinuities across the thermally thin-to-thick
    transition regime (Quintiere, 2006, Ch. 7).
    """
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def thermal_blend_weight(tau, alpha_eff, V_eff):
    """
    Thin->thick blend weight shared by FireModel and the neural surrogate features
    (single source of truth). Returns (weight in [0,1], tau/delta_thermal ratio).
    """
    delta_thermal = max(THERMAL_DEPTH_COEFF * alpha_eff * max(V_eff, 0.005) ** (-THERMAL_DEPTH_V_EXP), 1e-4)
    ratio = tau / delta_thermal
    return _smoothstep(ratio, THIN_THICK_RATIO_LO, THIN_THICK_RATIO_HI), ratio


class FireModel:
    """
    Hybrid Scale-Analysis & Reduced-Order 2D Flame Spread Model.

    Combines analytical boundary-layer heat transfer (de Ris, 1969; Fernandez-Pello,
    1983), Stefan-Boltzmann radiative balance (Bhattacharjee et al., 1996), and
    Saffire microgravity flight calibration (Jomaas et al., 2015) with a NumPy
    2D transient thermal-pyrolysis grid solver.
    """

    def __init__(self, materials):
        self.materials = materials

    def _extinct(self, name, mat, v_eff, T_eff, reason, detail, applic):
        """Returns standardized extinction dictionary when a physical quenching limit is breached."""
        return {
            "material": name,
            "flame_spread_mms": 0.0,
            "flame_spread_uncertainty_mms": 0.0,
            "v_eff_cms": v_eff,
            "flame_temp_K": T_eff,
            "effective_flame_temp_K": T_eff,
            "standoff_mm": 0.0,
            "q_conv_kWm2": 0.0,
            "q_rad_kWm2": 0.0,
            "q_surf_loss_kWm2": 0.0,
            "q_net_kWm2": 0.0,
            "status": f"EXTINCT ({reason})",
            "is_extinct": True,
            "extinction_reason": reason,
            "extinction_detail": detail,
            "ul94": mat.get("ul94", "N/A"),
            "note": mat.get("note", ""),
            "confidence_score": applic["score"],
            "confidence_level": applic["level"],
            "confidence_color": applic["level"],
            "confidence_issues": applic["issues"],
            "applicability_score": applic["score"],
            "applicability_level": applic["level"],
            "applicability_issues": applic["issues"],
        }

    def _compute_uncertainty_mms(self, material_name, base_spread, params, applic_score):
        """
        First-Order Taylor Series Uncertainty Propagation (ISO GUM / Kline-McClintock method):
            sigma_total = sqrt( sum_i ( (dV_f / dx_i) * sigma_xi )^2 + sigma_epistemic^2 )

        Reference:
        - Kline, S. J., & McClintock, F. A. (1953). "Describing uncertainties in
          single-sample experiments." Mechanical Engineering, 75, 3-8.
        """
        if base_spread <= 1e-6:
            return 0.0

        # Experimental measurement 1-sigma relative uncertainties (Saffire / BASS-II instrumentation)
        rel_uncertainties = {
            "oxygen_pct": 0.015,    # +-1.5% O2 sensor tolerance
            "flow_vel_cms": 0.05,   # +-5.0% hot-wire anemometer tolerance
            "thickness_mm": 0.02,   # +-2.0% sample manufacturing tolerance
            "pressure_kPa": 0.01,   # +-1.0% cabin pressure transducer tolerance
        }
        variance_sq = 0.0

        for key, rel_u in rel_uncertainties.items():
            val = params[key]
            if val <= 0:
                continue
            delta = val * 0.05
            p_hi = {**params, key: val + delta}
            p_lo = {**params, key: max(1e-4, val - delta)}
            s_hi = self._predict_core(material_name, **p_hi)["flame_spread_mms"]
            s_lo = self._predict_core(material_name, **p_lo)["flame_spread_mms"]
            # Central finite-difference sensitivity derivative dV_f / dx_i
            ds_dx = (s_hi - s_lo) / (2.0 * delta)
            sigma_x = val * rel_u
            variance_sq += (ds_dx * sigma_x) ** 2

        # Epistemic (model-form) uncertainty increases smoothly when extrapolating outside
        # the Saffire validation envelope (applicability_score < 100)
        epistemic_rel = 0.18 + 0.25 * (1.0 - max(0.0, min(100.0, applic_score)) / 100.0)
        epistemic_sigma = base_spread * epistemic_rel
        total_sigma = math.sqrt(variance_sq + epistemic_sigma ** 2)
        return round(total_sigma, 3)

    def _predict_core(self, material_name, g_level=0.0, pressure_kPa=101.3,
                      oxygen_pct=21.0, flow_vel_cms=20.0, thickness_mm=10.0,
                      configuration="Opposed", sample_width_mm=410.0,
                      geometry="plate", edge_boundary="free"):
        mat = self.materials[material_name]

        # Clamp physical inputs to non-negative / physically valid domains
        g_level = max(g_level, 0.0)
        pressure_kPa = max(pressure_kPa, 1.0)
        oxygen_pct = max(0.0, min(100.0, oxygen_pct))
        flow_vel_cms = max(flow_vel_cms, 0.0)
        thickness_mm = max(thickness_mm, 0.1)
        sample_width_mm = max(sample_width_mm, 1.0)

        config_norm = configuration.lower().strip()
        if config_norm not in ("opposed", "concurrent"):
            raise ValueError(f"Invalid configuration '{configuration}'. Must be 'Opposed' or 'Concurrent'.")

        geom_norm = geometry.lower().strip()
        if geom_norm not in ("plate", "rod", "cylinder", "sphere"):
            raise ValueError(f"Invalid geometry '{geometry}'. Must be 'plate', 'rod', 'cylinder', or 'sphere'.")

        edge_norm = edge_boundary.lower().strip()
        if edge_norm not in ("free", "clamped", "insulated"):
            raise ValueError(f"Invalid edge_boundary '{edge_boundary}'. Must be 'free', 'clamped', or 'insulated'.")

        g = g_level * 9.81                      # Gravitational acceleration [m/s^2]
        X_O2_nom = oxygen_pct / 100.0           # Nominal ambient O2 mole fraction [-]
        V_forced = max(flow_vel_cms / 100.0, 0.001)  # Forced convective flow velocity [m/s]
        tau = thickness_mm / 1000.0             # Fuel half-thickness / thickness [m]
        W = sample_width_mm / 1000.0            # Sample transverse width [m]
        T_inf = T_AMBIENT_K

        p_ratio = max(pressure_kPa / PRESSURE_REF_KPA, 0.05)

        # ----------------------------------------------------------------------
        # 1. BOUNDARY LAYER & QUASI-2D LATERAL OXYGEN ENTRAINMENT
        # ----------------------------------------------------------------------
        # Kinematic viscosity scales inversely with pressure (ideal gas density rho ~ P):
        #   nu_g = nu_0 / (P / P_ref)
        # Blasius laminar boundary-layer thickness:
        #   delta_f = x_ref / sqrt(Re),  Re = (V * x_ref) / nu_g
        # Ref: Schlichting, "Boundary-Layer Theory"; Fernandez-Pello & Hirano (1983)
        nu_g = NU_AIR_300K / max(p_ratio, 0.1)
        Re_prelim = max((V_forced * X_REF_M) / nu_g, 0.1)
        delta_f_prelim = X_REF_M / math.sqrt(Re_prelim)

        # Quasi-2D Lateral Oxygen Entrainment:
        # On narrow finite-width samples (<200 mm) with free edges, oxygen diffuses laterally
        # across the side boundary layers (~delta_f / W), enriching effective O2 at the flame sheet.
        # Ref: Olson et al. (1988) & BASS-II narrow-sample edge-diffusion observations.
        if geom_norm == "plate" and edge_norm == "free" and sample_width_mm < 200.0:
            entrain_boost = 1.0 + 0.25 * min(delta_f_prelim / max(W, 0.005), 0.6)
        elif geom_norm in ("rod", "cylinder", "sphere"):
            # Cylindrical/spherical geometry allows radial 3D diffusive O2 transport
            entrain_boost = 1.22 if geom_norm == "sphere" else 1.18
        else:
            entrain_boost = 1.0

        X_O2 = min(1.0, X_O2_nom * entrain_boost)
        eff_oxygen_pct = X_O2 * 100.0

        # ----------------------------------------------------------------------
        # 2. EFFECTIVE DIFFUSION FLAME TEMPERATURE (Dissociation & Pressure Law)
        # ----------------------------------------------------------------------
        # Flame temperature scales sub-linearly with O2 (due to high-T CO2/H2O
        # dissociation, exponent ~0.42) and saturates exponentially with pressure
        # (reaction-rate / Damkohler completeness factor).
        # Ref: Law, C. K. (2006), "Combustion Physics", Ch. 6; Quintiere (2006).
        T_eff = T_inf + (T_FLAME_MAX_K - T_inf) * ((X_O2 / X_O2_AIR) ** 0.42) * (
            1.0 - math.exp(-1.2 * (p_ratio ** 0.25))
        )

        # ----------------------------------------------------------------------
        # 3. MIXED CONVECTION EFFECTIVE VELOCITY (Forced + Buoyant Plume)
        # ----------------------------------------------------------------------
        # Buoyancy-induced velocity scale from Grashof/Froude balance:
        #   V_buoy = sqrt( 2 * g * x_ref * (T_eff - T_inf) / T_inf )
        # Combined vectorially with forced flow:
        #   V_eff = sqrt( V_forced^2 + V_buoy^2 )
        # Ref: Fernandez-Pello & Hirano (1983), CST 32, Eq. (12); Quintiere (2006).
        V_buoy = math.sqrt(2.0 * g * X_REF_M * max(T_eff - T_inf, 10.0) / T_inf) if g > 0.001 else 0.0
        V_eff = math.sqrt(V_forced ** 2 + V_buoy ** 2)
        v_eff_cms = round(V_eff * 100.0, 1)
        is_microgravity = g_level < MICROGRAVITY_THRESHOLD_G

        applic = self.applicability(material_name, g_level, oxygen_pct, flow_vel_cms, thickness_mm, pressure_kPa)

        # ----------------------------------------------------------------------
        # 4. FLAMMABILITY & EXTINCTION BOUNDARIES (LOI, Quench, Blow-off)
        # ----------------------------------------------------------------------
        # Sub-atmospheric pressure increases LOI slightly (~(P_ref/P)^0.12), while
        # microgravity lowers LOI (~0.85x) by removing buoyant convective heat losses.
        # Ref: ASTM D2863 (LOI); Olson et al. (1988), 22nd Symp. Combust., p. 1217.
        p_loi_correction = (1.0 / p_ratio) ** 0.12 if p_ratio < 1.0 else 1.0
        loi_factor = mat.get("LOI_microg_factor", LOI_MICROG_FACTOR_DEFAULT) if is_microgravity else 1.0
        loi_threshold = mat.get("LOI", 18.0) * loi_factor * p_loi_correction
        loi_margin = (eff_oxygen_pct - loi_threshold) / loi_threshold if loi_threshold > 0 else 1.0

        if eff_oxygen_pct < loi_threshold:
            return self._extinct(
                material_name, mat, v_eff_cms, 300.0, "Below Flammability Limit (LOI)",
                f"Effective LOI: {loi_threshold:.2f}%, Effective O2: {eff_oxygen_pct:.1f}%",
                applic,
            )

        # Quiescent Radiative Extinction Limit in Microgravity (Olson et al., 1988)
        if is_microgravity and flow_vel_cms < QUIESCENT_FLOW_CMS:
            return self._extinct(
                material_name, mat, v_eff_cms, round(T_eff, 1), "Quiescent Radiative Quench",
                f"Ventilation < {QUIESCENT_FLOW_CMS} cm/s. Radiative losses extinguish the diffusion sphere.",
                applic,
            )

        # Damkohler Aerodynamic Blow-off Limit: V_blowoff ~ sqrt(P / P_ref)
        # Ref: Fernandez-Pello & Hirano (1983), CST 32, p. 18.
        blowoff_limit = BLOWOFF_FLOW_CMS * math.sqrt(p_ratio)
        if flow_vel_cms > blowoff_limit:
            return self._extinct(
                material_name, mat, v_eff_cms, round(T_eff, 1), "Aerodynamic Blow-off",
                f"Flow ({flow_vel_cms:.1f} cm/s) > limit ({blowoff_limit:.1f} cm/s).",
                applic,
            )

        # ----------------------------------------------------------------------
        # 5. CONVECTIVE & RADIATIVE HEAT TRANSFER BALANCE AT PYROLYSIS FRONT
        # ----------------------------------------------------------------------
        # Gas thermal conductivity scales with temperature via Chapman-Enskog power law:
        #   k_g(T) = k_air,300K * (T_eff / 300)^0.7
        # Convective heat flux across flame standoff distance delta_f:
        #   q_conv = k_g * (T_eff - T_pyro) / delta_f
        # Ref: de Ris (1969), 12th Symp. Combust., Eq. (14).
        Re_eff = max((V_eff * X_REF_M) / nu_g, 0.1)
        delta_f = X_REF_M / math.sqrt(Re_eff)
        k_g = K_AIR_300K * (T_eff / 300.0) ** 0.7
        q_conv = (k_g * max(T_eff - mat["T_pyro"], 0)) / max(delta_f, 1e-4)

        # Stefan-Boltzmann solid surface radiative heat loss to ambient surroundings:
        #   q_surf,loss = epsilon * sigma * (T_pyro^4 - T_inf^4)
        # Ref: Bhattacharjee et al. (1996), J. Heat Transfer 118, Eq. (5).
        emissivity = mat.get("emissivity", 0.85)
        q_surf_rad_loss = emissivity * STEFAN_BOLTZMANN * (max(mat["T_pyro"], T_inf) ** 4 - T_inf ** 4)

        # Gas-phase radiative loss fraction (higher in 0g due to thick spherical flame)
        radiation_loss_fraction = (RAD_LOSS_COEFF_MICROG * p_ratio) if is_microgravity else 0.15
        q_gas_rad = q_conv * radiation_loss_fraction

        # Soot radiative feedback boost at elevated O2 concentrations (>21%)
        # Ref: Quintiere (2006), Ch. 10 (Radiative heat transfer from luminous flames).
        # Note: Under sustained burning, luminous flame radiation balances surface re-radiation
        # (de Ris, 1969; Quintiere, 2006), while q_surf_rad_loss governs radiative quenching
        # near extinction (Bhattacharjee et al., 1996).
        q_rad_feedback = 0.55 * q_conv * max(0.0, (X_O2 - X_O2_AIR) / X_O2_AIR)
        if p_ratio < 1.0:
            # Thicker flame standoff at low pressure raises radiative feedback share
            q_rad_feedback *= (1.0 / p_ratio) ** LOW_P_RAD_FEEDBACK_EXP
        q_net = q_conv - q_gas_rad + q_rad_feedback

        if (q_net - q_surf_rad_loss) <= 1.0:
            return self._extinct(
                material_name, mat, v_eff_cms, round(T_eff, 1), "Radiative Quench",
                f"Convective/flame heat flux ({q_net:.1f} W/m2) <= surface radiative loss ({q_surf_rad_loss:.1f} W/m2).",
                applic,
            )

        # Quasi-2D Lateral Conductive Heat Sink (Quenching Width Physics):
        # For samples clamped in conductive metal holders, lateral heat conduction
        # drains energy at rate ~ 4 * k_s * tau * dT_pyro / W^2.
        # Ref: Quintiere (2006), Ch. 7 (Critical quenching thickness/width).
        rho_s, c_s, k_s = mat["rho_s"], mat["c_s"], mat["k_s"]
        dT_pyro = max(mat["T_pyro"] - T_inf, 50.0)
        if geom_norm == "plate" and edge_norm == "clamped":
            q_lat_loss = (4.0 * k_s * tau * dT_pyro) / max(W ** 2, 1e-6)
            q_net -= q_lat_loss

        if (q_net - q_surf_rad_loss) <= 1.0:
            return self._extinct(
                material_name, mat, v_eff_cms, round(T_eff, 1), "Radiative / Conductive Quench",
                f"Net heat flux after lateral conduction ({q_net:.1f} W/m2) <= surface radiative loss ({q_surf_rad_loss:.1f} W/m2).",
                applic,
            )

        # ----------------------------------------------------------------------
        # 6. DE RIS / QUINTIERE FLAME SPREAD FORMULAS (THIN & THICK REGIMES)
        # ----------------------------------------------------------------------
        is_concurrent = config_norm == "concurrent"

        # Surface char layer insulation reduces effective conductivity by ~35% in thick
        # charring polymers/cellulosics. Ref: Quintiere (2006), Ch. 8.
        char_categories = {"Engineering Thermosets", "Wood & Bio-materials"}
        char_weight = _smoothstep(tau, 0.0005, 0.002)
        k_eff = k_s * (1.0 - 0.35 * char_weight) if mat.get("category", "") in char_categories else k_s

        # Regime A: Thermally Thin Solid Fuel (de Ris, 1969, Eq. 24; Williams, 1976):
        #   V_f,thin = (q_net * L_ph) / (rho_s * c_s * tau * (T_pyro - T_inf))
        # In concurrent flow, flame plume elongation scales with forced flow ~(V_eff / 0.20)^0.55
        # (Jomaas et al., 2015; Olson & Ferkul, 2019, Saffire-I vs Saffire-III S1).
        conc_vel_scale = ((V_eff / 0.20) ** 0.55) if (is_concurrent and is_microgravity) else 1.0
        f_conc_thin = (F_CONCURRENT_THIN_MICROG * conc_vel_scale) if is_microgravity else F_CONCURRENT_THIN_BUOYANT
        f_cfg_thin = f_conc_thin if is_concurrent else 1.0
        v_crit_thin = FLOW_STRAIN_CRITICAL_THIN_CMS / 100.0
        strain_thin = (v_crit_thin / V_forced) ** 0.58 if V_forced > v_crit_thin else 1.0
        L_ph_thin = L_PH_THIN_BASELINE_M * f_cfg_thin * strain_thin
        V_f_thin = (q_net * L_ph_thin) / (rho_s * c_s * tau * dT_pyro)

        # Regime B: Thermally Thick Solid Fuel (de Ris, 1969, Eq. 31; Quintiere, 2006, Eq. 7.26):
        #   V_f,thick = (q_net^2 * L_ph) / (k_eff * rho_s * c_s * (T_pyro - T_inf)^2)
        f_conc_thick = F_CONCURRENT_THICK_MICROG if is_microgravity else F_CONCURRENT_THICK_BUOYANT
        f_cfg_thick = f_conc_thick if is_concurrent else 1.0
        L_ph_thick = L_PH_THICK_FACTOR * delta_f * f_cfg_thick
        v_crit_thick = FLOW_STRAIN_CRITICAL_CMS / 100.0
        strain_thick = (v_crit_thick / V_forced) ** STRAIN_EXPONENT_THICK if V_forced > v_crit_thick else 1.0
        thermal_inertia = k_eff * rho_s * c_s
        V_f_thick = (q_net ** 2 * L_ph_thick * strain_thick) / (thermal_inertia * (dT_pyro ** 2))

        # Empirical thermal-depth blending: xi = tau / delta_thermal with
        #   delta_thermal = THERMAL_DEPTH_COEFF * alpha_s * V_eff^-THERMAL_DEPTH_V_EXP
        # Thin when xi < THIN_THICK_RATIO_LO, thick when xi > THIN_THICK_RATIO_HI
        # (see thermal_blend_weight and constants_provenance.json; low-confidence fit).
        alpha_s = k_eff / max(rho_s * c_s, 1.0)
        thick_blend, thermal_thickness_ratio = thermal_blend_weight(tau, alpha_s, V_eff)
        V_f_ms = (1.0 - thick_blend) * V_f_thin + thick_blend * V_f_thick

        # ----------------------------------------------------------------------
        # 7. GEOMETRY & NEAR-LIMIT CORRECTIONS
        # ----------------------------------------------------------------------
        if geom_norm == "plate":
            if edge_norm == "free" and sample_width_mm < 200.0:
                # Two-sided corner heating at free lateral edges accelerates front speed
                # on narrow sheets (<200 mm). Ref: BASS-II (50 mm) vs Saffire (410 mm) scaling.
                V_f_ms *= (200.0 / sample_width_mm) ** 0.20
        elif geom_norm in ("rod", "cylinder"):
            # Diameter-aware curvature: speed-up grows as the flame standoff becomes
            # comparable to the rod diameter (thickness_mm is the rod diameter here).
            ratio = min(delta_f / max(tau, 1e-4), ROD_CURVATURE_RATIO_MAX)
            V_f_ms *= ROD_CURVATURE_BASE * (1.0 + ROD_CURVATURE_SLOPE * ratio)
        elif geom_norm == "sphere":
            # Spherical PMMA geometry (BASS-II sphere tests) exhibits 3D radial curvature heat focus
            V_f_ms *= 1.85

        # Low-flow microgravity boost for thin fuels (BASS-II narrow sample regime, < 5 cm/s)
        thin_weight = 1.0 - thick_blend
        if thin_weight > 0.01 and flow_vel_cms < 5.0:
            boost = (5.0 / max(flow_vel_cms, 0.5)) ** 0.15
            V_f_ms *= (1.0 + thin_weight * (boost - 1.0))

        # Near-LOI Damkohler slowdown as O2 approaches extinction threshold
        if loi_margin < 0.15:
            V_f_ms *= max(0.10, (loi_margin / 0.15) ** 1.5)

        spread_mms = round(V_f_ms * 1000.0, 3)

        return {
            "material": material_name,
            "flame_spread_mms": spread_mms,
            "v_eff_cms": v_eff_cms,
            "flame_temp_K": round(T_eff, 1),
            "effective_flame_temp_K": round(T_eff, 1),
            "standoff_mm": round(delta_f * 1000.0, 2),
            "q_conv_kWm2": round(q_conv / 1000.0, 2),
            "q_rad_kWm2": round(q_gas_rad / 1000.0, 2),
            "q_surf_loss_kWm2": round(q_surf_rad_loss / 1000.0, 2),
            "q_net_kWm2": round(q_net / 1000.0, 2),
            "thermal_regime": "Thin" if thick_blend < 0.2 else ("Thick" if thick_blend > 0.8 else "Transitional"),
            "thick_blend_weight": round(thick_blend, 3),
            "status": "SUSTAINED FLAME SPREAD",
            "is_extinct": False,
            "sample_width_mm": sample_width_mm,
            "geometry": geom_norm,
            "edge_boundary": edge_norm,
            "extinction_reason": None,
            "extinction_detail": None,
            "ul94": mat.get("ul94", "HB"),
            "note": mat.get("note", ""),
            "confidence_score": applic["score"],
            "confidence_level": applic["level"],
            "confidence_color": applic["level"],
            "confidence_issues": applic["issues"],
            "applicability_score": applic["score"],
            "applicability_level": applic["level"],
            "applicability_issues": applic["issues"],
        }

    def predict(self, material_name, g_level=0.0, pressure_kPa=101.3,
                oxygen_pct=21.0, flow_vel_cms=20.0, thickness_mm=10.0,
                configuration="Opposed", sample_width_mm=410.0,
                geometry="plate", edge_boundary="free", **kwargs):
        """
        Main single-point prediction API returning steady-state flame spread rate [mm/s],
        heat fluxes [kW/m^2], 1-sigma propagated uncertainty, and Saffire applicability score.
        """
        if kwargs:
            raise TypeError(
                f"Unexpected keyword arguments: {list(kwargs.keys())}. "
                f"Expected: material_name, g_level, pressure_kPa, oxygen_pct, "
                f"flow_vel_cms, thickness_mm, configuration, sample_width_mm, geometry, edge_boundary."
            )

        if material_name not in self.materials:
            raise KeyError(f"Material '{material_name}' not in database.")

        params = {
            "g_level": g_level,
            "pressure_kPa": pressure_kPa,
            "oxygen_pct": oxygen_pct,
            "flow_vel_cms": flow_vel_cms,
            "thickness_mm": thickness_mm,
            "configuration": configuration,
            "sample_width_mm": sample_width_mm,
            "geometry": geometry,
            "edge_boundary": edge_boundary,
        }
        res = self._predict_core(material_name, **params)
        # Extinct results come from _extinct() and lack these; downstream insight logic needs them.
        res.setdefault("sample_width_mm", sample_width_mm)
        res.setdefault("geometry", geometry.lower().strip())
        res.setdefault("edge_boundary", edge_boundary.lower().strip())
        if not res["is_extinct"]:
            res["flame_spread_uncertainty_mms"] = self._compute_uncertainty_mms(
                material_name, res["flame_spread_mms"], params, res["applicability_score"]
            )
        else:
            res["flame_spread_uncertainty_mms"] = 0.0
        return res

    def applicability(self, material_name, g_level, oxygen_pct, flow_vel_cms, thickness_mm, pressure_kPa=101.3):
        """
        Model Applicability & Domain Validation Envelope (NASA-STD-7009A V&V Credibility Framework).
        Evaluates continuous distance outside the Saffire I-VI flight validation envelope:
          - Gravity: <= 0.01g (0g flight calibration)
          - Oxygen: 12% - 30%
          - Flow velocity: 0.5 - 30 cm/s
          - Thickness: 0.1 - 15 mm
          - Pressure: 55 - 105 kPa (Saffire-V 55 kPa exploration atmosphere to 101.3 kPa)
        """
        issues, penalty = [], 0.0

        if CORE_NASA_NAMES and material_name not in CORE_NASA_NAMES:
            penalty += 35.0
            issues.append("Material not directly Saffire-validated (extrapolated)")

        if g_level > 0.01:
            g_pen = min(30.0, 12.0 + 18.0 * min((g_level - 0.01) / 0.99, 1.0))
            penalty += g_pen
            issues.append(f"Gravity {g_level:.2f}g extrapolated (calibration data is 0g only)")

        if oxygen_pct < 12.0:
            penalty += min(20.0, 5.0 + 2.5 * (12.0 - oxygen_pct))
            issues.append(f"O2 {oxygen_pct:.1f}% below validated range (12-30%)")
        elif oxygen_pct > 30.0:
            penalty += min(20.0, 5.0 + 1.0 * (oxygen_pct - 30.0))
            issues.append(f"O2 {oxygen_pct:.1f}% above validated range (12-30%)")

        if flow_vel_cms < 0.5:
            penalty += min(15.0, 5.0 + 20.0 * (0.5 - flow_vel_cms))
            issues.append(f"Flow {flow_vel_cms:.2f} cm/s below validated range (0.5-30)")
        elif flow_vel_cms > 30.0:
            penalty += min(15.0, 5.0 + 0.5 * (flow_vel_cms - 30.0))
            issues.append(f"Flow {flow_vel_cms:.1f} cm/s above validated range (0.5-30)")

        if thickness_mm < 0.1 or thickness_mm > 15.0:
            dist = (0.1 - thickness_mm) * 50.0 if thickness_mm < 0.1 else (thickness_mm - 15.0) * 0.8
            penalty += min(12.0, 4.0 + dist)
            issues.append(f"Thickness {thickness_mm} mm outside validated range (0.1-15)")

        if pressure_kPa < 55.0 or pressure_kPa > 105.0:
            dist = (55.0 - pressure_kPa) * 0.5 if pressure_kPa < 55.0 else (pressure_kPa - 105.0) * 0.3
            penalty += min(20.0, 6.0 + dist)
            issues.append(f"Pressure {pressure_kPa:.1f} kPa outside validated range (55-105)")

        score = max(0, round(100.0 - penalty))
        if score >= 90:
            level = "HIGH"
        elif score >= 60:
            level = "MEDIUM"
        else:
            level = "LOW (Extrapolated)"
        return {"score": score, "level": level, "issues": issues}

    confidence = applicability

    def sensitivity(self, material_name, baseline, perturbation=0.10):
        """Local One-at-a-Time (OAT) finite-difference parameter sensitivity ranking."""
        base = self.predict(material_name, **baseline)["flame_spread_mms"]
        if base <= 1e-6:
            return []

        labels = {
            "g_level": "Gravity",
            "pressure_kPa": "Pressure",
            "oxygen_pct": "Oxygen",
            "flow_vel_cms": "Flow Speed",
            "thickness_mm": "Thickness",
            "sample_width_mm": "Sample Width (2D)",
        }
        results = []

        for param, value in baseline.items():
            if not isinstance(value, (int, float)):
                continue

            if param == "g_level":
                low_cfg = {**baseline, param: max(0.0, value - 0.05)}
                high_cfg = {**baseline, param: value + 0.05}
            elif value > 0:
                low_cfg = {**baseline, param: value * (1 - perturbation)}
                high_cfg = {**baseline, param: value * (1 + perturbation)}
            else:
                continue

            low = self.predict(material_name, **low_cfg)["flame_spread_mms"]
            high = self.predict(material_name, **high_cfg)["flame_spread_mms"]
            delta = high - low

            results.append({
                "parameter": labels.get(param, param),
                "impact": round(abs(delta), 4),
                "delta_pct": round((delta / base * 100.0) if base > 1e-6 else 0.0, 2),
            })

        return sorted(results, key=lambda x: x["impact"], reverse=True)

    def transient_simulate(self, material_name, duration_s=120, dt=1.0, **kwargs):
        """
        1D Transient Burn-Through Simulation (Conservation of Energy):
          1. Ignition preheat ramp-up: tau_ign ~ (rho_s * c_s * tau^2) / k_s
             (Fourier conduction time scale; Carslaw & Jaeger, 1959).
          2. Solid mass-loss regression rate via Heat of Gasification (H_g):
             r_dot = q_net / (rho_s * [c_s * (T_pyro - T_inf) + L_v])
             Ref: Tewarson, A. (2002), SFPE Handbook of Fire Protection Engineering, Ch. 3-4.
          3. Adaptive time-step refinement (dt <= 0.25 s) when remaining thickness < 1.5 mm
             to resolve burn-through acceleration cleanly.
        """
        if dt <= 0:
            raise ValueError("dt must be greater than 0")
        if duration_s <= 0:
            raise ValueError("duration_s must be greater than 0")

        mat = self.materials[material_name]
        rho_s, c_s = mat["rho_s"], mat["c_s"]
        dT_pyro = max(mat["T_pyro"] - T_AMBIENT_K, 50.0)
        # Heat of pyrolysis/gasification L_v [J/kg]; typical polymers ~1.5-2.0 MJ/kg (Tewarson, 2002)
        L_v = mat.get("L_v", 1.8e6)
        heat_of_gasification = c_s * dT_pyro + L_v

        # Fourier thermal diffusion time scale tau_ign = (rho * c * L^2) / k [s]
        tau_init = kwargs.get("thickness_mm", 10.0) / 1000.0
        k_s = mat["k_s"]
        tau_ign = max(1.0, min(15.0, (rho_s * c_s * min(tau_init, 0.003) ** 2) / max(k_s, 0.05)))

        remaining = kwargs.get("thickness_mm", 10.0)
        history, t, x_flame = [], 0.0, 0.0
        sim_kwargs = kwargs.copy()

        while t < duration_s and remaining > 0.01:
            step_dt = min(dt, 0.25) if remaining < 1.5 else dt
            if t + step_dt > duration_s:
                step_dt = duration_s - t

            sim_kwargs["thickness_mm"] = remaining
            r = self.predict(material_name, **sim_kwargs)

            if r["is_extinct"]:
                history.append({
                    "t": round(t, 2),
                    "x_flame_mm": round(x_flame, 2),
                    "spread_mms": 0.0,
                    "thickness_remaining": round(remaining, 3),
                    "status": "extinct",
                    "reason": r["extinction_reason"],
                })
                break

            ignition_ramp = 1.0 - math.exp(-t / tau_ign) if t > 0 else 0.05
            spread = round(r["flame_spread_mms"] * ignition_ramp, 3)

            history.append({
                "t": round(t, 2),
                "x_flame_mm": round(x_flame, 2),
                "spread_mms": spread,
                "thickness_remaining": round(remaining, 3),
                "regime": r["thermal_regime"],
                "status": "burning",
            })

            x_flame += spread * step_dt
            q_net_wm2 = r["q_net_kWm2"] * 1000.0 * ignition_ramp
            regression_rate_mms = (q_net_wm2 / max(rho_s * heat_of_gasification, 1.0)) * 1000.0
            remaining = max(remaining - regression_rate_mms * step_dt, 0.0)
            t += step_dt

        return history

    def simulate_2d(self, material_name, length_mm=300.0, width_mm=150.0,
                    nx=120, ny=60, duration_s=60.0, dt=0.25,
                    ignition_mode="line", ignition_pos=(0.15, 0.5),
                    edge_boundary="free", snapshot_interval_s=5.0, **kwargs):
        """
        Reduced-Order 2D Transient Thermal-Pyrolysis PDE Solver (NumPy-accelerated).

        Governing 2D Solid-Phase Energy Conservation Equation:
            rho_s * c_s * tau(x,y) * (dT/dt) = k_s * tau(x,y) * (d^2T/dx^2 + d^2T/dy^2)
                                               + Q_flame(x,y) - Q_rad,surf(x,y)

        Numerical & Physical Formulation:
        - Spatial discretization: 2nd-order central finite differences on (nx, ny) Cartesian grid.
        - Temporal integration: Explicit Euler with automatic Fourier stability sub-stepping
          (Fo = alpha_s * sub_dt / min(dx^2, dy^2) <= 0.20; Patankar, 1980).
        - Flame preheat flux kernel: Exponential boundary-layer heat flux decay calibrated to
          preserve exact energy integral: integral(Q_flame dx) = rho_s * c_s * tau_0 * dT_pyro * V_f,ref
          (Williams, 1976, 16th Symp. Combust.).
        - Transverse boundary conditions:
            * 'free': Zero-flux Neumann (dT/dy = 0) + lateral O2 entrainment edge boost.
            * 'clamped': Isothermal Dirichlet heat sink (T = T_inf) + edge quenching lag.
        """
        if material_name not in self.materials:
            raise KeyError(f"Material '{material_name}' not in database.")

        mat = self.materials[material_name]
        rho_s, c_s, k_s = mat["rho_s"], mat["c_s"], mat["k_s"]
        T_pyro = mat["T_pyro"]
        T_inf = T_AMBIENT_K
        dT_pyro = max(T_pyro - T_inf, 50.0)
        L_v = mat.get("L_v", 1.8e6)
        emissivity = mat.get("emissivity", 0.85)
        H_g = c_s * dT_pyro + L_v

        thickness_mm = kwargs.get("thickness_mm", 2.0)
        configuration = kwargs.get("configuration", "Opposed")
        config_norm = configuration.lower().strip()
        is_concurrent = config_norm == "concurrent"

        base_pred = self.predict(
            material_name,
            sample_width_mm=width_mm,
            edge_boundary=edge_boundary,
            **kwargs,
        )
        if base_pred["is_extinct"]:
            return {
                "material": material_name,
                "status": "EXTINCT",
                "reason": base_pred["extinction_reason"],
                "snapshots": [],
                "final_burned_area_pct": 0.0,
                "mean_front_speed_mms": 0.0,
            }

        V_f_ref = base_pred["flame_spread_mms"] / 1000.0
        q_net_ref = base_pred["q_net_kWm2"] * 1000.0
        standoff_m = max(base_pred["standoff_mm"] / 1000.0, 0.002)

        dx = (length_mm / 1000.0) / nx
        dy = (width_mm / 1000.0) / ny

        T = np.full((nx, ny), T_inf, dtype=np.float64)
        tau = np.full((nx, ny), thickness_mm / 1000.0, dtype=np.float64)
        state = np.zeros((nx, ny), dtype=np.int8)  # 0=unburned, 1=burning, 2=burned_out

        # Transverse profile: free-edge O2 entrainment boost vs clamped-edge conductive sink
        y_norm = np.linspace(-1.0, 1.0, ny)
        if edge_boundary == "free":
            transverse_boost = 1.0 + 0.18 * (y_norm ** 2)
        elif edge_boundary == "clamped":
            transverse_boost = 1.0 - 0.35 * (y_norm ** 4)
        else:
            transverse_boost = np.ones(ny, dtype=np.float64)

        ix0 = max(1, min(nx - 2, int(ignition_pos[0] * nx)))
        iy0 = max(1, min(ny - 2, int(ignition_pos[1] * ny)))

        if ignition_mode == "line":
            T[max(0, ix0 - 1):min(nx, ix0 + 2), :] = T_pyro + 40.0
            state[max(0, ix0 - 1):min(nx, ix0 + 2), :] = 1
        elif ignition_mode == "point":
            X_idx, Y_idx = np.ogrid[:nx, :ny]
            r_cells = max(2.0, 0.05 * min(nx, ny))
            mask = ((X_idx - ix0) ** 2 + (Y_idx - iy0) ** 2) <= r_cells ** 2
            T[mask] = T_pyro + 40.0
            state[mask] = 1
        elif ignition_mode == "edge":
            T[max(0, ix0 - 1):min(nx, ix0 + 2), :max(2, ny // 6)] = T_pyro + 40.0
            state[max(0, ix0 - 1):min(nx, ix0 + 2), :max(2, ny // 6)] = 1
        else:
            raise ValueError("ignition_mode must be 'line', 'point', or 'edge'.")

        # Explicit 2D Von Neumann / Fourier CFL stability constraint: Fo <= 0.20
        alpha_s = k_s / max(rho_s * c_s, 1.0)
        dt_diff_limit = 0.20 * min(dx ** 2, dy ** 2) / max(alpha_s, 1e-9)
        sub_steps = max(1, int(math.ceil(dt / dt_diff_limit)))
        sub_dt = dt / sub_steps

        L_forward = max(dx * 1.5, standoff_m * (2.2 if is_concurrent else 1.1))
        L_lateral = max(dy * 1.2, standoff_m * 0.85)

        snapshots = []
        front_positions = []
        t = 0.0
        next_snap = 0.0

        while t <= duration_s + 1e-9:
            burning_mask = (state == 1)
            if not np.any(burning_mask) and t > 0:
                break

            if t >= next_snap - 1e-6:
                active_x = np.where(state >= 1)[0]
                max_front_mm = float(active_x.max() * dx * 1000.0) if len(active_x) > 0 else 0.0
                mean_front_mm = float(active_x.mean() * dx * 1000.0) if len(active_x) > 0 else 0.0
                front_positions.append((t, max_front_mm))

                contour_mm = []
                for j in range(ny):
                    burned_j = np.where(state[:, j] >= 1)[0]
                    contour_mm.append(round(float(burned_j.max() * dx * 1000.0), 2) if len(burned_j) > 0 else 0.0)

                snapshots.append({
                    "t": round(t, 2),
                    "max_front_mm": round(max_front_mm, 2),
                    "mean_front_mm": round(mean_front_mm, 2),
                    "burned_area_pct": round(float(np.mean(state >= 1) * 100.0), 2),
                    "active_burning_pct": round(float(np.mean(state == 1) * 100.0), 2),
                    "front_contour_mm": contour_mm,
                    "temperature_K": np.round(T, 1).tolist(),
                })
                next_snap += snapshot_interval_s

            Q_flame = np.zeros((nx, ny), dtype=np.float64)
            burn_i, burn_j = np.where(burning_mask)

            if len(burn_i) > 0:
                X_m = np.arange(nx)[:, None] * dx
                Y_m = np.arange(ny)[None, :] * dy

                is_front = np.zeros_like(burning_mask)
                is_front[1:, :] |= burning_mask[1:, :] & (state[:-1, :] == 0)
                is_front[:-1, :] |= burning_mask[:-1, :] & (state[1:, :] == 0)
                is_front[:, 1:] |= burning_mask[:, 1:] & (state[:, :-1] == 0)
                is_front[:, :-1] |= burning_mask[:, :-1] & (state[:, 1:] == 0)
                if not np.any(is_front):
                    is_front = burning_mask

                fi, fj = np.where(is_front)
                tau_0 = thickness_mm / 1000.0
                q_preheat_planar = (rho_s * c_s * tau_0 * dT_pyro * V_f_ref) / max(L_forward, 1e-4)
                q_surf_avg = 0.4 * emissivity * STEFAN_BOLTZMANN * (max(T_pyro, T_inf) ** 4 - T_inf ** 4)
                flux_scale = q_preheat_planar + q_surf_avg

                for bi, bj in zip(fi, fj):
                    x_b = bi * dx
                    y_b = bj * dy
                    dx_grid = X_m - x_b
                    dy_grid = Y_m - y_b

                    if is_concurrent:
                        x_decay = np.where(dx_grid >= 0, dx_grid / L_forward, -dx_grid / (0.35 * L_forward))
                    else:
                        x_decay = np.abs(dx_grid) / L_forward
                    y_decay = np.abs(dy_grid) / L_lateral

                    kernel = np.exp(-(x_decay + y_decay))
                    Q_flame = np.maximum(Q_flame, flux_scale * transverse_boost[bj] * kernel)

            Q_loss = emissivity * STEFAN_BOLTZMANN * (np.maximum(T, T_inf) ** 4 - T_inf ** 4)
            Q_total = np.where(state < 2, Q_flame - Q_loss, 0.0)

            for _ in range(sub_steps):
                laplacian = np.zeros_like(T)
                laplacian[1:-1, 1:-1] = (
                    (T[2:, 1:-1] - 2.0 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / (dx ** 2)
                    + (T[1:-1, 2:] - 2.0 * T[1:-1, 1:-1] + T[1:-1, :-2]) / (dy ** 2)
                )

                eff_tau = np.maximum(tau, 1e-4)
                dT_dt = alpha_s * laplacian + Q_total / (rho_s * c_s * eff_tau)
                T = np.maximum(T_inf, T + dT_dt * sub_dt)

                # Enforce true Dirichlet / Neumann boundary conditions after step update
                if edge_boundary == "clamped":
                    T[:, 0] = T_inf
                    T[:, -1] = T_inf
                else:
                    T[:, 0] = T[:, 1]
                    T[:, -1] = T[:, -2]

                T[0, :] = T[1, :]
                T[-1, :] = T[-2, :]

            newly_ignited = (state == 0) & (T >= T_pyro)
            state[newly_ignited] = 1

            burning_now = (state == 1)
            r_dot_grid = np.broadcast_to((q_net_ref * transverse_boost[None, :]) / max(rho_s * H_g, 1.0), (nx, ny))
            tau[burning_now] = np.maximum(0.0, tau[burning_now] - r_dot_grid[burning_now] * dt)

            burned_out = burning_now & (tau <= 2e-5)
            state[burned_out] = 2

            t += dt

        if len(front_positions) >= 2 and (front_positions[-1][0] - front_positions[0][0]) > 0:
            mean_speed = (front_positions[-1][1] - front_positions[0][1]) / (front_positions[-1][0] - front_positions[0][0])
        else:
            mean_speed = 0.0

        return {
            "material": material_name,
            "status": "COMPLETED",
            "grid_shape": [nx, ny],
            "dimensions_mm": [length_mm, width_mm],
            "edge_boundary": edge_boundary,
            "ignition_mode": ignition_mode,
            "reference_1d_spread_mms": base_pred["flame_spread_mms"],
            "mean_2d_front_speed_mms": round(max(0.0, mean_speed), 3),
            "final_burned_area_pct": snapshots[-1]["burned_area_pct"] if snapshots else 0.0,
            "snapshots": snapshots,
        }