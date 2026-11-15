import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from backend.fire_model import (
        T_FLAME_MAX_K, T_AMBIENT_K, NU_AIR_300K, X_REF_M,
        F_CONCURRENT_THIN_MICROG, F_CONCURRENT_THICK_MICROG,
        F_CONCURRENT_THIN_BUOYANT, F_CONCURRENT_THICK_BUOYANT,
        LOI_MICROG_FACTOR_DEFAULT, _smoothstep, thermal_blend_weight,
        FLOW_STRAIN_CRITICAL_CMS, FLOW_STRAIN_CRITICAL_THIN_CMS,
        LOW_P_RAD_FEEDBACK_EXP, ROD_CURVATURE_RATIO_MAX,
        LATERAL_O2_ENTRAIN_COEFF, NARROW_SHEET_EDGE_EXP,
    )
except ImportError:
    from fire_model import (
        T_FLAME_MAX_K, T_AMBIENT_K, NU_AIR_300K, X_REF_M,
        F_CONCURRENT_THIN_MICROG, F_CONCURRENT_THICK_MICROG,
        F_CONCURRENT_THIN_BUOYANT, F_CONCURRENT_THICK_BUOYANT,
        LOI_MICROG_FACTOR_DEFAULT, _smoothstep, thermal_blend_weight,
        FLOW_STRAIN_CRITICAL_CMS, FLOW_STRAIN_CRITICAL_THIN_CMS,
        LOW_P_RAD_FEEDBACK_EXP, ROD_CURVATURE_RATIO_MAX,
        LATERAL_O2_ENTRAIN_COEFF, NARROW_SHEET_EDGE_EXP,
    )

INPUT_DIM = 41
OUTPUT_DIM = 3
N_ENSEMBLE = 5
CHARRING_CATS = {"Engineering Thermosets", "Wood & Bio-materials"}
EPS = 1e-6


def engineer_features(
    g, o2, flow, thick, mat,
    configuration="Opposed",
    pressure_kPa=101.3,
    sample_width_mm=410.0,
    geometry="plate",
    edge_boundary="free",
):
    T_inf = T_AMBIENT_K
    X_O2_nom = o2 / 100.0
    V_forced = max(flow / 100.0, 0.001)
    tau = max(thick, 0.1) / 1000.0
    W = max(sample_width_mm, 1.0) / 1000.0
    g_acc = g * 9.81
    p_ratio = max(pressure_kPa / 101.3, 0.05)

    geom = geometry.lower().strip()
    edge = edge_boundary.lower().strip()

    nu_g = NU_AIR_300K / max(p_ratio, 0.1)
    Re_prelim = max((V_forced * X_REF_M) / nu_g, 0.1)
    delta_f_prelim = X_REF_M / math.sqrt(Re_prelim)

    if geom == "plate" and edge == "free" and sample_width_mm < 200.0:
        entrain_boost = 1.0 + LATERAL_O2_ENTRAIN_COEFF * min(delta_f_prelim / max(W, 0.005), 0.6)
    elif geom in ("rod", "cylinder", "sphere"):
        entrain_boost = 1.22 if geom == "sphere" else 1.18
    else:
        entrain_boost = 1.0

    X_O2 = min(1.0, X_O2_nom * entrain_boost)
    eff_o2_pct = X_O2 * 100.0

    T_eff = T_inf + (T_FLAME_MAX_K - T_inf) * ((X_O2 / 0.21) ** 0.42) * (
        1.0 - math.exp(-1.2 * (p_ratio ** 0.25))
    )

    V_buoy = math.sqrt(2.0 * g_acc * X_REF_M * max(T_eff - T_inf, 10.0) / T_inf) if g_acc > 0.001 else 0.0
    V_eff = math.sqrt(V_forced ** 2 + V_buoy ** 2)
    Re_eff = max((V_eff * X_REF_M) / nu_g, 0.1)
    delta_f = X_REF_M / math.sqrt(Re_eff)
    dT_pyro = max(mat["T_pyro"] - T_inf, 50.0)
    thermal_mass = mat["rho_s"] * mat["c_s"] * tau

    is_microg = g < 0.01
    is_concurrent = configuration.lower().strip() == "concurrent"

    category = mat.get("category", "")
    is_charring = 1.0 if category in CHARRING_CATS else 0.0
    is_foam = 1.0 if "Foam" in category else 0.0
    is_liquid = 1.0 if "Liquid" in category else 0.0
    is_metal = 1.0 if "Metal" in category else 0.0
    is_textile = 1.0 if "Fabric" in category else 0.0

    char_weight = _smoothstep(tau, 0.0005, 0.002)
    k_eff = mat["k_s"] * (1.0 - 0.35 * char_weight) if is_charring else mat["k_s"]
    alpha_eff = k_eff / max(mat["rho_s"] * mat["c_s"], 1.0)
    thick_blend, thermal_ratio = thermal_blend_weight(tau, alpha_eff, V_eff)

    if is_concurrent:
        f_thin = F_CONCURRENT_THIN_MICROG if is_microg else F_CONCURRENT_THIN_BUOYANT
        f_thick = F_CONCURRENT_THICK_MICROG if is_microg else F_CONCURRENT_THICK_BUOYANT
        flow_factor = (1.0 - thick_blend) * f_thin + thick_blend * f_thick
    else:
        flow_factor = 1.0

    is_plate = 1.0 if geom == "plate" else 0.0
    is_rod = 1.0 if geom in ("rod", "cylinder") else 0.0
    is_sphere = 1.0 if geom == "sphere" else 0.0
    is_edge_free = 1.0 if edge == "free" else 0.0
    is_edge_clamped = 1.0 if edge == "clamped" else 0.0

    log_width = math.log(max(sample_width_mm, 1.0) + EPS)
    loi_factor = mat.get("LOI_microg_factor", LOI_MICROG_FACTOR_DEFAULT)
    emissivity = mat.get("emissivity", 0.85)
    alpha_s = mat["k_s"] / max(mat["rho_s"] * mat["c_s"], 1.0)

    p_loi_corr = (1.0 / p_ratio) ** 0.12 if p_ratio < 1.0 else 1.0
    loi_thr = mat.get("LOI", 18.0) * (loi_factor if is_microg else 1.0) * p_loi_corr
    loi_margin = (eff_o2_pct - loi_thr) / loi_thr if loi_thr > 0 else 1.0
    loi_margin = max(-0.5, min(2.0, loi_margin))
    low_p_factor = math.log((1.0 / p_ratio) ** LOW_P_RAD_FEEDBACK_EXP) if p_ratio < 1.0 else 0.0
    rod_ratio = min(delta_f / tau, ROD_CURVATURE_RATIO_MAX) if is_rod else 0.0
    narrow_boost = math.log((200.0 / sample_width_mm) ** NARROW_SHEET_EDGE_EXP) if (
        geom == "plate" and edge == "free" and sample_width_mm < 200.0) else 0.0
    strain_thin = math.log(max(V_forced / (FLOW_STRAIN_CRITICAL_THIN_CMS / 100.0), 1.0))
    strain_thick = math.log(max(V_forced / (FLOW_STRAIN_CRITICAL_CMS / 100.0), 1.0))

    return [
        g, o2, flow, thick,
        mat["rho_s"], mat["c_s"], mat["k_s"], mat["T_pyro"], mat["LOI"],
        math.log(max(T_eff - mat["T_pyro"], EPS)),
        math.log(V_eff + EPS),
        math.log(dT_pyro + EPS),
        math.log(thermal_mass + EPS),
        math.log(mat["k_s"] + EPS),
        T_eff,
        p_ratio,
        flow_factor,
        is_charring, is_foam, is_liquid, is_metal, is_textile,
        log_width, is_plate, is_rod, is_sphere, loi_factor,
        is_edge_free, is_edge_clamped, emissivity * math.log(alpha_s + EPS),
        math.log(delta_f + EPS),
        math.log(max(thermal_ratio, EPS)),
        thick_blend,
        rod_ratio,
        low_p_factor,
        narrow_boost,
        loi_margin,
        strain_thin,
        strain_thick,
        1.0 if is_microg else 0.0,
        1.0 if is_concurrent else 0.0,
    ]


class SurrogateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


class EnsembleSurrogate:
    def __init__(self, model_dir=None, device=None):
        if model_dir is None:
            model_dir = Path(__file__).parent / "models"
        model_dir = Path(model_dir)

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        scaler_path = model_dir / "scalers.npz"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"scalers.npz not found at {scaler_path}. "
                "The .pt ensemble weights are only valid with the .npz scaler archive."
            )

        data = np.load(scaler_path)
        self.mean_X = data["mean_X"].astype(np.float32)
        self.scale_X = data["scale_X"].astype(np.float32)
        self.mean_y = data["mean_y"].astype(np.float32)
        self.scale_y = data["scale_y"].astype(np.float32)

        if self.mean_X.shape[0] != INPUT_DIM:
            raise ValueError(
                f"Scaler expects {self.mean_X.shape[0]} features, "
                f"but INPUT_DIM={INPUT_DIM}. Run `python -m ai.train_surrogate` to retrain."
            )

        self.models = []
        for i in range(N_ENSEMBLE):
            model_file = model_dir / f"surrogate_{i}.pt"
            if not model_file.exists():
                raise FileNotFoundError(f"Model file {model_file} not found.")
            net = SurrogateNet().to(self.device)
            net.load_state_dict(torch.load(model_file, map_location=self.device, weights_only=True))
            net.eval()
            self.models.append(net)

    @torch.no_grad()
    def predict(
        self, g, o2, flow, thick, mat,
        configuration="Opposed",
        pressure_kPa=101.3,
        sample_width_mm=410.0,
        geometry="plate",
        edge_boundary="free",
    ):
        feats = np.array(
            engineer_features(
                g, o2, flow, thick, mat,
                configuration=configuration,
                pressure_kPa=pressure_kPa,
                sample_width_mm=sample_width_mm,
                geometry=geometry,
                edge_boundary=edge_boundary,
            ),
            dtype=np.float32,
        )
        x_norm = (feats - self.mean_X) / self.scale_X
        x = torch.tensor(x_norm[None, :], dtype=torch.float32, device=self.device)

        preds = torch.stack([m(x)[0] for m in self.models]).cpu().numpy()
        preds_orig = preds * self.scale_y + self.mean_y

        spread_linear = np.power(10.0, preds_orig[:, 0])
        flame_temp_mean = float(preds_orig[:, 1].mean())
        flame_temp_std = float(preds_orig[:, 1].std())

        return {
            "spread_mms": {
                "mean": float(spread_linear.mean()),
                "std": float(spread_linear.std()),
            },
            "flame_temp_K": {
                "mean": flame_temp_mean,
                "std": flame_temp_std,
            },
            "effective_flame_temp_K": {
                "mean": flame_temp_mean,
                "std": flame_temp_std,
            },
            "v_eff_cms": {
                "mean": float(preds_orig[:, 2].mean()),
                "std": float(preds_orig[:, 2].std()),
            },
        }
