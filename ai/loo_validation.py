import sys
import csv
import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backend.fire_model as fm
from backend.fire_model import FireModel
from backend.materials import MAT_DICT

CSV_PATH = Path(__file__).parent / "calibration_data_1d.csv"

FUEL_TO_MAT = {
    "SIBAL": "SIBAL Standard Fabric", "PMMA": "PMMA (Acrylic / Plexiglas)",
    "PMMA_Rod": "PMMA (Acrylic / Plexiglas)", "PMMA_Flat": "PMMA (Acrylic / Plexiglas)",
    "PMMA_Sheet": "PMMA (Acrylic / Plexiglas)", "PMMA_Sphere": "PMMA (Acrylic / Plexiglas)",
    "Nomex_HT9040": "Nomex 450 (Aramid Fabric)",
}

PARAMS = ["L_PH_THIN_BASELINE_M", "F_CONCURRENT_THIN_MICROG", "STRAIN_EXPONENT_THICK", "RAD_LOSS_COEFF_MICROG"]

def load_points():
    pts = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r["tier"] in ["core", "extended"]:
                pts.append({
                    "tier": r["tier"], "mat": FUEL_TO_MAT.get(r["fuel"]),
                    "config": r["config"], "g": float(r["g_level"]), "p": float(r["pressure_kPa"]),
                    "o2": float(r["o2_pct"]), "flow": float(r["flow_cms"]),
                    "thick": float(r["thickness_mm"]), "width": float(r["sample_width_mm"]),
                    "geom": r["geometry"], "actual": float(r["spread_mms"])
                })
    return pts

def evaluate_point(model, pt):
    res = model.predict(
        pt["mat"], g_level=pt["g"], pressure_kPa=pt["p"], oxygen_pct=pt["o2"],
        flow_vel_cms=pt["flow"], thickness_mm=pt["thick"], configuration=pt["config"],
        sample_width_mm=pt["width"], geometry=pt["geom"]
    )
    return 0.0 if res["is_extinct"] else res["flame_spread_mms"]

def main():
    points = load_points()
    n = len(points)
    
    # Backup original constants
    orig_consts = {name: getattr(fm, name) for name in PARAMS}
    
    loo_errors_all = []
    loo_errors_1d = []
    
    print(f"Running LOO Validation on {n} points (Refitting {len(PARAMS)} parameters)...")
    
    try:
        for i in range(n):
            test_pt = points[i]
            train_pts = [p for j, p in enumerate(points) if j != i]
            
            def loss(params):
                for name, val in zip(PARAMS, params):
                    setattr(fm, name, val)
                model = FireModel(MAT_DICT)
                errs = []
                for p in train_pts:
                    pred = evaluate_point(model, p)
                    errs.append(abs(pred - p["actual"]) / p["actual"] if pred > 0 else 1.0)
                return np.mean(errs)
            
            res = differential_evolution(loss, bounds=[(0.005, 0.015), (0.1, 0.3), (0.5, 2.5), (0.1, 0.4)], maxiter=50, seed=42, polish=True)
            
            for name, val in zip(PARAMS, res.x):
                setattr(fm, name, val)
                
            model = FireModel(MAT_DICT)
            pred = evaluate_point(model, test_pt)
            err_pct = abs(pred - test_pt["actual"]) / test_pt["actual"] * 100
            
            loo_errors_all.append(err_pct)
            if test_pt["tier"] == "core":
                loo_errors_1d.append(err_pct)
                
            print(f"Fold {i+1}/{n} ({test_pt['tier']}) | Actual: {test_pt['actual']:.3f} | Pred: {pred:.3f} | Error: {err_pct:.1f}%")
    finally:
        for name, val in orig_consts.items():
            setattr(fm, name, val)

    errs = np.array(loo_errors_all)
    band = np.percentile(errs, 95)
    coverage = np.sum(errs <= band)
    
    print("\n=== LOO CROSS-VALIDATION RESULTS ===")
    print(f"True 1D Planar Out-of-Sample MAPE (Core 5 points): {np.mean(loo_errors_1d):.2f}%")
    print(f"All-13 Out-of-Sample MAPE: {np.mean(loo_errors_all):.2f}%")
    print(f"Band (95th pct of LOO errors): ±{band:.2f}% | covers {coverage}/{len(errs)}")

if __name__ == "__main__":
    main()