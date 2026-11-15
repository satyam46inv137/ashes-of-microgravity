"""
Tier-aware calibration of the physics engine against flight data.

Usage:
    python -m ai.calibrate_constants

Reports:
    core             -- inside flight-validated envelope (headline number)
    extended         -- outside envelope (strain, near-LOI, low-pressure)
    preliminary      -- pending NTRS validation (excluded from MAPE)
    known_limitation -- documented model failure modes (excluded from MAPE)
"""

import csv
import logging
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from backend.fire_model import FireModel
from backend.materials import MAT_DICT

logging.basicConfig(level=logging.WARNING, format="%(message)s")

CALIBRATION_CSV = Path(__file__).parent / "calibration_data_1d.csv"

FUEL_TO_MATERIAL = {
    "SIBAL": "SIBAL Standard Fabric",
    "PMMA": "PMMA (Acrylic / Plexiglas)",
    "PMMA_Flat": "PMMA (Acrylic / Plexiglas)",
    "PMMA_Sheet": "PMMA (Acrylic / Plexiglas)",
    "PMMA_Rod": "PMMA (Acrylic / Plexiglas)",
    "PMMA_Sphere": "PMMA (Acrylic / Plexiglas)",
    "Nomex_HT9040": "Nomex 450 (Aramid Fabric)",
}


def load_points():
    if not CALIBRATION_CSV.exists():
        raise FileNotFoundError(f"Calibration file not found: {CALIBRATION_CSV}")

    points = []
    with open(CALIBRATION_CSV) as f:
        for row in csv.DictReader(f):
            mat_name = FUEL_TO_MATERIAL.get(row["fuel"])
            if not mat_name:
                continue
            points.append({
                "tier": row["tier"],
                "source": row["source"],
                "mission": row["mission"],
                "fuel": row["fuel"],
                "material": mat_name,
                "config": row["config"],
                "g": float(row["g_level"]),
                "p": float(row["pressure_kPa"]),
                "o2": float(row["o2_pct"]),
                "flow": float(row["flow_cms"]),
                "thick": float(row["thickness_mm"]),
                "width": float(row["sample_width_mm"]),
                "geom": row["geometry"],
                "actual": float(row["spread_mms"]),
            })
    return points


def evaluate(model, points):
    rows = []
    preds, actuals = [], []

    for pt in points:
        try:
            res = model.predict(
                pt["material"],
                g_level=pt["g"],
                pressure_kPa=pt["p"],
                oxygen_pct=pt["o2"],
                flow_vel_cms=pt["flow"],
                thickness_mm=pt["thick"],
                configuration=pt["config"],
                sample_width_mm=pt["width"],
                geometry=pt["geom"],
            )
        except Exception as e:
            rows.append((pt, None, f"ERR: {e}"))
            continue

        pred = 0.0 if res["is_extinct"] else res["flame_spread_mms"]
        preds.append(pred)
        actuals.append(pt["actual"])
        err_pct = abs(pred - pt["actual"]) / pt["actual"] * 100 if pt["actual"] > 0 else 100
        rows.append((pt, pred, f"{err_pct:.1f}%"))

    if not preds:
        return float("nan"), float("nan"), rows

    preds_arr = np.array(preds)
    actuals_arr = np.array(actuals)
    ape = np.abs(preds_arr - actuals_arr) / np.where(actuals_arr > 0, actuals_arr, 1e-9) * 100
    mape = float(np.mean(ape))

    ss_res = np.sum((actuals_arr - preds_arr) ** 2)
    ss_tot = np.sum((actuals_arr - actuals_arr.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return mape, r2, rows


def print_report(rows, mape, r2, label):
    print(f"\n{'=' * 95}")
    print(f"  {label}")
    if not rows or (isinstance(mape, float) and math.isnan(mape)):
        print(f"  (No points in this tier — skipped)")
        print(f"{'=' * 95}\n")
        return
    print(f"  MAPE = {mape:.2f}%  |  R2 = {r2:.3f}  |  Points = {len(rows)}")
    print(f"{'=' * 95}")
    print(f"{'Mission':<14} {'Fuel':<14} {'Actual':>8} {'Predicted':>10} {'Error':>8} {'Tier'}")
    print(f"{'-' * 95}")
    for pt, pred, err in rows:
        pred_str = f"{pred:.4f}" if pred is not None else "-"
        print(f"{pt['mission']:<14} {pt['fuel'][:12]:<14} {pt['actual']:>8.4f} {pred_str:>10} {err:>8} {pt['tier']}")
    print(f"{'-' * 95}\n")


def main():
    points = load_points()

    by_tier = {}
    for pt in points:
        by_tier.setdefault(pt["tier"], []).append(pt)

    summary = ", ".join(f"{len(v)} {k}" for k, v in by_tier.items())
    print(f"\nLoaded {len(points)} calibration points ({summary})")

    model = FireModel(MAT_DICT)

    core_mape, core_r2, core_rows = evaluate(model, by_tier.get("core", []))
    print_report(core_rows, core_mape, core_r2, "CORE VALIDATION (inside flight-validated envelope)")

    ext_mape, ext_r2, ext_rows = evaluate(model, by_tier.get("extended", []))
    print_report(ext_rows, ext_mape, ext_r2, "EXTENDED VALIDATION (outside envelope - strain/LOI/pressure regimes)")

    if "preliminary" in by_tier:
        p_mape, p_r2, p_rows = evaluate(model, by_tier["preliminary"])
        print_report(p_rows, p_mape, p_r2, "PRELIMINARY (pending NTRS validation - NOT counted in headline MAPE)")

    if "known_limitation" in by_tier:
        k_mape, k_r2, k_rows = evaluate(model, by_tier["known_limitation"])
        print_report(k_rows, k_mape, k_r2, "KNOWN LIMITATION (documented model failure mode - NOT counted in headline MAPE)")

    core_and_ext = by_tier.get("core", []) + by_tier.get("extended", [])
    combined_mape, combined_r2, _ = evaluate(model, core_and_ext)

    print("=" * 95)
    print("  FINAL METRICS FOR README")
    print("=" * 95)
    print(f"  Core MAPE:      {core_mape:.2f}%  (n={len(core_rows)}, R2={core_r2:.3f})")
    print(f"  Extended MAPE:  {ext_mape:.2f}%  (n={len(ext_rows)}, R2={ext_r2:.3f})")
    print(f"  Combined MAPE:  {combined_mape:.2f}%  (n={len(core_and_ext)}, R2={combined_r2:.3f})")
    print(f"  Sources: Saffire-I/II/III/IV/V, BASS, BASS-II, SoFIE")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()