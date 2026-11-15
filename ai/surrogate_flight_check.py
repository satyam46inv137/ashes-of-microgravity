"""
Direct check of the trained neural surrogate on the flight calibration runs.

For every run in ai/calibration_data_1d.csv this prints, side by side:
    flight actual | physics engine | surrogate (mean +/- std of the 5 members)
and summarises three errors:
    1. physics   vs flight   (in-sample: the physics constants were tuned on these runs)
    2. surrogate vs flight   (does the emulator inherit / add error at flight conditions?)
    3. surrogate vs physics  (emulator fidelity; this is what the 30% OOD flag monitors)

WHAT THIS DOES *NOT* SHOW
-------------------------
The flight runs are the calibration data of the physics engine, and the surrogate is trained
to reproduce the physics engine. So this is NOT an out-of-sample validation of either model.
It tells you how faithfully the emulator reproduces the physics (and flight data) at the
conditions that matter most. For an honest out-of-sample number for the physics see
ai/nested_loo.py. (The surrogate was not re-trained per fold, so it has no nested estimate.)

Run from the repo root (needs the trained weights in ai/models/ and torch installed):
    python -m ai.surrogate_flight_check
    python -m ai.surrogate_flight_check --include-excluded --out ai/surrogate_flight_check.json
"""
import argparse
import json

import numpy as np

from ai.nested_loo import FUEL_ALIASES, fm, load_materials, load_rows  # noqa: F401  (shared CSV loader)

DEFAULTS = dict(g_level=0.0, pressure_kPa=101.3, oxygen_pct=21.0, flow_vel_cms=20.0,
                thickness_mm=10.0, configuration="Opposed", sample_width_mm=410.0,
                geometry="plate", edge_boundary="free")


def _ape(pred, ref):
    return abs(pred - ref) / ref * 100.0


def _summary(vals):
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return dict(n=0)
    return dict(n=int(vals.size), MAPE=float(vals.mean()), median=float(np.median(vals)),
                max=float(vals.max()), within_30pct=float((vals <= 30).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="ai/calibration_data_1d.csv")
    ap.add_argument("--include-excluded", action="store_true",
                    help="also evaluate the 2 documented exclusions (MIST_04, B_FLIGHT_12)")
    ap.add_argument("--out", default="ai/surrogate_flight_check.json")
    args = ap.parse_args()

    from ai.surrogate import EnsembleSurrogate  # imported here so --help works without torch

    materials = load_materials()
    physics = fm.FireModel(materials)
    sur = EnsembleSurrogate()
    rows = load_rows(args.csv, None, args.include_excluded)

    out_rows = []
    print(f"{'run':<18}{'flight':>9}{'physics':>10}{'surrogate':>11}{'+/-std':>9}"
          f"{'phys/fl%':>10}{'sur/fl%':>9}{'sur/phys%':>10}")
    for r in rows:
        kw = {**DEFAULTS, **r["kwargs"]}
        mat = materials[r["material"]]
        res = (physics._predict_with_raw(r["material"], **kw)
               if hasattr(physics, "_predict_with_raw") else physics.predict(r["material"], **kw))
        if res.get("is_extinct"):
            print(f"{r['run_id']:<18}{r['actual']:>9.3f}   physics predicts EXTINCTION (surrogate not queried)")
            out_rows.append(dict(run_id=r["run_id"], flight=r["actual"], physics=None, surrogate=None))
            continue
        phys = res.get("flame_spread_mms_raw", res["flame_spread_mms"])
        ai = sur.predict(kw["g_level"], kw["oxygen_pct"], kw["flow_vel_cms"], kw["thickness_mm"], mat,
                         configuration=kw["configuration"], pressure_kPa=kw["pressure_kPa"],
                         sample_width_mm=kw["sample_width_mm"], geometry=kw["geometry"],
                         edge_boundary=kw["edge_boundary"])
        s_mean, s_std = ai["spread_mms"]["mean"], ai["spread_mms"]["std"]
        e_pf, e_sf, e_sp = _ape(phys, r["actual"]), _ape(s_mean, r["actual"]), _ape(s_mean, phys)
        print(f"{r['run_id']:<18}{r['actual']:>9.3f}{phys:>10.3f}{s_mean:>11.3f}{s_std:>9.3f}"
              f"{e_pf:>10.1f}{e_sf:>9.1f}{e_sp:>10.1f}")
        out_rows.append(dict(run_id=r["run_id"], flight=r["actual"], physics=phys, surrogate=s_mean,
                             surrogate_std=s_std, err_physics_vs_flight=e_pf,
                             err_surrogate_vs_flight=e_sf, err_surrogate_vs_physics=e_sp))

    ok = [o for o in out_rows if o.get("physics") is not None]
    summary = dict(
        physics_vs_flight_in_sample=_summary([o["err_physics_vs_flight"] for o in ok]),
        surrogate_vs_flight=_summary([o["err_surrogate_vs_flight"] for o in ok]),
        surrogate_vs_physics=_summary([o["err_surrogate_vs_physics"] for o in ok]),
    )
    print("\nSummary (APE, %):")
    for k, v in summary.items():
        if v["n"]:
            print(f"  {k:<30} MAPE={v['MAPE']:.2f}  median={v['median']:.2f}  max={v['max']:.2f}  "
                  f"within 30%: {v['within_30pct']:.0%}  (n={v['n']})")
    print("\nHow to read this:")
    print("  - surrogate_vs_physics small -> the emulator reproduces the physics well at flight conditions.")
    print("  - surrogate_vs_flight ~ physics_vs_flight -> the surrogate simply inherits the physics error.")
    print("  - This is IN-SAMPLE for the physics constants; it is not independent validation of either model.")

    with open(args.out, "w") as f:
        json.dump(dict(rows=out_rows, summary=summary, caveat="in-sample for physics; emulator fidelity check"),
                  f, indent=2, default=float)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()