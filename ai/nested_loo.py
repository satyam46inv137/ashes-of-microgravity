"""
Fully nested Leave-One-Out (LOO) and Leave-One-Mission-Out (LOMO) validation
for backend/fire_model.py.

WHY THIS EXISTS
---------------
The old ai/loo_validation.py refits only 4 parameters per fold; the other
empirical constants stay at values that were tuned on all 13 flight runs, so
the held-out run has already "seen" the test point through them (leakage).

Here, in EVERY fold:
  1. all fitted constants are reset and refit using ONLY the training runs,
  2. the optimiser starts from / is bounded by a-priori ranges that do NOT
     depend on the all-data tuned values,
  3. the held-out run(s) are predicted with the fold's own parameters.

HOW IT WORKS
------------
fire_model.py reads its constants as module globals at call time, so we just
setattr() them on the module for each evaluation (no changes to fire_model.py
are required). Run from the repo root:

    python -m ai.nested_loo --csv ai/calibration_data_1d.csv --mode both --workers 4

ADAPT BEFORE RUNNING (2 places marked ### ADAPT)
------------------------------------------------
  - load_materials(): how to get your materials dict
  - PARAM_SPECS bounds: set from literature / physical reasoning, NOT around
    the values you already tuned (that would re-introduce leakage)
"""
import argparse
import contextlib
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

try:
    import backend.fire_model as fm
except ModuleNotFoundError:  # running from inside backend/
    import fire_model as fm

# ----------------------------------------------------------------------------
# Parameters refit inside every fold: (module constant, lo, hi, log_scale)
# ### ADAPT: bounds must be a-priori (literature / physics), wide, and
# ### independent of your tuned values. Midpoint of the bounds is the prior.
# ----------------------------------------------------------------------------
PARAM_SPECS = [
    ("L_PH_THIN_BASELINE_M",     0.001, 0.020, True),
    ("F_CONCURRENT_THIN_MICROG", 0.03,  0.80,  True),
    ("STRAIN_EXPONENT_THICK",    0.3,   2.2,   False),
    ("RAD_LOSS_COEFF_MICROG",    0.02,  0.70,  False),
    ("ROD_CURVATURE_BASE",       1.0,   2.5,   False),
    ("ROD_CURVATURE_SLOPE",      0.0,   1.0,   False),
    ("LOW_P_RAD_FEEDBACK_EXP",   0.0,   1.5,   False),
    ("THERMAL_DEPTH_COEFF",      100.0, 12000.0, True),
    ("THERMAL_DEPTH_V_EXP",      0.0,   0.6,   False),
    ("THIN_THICK_RATIO_LO",      0.2,   1.2,   False),
    ("THIN_THICK_RATIO_HI",      1.2,   3.0,   False),
]
SPEC_NAMES = [s[0] for s in PARAM_SPECS]

# CSV schema of ai/calibration_data_1d.csv
#   csv column -> FireModel keyword
KW_MAP = {"g_level": "g_level", "pressure_kPa": "pressure_kPa", "o2_pct": "oxygen_pct",
          "flow_cms": "flow_vel_cms", "thickness_mm": "thickness_mm", "config": "configuration",
          "sample_width_mm": "sample_width_mm", "geometry": "geometry"}
EXCLUDE_TIERS = {"preliminary", "known_limitation"}   # skipped unless --include-excluded

# CSV "fuel" names -> names in backend/materials.py (MAT_DICT keys)
FUEL_ALIASES = {
    "SIBAL": "SIBAL Standard Fabric",
    "PMMA": "PMMA (Acrylic / Plexiglas)",
    "PMMA_Rod": "PMMA (Acrylic / Plexiglas)",
    "PMMA_Sheet": "PMMA (Acrylic / Plexiglas)",
    "PMMA_Flat": "PMMA (Acrylic / Plexiglas)",
    "Nomex_HT9040": "Nomex 450 (Aramid Fabric)",
}

SPREAD_FLOOR_MMS = 1e-4  # prediction used when the model says "extinct"

_FACTORY_DEFAULTS = {n: getattr(fm, n) for n in SPEC_NAMES}  # tuned (all-data) values
_MATERIALS = None


def load_materials():
    """### ADAPT: return the dict {name: property_dict} used by FireModel."""
    global _MATERIALS
    if _MATERIALS is None:
        try:
            import backend.materials as m
        except ModuleNotFoundError:
            import materials as m
        for attr in ("MAT_DICT", "MATERIALS", "MATERIAL_DB"):
            if hasattr(m, attr):
                _MATERIALS = getattr(m, attr)
                break
        else:
            sys.exit("load_materials(): couldn't find the materials dict; edit it.")
    return _MATERIALS


@contextlib.contextmanager
def patched(params):
    old = {n: getattr(fm, n) for n in params}
    for n, v in params.items():
        setattr(fm, n, v)
    try:
        yield
    finally:
        for n, v in old.items():
            setattr(fm, n, v)


def u_to_params(u):
    out = {}
    for ui, (name, lo, hi, log) in zip(u, PARAM_SPECS):
        out[name] = (math.exp(math.log(lo) + ui * (math.log(hi) - math.log(lo)))
                     if log else lo + ui * (hi - lo))
    return out


def predict_rows(rows, params):
    model = fm.FireModel(load_materials())
    preds = []
    with patched(params):
        for r in rows:
            try:
                res = model._predict_core(r["material"], **r["kwargs"])
                p = SPREAD_FLOOR_MMS if res["is_extinct"] else max(res["flame_spread_mms"], SPREAD_FLOOR_MMS)
            except Exception:
                p = SPREAD_FLOOR_MMS
            preds.append(p)
    return np.array(preds)


def fit_params(train_rows, prior_w, seed, maxiter, popsize):
    actual = np.array([r["actual"] for r in train_rows])

    def loss(u):
        p = u_to_params(u)
        if p["THIN_THICK_RATIO_LO"] >= p["THIN_THICK_RATIO_HI"]:
            return 1e3
        pred = predict_rows(train_rows, p)
        data = np.mean(np.log(pred / actual) ** 2)           # log-ratio ~ relative error
        prior = np.mean((np.asarray(u) - 0.5) ** 2)          # shrink toward a-priori midpoint
        return data + prior_w * prior

    res = differential_evolution(
        loss, bounds=[(0.0, 1.0)] * len(PARAM_SPECS), seed=seed,
        maxiter=maxiter, popsize=popsize, tol=1e-6, polish=False,
        updating="immediate",
    )
    return u_to_params(res.x), res.x, float(res.fun)


def _run_fold(args):
    fold_id, train, test, prior_w, seed, maxiter, popsize = args
    params, u, fun = fit_params(train, prior_w, seed, maxiter, popsize)
    pred = predict_rows(test, params)
    return dict(fold=fold_id, params=params, u=u.tolist(), train_loss=fun,
                test_ids=[r["run_id"] for r in test],
                test_actual=[r["actual"] for r in test],
                test_pred=pred.tolist())


def mission_of(run_id):
    s = str(run_id)
    m = re.search(r"(saffire[-_ ]?(?:VI|V|IV|III|II|I)(?![A-Za-z])|bass[-_ ]?II(?![A-Za-z])|sofie|mist)", s, re.I)
    if m:
        return re.sub(r"[-_ ]", "", m.group(1)).upper()
    return "UNKNOWN"


def load_rows(csv, mission_col, include_excluded):
    df = pd.read_csv(csv)
    rows = []
    for _, d in df.iterrows():
        if not include_excluded and str(d["tier"]).strip().lower() in EXCLUDE_TIERS:
            continue
        kw = {m: d[c] for c, m in KW_MAP.items() if c in df.columns and pd.notna(d[c])}
        rid = f"{d['source']}_{d['mission']}"            # e.g. Saffire-I_S1
        rows.append(dict(
            run_id=rid, material=FUEL_ALIASES.get(d["fuel"], d["fuel"]), actual=float(d["spread_mms"]), kwargs=kw,
            mission=str(d[mission_col]) if mission_col else str(d["source"])))  # group = Saffire-I, BASS-II...
    return rows


def metrics(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    ape = np.abs(pred - actual) / actual * 100.0
    return dict(n=len(ape), MAPE=float(ape.mean()), median_APE=float(np.median(ape)),
                P95_APE=float(np.percentile(ape, 95)), max_APE=float(ape.max()),
                within_20pct=float((ape <= 20).mean()), within_30pct=float((ape <= 30).mean()))


def run_cv(name, rows, groups, args):
    jobs = []
    for i, g in enumerate(groups):
        ids = set(g)
        train = [r for r in rows if r["run_id"] not in ids]
        test = [r for r in rows if r["run_id"] in ids]
        jobs.append((f"{name}:{i}", train, test, args.prior_weight, args.seed + i,
                     args.maxiter, args.popsize))
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            folds = list(ex.map(_run_fold, jobs))
    else:
        folds = [_run_fold(j) for j in jobs]

    act = [a for f in folds for a in f["test_actual"]]
    prd = [p for f in folds for p in f["test_pred"]]
    ids = [i for f in folds for i in f["test_ids"]]

    # Baselines evaluated on the SAME folds
    geo = []
    for f in folds:
        tr = [r["actual"] for r in rows if r["run_id"] not in set(f["test_ids"])]
        geo += [math.exp(np.mean(np.log(tr)))] * len(f["test_ids"])
    by_id = {r["run_id"]: r for r in rows}

    U = np.array([f["u"] for f in folds])
    stab = {}
    for j, n in enumerate(SPEC_NAMES):
        vals = [f["params"][n] for f in folds]
        stab[n] = dict(mean=float(np.mean(vals)), cv=float(np.std(vals) / (abs(np.mean(vals)) + 1e-12)),
                       frac_at_bound=float(np.mean((U[:, j] < 0.02) | (U[:, j] > 0.98))))
    return dict(nested=metrics(act, prd), baseline_geomean_of_train=metrics(act, geo),
                per_run=[dict(run_id=i, actual=a, pred=p, ape=abs(p - a) / a * 100)
                         for i, a, p in zip(ids, act, prd)],
                param_stability=stab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="ai/calibration_data_1d.csv")
    ap.add_argument("--mode", choices=["loo", "lomo", "both"], default="both")
    ap.add_argument("--mission-col", default=None)
    ap.add_argument("--include-excluded", action="store_true",
                    help="also use MIST_04 / B_FLIGHT_12 (all-15 reporting)")
    ap.add_argument("--prior-weight", type=float, default=0.0)
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ai/nested_loo_results.json")
    args = ap.parse_args()

    rows = load_rows(args.csv, args.mission_col, args.include_excluded)
    print(f"{len(rows)} runs | missions: " +
          ", ".join(f"{m}={sum(r['mission']==m for r in rows)}" for m in sorted({r['mission'] for r in rows})))

    out = dict(config=vars(args), n_params=len(PARAM_SPECS), n_runs=len(rows))

    # In-sample reference with the all-data tuned values (the optimistic number)
    pred = predict_rows(rows, _FACTORY_DEFAULTS)
    out["in_sample_with_tuned_constants"] = metrics([r["actual"] for r in rows], pred)

    if args.mode in ("loo", "both"):
        out["nested_loo"] = run_cv("loo", rows, [[r["run_id"]] for r in rows], args)
    if args.mode in ("lomo", "both"):
        missions = sorted({r["mission"] for r in rows})
        out["nested_lomo"] = run_cv("lomo", rows,
                                    [[r["run_id"] for r in rows if r["mission"] == m] for m in missions], args)

    for k in ("in_sample_with_tuned_constants", "nested_loo", "nested_lomo"):
        if k not in out:
            continue
        m = out[k] if k.startswith("in_") else out[k]["nested"]
        print(f"\n{k}: MAPE={m['MAPE']:.2f}%  median={m['median_APE']:.2f}%  "
              f"P95={m['P95_APE']:.2f}%  max={m['max_APE']:.2f}%  <=20%: {m['within_20pct']:.0%}")
        if not k.startswith("in_"):
            for r in out[k]["per_run"]:
                print(f"   {r['run_id']:<22} actual={r['actual']:.3f} pred={r['pred']:.3f} err={r['ape']:.1f}%")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()