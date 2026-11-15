import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# save as check_surrogate_drift.py
import random, numpy as np
from backend.fire_model import FireModel
from backend.materials import MAT_DICT
from ai.surrogate import EnsembleSurrogate

model = FireModel(MAT_DICT)
surr = EnsembleSurrogate()

random.seed(42)
mismatches = 0
deltas = []
tested = 0

for _ in range(200):
    mat = random.choice(list(MAT_DICT.values()))
    g     = random.uniform(0, 1.0)
    o2    = random.uniform(15, 30)
    flow  = random.uniform(1, 30)
    thick = random.uniform(0.1, 10)
    cfg   = random.choice(["Opposed", "Concurrent"])
    
    try:
        phys = model.predict(mat["name"], g_level=g, oxygen_pct=o2,
                             flow_vel_cms=flow, thickness_mm=thick,
                             configuration=cfg)
    except Exception:
        continue
    
    if phys["is_extinct"]:
        continue
    
    tested += 1
    ai = surr.predict(g, o2, flow, thick, mat, configuration=cfg)
    ai_spread = ai["spread_mms"]["mean"]
    delta = abs(ai_spread - phys["flame_spread_mms"]) / phys["flame_spread_mms"]
    deltas.append(delta)
    if delta > 0.30:
        mismatches += 1

deltas = np.array(deltas)
print(f"Tested: {tested} samples")
print(f"Median delta: {np.median(deltas)*100:.2f}%")
print(f"95th pct delta: {np.percentile(deltas, 95)*100:.2f}%")
print(f"Mismatches (>30%): {mismatches}/{tested}")
print()
if mismatches / max(tested, 1) < 0.05:
    print("✅ RETRAIN NOT NEEDED — surrogate is in sync with fire_model.py")
elif mismatches / tested < 0.15:
    print("⚠️  RETRAIN RECOMMENDED — small but noticeable drift")
else:
    print("❌ RETRAIN REQUIRED — surrogate is significantly stale")