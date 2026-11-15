"""
Training pipeline for the 5-model Physics-Informed Neural Surrogate Ensemble.
Generates 35,000 stratified samples across microgravity, lunar/Martian, and terrestrial
regimes, including Quasi-2D sample widths and edge boundary conditions.
"""

import copy
import logging
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

try:
    from backend.fire_model import FireModel
    from backend.materials import MAT_DICT, MATERIALS_118
except ImportError:
    from fire_model import FireModel
    from materials import MAT_DICT, MATERIALS_118

try:
    from ai.surrogate import SurrogateNet, N_ENSEMBLE, engineer_features
except ImportError:
    from surrogate import SurrogateNet, N_ENSEMBLE, engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

N_SAMPLES = 35000
EPOCHS = 300            # real epochs now (each = ~N_train/BATCH_SIZE optimizer steps)
BATCH_SIZE = 256        # mini-batch training (was full-batch: 1 step per "epoch")
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 25
LOSS_WEIGHTS = (1.0, 0.25, 0.25)   # log10(spread) matters most; T_flame / V_eff are near-deterministic
SPREAD_MIN = 0.005
SPREAD_MAX = 35.0
SEED = 42


def generate_training_data(model, n_samples=N_SAMPLES):
    X, y = [], []
    skipped_extinct = 0
    skipped_error = 0

    for _ in tqdm(range(n_samples), desc="Generating"):
        mat = random.choice(MATERIALS_118)

        # Stratified gravity sampling: 30% microgravity (0-0.01g), 35% Moon/Mars, 35% Earth/hyper-g
        r_g = random.random()
        if r_g < 0.30:
            g = random.uniform(0.0, 0.01)
        elif r_g < 0.65:
            g = random.uniform(0.01, 0.40)
        else:
            g = random.uniform(0.40, 1.50)

        p = random.uniform(40.0, 120.0)
        o2 = random.uniform(12.0, 35.0)
        flow = random.uniform(0.5, 48.0)
        # Stratified thickness sampling to cover thin (<1mm), transitional (0.5-2mm), and thick (>2mm)
        thick = random.uniform(0.1, 2.5) if random.random() < 0.45 else random.uniform(2.5, 20.0)
        cfg = random.choice(["Opposed", "Concurrent"])

        geom = random.choice(["plate", "plate", "rod", "sphere"])
        if geom == "plate":
            width = random.uniform(40.0, 410.0)
            edge = random.choice(["free", "free", "clamped", "insulated"])
        else:
            width = 410.0
            edge = "free"

        try:
            r = model.predict(
                mat["name"],
                g_level=g,
                pressure_kPa=p,
                oxygen_pct=o2,
                flow_vel_cms=flow,
                thickness_mm=thick,
                configuration=cfg,
                sample_width_mm=width,
                geometry=geom,
                edge_boundary=edge,
            )
        except Exception:
            skipped_error += 1
            continue

        if r["is_extinct"]:
            skipped_extinct += 1
            continue

        spread = r["flame_spread_mms"]
        if spread < SPREAD_MIN or spread > SPREAD_MAX:
            continue

        feats = engineer_features(
            g, o2, flow, thick, mat,
            configuration=cfg,
            pressure_kPa=p,
            sample_width_mm=width,
            geometry=geom,
            edge_boundary=edge,
        )
        y.append([np.log10(spread), r["flame_temp_K"], r["v_eff_cms"]])
        X.append(feats)

    logger.info(f"Data generation complete: {len(X)} usable, {skipped_extinct} extinct, {skipped_error} errors")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def spread_metrics(pred_log, true_log):
    """Metrics in physical units (mm/s), computed from log10(spread) predictions."""
    pred_log = np.asarray(pred_log, dtype=np.float64)
    true_log = np.asarray(true_log, dtype=np.float64)
    if len(true_log) == 0:
        return {"n": 0, "rmse_log10": float("nan"), "mape": float("nan"), "within20": float("nan")}
    ape = np.abs(10.0 ** pred_log - 10.0 ** true_log) / 10.0 ** true_log * 100.0
    return {
        "n": int(len(true_log)),
        "rmse_log10": float(np.sqrt(np.mean((pred_log - true_log) ** 2))),
        "mape": float(ape.mean()),
        "within20": float(np.mean(ape < 20.0) * 100.0),
    }


def regime_masks(X):
    """Boolean masks over raw (unscaled) engineered features. Indices follow engineer_features()."""
    g, thick = X[:, 0], X[:, 3]
    is_rod_or_sphere = (X[:, 24] + X[:, 25]) > 0.5
    return {
        "microgravity (g<0.01)": g < 0.01,
        "partial/1g (g>=0.4)": g >= 0.4,
        "thin (<1 mm)": thick < 1.0,
        "thick (>=1 mm)": thick >= 1.0,
        "rod / sphere": is_rod_or_sphere,
        "concurrent": X[:, 40] > 0.5,
    }


def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = FireModel(MAT_DICT)
    logger.info("Generating training data from calibrated Quasi-2D physics engine...")
    X, y = generate_training_data(model)

    spread_log = y[:, 0]
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Spread range: {10**spread_log.min():.4f} - {10**spread_log.max():.2f} mm/s")

    # 70 / 15 / 15 split. Validation drives early stopping + LR schedule only;
    # the test split is never touched until the final report.
    X_train_raw, X_tmp, y_train_raw, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)

    scaler_X = StandardScaler().fit(X_train_raw)
    scaler_y = StandardScaler().fit(y_train_raw)

    X_train = scaler_X.transform(X_train_raw).astype(np.float32)
    X_val = scaler_X.transform(X_val_raw).astype(np.float32)
    X_test = scaler_X.transform(X_test_raw).astype(np.float32)
    y_train = scaler_y.transform(y_train_raw).astype(np.float32)
    y_val = scaler_y.transform(y_val_raw).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device} | LR={LR} | EPOCHS={EPOCHS} | BATCH={BATCH_SIZE} | PATIENCE={PATIENCE}")

    Xt = torch.tensor(X_train).to(device)
    yt = torch.tensor(y_train).to(device)
    Xv = torch.tensor(X_val).to(device)
    yv = torch.tensor(y_val).to(device)
    Xte = torch.tensor(X_test).to(device)
    w = torch.tensor(LOSS_WEIGHTS, dtype=torch.float32, device=device)

    def weighted_mse(pred, target):
        return (((pred - target) ** 2) * w).mean()

    save_dir = Path(__file__).resolve().parent / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    n_train = Xt.shape[0]
    val_losses = []
    member_test_preds = []

    for i in range(N_ENSEMBLE):
        torch.manual_seed(SEED + i * 42)
        net = SurrogateNet().to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

        best_val = float("inf")
        best_state = copy.deepcopy(net.state_dict())
        patience_counter = 0
        epochs_run = 0

        for epoch in range(EPOCHS):
            net.train()
            perm = torch.randperm(n_train, device=device)
            for start_idx in range(0, n_train, BATCH_SIZE):
                idx = perm[start_idx:start_idx + BATCH_SIZE]
                optimizer.zero_grad()
                loss = weighted_mse(net(Xt[idx]), yt[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            net.eval()
            with torch.no_grad():
                val_loss = weighted_mse(net(Xv), yv).item()
            scheduler.step(val_loss)
            epochs_run = epoch + 1

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(net.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > PATIENCE:
                    break

        net.load_state_dict(best_state)
        torch.save(best_state, save_dir / f"surrogate_{i}.pt")
        val_losses.append(best_val)

        net.eval()
        with torch.no_grad():
            member_test_preds.append(net(Xte).cpu().numpy() * scaler_y.scale_ + scaler_y.mean_)
        m = spread_metrics(member_test_preds[-1][:, 0], y_test_raw[:, 0])
        logger.info(f"Model {i}: val_loss={best_val:.5f} | epochs={epochs_run} | "
                    f"test MAPE={m['mape']:.2f}% RMSE(log10)={m['rmse_log10']:.4f}")

    np.savez(
        save_dir / "scalers.npz",
        mean_X=scaler_X.mean_.astype(np.float32),
        scale_X=scaler_X.scale_.astype(np.float32),
        mean_y=scaler_y.mean_.astype(np.float32),
        scale_y=scaler_y.scale_.astype(np.float32),
    )
    joblib.dump(scaler_X, save_dir / "scaler_X.pkl")
    joblib.dump(scaler_y, save_dir / "scaler_y.pkl")

    # ---- Final held-out report (ensemble) ----
    P = np.stack(member_test_preds)
    ens_log = P[:, :, 0].mean(axis=0)
    ens_std = P[:, :, 0].std(axis=0)
    true_log = y_test_raw[:, 0]

    overall = spread_metrics(ens_log, true_log)
    logger.info("=" * 78)
    logger.info(f"HELD-OUT TEST (n={overall['n']}) ensemble: MAPE={overall['mape']:.2f}% | "
                f"median-friendly within-20%={overall['within20']:.1f}% | RMSE(log10)={overall['rmse_log10']:.4f}")
    err = np.abs(ens_log - true_log)
    logger.info(f"Ensemble uncertainty: corr(std,|err|)={np.corrcoef(ens_std, err)[0, 1]:.2f} | "
                f"|err|<=2*std for {np.mean(err <= 2 * ens_std) * 100:.0f}% of points")
    for name, mask in regime_masks(X_test_raw).items():
        rm = spread_metrics(ens_log[mask], true_log[mask])
        logger.info(f"  {name:<24} n={rm['n']:>5}  MAPE={rm['mape']:6.2f}%  within-20%={rm['within20']:5.1f}%")
    logger.info("=" * 78)
    logger.info(f"Training complete! Mean val loss: {np.mean(val_losses):.5f}")

    # ---- Persist held-out report as JSON artifact for README / provenance ----
    import json
    heldout_report = {
        "overall": overall,
        "per_regime": {
            name: spread_metrics(ens_log[mask], true_log[mask])
            for name, mask in regime_masks(X_test_raw).items()
        },
        "n_train": int(len(X_train_raw)),
        "n_val": int(len(X_val_raw)),
        "n_test": int(len(X_test_raw)),
        "seed": SEED,
        "n_samples_total": N_SAMPLES,
        "ensemble_members": N_ENSEMBLE,
    }
    report_path = save_dir / "heldout_report.json"
    with open(report_path, "w") as f:
        json.dump(heldout_report, f, indent=2)
    logger.info(f"Held-out report saved to {report_path}")


if __name__ == "__main__":
    train()