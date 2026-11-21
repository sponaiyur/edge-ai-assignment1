"""
train.py -- STARTER. Clean the data, fit baselines, train the network, save it for the server.

    python train.py --team 7 --hidden 32,32
Writes model/model.pt and model/config.json, which app/main.py loads.

Look for TODO. The training loop is given; the decisions are yours.
"""
import argparse
import json
import os
import copy

import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet
from report import save_training_report

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--hidden", default="32,32")
ap.add_argument("--epochs", type=int, default=1500)
a = ap.parse_args()
hidden = tuple(int(h) for h in a.hidden.split(","))

df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")

# Refer notebooks/EDA.ipynb for detailed explanation and thought process for bad
rows_before=len(df)
bad = (
    (df["battery_v"] == 0) |
    (df["temp_c"] == -127) |
    (df["wifi_rssi_dbm"] == 0)
)
glitch_rows = int(bad.sum())

print(f"{len(df)} rows, {bad.sum()} glitch rows dropped")
df = df[~bad].copy()
rows_after=len(df)

rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)

mu = X.mean(axis=0)
sd = X.std(axis=0)

# Prevent division by zero
sd[sd == 0] = 1.0

# mu, sd = np.zeros(X.shape[1], np.float32), np.ones(X.shape[1], np.float32)

mae = lambda p, t: float(np.mean(np.abs(p - t)))
mean_mae = mae(np.full_like(yv, y.mean()), yv)
print(f"baseline (predict the mean): {mean_mae:.2f} days")

X_bias = np.column_stack((X, np.ones(len(X))))
Xv_bias = np.column_stack((Xv, np.ones(len(Xv))))

weights, *_ = np.linalg.lstsq(X_bias, y, rcond=None)

linear_pred = Xv_bias @ weights

linear_mae = mae(linear_pred, yv)
print(f"linear regression: {linear_mae:.2f} days")

torch.manual_seed(0)
net = BatteryNet(hidden)
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
Xvt = torch.tensor((Xv - mu) / sd)

history = []
best_val_mae = float("inf")
best_epoch = None
best_state = None

for ep in range(a.epochs):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 128):
        b = perm[i:i + 128]
        loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        train_pred=net(Xt).numpy()
        val_pred = net(Xvt).numpy()
        train_epoch_mae = mae(train_pred, y)
        val_epoch_mae = mae(val_pred, yv)

        #recording every epoch - internal documentation
    history.append({"epoch": ep+1, "train_mae":train_epoch_mae, "val_mae": val_epoch_mae})
    #print(f"  epoch {ep:4d}  val MAE {mae(net(Xvt).numpy(), yv):.2f} days")
    if val_epoch_mae < best_val_mae:
        best_val_mae = val_epoch_mae
        best_epoch = ep + 1
        best_state = copy.deepcopy(net.state_dict())
    if ep % 300 == 0 or ep == a.epochs - 1:
        print(f"epoch {ep:4d} | train MAE {train_epoch_mae:.2f} | val MAE {val_epoch_mae:.2f}")

# Restore best checkpoint
net.load_state_dict(best_state)

val_mae = best_val_mae

print(f"Best epoch: {best_epoch}")
print(f"Best validation MAE: {val_mae:.2f} days")

# Prepare experiment report
report_data = {
    "dataset": {
        "team": a.team,
        "rows_before_cleaning": rows_before,
        "glitch_rows_removed": glitch_rows,
        "rows_after_cleaning": rows_after,
        "glitch_rules": {
            "battery_v": "== 0",
            "temp_c": "== -127",
            "wifi_rssi_dbm": "== 0"
        }
    },

    "split": {
        "train_rows": len(X),
        "validation_rows": len(Xv),
        "random_seed": 0
    },

    "features": FEATURES,

    "normalisation": {
        "method": "z-score",
        "mu": mu.tolist(),
        "sd": sd.tolist()
    },

    "baselines": {
        "mean_mae": mean_mae,
        "linear_regression_mae": linear_mae
    },

    "training": {
        "epochs_requested": a.epochs,
        "learning_rate": 3e-3,
        "batch_size": 128,
        "optimizer": "Adam",
        "loss": "MSE",
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "final_train_mae": history[-1]["train_mae"],
        "final_val_mae": history[-1]["val_mae"],
        "history": history
    }
}

# Save experiment report
save_training_report(hidden, report_data)

os.makedirs("model", exist_ok=True)
torch.save(net.state_dict(), "model/model.pt")
json.dump(dict(team=a.team, hidden=list(hidden), mu=mu.tolist(), sd=sd.tolist(),
               features=FEATURES, val_mae=round(val_mae, 2)), open("model/config.json", "w"), indent=2)
print("saved model/model.pt and model/config.json")
