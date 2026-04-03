import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

FILE = "experiment dataset.xlsx"

X_COLS = ["C", "H", "O", "N", "H/C", "O/C"]
Y_COLS = ["PET", "PE&PP", "PVC", "PS", "PA", "PC"]

df = pd.read_excel(FILE)
df.columns = [str(c).strip() for c in df.columns]

df = df.dropna(subset=X_COLS + Y_COLS).reset_index(drop=True)

X = df[X_COLS].to_numpy(dtype=float)
y = df[Y_COLS].to_numpy(dtype=float)

n = len(df)
print(f"Total experimental samples: {n}")

loo = LeaveOneOut()

y_true_all = []
y_pred_all = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ✅ 完全使用论文参数
    model = RandomForestRegressor(
        n_estimators=67,
        max_depth=27,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_true_all.append(y_test.reshape(-1))
    y_pred_all.append(y_pred.reshape(-1))

y_true_all = np.vstack(y_true_all)
y_pred_all = np.vstack(y_pred_all)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


r2_overall = r2_score(y_true_all, y_pred_all, multioutput="variance_weighted")
rmse_overall = rmse(y_true_all, y_pred_all)
mae_overall = mean_absolute_error(y_true_all, y_pred_all)

print("\n===== Experimental-only LOOCV Results =====")
print(f"Overall R2 (variance_weighted): {r2_overall:.4f}")
print(f"Overall RMSE: {rmse_overall:.4f}")
print(f"Overall MAE: {mae_overall:.4f}")

print("\n===== Per-Polymer Metrics =====")
for j, name in enumerate(Y_COLS):
    r2_j = r2_score(y_true_all[:, j], y_pred_all[:, j])
    rmse_j = rmse(y_true_all[:, j], y_pred_all[:, j])
    mae_j = mean_absolute_error(y_true_all[:, j], y_pred_all[:, j])
    print(f"{name}:  R2={r2_j:.4f}  RMSE={rmse_j:.4f}  MAE={mae_j:.4f}")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_metrics_dict(y_true, y_pred, y_cols):
    results = {}

    for j, name in enumerate(y_cols):
        results[name] = {
            "R2": r2_score(y_true[:, j], y_pred[:, j]),
            "RMSE": rmse(y_true[:, j], y_pred[:, j]),
            "MAE": mean_absolute_error(y_true[:, j], y_pred[:, j]),
        }

    results["Overall"] = {
        "R2": r2_score(y_true, y_pred, multioutput="variance_weighted"),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
    }

    return results

loo = LeaveOneOut()

y_true_test_all = []
y_pred_test_all = []

y_true_train_all = []
y_pred_train_all = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model_fold = clone(model)
    model_fold.fit(X_train, y_train)

    y_test_pred = model_fold.predict(X_test)
    y_train_pred = model_fold.predict(X_train)

    y_true_test_all.append(y_test)
    y_pred_test_all.append(y_test_pred)

    y_true_train_all.append(y_train)
    y_pred_train_all.append(y_train_pred)

y_true_test_all = np.vstack(y_true_test_all)
y_pred_test_all = np.vstack(y_pred_test_all)

y_true_train_all = np.vstack(y_true_train_all)
y_pred_train_all = np.vstack(y_pred_train_all)

Y_COLS = ['PET', 'PE&PP', 'PVC', 'PS', 'PA', 'PC']

train_metrics = get_metrics_dict(y_true_train_all, y_pred_train_all, Y_COLS)
test_metrics = get_metrics_dict(y_true_test_all, y_pred_test_all, Y_COLS)

rows = []
for name in Y_COLS + ["Overall"]:
    rows.append([
        name,
        train_metrics[name]["R2"],
        train_metrics[name]["RMSE"],
        train_metrics[name]["MAE"],
        test_metrics[name]["R2"],
        test_metrics[name]["RMSE"],
        test_metrics[name]["MAE"],
    ])

loocv_metrics_df = pd.DataFrame(
    rows,
    columns=[
        "Category",
        "Train_R2", "Train_RMSE", "Train_MAE",
        "Test_R2", "Test_RMSE", "Test_MAE"
    ]
)

print("\n===== LOOCV Metrics Table =====")
print(loocv_metrics_df.round(4))

loocv_metrics_df.round(4).to_excel("LOOCV_metrics_side_by_side.xlsx", index=False)