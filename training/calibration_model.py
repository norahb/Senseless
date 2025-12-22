
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# === CONFIG ===
use_case = "co2"   # change per run if needed

# === Load CSVs ===
gt = pd.read_csv(f"./data/sensor_data/{use_case}/{use_case.capitalize()}_images_June2025.csv")
ssl = pd.read_csv(f"./output/{use_case}/{use_case}_ssl_images_labels.csv")
sensor = pd.read_csv(f"./output/{use_case}/{use_case}_sensors_images_labels.csv")

# Ensure consistent column names
gt.columns = gt.columns.str.strip()
ssl.columns = ssl.columns.str.strip()
sensor.columns = sensor.columns.str.strip()

# --- Rename columns ---
ssl.rename(columns={
    "Label_ssl": "SSL_Label",
    "Confidence_Score": "SSL_Conf"
}, inplace=True)

sensor.rename(columns={
    "Label": "Sensor_Label",
    "Confidence_Score_nonvision": "Sensor_Conf"
}, inplace=True)

# --- Merge files ---
df = gt.merge(ssl[["Image_Name", "SSL_Label", "SSL_Conf"]], on="Image_Name", how="inner")
df = df.merge(sensor[["Image_Name", "Sensor_Label", "Sensor_Conf", "True_Label"]], on="Image_Name", how="inner")

# --- Correctness flags ---
df["SSL_Correct"] = (df["SSL_Label"].str.lower() == df["True_Label"].str.lower()).astype(int)
df["Sensor_Correct"] = (df["Sensor_Label"].str.lower() == df["True_Label"].str.lower()).astype(int)

print(f"Dataset size: {len(df)} images")
print(df[["SSL_Conf", "SSL_Correct", "Sensor_Conf", "Sensor_Correct"]].head())

# === Fit Isotonic models ===
ssl_iso = IsotonicRegression(out_of_bounds="clip").fit(df["SSL_Conf"], df["SSL_Correct"])
sensor_iso = IsotonicRegression(out_of_bounds="clip").fit(df["Sensor_Conf"], df["Sensor_Correct"])

# === Evaluate calibration quality ===
for name, model, confs, correct in [
    ("SSL-Isotonic", ssl_iso, df["SSL_Conf"], df["SSL_Correct"]),
    ("Sensor-Isotonic", sensor_iso, df["Sensor_Conf"], df["Sensor_Correct"])
]:
    calibrated = model.transform(confs)
    brier = brier_score_loss(correct, calibrated)
    auc = roc_auc_score(correct, calibrated)
    print(f"{name} -> Brier={brier:.3f}, AUC={auc:.3f}")

# === Save models ===
model_dir = f"models/{use_case}"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(ssl_iso, f"{model_dir}/ssl_isotonic.pkl")
joblib.dump(sensor_iso, f"{model_dir}/sensor_isotonic.pkl")
print(f"✅ Isotonic models saved under {model_dir}")

# === Reliability + Coverage plotting ===
def plot_reliability_and_coverage(raw_conf, calibrated_conf, correct, title, target_acc=90):
    plt.figure(figsize=(12,5))

    # Reliability curve
    plt.subplot(1,2,1)
    prob_true_raw, prob_pred_raw = calibration_curve(correct, raw_conf, n_bins=10)
    plt.plot(prob_pred_raw, prob_true_raw, marker='o', label="Raw")

    prob_true_cal, prob_pred_cal = calibration_curve(correct, calibrated_conf, n_bins=10)
    plt.plot(prob_pred_cal, prob_true_cal, marker='o', label="Calibrated")

    plt.plot([0,1],[0,1],'k--', label="Perfect")
    plt.title(f"{title} Reliability")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.legend()

    # Coverage–Accuracy tradeoff
    plt.subplot(1,2,2)

    thresholds = np.linspace(0,1,50)
    # coverages, accuracies = [], []
    # optimal_thresh = None

    # for t in thresholds:
    #     mask = calibrated_conf >= t
    #     if mask.sum() > 0:
    #         cov = mask.mean()*100
    #         acc = correct[mask].mean()*100
            
    #         coverages.append(cov)
    #         accuracies.append(acc)
    #         if optimal_thresh is None and acc >= target_acc:
    #             optimal_thresh = t
    #     else:
    #         coverages.append(0)
    #         accuracies.append(np.nan)

    optimal_thresh = None
    best_score = -1
    coverages, accuracies = [], []

    for t in thresholds:
        mask = calibrated_conf >= t
        if mask.sum() > 0:
            cov = mask.mean() * 100
            acc = correct[mask].mean() * 100
            coverages.append(cov)
            accuracies.append(acc)

            # Balance: maximize accuracy × coverage
            score = (acc / 100.0) * (cov / 100.0)  # normalize to 0-1
            if acc >= target_acc and score > best_score:
                best_score = score
                optimal_thresh = t
        else:
            coverages.append(0)
            accuracies.append(np.nan)

    plt.plot(coverages, accuracies, marker='o')
    plt.axhline(target_acc, color='red', linestyle='--', label=f"{target_acc}% target")
    plt.title(f"{title} Coverage–Accuracy")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"output/{use_case}/{title.lower()}_reliability_coverage.png")
    plt.close()

    if optimal_thresh is not None:
        print(f"✅ {title} optimal cutoff = {optimal_thresh:.2f} (≥{target_acc}% acc)")
    else:
        print(f"⚠️ {title} could not reach {target_acc}% accuracy")
    return optimal_thresh

# Apply to SSL
calibrated_ssl = ssl_iso.transform(df["SSL_Conf"])
ssl_cutoff = plot_reliability_and_coverage(df["SSL_Conf"], calibrated_ssl, df["SSL_Correct"], "SSL")

# Apply to Sensor
calibrated_sensor = sensor_iso.transform(df["Sensor_Conf"])
sensor_cutoff = plot_reliability_and_coverage(df["Sensor_Conf"], calibrated_sensor, df["Sensor_Correct"], "Sensor")

# === Save chosen thresholds ===
thresholds = {"ssl_cutoff": float(ssl_cutoff) if ssl_cutoff else None,
              "sensor_cutoff": float(sensor_cutoff) if sensor_cutoff else None}
import json
with open(f"{model_dir}/calibration_thresholds.json","w") as f:
    json.dump(thresholds, f, indent=2)

print("✅ Thresholds saved:", thresholds)
