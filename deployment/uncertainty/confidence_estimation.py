import os
import json
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys
from scipy.stats import norm
from .alert_manager import generate_alert_for_segments
from sklearn.isotonic import IsotonicRegression

import joblib

# Add training path for adaptive model imports
training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'training'))
if training_path not in sys.path:
    sys.path.insert(0, training_path)

try:
    from non_vision_subsystem.adaptive_autoencoder import AdaptiveUnsupervisedAutoencoder
except ImportError:
    print("⚠️ Warning: Could not import AdaptiveUnsupervisedAutoencoder")

def load_adaptive_model_for_inference(config):
    model_dir = Path(config.sensor_model_path)
    config_path = model_dir / 'production_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Production config not found: {config_path}")
    with open(config_path, 'r') as f:
        production_config = json.load(f)
    if production_config.get('model_type') != 'adaptive_autoencoder':
        raise ValueError(f"Expected adaptive_autoencoder, got {production_config.get('model_type')}")
    model_path = model_dir / f'adaptive_{config.use_case}'
    if not Path(f"{model_path}.pkl").exists():
        raise FileNotFoundError(f"Adaptive model not found: {model_path}.pkl")
    model = AdaptiveUnsupervisedAutoencoder(
        sensor_names=config.use_case_config.sensor_cols,
        case_name=config.use_case
    )
    model.load_adaptive_model(str(model_path))
    optimal_sensitivity = production_config.get('optimal_sensitivity', 'medium')
    return model, production_config, optimal_sensitivity

def load_error_distributions(config, sensor_data=None):
    """
    Load error distribution parameters (Gaussian or KDE) from training.
    If the file is missing, calculate them dynamically if sensor_data is provided.
    """
    error_dist_path = os.path.join("models", config.use_case, f"{config.use_case}_error_distributions.json")
    if not os.path.exists(error_dist_path):
        print(f"⚠️ Error distributions not found: {error_dist_path}")
        if sensor_data is None:
            print("❌ No sensor data provided for dynamic error distribution calculation.")
            return {}

        print("🔄 Calculating error distributions dynamically...")
        error_dists = {}
        for sensor in config.use_case_config.sensor_cols:
            if sensor in sensor_data.columns:
                sensor_values = sensor_data[sensor].dropna()  # Remove NaN values
                mean = sensor_values.mean()
                std = sensor_values.std()
                error_dists[sensor] = {"type": "gaussian", "mean": mean, "std": std}
                print(f"   Sensor: {sensor}, Mean: {mean:.3f}, Std: {std:.3f}")
            else:
                print(f"⚠️ Sensor column '{sensor}' not found in the provided data.")

        # Save the calculated error distributions to a JSON file for future use
        os.makedirs(os.path.dirname(error_dist_path), exist_ok=True)
        with open(error_dist_path, "w") as f:
            json.dump(error_dists, f, indent=4)
        print(f"✅ Saved calculated error distributions to {error_dist_path}")

        return error_dists

    # Load error distributions from the JSON file
    with open(error_dist_path, "r") as f:
        error_dists = json.load(f)

    print(f"✅ Loaded error distributions from {error_dist_path}")
    print(f"   Sensors in error distributions: {list(error_dists.keys())}")
    for sensor, dist in error_dists.items():
        dist_type = dist.get("type", "unknown")
        if dist_type == "gaussian":
            print(f"   Sensor: {sensor}, Type: Gaussian, Mean: {dist.get('mean', 'N/A')}, Std: {dist.get('std', 'N/A')}")
        elif dist_type == "kde":
            print(f"   Sensor: {sensor}, Type: KDE, Points: {len(dist.get('xs', []))}")
        else:
            print(f"   Sensor: {sensor}, Type: {dist_type} (unsupported)")

    return error_dists

def calculate_confidence_with_calibration(reconstruction_errors, error_dists, sensor_cols, sensor_iso=None):
    """
    Calculate confidence using error distribution CDF, then calibrate with isotonic regression if available.
    """
    confidences = []
    for i in range(reconstruction_errors.shape[0]):
        sensor_conf = []
        for j, sensor in enumerate(sensor_cols):
            err = reconstruction_errors[i, j]
            dist = error_dists.get(sensor, None)
            if dist:
                mu = dist.get("mean", 0)
                sigma = dist.get("std", 1e-6)
                conf = norm.cdf(err, loc=mu, scale=sigma)
                sensor_conf.append(conf)
            else:
                sensor_conf.append(0.5)
        # Aggregate sensor confidences (mean)
        mean_conf = np.mean(sensor_conf)
        confidences.append(mean_conf)
    confidences = np.array(confidences)
    # Print confidence stats before calibration
    print(f"   Raw confidence stats: min={confidences.min():.3f}, max={confidences.max():.3f}, mean={confidences.mean():.3f}")
    # Apply isotonic calibration if available
    # if sensor_iso is not None:
    #     confidences = sensor_iso.transform(confidences)
    #     print(f"   Isotonic calibration applied. Calibrated stats: min={confidences.min():.3f}, max={confidences.max():.3f}, mean={confidences.mean():.3f}")
    # else:
    #     print("   No isotonic calibration applied.")
    return confidences

def evaluate_predictions(config, sensor_df):
    try:
        ground_truth_df = pd.read_csv(config.incoming_sensor_data_path)
        if 'Status' not in ground_truth_df.columns:
            print("⚠️ No 'Status' column found in ground truth data - skipping evaluation")
            return
        if len(sensor_df) != len(ground_truth_df):
            print(f"⚠️ Different data lengths: {len(sensor_df)} vs {len(ground_truth_df)} - skipping evaluation")
            return
        y_true = ground_truth_df['Status'].values
        y_pred = sensor_df['Sensor_Status'].values
        valid_mask = y_pred != 'Sensor_Error'
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        y_true_binary = (y_true_valid != 'Normal').astype(int)
        y_pred_binary = (y_pred_valid != 'Normal').astype(int)
        correct = (y_true_binary == y_pred_binary).sum()
        total = len(y_true_binary)
        accuracy = correct / total
        true_normal = (y_true_binary == 0).sum()
        true_anomaly = (y_true_binary == 1).sum()
        pred_normal = (y_pred_binary == 0).sum()
        pred_anomaly = (y_pred_binary == 1).sum()
        tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
        fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
        fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()
        tn = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
        print(f"\n📊 QUICK EVALUATION RESULTS:")
        print(f"   🎯 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   📈 Correct: {correct}/{total}")
        print(f"   📋 Ground Truth: {true_normal} Normal, {true_anomaly} Anomaly")
        print(f"   🔍 Predicted: {pred_normal} Normal, {pred_anomaly} Anomaly")
        print(f"   ✅ True Positives (caught anomalies): {tp}")
        print(f"   ❌ False Positives (false alarms): {fp}")
        print(f"   ⚠️  False Negatives (missed anomalies): {fn}")
        print(f"   ✓  True Negatives (correct normals): {tn}")
        if fn > 0:
            print(f"   🚨 WARNING: {fn} real anomalies were missed!")
    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")

def get_anomaly_message(use_case):
    messages = {
        "door": "Door left open",
        "co2": "Unauthorized person detected", 
        "appliance": "Unattended stove",
        "abnormal_object": "Object blocking hallway"
    }
    return messages.get(use_case, "anomaly detected")

def get_location(use_case):
    locations = {
        "door": "Entrance",
        "co2": "Room",
        "appliance": "Kitchen", 
        "abnormal_object": "Hallway"
    }
    return locations.get(use_case, "Room")

def run(config, logits_tensor=None, sensor_df=None, train_isotonic=False):
    print("\n🧠 Estimating confidence...")
    start_time = time.time()

    # Load sensor_df if not provided
    if sensor_df is None:
        if not os.path.exists(config.decision_log_path):
            print(f"❌ Decision log not found: {config.decision_log_path}")
            return pd.DataFrame()
        sensor_df = pd.read_csv(config.decision_log_path)

    evaluate_predictions(config, sensor_df)

    # Try to load isotonic calibration model and threshold
    sensor_iso = None
    calibration_path = os.path.join("models", config.use_case, "sensor_isotonic.pkl")
    if os.path.exists(calibration_path):
        try:
            sensor_iso = joblib.load(calibration_path)
        except Exception as e:
            print(f"⚠️ Could not load isotonic calibration: {e}")

    # Load adaptive model and error distributions
    try:
        model, production_config, optimal_sensitivity = load_adaptive_model_for_inference(config)
        error_dists = load_error_distributions(config, sensor_data=sensor_df)
        sensor_cols = config.use_case_config.sensor_cols

        valid_rows = sensor_df[sensor_df["Sensor_Status"] != "Sensor_Error"]
        valid_indices = valid_rows.index

        if len(valid_indices) == 0:
            print("⚠️ No valid rows for confidence estimation")
            return sensor_df

        X_valid = valid_rows[sensor_cols].values
        nan_mask = pd.isna(X_valid).any(axis=1)
        if nan_mask.any():
            print(f"⚠️ Removing {nan_mask.sum()} rows with NaN values")
            X_valid = X_valid[~nan_mask]
            valid_indices_clean = valid_indices[~nan_mask]
        else:
            valid_indices_clean = valid_indices

        if len(X_valid) == 0:
            print("⚠️ No valid data after cleaning")
            return sensor_df

        # Ensure 2D shape for single-feature use cases
        if X_valid.ndim == 1:
            X_valid = X_valid.reshape(-1, 1)

        X_scaled = model.scaler.transform(X_valid)
        reconstructed = model.autoencoder.predict(X_scaled)

        # Ensure 2D shape for reconstructed data
        if reconstructed.ndim == 1:
            reconstructed = reconstructed.reshape(-1, 1)

        reconstruction_errors = np.power(X_scaled - reconstructed, 2)
        print(f"Reconstruction errors: min={reconstruction_errors.min()}, max={reconstruction_errors.max()}, mean={reconstruction_errors.mean()}")
        # --- Combined confidence calculation ---
        confidence_scores = calculate_confidence_with_calibration(
            reconstruction_errors, error_dists, sensor_cols, sensor_iso=sensor_iso
        )

        # Retrain isotonic calibration if requested
        if train_isotonic:
            print("🔄 Retraining isotonic calibration model on deployment data...")
            sensor_iso = IsotonicRegression(out_of_bounds="clip")
            # Map 'Status' column to numeric labels
            status_mapping = {"Normal": 0, "Anomaly": 1}
            ground_truth_labels = valid_rows["Status"].map(status_mapping).values
            sensor_iso.fit(confidence_scores, ground_truth_labels)
            joblib.dump(sensor_iso, calibration_path)
            print(f"✅ Isotonic calibration model retrained and saved to {calibration_path}")

        # Calculate dynamic threshold
        threshold = np.percentile(confidence_scores, 30)
        low_conf_mask = confidence_scores < threshold

        sensor_df["Confidence"] = np.nan
        sensor_df["Low_Confidence"] = False
        sensor_df.loc[valid_indices_clean, "Confidence"] = confidence_scores
        sensor_df.loc[valid_indices_clean, "Low_Confidence"] = low_conf_mask

        os.makedirs(os.path.dirname(config.decision_log_path), exist_ok=True)
        sensor_df.to_csv(config.decision_log_path, index=False)

        avg_confidence = np.mean(confidence_scores)
        low_conf_count = sum(low_conf_mask)
        low_conf_percentage = (low_conf_count / len(confidence_scores)) * 100

        generate_alert_for_segments(sensor_df, config.use_case, confidence_threshold=threshold)

        confidence_time = time.time() - start_time
        print(f"⏱️ Confidence estimation time: {confidence_time:.2f}s")
        print(f"✅ Confidence estimation completed:")
        print(f"   📊 Average confidence: {avg_confidence:.3f}")
        print(f"   🔍 Low confidence samples: {low_conf_count}/{len(confidence_scores)} ({low_conf_percentage:.1f}%)")
    
    except Exception as e:
        print(f"❌ Error during confidence estimation: {e}")
        sensor_df["Confidence"] = 0.5
        sensor_df["Low_Confidence"] = True
        os.makedirs(os.path.dirname(config.decision_log_path), exist_ok=True)
        sensor_df.to_csv(config.decision_log_path, index=False)
        print("⚠️ Added fallback confidence values to prevent pipeline crash")

    return sensor_df
