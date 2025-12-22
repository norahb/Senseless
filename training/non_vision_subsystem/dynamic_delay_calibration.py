# non_vision_subsystem/dynamic_delay_calibration.py

import os
import json
import pandas as pd
from .sensor_delay_detector import SensorDelayDetector
from .adaptive_autoencoder import AdaptiveUnsupervisedAutoencoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_bland_altman(event_times, detection_times, sensor, output_dir, use_case):
    # Ensure pandas Series
    event_times = pd.Series(event_times).reset_index(drop=True)
    detection_times = pd.Series(detection_times).reset_index(drop=True)

    # Differences in seconds
    diffs = (detection_times - event_times).apply(lambda x: x.total_seconds())
    means = event_times + (detection_times - event_times) / 2

    mean_diff = np.mean(diffs)
    sd = np.std(diffs)

    plt.figure(figsize=(8,6))
    plt.scatter(means, diffs, alpha=0.5)
    plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Bias')
    plt.axhline(mean_diff + 1.96*sd, color='gray', linestyle=':')
    plt.axhline(mean_diff - 1.96*sd, color='gray', linestyle=':')
    plt.title(f"Bland–Altman Plot ({use_case} - {sensor})")
    plt.xlabel("Mean Event Time")
    plt.ylabel("Difference (s)")
    plt.legend()
    path = f"{output_dir}/{use_case}_{sensor}_bland_altman.png"
    plt.savefig(path); plt.close()
    return path

def plot_cumulative_error(errors, sensor, output_dir, use_case):
    sorted_err = np.sort(errors)
    cum = np.arange(1, len(sorted_err)+1)/len(sorted_err)
    plt.figure(figsize=(8,6))
    plt.step(sorted_err, cum)
    plt.xlabel("Detection Error (s)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"Cumulative Error Distribution ({use_case} - {sensor})")
    plt.grid(True)
    path = f"{output_dir}/{use_case}_{sensor}_cumulative_error.png"
    plt.savefig(path); plt.close()
    return path

def plot_error_histogram(errors, sensor, output_dir, use_case):
    plt.figure(figsize=(8,6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel("Error (s)")
    plt.ylabel("Count")
    plt.title(f"Error Histogram ({use_case} - {sensor})")
    path = f"{output_dir}/{use_case}_{sensor}_error_hist.png"
    plt.savefig(path); plt.close()
    return path

def compute_delay_metrics(event_times, detection_times, max_window=120):
    """Compute delay/error metrics given event and detection times."""
    delays = []
    for event_time in event_times:
        future = detection_times[detection_times > event_time]
        future = future[future <= event_time + pd.Timedelta(seconds=max_window)]
        if not future.empty:
            d = (future.iloc[0] - event_time).total_seconds()
            delays.append(d)

    if delays:
        delays = np.array(delays)
        return {
            "mean_delay": float(np.mean(delays)),
            "median_delay": float(np.median(delays)),
            "std_delay": float(np.std(delays)),
            "rmse": float(np.sqrt(mean_squared_error(np.zeros(len(delays)), delays))),
            "mae": float(mean_absolute_error(np.zeros(len(delays)), delays)),
            "r2": float(r2_score(np.zeros(len(delays)), delays)),
            "events_detected": len(delays),
            "total_events": len(event_times)
        }
    else:
        return {
            "mean_delay": 0.0, "median_delay": 0.0, "std_delay": 0.0,
            "rmse": 0.0, "mae": 0.0, "r2": 0.0,
            "events_detected": 0, "total_events": len(event_times)
        }

def run(config):
    print(f"\n[DYNAMIC DELAY] 🧠 Running onsite delay calibration for: {config.name}")

    model_dir = f"models/{config.name}"
    os.makedirs(model_dir, exist_ok=True)
    output_path = os.path.join(model_dir, f"{config.name}_calibrated_delays.json")

    # 🛑 Skip calibration if all sensors are IRS
    if all(sensor_type.upper() == "IRS" for sensor_type in config.sensor_metadata.values()):
        print("🛑 Skipping dynamic delay calibration: all sensors are IRS.")

        calibrated_delays = {
            "lab_baseline_delays": {s: 0.0 for s in config.sensor_cols},
            "measured_onsite_delays": {s: 0.0 for s in config.sensor_cols},
            "lab+onsite_delays": {s: 0.0 for s in config.sensor_cols},
            "final_calibrated_delays_with_reference_sensor": {s: 0.0 for s in config.sensor_cols},
            "human_adjusted_delays": {}
        }

        with open(output_path, 'w') as f:
            json.dump(calibrated_delays, f, indent=2)
        print(f"✅ Saved default 0.0 delay calibration file to: {output_path}")
        return

    # === LOAD SENSOR DATA ===
    df = pd.read_csv(config.sensor_data_path)
    df.columns = df.columns.str.strip()
    df["Timestamp"] = pd.to_datetime(df[config.date_col] + " " + df[config.time_col], dayfirst=True).dt.tz_localize(None)
    df = df.dropna(subset=config.sensor_cols + [config.status_col]).reset_index(drop=True)
    df[config.sensor_cols] = df[config.sensor_cols].interpolate().bfill()

    sensor_data = {}
    for col in config.sensor_cols:
        sensor_data[col] = pd.DataFrame({
            "timestamp": df["Timestamp"],
            "value": pd.to_numeric(df[col], errors="coerce")
        }).dropna()

    # === LOAD MODELS AND THRESHOLDS ===
    model, scaler, thresholds = None, None, {}

    try:
        adaptive_model_path = os.path.join(model_dir, f"adaptive_{config.name}")
        # if os.path.exists(adaptive_model_path):
        print("🔄 Loading full adaptive autoencoder model...")

        model = AdaptiveUnsupervisedAutoencoder(
                sensor_names=config.sensor_cols,
                case_name=config.name
            )
        model = model.load_adaptive_model(adaptive_model_path)

        scaler = model.scaler
        thresholds = getattr(model, "sensor_thresholds", {})
        print(f"Loaded thresholds: {thresholds}")
        print("✅ Successfully loaded full adaptive model with home-optimized detection")

    except Exception as e:
        print(f"❌ Model load failed for {config.name}: {e}")
        raise

    # === LOAD SENSITIVITY FROM production_config.json ===
    prod_config_path = os.path.join(model_dir, "production_config.json")
    if not os.path.exists(prod_config_path):
        raise FileNotFoundError(f"production_config.json missing for {config.name}")

    with open(prod_config_path, "r") as f:
        prod_config = json.load(f)

    sensitivity = prod_config.get("optimal_sensitivity")
    if not sensitivity:
        raise ValueError(f"No optimal_sensitivity in {prod_config_path}")

    print(f"✅ Loaded sensitivity for {config.name}: {sensitivity}")

    # Check for delay file (lab baseline delays)
    delay_file_path = os.path.join(model_dir, f"{config.name}_delays.json")
    if not os.path.exists(delay_file_path):
        print(f"⚠️ Lab delay file not found: {delay_file_path}")
        print("🔄 Creating default lab delays...")
        
        # Create default lab delays
        lab_delays = {sensor: 0.0 for sensor in config.sensor_cols}
        with open(delay_file_path, 'w') as f:
            json.dump(lab_delays, f, indent=2)
        print(f"✅ Created default lab delays at: {delay_file_path}")

    # === INITIALIZE DELAY DETECTOR ===
    delay_detector = SensorDelayDetector(
        delay_file_path=delay_file_path, 
        sensor_metadata=config.sensor_metadata
    )

    # === RUN DETECTION + DELAY CALCULATION ===
    try:
        delay_detector.detect_anomalies(
            sensor_data, 
            threshold_dict=thresholds, 
            model=model, 
            scaler=scaler,
            sensitivity=sensitivity
        )
        
        calibrated_delays = delay_detector.calculate_delays(
            sensor_data, 
            use_case=config.name
        )

        # === DELAY METRICS EVALUATION ===
        # Ground-truth anomaly events
        gt_event_times = df[df[config.status_col] == config.anomaly_value]["Timestamp"].reset_index(drop=True)

        # Detection times from delay detector (use threshold crossing)
        det_times = []
        for events in delay_detector.detected_anomalies.items():
            for e in events:
                if "threshold_crossing_time" in e:
                    det_times.append(e["threshold_crossing_time"])
        det_times = pd.Series(det_times).sort_values().reset_index(drop=True)

        # Compute metrics
        delay_metrics = compute_delay_metrics(gt_event_times, det_times, max_window=config.max_detection_window)

        # Save to JSON
        metrics_path = os.path.join(model_dir, f"{config.name}_delay_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(delay_metrics, f, indent=2)

        # Print summary
        print("\n📊 Delay Metrics:")
        for k, v in delay_metrics.items():
            print(f"   {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}")
        print(f"✅ Saved delay metrics to {metrics_path}")

        # === VISUAL EVALUATION PLOTS ===
        if not det_times.empty and not gt_event_times.empty:
            delays = []
            matched_event_times = []
            matched_detection_times = []
            for event_time in gt_event_times:
                future = det_times[det_times > event_time]
                future = future[future <= event_time + pd.Timedelta(seconds=config.max_detection_window)]
                if not future.empty:
                    matched_event_times.append(event_time)
                    matched_detection_times.append(future.iloc[0])
                    delays.append((future.iloc[0] - event_time).total_seconds())

            if delays:
                delays = np.array(delays)
                ba_path = plot_bland_altman(matched_event_times, matched_detection_times,
                                            sensor="overall", output_dir=model_dir, use_case=config.name)
                cum_path = plot_cumulative_error(delays, sensor="overall",
                                                output_dir=model_dir, use_case=config.name)
                hist_path = plot_error_histogram(delays, sensor="overall",
                                                output_dir=model_dir, use_case=config.name)

                print(f"📈 Saved plots:\n   Bland–Altman → {ba_path}\n   Cumulative error → {cum_path}\n   Histogram → {hist_path}")


        # === SAVE FINAL CALIBRATED DELAYS ===
        delay_detector.save_delays(filepath=output_path)
        print("\n✅ Calibrated delays saved to:", output_path)
        
    except Exception as e:
        print(f"❌ Error during delay calibration: {e}")
        print("🔄 Creating fallback calibrated delays...")
        
        # Create fallback calibrated delays
        fallback_delays = {
            "lab_baseline_delays": {s: 0.0 for s in config.sensor_cols},
            "measured_onsite_delays": {s: 0.0 for s in config.sensor_cols},
            "lab+onsite_delays": {s: 0.0 for s in config.sensor_cols},
            "final_calibrated_delays_with_reference_sensor": {s: 0.0 for s in config.sensor_cols},
            "human_adjusted_delays": {}
        }
        
        with open(output_path, 'w') as f:
            json.dump(fallback_delays, f, indent=2)
        print(f"✅ Saved fallback calibrated delays to: {output_path}")