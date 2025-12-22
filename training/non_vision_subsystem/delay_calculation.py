# non_vision_subsystem/delay_calculation.py

import pandas as pd
import numpy as np
import json
import os
from .model_compatibility import ModelCompatibilityManager

def run(config):
    print(f"\n[DELAY] ⏰ Calculating delays for use case: {config.name}")

    # LOAD DATA
    df = pd.read_csv(config.sensor_data_path)
    df.columns = df.columns.str.strip()
    df['Timestamp'] = pd.to_datetime(df[config.date_col] + ' ' + df[config.time_col], dayfirst=True)
    df = df.dropna(subset=config.sensor_cols + [config.status_col]).reset_index(drop=True)
    df[config.sensor_cols] = df[config.sensor_cols].interpolate().bfill()

    # LOAD SAVED MODELS + THRESHOLDS USING COMPATIBILITY MANAGER
    model_manager = ModelCompatibilityManager(config)
    
    try:
        model, scaler, thresholds, feature_importance = model_manager.load_model_components()
        print(f"✅ Loaded {model_manager.model_format} model format")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # PREPARE DATA
    X_scaled = scaler.transform(df[config.sensor_cols].values)
    reconstructed = model.autoencoder.predict(X_scaled)
    reconstruction_errors = np.power(X_scaled - reconstructed, 2)

    # DETECT ANOMALIES
    df_anomalies = pd.DataFrame({'Timestamp': df['Timestamp']})
    for i, s in enumerate(config.sensor_cols):
        sensor_threshold = model_manager.get_threshold_for_sensor(thresholds, s)
        df_anomalies[f'Anomaly_{s}'] = (reconstruction_errors[:, i] > sensor_threshold).astype(int)

    # CALCULATE DELAYS
    gt_event_times = df[df[config.status_col] == config.anomaly_value]['Timestamp'].reset_index(drop=True)

    def calc_delay(detections, events):
        delays = []
        for e in events:
            future = detections[detections > e]
            if not future.empty:
                delay = (future.iloc[0] - e).total_seconds()
                if delay <= config.max_detection_window:
                    delays.append(delay)
        return np.mean(delays) if delays else np.nan

    delays = {}
    print("\n⏰ Average Delay per Sensor:")
    for s in config.sensor_cols:
        if config.sensor_metadata[s] == 'IRS':
            avg_delay = 0
        else:
            detections = df['Timestamp'][df_anomalies[f'Anomaly_{s}'] == 1].reset_index(drop=True)
            avg_delay = calc_delay(detections, gt_event_times)
        delays[s] = avg_delay
        print(f"  {s}: {avg_delay:.2f} sec")

    # Save delays to models directory
    model_dir = f"models/{config.name}"
    delays_path = f"{model_dir}/{config.name}_delays.json"
    with open(delays_path, 'w') as f:
        json.dump(delays, f)
    print(f"✅ Saved delays to {delays_path}")