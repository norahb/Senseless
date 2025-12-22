import numpy as np
import pandas as pd

def run_rule_based_detector(config):
    """Rule-based anomaly detection for abnormal_object use case"""
    
    # Load sensor data
    df = pd.read_csv(config.incoming_sensor_data_path)
    df["Sensor_Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S")
    df.drop(columns=["Timestamp"], inplace=True, errors='ignore')
    
    # Extract sensor features
    sensor_cols = config.use_case_config.sensor_cols
    X = df[sensor_cols].values
    
    # Rule-based logic: Object present if distance thresholds NOT met
    # S1_distance normal range: 129-131, S2_distance normal range: 204-207
    anomaly_flags = []
    status_labels = []
    
    for row in X:
        s1_distance = row[0]  # Assuming first sensor is S1_distance
        s2_distance = row[1] if len(row) > 1 else s1_distance  # Handle duplicate sensor case
        
        # Apply rule: Normal if both sensors in range
        is_normal = (129 <= s1_distance <= 131) and (204 <= s2_distance <= 207)
        
        anomaly_flags.append(0 if is_normal else 1)
        status_labels.append("Normal" if is_normal else "Anomaly")
    
    # Update dataframe
    df["Anomaly_Prediction"] = anomaly_flags
    df["Sensor_Status"] = status_labels
    df["Corrupt_Sensors"] = "None"  # Rule-based doesn't detect sensor corruption
    
    # Save results
    import os
    os.makedirs(os.path.dirname(config.decision_log_path), exist_ok=True)
    df.to_csv(config.decision_log_path, index=False)
    
    anomaly_count = sum(anomaly_flags)
    print(f"Detected {anomaly_count} anomalies out of {len(df)} samples using rule-based detection")
    
    # Create simple logits tensor for compatibility
    import torch
    logits_tensor = torch.tensor([[0.5] for _ in range(len(df))]).float()
    
    return df, logits_tensor