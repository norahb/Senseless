
import os
import sys
import torch
import numpy as np
import pandas as pd
import joblib
# from non_vision_subsystem.enhanced_autoencoder import EnhancedAutoencoder
from pathlib import Path
import json
from .rule_based_detector import run_rule_based_detector

# Add the training root directory so that 'non_vision_subsystem' is importable
training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training'))
if training_path not in sys.path:
    sys.path.insert(0, training_path)
from non_vision_subsystem.adaptive_autoencoder import AdaptiveUnsupervisedAutoencoder


def detect_model_format(config):
    """Detect if using adaptive autoencoder or legacy model"""
    model_dir = Path(config.sensor_model_path).parent
    production_config_path = model_dir / 'production_config.json'
    adaptive_model_path = model_dir / f'adaptive_{config.use_case}.pkl'
    
    if production_config_path.exists() and adaptive_model_path.exists():
        return 'adaptive'
    return 'legacy'


def detect_outliers_zscore(data, threshold=3.5):
    """
    Detects outliers in a 2D NumPy array using Z-score.
    Returns a boolean mask: True for outliers, False otherwise.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_scores = (data - mean) / (std + 1e-8)  # avoid divide-by-zero
    return np.abs(z_scores) > threshold


def is_physically_impossible(sensor_data, sensor_cols, thresholds):
    """Check if sensor readings are physically impossible based on configured thresholds"""
    impossible_mask = np.zeros(sensor_data.shape[0], dtype=bool)
    
    for i, sensor_name in enumerate(sensor_cols):
        if sensor_name in thresholds:
            sensor_values = sensor_data[:, i]
            min_val = thresholds[sensor_name]["min"]
            max_val = thresholds[sensor_name]["max"]
            
            # Mark as impossible if outside physical range
            sensor_impossible = (sensor_values < min_val) | (sensor_values > max_val)
            impossible_mask |= sensor_impossible
            
            # Debug info for significant violations
            if sensor_impossible.any():
                extreme_values = sensor_values[sensor_impossible]
                print(f"🌡️ {sensor_name} outside range [{min_val}, {max_val}]: {len(extreme_values)} values, "
                      f"range: [{extreme_values.min():.1f}, {extreme_values.max():.1f}]")
    
    return impossible_mask


def enhanced_outlier_detection(sensor_data, sensor_cols, config):
    """Enhanced outlier detection using both Z-score and physical impossibility"""
    
    # Get thresholds from config
    thresholds = config.use_case_config.sensor_error_thresholds
    mild_zscore = config.use_case_config.mild_outlier_zscore
    extreme_zscore = config.use_case_config.extreme_corruption_zscore
    
    print(f"🔧 Using {config.use_case} thresholds: mild_z={mild_zscore}, extreme_z={extreme_zscore}")
    
    # Step 1: Check physical impossibility
    physical_impossible_mask = is_physically_impossible(sensor_data, sensor_cols, thresholds)
    
    # Step 2: Check statistical outliers
    mild_outlier_mask = detect_outliers_zscore(sensor_data, threshold=mild_zscore)
    extreme_outlier_mask = detect_outliers_zscore(sensor_data, threshold=extreme_zscore)
    extreme_statistical_mask = extreme_outlier_mask.any(axis=1)
    
    # Step 3: Combine conditions for corruption detection
    # For extreme corruption: must be BOTH physically impossible AND extreme statistical outlier
    extreme_corruption_mask = physical_impossible_mask & extreme_statistical_mask
    
    print(f"🔍 Physical impossible: {physical_impossible_mask.sum()}")
    print(f"🔍 Extreme statistical outliers: {extreme_statistical_mask.sum()}")
    print(f"🔍 Final corruption (both conditions): {extreme_corruption_mask.sum()}")
    
    # Special case: if no extreme corruption but we have physical impossibilities,
    # these might be legitimate extreme anomalies (like your appliance case)
    anomaly_candidates = physical_impossible_mask & ~extreme_statistical_mask
    if anomaly_candidates.any():
        print(f"🎯 Potential extreme anomalies (physical but not statistical): {anomaly_candidates.sum()}")
    
    return mild_outlier_mask, extreme_corruption_mask


def run_sensor_model(config):

    # Check if this is abnormal_object use case - use rule-based detection
    if config.use_case.lower() == "abnormal_object":
        print("🤖 Using rule-based detection for abnormal_object use case")
        return run_rule_based_detector(config)
    
    # === Load trained sensor model ===
    model_path = config.sensor_model_path
    print(f"🔍 Loading sensor model from: {model_path}")

    # Check if it's a directory (new format) or file (legacy format)
    if os.path.isdir(model_path):
        # Try different possible locations for production_config.json
        possible_config_paths = [
            os.path.join(model_path, 'production_config.json'),  
            os.path.join(model_path, config.use_case, 'production_config.json'),  
        ]
        
        production_config_path = None
        for path in possible_config_paths:
            if os.path.exists(path):
                production_config_path = path
                break
        
        if production_config_path:
            print(f"🔍 Using production config path: {production_config_path}")

            # Load adaptive model
            print("🔄 Loading adaptive autoencoder model...")
            
            with open(production_config_path, 'r') as f:
                production_config = json.load(f)
            
            model = AdaptiveUnsupervisedAutoencoder(
                sensor_names=config.use_case_config.sensor_cols,
                case_name=config.use_case
            )
            
            # Try to find the adaptive model file in the same directory as config
            config_dir = os.path.dirname(production_config_path)
            adaptive_model_path = os.path.join(config_dir, f'adaptive_{config.use_case}')
            
            model.load_adaptive_model(adaptive_model_path)
            print(f"🔍 Model loaded from: {adaptive_model_path}.pkl")
            print(f"🔍 Model file modified: {os.path.getmtime(f'{adaptive_model_path}.pkl')}")
            
            optimal_sensitivity = production_config.get('optimal_sensitivity', 'medium')
            print(f"✅ Using original trained sensitivity: {optimal_sensitivity}")

            model_type = 'adaptive'
            
        else:
            raise FileNotFoundError(f"Production config not found. Checked: {possible_config_paths}")
            
    else:
        # Legacy model format
        print("🔄 Loading legacy model...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Sensor model not found at: {model_path}")
        
        model = joblib.load(model_path)
        model_type = 'legacy'
        optimal_sensitivity = None

    # Add this debug to sensor_inference.py after model loading:
    print(f"🔍 Available methods: {model.get_available_methods() if hasattr(model, 'get_available_methods') else 'Unknown'}")

    # === Load incoming sensor data ===
    data_path = config.incoming_sensor_data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Sensor input CSV not found: {data_path}")
    df = pd.read_csv(data_path)

    # === Convert and retain timestamp for fallback use ===
    df["Sensor_Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S")
    df.drop(columns=["Timestamp"], inplace=True, errors='ignore')

    # === Extract sensor features in training order ===
    sensor_cols = config.use_case_config.sensor_cols
    X = df[sensor_cols].values

    # === Enhanced outlier detection using config thresholds ===
    mild_outlier_mask, corruption_mask = enhanced_outlier_detection(X, sensor_cols, config)
    
    # === Step 1: Handle mild outliers with interpolation ===
    X_cleaned = X.astype(float)
    X_cleaned[mild_outlier_mask] = np.nan
    X_cleaned_df = pd.DataFrame(X_cleaned, columns=sensor_cols)
    X_filled = X_cleaned_df.interpolate(limit_direction='both', axis=0).bfill().ffill()
    X_final = X_filled.values

    # === Step 2: Exclude corrupted samples from anomaly detection ===
    X_valid = X_final[~corruption_mask]
    valid_indices = np.where(~corruption_mask)[0]

    if len(X_valid) > 0:
        if model_type == 'adaptive':
            # Use adaptive model prediction
            anomaly_flags, anomaly_scores, _ = model.predict_anomalies_home_optimized(
                X_valid, adapt=False, sensitivity=optimal_sensitivity, verbose=False
            )
            # For compatibility with existing code
            anomaly_scores = {'overall': anomaly_scores}
        else:
            # Legacy model prediction
            anomaly_scores, anomaly_flags, _ = model.predict(X_valid)
    else:
        anomaly_scores = {}
        anomaly_flags = np.array([])

    # === Step 3: Build full anomaly flag and label list ===
    full_flags = np.zeros(len(X_final), dtype=int)
    status_labels = []

    for idx in range(len(X_final)):
        if corruption_mask[idx]:
            status_labels.append("Sensor_Error")
        elif idx in valid_indices:
            model_idx = np.where(valid_indices == idx)[0][0]
            is_anomaly = anomaly_flags[model_idx]
            status_labels.append("Anomaly" if is_anomaly else "Normal")
            full_flags[idx] = is_anomaly
        else:
            status_labels.append("Unknown")

    # === Step 4: Estimate confidence (optional) using average score ===
    if model_type == 'adaptive':
        # For adaptive models, use a simple approach for compatibility
        if len(valid_indices) > 0:
            # Create simple confidence scores based on predictions
            avg_scores = anomaly_flags.astype(float)  # Convert predictions to float scores
        else:
            avg_scores = np.array([0.0])
    else:
        # Legacy model processing
        if isinstance(anomaly_scores, dict):
            avg_scores = np.mean(list(anomaly_scores.values()), axis=0) if anomaly_scores else [0.0] * len(sensor_cols)
        else:
            avg_scores = anomaly_scores  # fallback

    # Ensure avg_scores is always a 1D array
    if len(avg_scores) == 0:
        avg_scores = np.array([0.0])
    elif isinstance(avg_scores, (int, float)):
        avg_scores = np.array([avg_scores])

    logits_tensor = torch.tensor(avg_scores).unsqueeze(1).float()

    # === Step 5: Save output ===
    df["Anomaly_Prediction"] = full_flags
    df["Sensor_Status"] = status_labels

    if not os.path.exists(os.path.dirname(config.decision_log_path)):
        os.makedirs(os.path.dirname(config.decision_log_path))

    # === Step 6: Enhanced sensor error reporting ===
    # Instead of just reporting which sensors are corrupt per row,
    # also track which thresholds were violated
    sensor_error_info = []
    threshold_violations = []

    for row_idx in range(len(X)):
        if corruption_mask[row_idx]:
            # Find which sensors violated thresholds
            violated_sensors = []
            violation_details = []
            
            for col_idx, sensor_name in enumerate(sensor_cols):
                if sensor_name in config.use_case_config.sensor_error_thresholds:
                    value = X[row_idx, col_idx]
                    min_val = config.use_case_config.sensor_error_thresholds[sensor_name]["min"]
                    max_val = config.use_case_config.sensor_error_thresholds[sensor_name]["max"]
                    
                    if value < min_val or value > max_val:
                        violated_sensors.append(sensor_name)
                        violation_details.append(f"{sensor_name}={value:.1f}[{min_val},{max_val}]")
            
            sensor_error_info.append(",".join(violated_sensors) if violated_sensors else "Statistical")
            threshold_violations.append(";".join(violation_details) if violation_details else "Z-score")
        else:
            sensor_error_info.append("None")
            threshold_violations.append("None")

    df["Corrupt_Sensors"] = sensor_error_info
    df["Threshold_Violations"] = threshold_violations

    df.to_csv(config.decision_log_path, index=False)
    print(f"Detected {df['Anomaly_Prediction'].sum()} anomalies out of {len(df)} samples")
    
    # Enhanced status breakdown
    status_counts = df['Sensor_Status'].value_counts()
    print(f"📊 Status breakdown: {dict(status_counts)}")
    
    # Report on threshold violations if any
    sensor_errors = (df['Sensor_Status'] == 'Sensor_Error').sum()
    if sensor_errors > 0:
        print(f"⚠️ Sensor errors detected: {sensor_errors} samples")
        violation_summary = df[df['Sensor_Status'] == 'Sensor_Error']['Corrupt_Sensors'].value_counts()
        print(f"📊 Violation breakdown: {dict(violation_summary)}")

    return df, logits_tensor