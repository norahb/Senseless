import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import json
import os
import sys

def detect_anomalies(detector, X_data, config):
    """Detect anomalies using the specified detection strategy."""
    # Determine detection strategy
    prod_config_path = f"../models/{config.name}/production_config.json"
    if os.path.exists(prod_config_path):
        with open(prod_config_path, 'r') as f:
            prod_config = json.load(f)
        strategy = prod_config.get("method_selection", "ensemble")
        optimal_sensitivity = prod_config.get("optimal_sensitivity", "medium")
    else:
        strategy = "ensemble"
        optimal_sensitivity = "medium"

    if strategy == "ensemble":
        # Multi-sensitivity ensemble detection
        sensitivity_levels = ['high', 'medium', 'low']
        sensitivity_weights = {'high': 1.0, 'medium': 1.5, 'low': 2.0}
        weighted_scores = np.zeros(len(X_data))

        for sensitivity in sensitivity_levels:
            predictions, _, _ = detector.predict_anomalies_home_optimized(
                X_data, adapt=False, sensitivity=sensitivity, verbose=False
            )
            weight = sensitivity_weights[sensitivity]
            weighted_scores += predictions * weight

        ensemble_predictions = (weighted_scores >= 1.0).astype(int)
        return ensemble_predictions
    else:
        # Single optimal sensitivity detection
        predictions, _, _ = detector.predict_anomalies_home_optimized(
            X_data, adapt=False, sensitivity=optimal_sensitivity, verbose=False
        )
        return predictions

def run_no_delay_detection(config):
    """Anomaly detection without delay adjustment."""
    print(f"\n[NO-DELAY] Anomaly detection: {config.name}")

    # Load sensor data
    df = pd.read_csv(config.sensor_data_path)
    df.columns = df.columns.str.strip()
    df['Timestamp'] = pd.to_datetime(df[config.date_col] + ' ' + df[config.time_col], dayfirst=True)
    df = df.dropna(subset=config.sensor_cols + [config.status_col]).reset_index(drop=True)
    df[config.sensor_cols] = df[config.sensor_cols].interpolate().bfill()
    y_true = df[config.status_col].apply(lambda x: 0 if x == config.normal_value else 1).values

    # Create detector
    from adaptive_autoencoder import AdaptiveUnsupervisedAutoencoder
    detector = AdaptiveUnsupervisedAutoencoder(
        sensor_names=config.sensor_cols,
        case_name=config.name
    )
    detector.load_adaptive_model(f"../models/{config.name}/adaptive_{config.name}")

    # Detect anomalies
    X_data = df[config.sensor_cols].values
    predictions = detect_anomalies(detector, X_data, config)

    # Label anomalies
    df['Raw_Anomaly_Flag'] = predictions
    df['Final_Label'] = df['Raw_Anomaly_Flag'].apply(
        lambda x: config.anomaly_value if x == 1 else config.normal_value
    )

    # Evaluation
    print("Detection Results:")
    print(classification_report(y_true, predictions, target_names=["Normal", "Anomaly"]))

    # Save results
    output_dir = f"../output/{config.name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{config.name}_sensor_labeled_data_no_delay.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    return df

def run_no_delay_image_alignment(config):
    """Align images with sensor labels without delay adjustment."""
    print(f"\n[ALIGN-NO-DELAY] Image alignment: {config.name}")

    # Load data
    image_df = pd.read_csv(config.image_data_path)
    image_df['Timestamp'] = pd.to_datetime(image_df['Timestamp'], dayfirst=True)

    sensor_df = pd.read_csv(f"../output/{config.name}/{config.name}_sensor_labeled_data_no_delay.csv")
    sensor_df['Timestamp'] = pd.to_datetime(sensor_df['Timestamp'])

    # Alignment logic
    primary_tolerance_sec = 30
    aligned_records = []
    match_stats = {'Aligned': 0, 'Unaligned': 0}

    for _, img_row in image_df.iterrows():
        img_time = img_row['Timestamp']
        image_name = img_row['Image_Name']

        # Find closest match within the primary tolerance window
        primary_start = img_time - pd.Timedelta(seconds=primary_tolerance_sec)
        primary_end = img_time + pd.Timedelta(seconds=primary_tolerance_sec)
        primary_match = sensor_df[
            (sensor_df['Timestamp'] >= primary_start) &
            (sensor_df['Timestamp'] <= primary_end)
        ]

        if not primary_match.empty:
            time_diffs = (primary_match['Timestamp'] - img_time).abs()
            closest_idx = time_diffs.idxmin()
            match_row = primary_match.loc[closest_idx]
            aligned_label = match_row['Final_Label']
            match_stats['Aligned'] += 1
        else:
            aligned_label = 'Unknown'
            match_stats['Unaligned'] += 1

        record = {
            'Image_Name': image_name,
            'Image_Timestamp': img_time,
            'Aligned_Label': aligned_label,
            'True_Label': img_row['Status']
        }
        aligned_records.append(record)

    # Create DataFrame
    aligned_df = pd.DataFrame(aligned_records)
    aligned_df['Correct'] = aligned_df['Aligned_Label'] == aligned_df['True_Label']

    # Calculate alignment accuracy
    accuracy = aligned_df['Correct'].mean() * 100
    print(f"Alignment Accuracy: {accuracy:.2f}%")

    # Save results
    output_dir = f"../output/{config.name}"
    output_file = f"{output_dir}/{config.name}_sensors_images_labels_no_delay.csv"
    aligned_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    return aligned_df

def run_complete_no_delay_pipeline(config):
    """Run complete no-delay pipeline."""
    print(f"Running complete NO-DELAY pipeline: {config.name}")
    
    # Step 1: Detection
    sensor_df = run_no_delay_detection(config)
    
    # Step 2: Alignment
    aligned_df = run_no_delay_image_alignment(config)
    
    print(f"No-delay pipeline completed for {config.name}")
    return sensor_df, aligned_df

if __name__ == "__main__":
    # Add parent directory to path for config import
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from config.config_manager import ConfigManager
    
    if len(sys.argv) != 2:
        print("Usage: python no_delay_alignment.py <use_case_name>")
        print("Example: python no_delay_alignment.py door_case")
        sys.exit(1)
    
    use_case_name = sys.argv[1]
    config = ConfigManager.get_config(use_case_name)
    
    run_complete_no_delay_pipeline(config)