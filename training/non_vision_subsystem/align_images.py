

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import json
import os

def run(config):
    print(f"[ALIGN] Timestamp alignment: {config.name}")

    # Load image dataset
    image_df = pd.read_csv(config.image_data_path)
    image_df['Timestamp'] = pd.to_datetime(image_df['Timestamp'], format='mixed', dayfirst=True)

    # Load sensor data with period-based labels
    sensor_df = pd.read_csv(f"output/{config.name}/{config.name}_sensor_labeled_data.csv")

    sensor_df['Timestamp'] = pd.to_datetime(sensor_df['Timestamp'], format='mixed', dayfirst=True)

    print(f"Loaded {len(image_df)} images and {len(sensor_df)} sensor records")

    # # Simple alignment windows since backward labeling created proper periods
    # primary_tolerance_sec = 2  # ±2 seconds for exact matches
    # fallback_tolerance_sec = 10  # ±10 seconds for timing jitter

        # Simple alignment windows since backward labeling created proper periods
    primary_tolerance_sec = 2  # ±2 seconds for exact matches

    # Load calibrated delays
    model_dir = f"models/{config.name}"
    delay_file = os.path.join(model_dir, f"{config.name}_calibrated_delays.json")

    if os.path.exists(delay_file):
        with open(delay_file, "r") as f:
            delay_data = json.load(f)

        delays = delay_data.get("final_calibrated_delays_with_reference_sensor", {})
        if delays:
            max_delay = max(delays.values())
            # If max delay > 2, use it; otherwise, fallback = 5
            fallback_tolerance_sec = max_delay if max_delay > 2 else 5
        else:
            fallback_tolerance_sec = 5
    else:
        fallback_tolerance_sec = 5

    print(f"[ALIGN] Using tolerances: primary={primary_tolerance_sec}s, "
          f"fallback={fallback_tolerance_sec:.2f}s")


    aligned_records = []
    match_stats = {'Primary': 0, 'Fallback': 0, 'Unknown': 0}

    for idx, img_row in image_df.iterrows():
        img_time = img_row['Timestamp']
        image_name = img_row['Image_Name']

        # PRIMARY MATCH: Close timestamp matching
        primary_start = img_time - pd.Timedelta(seconds=primary_tolerance_sec)
        primary_end = img_time + pd.Timedelta(seconds=primary_tolerance_sec)
        
        primary_match = sensor_df[
            (sensor_df['Timestamp'] >= primary_start) & 
            (sensor_df['Timestamp'] <= primary_end)
        ]

        if not primary_match.empty:
            # Select closest within primary window
            time_diffs = (primary_match['Timestamp'] - img_time).abs()
            closest_idx = time_diffs.idxmin()
            match_row = primary_match.loc[closest_idx]
            match_type = 'Primary'
            time_diff = (match_row['Timestamp'] - img_time).total_seconds()
            
        else:
            # FALLBACK MATCH: Slightly wider window for timing issues
            fallback_start = img_time - pd.Timedelta(seconds=fallback_tolerance_sec)
            fallback_end = img_time + pd.Timedelta(seconds=fallback_tolerance_sec)
            
            fallback_match = sensor_df[
                (sensor_df['Timestamp'] >= fallback_start) & 
                (sensor_df['Timestamp'] <= fallback_end)
            ]

            if not fallback_match.empty:
                time_diffs = (fallback_match['Timestamp'] - img_time).abs()
                closest_idx = time_diffs.idxmin()
                match_row = fallback_match.loc[closest_idx]
                match_type = 'Fallback'
                time_diff = (match_row['Timestamp'] - img_time).total_seconds()
            else:
                match_row = None
                match_type = 'Unknown'
                time_diff = None

        # Extract match information
        if match_row is not None:
            aligned_label = match_row['Overall_Final_Label']
            aligned_validity = match_row['Overall_Final_Validity']
            aligned_conf = match_row['Overall_Confidence_Score']
            
            # Additional sensor-level information
            sensor_labels = {}
            sensor_confidences = {}
            for sensor in config.sensor_cols:
                sensor_labels[f'{sensor}_Label'] = match_row.get(f'{sensor}_Final_Label', 'Unknown')
                sensor_confidences[f'{sensor}_Confidence'] = match_row.get(f'{sensor}_Confidence_Score', 0.0)
                
        else:
            aligned_label = 'Unknown'
            aligned_validity = 'Unknown'
            aligned_conf = 0.0
            sensor_labels = {}
            sensor_confidences = {}

        match_stats[match_type] += 1

        # Build record
        record = {
            'Image_Name': image_name,
            'Image_Timestamp': img_time,
            'Sensor_Timestamp': match_row['Timestamp'] if match_row is not None else None,
            'Label': aligned_label,
            'Aligned_Validity': aligned_validity,
            'Confidence_Score_nonvision': aligned_conf,
            'Match_Type': match_type,
            'Time_Difference_Sec': time_diff,
            'True_Label': img_row['Status']
        }
        
        record.update(sensor_labels)
        record.update(sensor_confidences)
        aligned_records.append(record)

    # Create DataFrame
    aligned_df = pd.DataFrame(aligned_records)
    aligned_df['Correct'] = aligned_df.apply(
        lambda row: 'Yes' if row['Label'] == row['True_Label'] else 'No', axis=1
    )

    # Analysis
    print(f"\nAlignment Results Summary:")
    total_images = len(aligned_df)
    
    print(f"Match Type Distribution:")
    for match_type, count in match_stats.items():
        percentage = (count / total_images) * 100
        print(f"   {match_type}: {count:,} ({percentage:.1f}%)")

    # Time difference analysis
    successful_matches = aligned_df[aligned_df['Time_Difference_Sec'].notna()]
    if not successful_matches.empty:
        time_diffs = successful_matches['Time_Difference_Sec']
        print(f"\nTime Difference Analysis:")
        print(f"   Count: {len(time_diffs):,}")
        print(f"   Mean: {time_diffs.mean():.1f} seconds")
        print(f"   Median: {time_diffs.median():.1f} seconds")
        print(f"   Range: [{time_diffs.min():.1f}, {time_diffs.max():.1f}] seconds")
        
        within_5s = np.sum(np.abs(time_diffs) <= 5)
        within_10s = np.sum(np.abs(time_diffs) <= 10)
        
        print(f"   Within ±5s: {within_5s} ({within_5s/len(time_diffs)*100:.1f}%)")
        print(f"   Within ±10s: {within_10s} ({within_10s/len(time_diffs)*100:.1f}%)")

    # Save results
    output_dir = f"output/{config.name}"
    # os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/{config.name}_sensors_images_labels.csv"
    aligned_df.to_csv(output_file, index=False)
    print(f"Saved alignment results to {output_file}")

    # Evaluation
    valid_mask = aligned_df['Label'] != 'Unknown'
    if valid_mask.sum() > 0:
        y_true = aligned_df.loc[valid_mask, 'True_Label'].apply(
            lambda x: 0 if x == config.normal_value else 1
        )
        y_pred = aligned_df.loc[valid_mask, 'Label'].apply(
            lambda x: 0 if x == config.normal_value else 1
        )

        print("\nImage-Sensor Alignment Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
        
        aligned_only = aligned_df[aligned_df['Label'] != 'Unknown']
        accuracy = (aligned_only['Label'] == aligned_only['True_Label']).mean() * 100
        print(f"\nOverall Alignment Accuracy: {accuracy:.1f}%")
        
        # Confidence analysis
        high_conf_matches = aligned_only[aligned_only['Confidence_Score_nonvision'] > 0.7]
        medium_conf_matches = aligned_only[(aligned_only['Confidence_Score_nonvision'] > 0.4) & 
                                         (aligned_only['Confidence_Score_nonvision'] <= 0.7)]
        low_conf_matches = aligned_only[aligned_only['Confidence_Score_nonvision'] <= 0.4]
        
        print(f"\nConfidence-Based Accuracy Analysis:")
        
        if not high_conf_matches.empty:
            high_conf_accuracy = (high_conf_matches['Label'] == high_conf_matches['True_Label']).mean() * 100
            print(f"   High Confidence (>0.7): {high_conf_accuracy:.1f}% ({len(high_conf_matches):,} samples)")
        
        if not medium_conf_matches.empty:
            med_conf_accuracy = (medium_conf_matches['Label'] == medium_conf_matches['True_Label']).mean() * 100
            print(f"   Medium Confidence (0.4-0.7): {med_conf_accuracy:.1f}% ({len(medium_conf_matches):,} samples)")
            
        if not low_conf_matches.empty:
            low_conf_accuracy = (low_conf_matches['Label'] == low_conf_matches['True_Label']).mean() * 100
            print(f"   Low Confidence (≤0.4): {low_conf_accuracy:.1f}% ({len(low_conf_matches):,} samples)")

        # Match type accuracy analysis
        print(f"\nAccuracy by Match Type:")
        for match_type in ['Primary', 'Fallback']:
            type_matches = aligned_only[aligned_only['Match_Type'] == match_type]
            if not type_matches.empty:
                type_accuracy = (type_matches['Label'] == type_matches['True_Label']).mean() * 100
                avg_conf = type_matches['Confidence_Score_nonvision'].mean()
                avg_time_diff = type_matches['Time_Difference_Sec'].abs().mean()
                print(f"   {match_type}: {type_accuracy:.1f}% accuracy, "
                      f"Avg confidence: {avg_conf:.3f}, "
                      f"Avg time diff: {avg_time_diff:.1f}s "
                      f"({len(type_matches):,} samples)")

    else:
        print("No successful alignments found!")

    # Diagnostic information
    print(f"\nAlignment Diagnostic Info:")
    print(f"   Image timestamp range: {image_df['Timestamp'].min()} to {image_df['Timestamp'].max()}")
    print(f"   Sensor timestamp range: {sensor_df['Timestamp'].min()} to {sensor_df['Timestamp'].max()}")
    
    # Check temporal overlap
    img_start, img_end = image_df['Timestamp'].min(), image_df['Timestamp'].max()
    sensor_start, sensor_end = sensor_df['Timestamp'].min(), sensor_df['Timestamp'].max()
    
    overlap_start = max(img_start, sensor_start)
    overlap_end = min(img_end, sensor_end)
    
    if overlap_start <= overlap_end:
        overlap_duration = (overlap_end - overlap_start).total_seconds() / 3600
        print(f"   Temporal overlap: {overlap_duration:.1f} hours")
    else:
        print("   No temporal overlap between image and sensor data!")

    print(f"\nImage alignment completed with simple timestamp matching")
    
    return aligned_df