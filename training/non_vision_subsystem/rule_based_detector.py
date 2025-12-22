

import os
import json
import pandas as pd
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def print_detailed_evaluation(y_true, y_pred, target_names=None, title="Classification Results"):
    """
    Print comprehensive evaluation metrics including accuracy and classification report
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels  
    target_names : list, optional
        Target class names for display
    title : str
        Title for the evaluation section
    """
    if target_names is None:
        target_names = ["Normal", "Anomaly"]
    
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")
    
    # Basic accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔍 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 {target_names[0]:<8} {target_names[1]:<8}")
    print(f"Actual {target_names[0]:<8} {cm[0,0]:<8} {cm[0,1]:<8}")
    print(f"       {target_names[1]:<8} {cm[1,0]:<8} {cm[1,1]:<8}")
    
    # Additional metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    print(f"\n📈 Per-Class Metrics Summary:")
    for i, class_name in enumerate(target_names):
        print(f"   {class_name}:")
        print(f"     - Precision: {precision[i]:.4f}")
        print(f"     - Recall:    {recall[i]:.4f}")
        print(f"     - F1-Score:  {f1[i]:.4f}")
        print(f"     - Support:   {support[i]}")
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print(f"\n🔄 Macro Averages:")
    print(f"   - Precision: {macro_precision:.4f}")
    print(f"   - Recall:    {macro_recall:.4f}")
    print(f"   - F1-Score:  {macro_f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1.tolist(),
        'support': support.tolist(),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'confusion_matrix': cm.tolist()
    }

def print_rule_based_analysis(df, config):
    """
    Print analysis of the rule-based detection logic
    """
    print(f"\n📋 Rule-Based Detection Analysis:")
    print(f"   Rule Logic: Object present if distance thresholds NOT met")
    print(f"   S1_distance normal range: 129-131")
    print(f"   S2_distance normal range: 204-207")
    
    # Distribution analysis
    total_samples = len(df)
    normal_count = sum(df['Overall_Final_Label'] == config.normal_value)
    anomaly_count = sum(df['Overall_Final_Label'] == config.anomaly_value)
    
    print(f"\n📊 Detection Distribution:")
    print(f"   Total samples: {total_samples}")
    print(f"   Normal detected: {normal_count} ({normal_count/total_samples*100:.1f}%)")
    print(f"   Anomaly detected: {anomaly_count} ({anomaly_count/total_samples*100:.1f}%)")
    
    # Sensor value analysis
    print(f"\n🔍 Sensor Value Analysis:")
    for sensor in config.sensor_cols:
        sensor_values = df[sensor]
        print(f"   {sensor}:")
        print(f"     - Min: {sensor_values.min():.2f}")
        print(f"     - Max: {sensor_values.max():.2f}")
        print(f"     - Mean: {sensor_values.mean():.2f}")
        print(f"     - Std: {sensor_values.std():.2f}")
        
        # Check how often each sensor is in normal range
        if sensor == 'S1_distance':
            in_range = ((sensor_values >= 129) & (sensor_values <= 131)).sum()
        elif sensor == 'S2_distance':
            in_range = ((sensor_values >= 204) & (sensor_values <= 207)).sum()
        else:
            in_range = 0
            
        print(f"     - In normal range: {in_range} ({in_range/len(sensor_values)*100:.1f}%)")

def run(config):
    print(f"\n🤖 Running rule-based anomaly detection for: {config.name}")
    
    # Start timing the rule-based detection process
    start_time = time.time()
    
    # ✅ Load sensor data
    df = pd.read_csv(config.sensor_data_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Handle missing values if any
    if df[config.sensor_cols + [config.status_col]].isnull().any().any():
        print("⚠️ Found missing values, cleaning data...")
        df = df.dropna(subset=config.sensor_cols + [config.status_col])
        df = df.reset_index(drop=True)
    
    print(f"📊 Loaded {len(df)} samples with {len(config.sensor_cols)} sensors")
    
    # 🧠 Apply rule-based logic (example: object present if distance < threshold)
    print(f"\n🔧 Applying rule-based detection logic...")
    rule_start_time = time.time()
    
    df["Overall_Final_Label"] = df.apply(
        lambda row: config.normal_value
        if (129 <= row["S1_distance"] <= 131 and 204 <= row["S2_distance"] <= 207)
        else config.anomaly_value,
        axis=1
    )

    df["Overall_Confidence_Score"] = 1.0  # Rule-based has 100% confidence
    df["Overall_Final_Validity"] = "Valid"
    
    rule_end_time = time.time()
    rule_detection_time = rule_end_time - rule_start_time
    
    # Print rule-based analysis
    if not config.quiet_mode:
        print_rule_based_analysis(df, config)
    
    # 📊 EVALUATION: Compare predictions with ground truth
    evaluation_results = None
    if config.status_col in df.columns:
        print(f"\n🎯 Evaluating rule-based detection against ground truth...")
        
        # Prepare ground truth and predictions
        y_true = df[config.status_col].apply(
            lambda x: 0 if x == config.normal_value else 1
        ).values
        
        y_pred = df['Overall_Final_Label'].apply(
            lambda x: 0 if x == config.normal_value else 1
        ).values
        
        # Print detailed evaluation
        evaluation_results = print_detailed_evaluation(
            y_true, y_pred, 
            target_names=["Normal", "Anomaly"],
            title=f"Rule-Based Detection Results - {config.name.upper()}"
        )
        
        # Save evaluation results
        output_dir = os.path.join("output", config.name)
        os.makedirs(output_dir, exist_ok=True)
        
        eval_file = os.path.join(output_dir, f"{config.name}_rule_based_evaluation.json")
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print(f"\n💾 Evaluation results saved to: {eval_file}")
        
        # Quick summary for quiet mode
        if config.quiet_mode:
            accuracy = evaluation_results['accuracy']
            macro_f1 = evaluation_results['macro_f1']
            print(f"\n✅ Quick Results: Accuracy={accuracy:.1%}, Macro-F1={macro_f1:.3f}")
    
    else:
        print("⚠️ No ground truth column found, skipping evaluation")

    # 💾 Save labeled sensor data
    output_dir = os.path.join("output", config.name)
    os.makedirs(output_dir, exist_ok=True)
    labeled_data_path = os.path.join(output_dir, f"{config.name}_sensor_labeled_data.csv")
    df.to_csv(labeled_data_path, index=False)
    
    if not config.quiet_mode:
        print(f"\n💾 Labeled sensor data saved to: {labeled_data_path}")

    # 🛑 Skip delay calibration if all sensors are IRS
    if all(sensor_type.upper() == "IRS" for sensor_type in config.sensor_metadata.values()):
        if not config.quiet_mode:
            print("\n🛑 Skipping dynamic delay calibration: all sensors are IRS.")

        calibrated_delays = {
            "lab_baseline_delays": {s: 0.0 for s in config.sensor_cols},
            "measured_onsite_delays": {s: 0.0 for s in config.sensor_cols},
            "lab+onsite_delays": {s: 0.0 for s in config.sensor_cols},
            "final_calibrated_delays_with_reference_sensor": {s: 0.0 for s in config.sensor_cols},
            "human_adjusted_delays": {}
        }

        model_dir = os.path.join("models", config.name)
        os.makedirs(model_dir, exist_ok=True)
        delay_file = os.path.join(model_dir, f"{config.name}_calibrated_delays.json")

        with open(delay_file, 'w') as f:
            json.dump(calibrated_delays, f, indent=2)

        if not config.quiet_mode:
            print(f"✅ Saved default 0.0 delay calibration file to: {delay_file}")
    
    # Calculate total execution time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print timing results
    print(f"\n⏱️ Rule-Based Detection Timing:")
    print(f"   Core detection logic: {rule_detection_time:.2f} seconds")
    print(f"   Total execution time: {total_time:.2f} seconds")
    print(f"✅ Rule-based detection completed for {config.name}")
    
    # Return results for pipeline integration
    return {
        'success': True,
        'samples_processed': len(df),
        'normal_detected': sum(df['Overall_Final_Label'] == config.normal_value),
        'anomaly_detected': sum(df['Overall_Final_Label'] == config.anomaly_value),
        'evaluation_results': evaluation_results,
        'rule_detection_time': rule_detection_time,
        'total_execution_time': total_time
    }