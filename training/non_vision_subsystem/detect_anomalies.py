import pandas as pd
import numpy as np
import json
from sklearn.metrics import classification_report
import os
from dataclasses import dataclass
from scipy.stats import norm, gaussian_kde
from pathlib import Path
import joblib

@dataclass
class UseCaseCharacteristics:
    mean_confidence: float
    consensus_agreement_rate: float
    sensor_reliability_score: float
    detection_difficulty: str

def calibrate_error_distributions(detector, X_data, sensor_names, method="gaussian", save_path=None):
    """
    Fit error distributions per sensor using the trained AE.
    Optionally save to JSON for reuse.
    """
    X_scaled = detector.scaler.transform(X_data)
    reconstructions = detector.autoencoder.predict(X_scaled)

    # Ensure 2D shape for single-sensor case ---
    if X_scaled.ndim == 1:
        X_scaled = X_scaled.reshape(-1, 1)
    if reconstructions.ndim == 1:
        reconstructions = reconstructions.reshape(-1, 1)

    reconstruction_errors = (X_scaled - reconstructions) ** 2

    error_distributions = {}
    for i, sensor in enumerate(sensor_names):
        sensor_errors = reconstruction_errors[:, i]

        if method == "gaussian":
            mu, sigma = np.mean(sensor_errors), np.std(sensor_errors) + 1e-8
            error_distributions[sensor] = {"type": "gaussian", "mean": float(mu), "std": float(sigma)}

        elif method == "kde":
            kde = gaussian_kde(sensor_errors)
            xs = np.linspace(min(sensor_errors), max(sensor_errors), 200)
            cdf_vals = [kde.integrate_box_1d(-np.inf, e) for e in xs]
            error_distributions[sensor] = {
                "type": "kde",
                "xs": xs.tolist(),
                "cdf_vals": cdf_vals
            }

    detector.error_distributions = error_distributions

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(error_distributions, f, indent=2)
        print(f"✅ Saved error distributions to {save_path}")

    return error_distributions

def load_error_distributions(detector, load_path):
    """Load previously saved error distributions into detector."""
    if not Path(load_path).exists():
        raise FileNotFoundError(f"No saved error distributions at {load_path}")

    with open(load_path, "r") as f:
        detector.error_distributions = json.load(f)

    print(f"✅ Loaded error distributions from {load_path}")
    return detector.error_distributions

def select_detection_strategy(config):
    """
    Determine whether to use ensemble or single optimal sensitivity based on training results.
    Now also respects the method_selection saved in production_config.json.
    """
    prod_config_path = f"models/{config.name}/production_config.json"
    
    if not os.path.exists(prod_config_path):
        return "ensemble", "medium", 0.70
    
    try:
        with open(prod_config_path, 'r') as f:
            prod_config = json.load(f)
    except Exception:
        return "ensemble", "medium", 0.70
    
    # Check method_selection first ===
    method_selection = prod_config.get("method_selection", None)
    if method_selection and method_selection != "ensemble":
        # Single method was chosen at training time (e.g., autoencoder only)
        best_sensitivity = prod_config.get("optimal_sensitivity", "medium")
        best_accuracy = (
            prod_config.get("final_performance", {}).get("test_accuracy", 0.0)
        )
        print(f"[DEBUG] Single-method training detected → {method_selection.upper()}")
        return "single_optimal", best_sensitivity, best_accuracy
    
    # === Otherwise, use sensitivity comparison ===
    source_used = None
    sensitivity_comparison = {}
    
    if "sensitivity_comparison" in prod_config:
        if "selection_score" in prod_config or "final_performance" in prod_config:
            sensitivity_comparison = prod_config["sensitivity_comparison"]
            source_used = "single training run"
        else:
            sensitivity_comparison = prod_config["sensitivity_comparison"]
            source_used = "averaged (last 3 cycles)"
    elif "last_cycle_performance" in prod_config:
        sensitivity_comparison = prod_config["last_cycle_performance"]
        source_used = "last-cycle performance"
    
    if not sensitivity_comparison:
        return "ensemble", "medium", 0.70
    
    # Find best individual sensitivity
    best_sensitivity = None
    best_accuracy = 0.0
    
    for sensitivity, metrics in sensitivity_comparison.items():
        test_accuracy = metrics.get('test_accuracy') or metrics.get('accuracy', 0.0)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_sensitivity = sensitivity
    
    if best_sensitivity is None:
        best_sensitivity, best_accuracy = "medium", 0.0
    
    # Decision logic
    if best_accuracy > 0.90:
        strategy = "single_optimal"
        reason = f"excellent individual performance ({best_accuracy:.1%})"
    elif best_accuracy >= 0.70:
        strategy = "ensemble"
        reason = f"good candidate for ensemble improvement ({best_accuracy:.1%})"
    else:
        strategy = "single_optimal"
        reason = f"low accuracy, ensemble unlikely to help ({best_accuracy:.1%})"
    
    print(f"Detection strategy: {strategy} - {reason}")
    print(f"Best individual: {best_sensitivity.upper()} ({best_accuracy:.1%})")
    print(f"[DEBUG] Sensitivity source: {source_used}")
    
    return strategy, best_sensitivity, best_accuracy

def calculate_adaptive_confidence_scores(detector, X_data, sensor_names):
    """Probability-based confidence using saved/calibrated error distributions."""
    X_scaled = detector.scaler.transform(X_data)
    reconstructions = detector.autoencoder.predict(X_scaled)
    reconstruction_errors = (X_scaled - reconstructions) ** 2

    sensor_confidences = {}
    for i, sensor in enumerate(sensor_names):
        sensor_errors = reconstruction_errors[:, i]

        if hasattr(detector, "error_distributions") and sensor in detector.error_distributions:
            dist_info = detector.error_distributions[sensor]

            if dist_info["type"] == "gaussian":
                mu, sigma = dist_info["mean"], dist_info["std"]
                p_vals = norm.cdf(sensor_errors, loc=mu, scale=sigma)
                confidence_scores = np.clip(p_vals, 0.01, 0.99)

            elif dist_info["type"] == "kde":
                xs, cdf_vals = np.array(dist_info["xs"]), np.array(dist_info["cdf_vals"])
                confidence_scores = np.interp(sensor_errors, xs, cdf_vals)
                confidence_scores = np.clip(confidence_scores, 0.01, 0.99)

        else:
            confidence_scores = np.full_like(sensor_errors, 0.5, dtype=float)

        sensor_confidences[sensor] = confidence_scores

    confidence_matrix = np.column_stack([sensor_confidences[s] for s in sensor_names])
    if hasattr(detector, "feature_importance"):
        weights = [detector.feature_importance.get(sensor, 1.0) for sensor in sensor_names]
        weights = np.array(weights) / np.sum(weights)
        overall_confidence = np.average(confidence_matrix, weights=weights, axis=1)
    else:
        overall_confidence = np.mean(confidence_matrix, axis=1)

    return sensor_confidences, overall_confidence

def multi_sensitivity_ensemble_detection(detector, X_data, config):
    """Multi-sensitivity ensemble approach with weighted voting."""
    sensitivity_levels = ['high', 'medium', 'low']
    sensitivity_weights = {'high': 1.0, 'medium': 1.5, 'low': 2.0}
    consensus_threshold = 1.0
    
    sensitivity_results = {}
    all_predictions = {}
    all_confidences = {}
    all_overall_confidences = {}
    
    for sensitivity in sensitivity_levels:
        predictions, insights, drift_info = detector.predict_anomalies_home_optimized(
            X_data, adapt=False, sensitivity=sensitivity, verbose=False
        )
        
        try:
            sensor_confidences, overall_confidence = calculate_adaptive_confidence_scores(
                detector, X_data, config.sensor_cols
            )
        except Exception:
            sensor_confidences = {}
            for sensor in config.sensor_cols:
                base_conf = 0.3 + 0.4 * predictions
                sensor_confidences[sensor] = np.clip(base_conf, 0.1, 0.9)
            overall_confidence = np.mean([sensor_confidences[s] for s in config.sensor_cols], axis=0)
        
        sensitivity_results[sensitivity] = {
            'predictions': predictions,
            'detection_rate': np.mean(predictions),
            'anomaly_count': np.sum(predictions)
        }
        
        all_predictions[sensitivity] = predictions
        all_confidences[sensitivity] = sensor_confidences
        all_overall_confidences[sensitivity] = overall_confidence
    
    # Apply weighted voting consensus
    n_samples = len(X_data)
    weighted_scores = np.zeros(n_samples)
    
    for sensitivity in sensitivity_levels:
        weight = sensitivity_weights[sensitivity]
        predictions = all_predictions[sensitivity]
        weighted_scores += predictions * weight
    
    ensemble_predictions = (weighted_scores >= consensus_threshold).astype(int)
    
    # Combine confidence scores with agreement bonus
    ensemble_sensor_confidences = {}
    ensemble_overall_confidence = np.zeros(n_samples)
    
    for sensor in config.sensor_cols:
        sensor_conf_weighted = np.zeros(n_samples)
        total_weight = 0
        
        for sensitivity in sensitivity_levels:
            weight = sensitivity_weights[sensitivity]
            sensor_conf_weighted += all_confidences[sensitivity][sensor] * weight
            total_weight += weight
        
        sensor_conf_weighted /= total_weight
        ensemble_sensor_confidences[sensor] = sensor_conf_weighted
    
    for i in range(n_samples):
        overall_conf_weighted = 0
        total_weight = 0
        
        for sensitivity in sensitivity_levels:
            weight = sensitivity_weights[sensitivity]
            overall_conf_weighted += all_overall_confidences[sensitivity][i] * weight
            total_weight += weight
        
        overall_conf_weighted /= total_weight
        
        # Agreement analysis
        sample_predictions = [all_predictions[sens][i] for sens in sensitivity_levels]
        
        agreement_bonus = 0
        if all(pred == sample_predictions[0] for pred in sample_predictions):
            agreement_bonus = 0.1
        elif sum(sample_predictions) >= 2:
            agreement_bonus = 0.05
        
        final_confidence = overall_conf_weighted + agreement_bonus
        ensemble_overall_confidence[i] = np.clip(final_confidence, 0.05, 0.95)
    
    # Calculate consensus statistics
    full_agreement = 0
    partial_agreement = 0
    no_agreement = 0
    
    for i in range(n_samples):
        sample_preds = [all_predictions[sens][i] for sens in sensitivity_levels]
        unique_preds = len(set(sample_preds))
        
        if unique_preds == 1:
            full_agreement += 1
        elif sum(sample_preds) >= 2:
            partial_agreement += 1
        else:
            no_agreement += 1
    
    ensemble_results = {
        'ensemble_predictions': ensemble_predictions,
        'ensemble_sensor_confidences': ensemble_sensor_confidences,
        'ensemble_overall_confidence': ensemble_overall_confidence,
        'sensitivity_details': sensitivity_results,
        'consensus_stats': {
            'full_agreement': full_agreement,
            'partial_agreement': partial_agreement,
            'no_agreement': no_agreement
        }
    }
    
    return ensemble_results

def calculate_use_case_characteristics(ensemble_results, period_confidences):
    """Calculate use case characteristics from ensemble results."""
    consensus_stats = ensemble_results.get('consensus_stats', {})
    total_samples = sum([
        consensus_stats.get('full_agreement', 0),
        consensus_stats.get('partial_agreement', 0),
        consensus_stats.get('no_agreement', 0)
    ])
    
    if total_samples > 0:
        consensus_agreement_rate = (
            consensus_stats.get('full_agreement', 0) + 
            consensus_stats.get('partial_agreement', 0) * 0.5
        ) / total_samples
    else:
        consensus_agreement_rate = 0.5
    
    sensitivity_details = ensemble_results.get('sensitivity_details', {})
    detection_rates = []
    for sens in ['high', 'medium', 'low']:
        if sens in sensitivity_details:
            detection_rates.append(sensitivity_details[sens]['detection_rate'])
    
    if len(detection_rates) >= 2:
        detection_spread = max(detection_rates) - min(detection_rates)
        sensor_reliability_score = max(0, 1.0 - detection_spread)
    else:
        sensor_reliability_score = 0.5
    
    mean_confidence = np.mean(period_confidences) if period_confidences else 0.5
    
    if mean_confidence > 0.7 and consensus_agreement_rate > 0.7:
        detection_difficulty = 'easy'
    elif mean_confidence > 0.4 and consensus_agreement_rate > 0.5:
        detection_difficulty = 'medium'
    else:
        detection_difficulty = 'hard'
    
    return UseCaseCharacteristics(
        mean_confidence=mean_confidence,
        consensus_agreement_rate=consensus_agreement_rate,
        sensor_reliability_score=sensor_reliability_score,
        detection_difficulty=detection_difficulty
    )

def calculate_adaptive_confidence_threshold(duration_hours, characteristics):
    """Calculate adaptive confidence threshold based on use case characteristics."""
    base_thresholds = {
        'easy': 0.3,
        'medium': 0.2,
        'hard': 0.1
    }
    
    base_threshold = base_thresholds[characteristics.detection_difficulty]
    
    confidence_adjustment = (characteristics.mean_confidence - 0.5) * 0.2
    consensus_adjustment = (characteristics.consensus_agreement_rate - 0.5) * 0.1
    reliability_adjustment = (characteristics.sensor_reliability_score - 0.5) * 0.1
    
    if duration_hours > 8:
        duration_multiplier = 0.3
    elif duration_hours > 2:
        duration_multiplier = 0.5
    elif duration_hours > 0.5:
        duration_multiplier = 0.7
    elif duration_hours > 0.1:
        duration_multiplier = 1.0
    else:
        duration_multiplier = 1.5
    
    adjusted_threshold = base_threshold + confidence_adjustment + consensus_adjustment + reliability_adjustment
    final_threshold = adjusted_threshold * duration_multiplier
    
    return np.clip(final_threshold, 0.05, 0.8)

def adaptive_period_merging(periods, case_name="door"):
    """Merge nearby periods based on use case-specific behavior."""
    if not periods:
        return periods
    
    sorted_periods = sorted(periods, key=lambda x: x[2])
    merged = []
    current_period = sorted_periods[0]
    
    max_gap_mapping = {
        "door": 300,
        "appliance": 120,
        "co2": 600
    }
    max_gap = max_gap_mapping.get(case_name.lower(), 180)
    
    for next_period in sorted_periods[1:]:
        current_end = current_period[3]
        next_start = next_period[2]
        gap_seconds = (next_start - current_end).total_seconds()
        
        if gap_seconds <= max_gap:
            merged_start_idx = current_period[0]
            merged_end_idx = next_period[1]
            merged_start_time = current_period[2]
            merged_end_time = next_period[3]
            merged_duration = (merged_end_time - merged_start_time).total_seconds()
            
            current_duration = current_period[4]
            next_duration = next_period[4]
            total_duration = current_duration + next_duration
            
            if total_duration > 0:
                merged_confidence = (
                    (current_period[5] * current_duration + next_period[5] * next_duration) 
                    / total_duration
                )
            else:
                merged_confidence = (current_period[5] + next_period[5]) / 2
            
            current_period = (
                merged_start_idx, merged_end_idx, merged_start_time, 
                merged_end_time, merged_duration, merged_confidence
            )
        else:
            merged.append(current_period)
            current_period = next_period
    
    merged.append(current_period)
    return merged

def detect_continuous_anomaly_periods(df, predictions, sensor_confidences, config, ensemble_results):
    """Detect continuous anomaly periods with adaptive confidence filtering."""
    try:
        with open(f"models/{config.name}/{config.name}_calibrated_delays.json", 'r') as f:
            delays = json.load(f)
        final_delays = delays.get("final_calibrated_delays_with_reference_sensor", {})
    except FileNotFoundError:
        final_delays = {s: 0.0 for s in config.sensor_cols}
    
    sampling_interval = df['Timestamp'].diff().median().total_seconds()
    min_anomaly_duration = max(5, sampling_interval * 3)
    
    all_period_confidences = []
    
    for s in config.sensor_cols:
        sensor_delay = final_delays.get(s, 0.0)
        anomaly_periods = []
        in_anomaly = False
        current_start = None
        
        for i, pred in enumerate(predictions):
            if pred == 1 and not in_anomaly:
                current_start = i
                in_anomaly = True
            elif pred == 0 and in_anomaly:
                current_end = i - 1
                start_time = df.loc[current_start, 'Timestamp']
                end_time = df.loc[current_end, 'Timestamp']
                duration = (end_time - start_time).total_seconds()
                
                if duration >= min_anomaly_duration:
                    period_confidences = sensor_confidences[s][current_start:current_end+1]
                    avg_confidence = np.mean(period_confidences)
                    all_period_confidences.append(avg_confidence)
                    
                    anomaly_periods.append((
                        current_start, current_end, start_time, end_time, duration, avg_confidence
                    ))
                
                in_anomaly = False
                current_start = None
        
        if in_anomaly and current_start is not None:
            current_end = len(predictions) - 1
            start_time = df.loc[current_start, 'Timestamp']
            end_time = df.loc[current_end, 'Timestamp']
            duration = (end_time - start_time).total_seconds()
            
            if duration >= min_anomaly_duration:
                period_confidences = sensor_confidences[s][current_start:current_end+1]
                avg_confidence = np.mean(period_confidences)
                all_period_confidences.append(avg_confidence)
                
                anomaly_periods.append((
                    current_start, current_end, start_time, end_time, duration, avg_confidence
                ))
        
        # Merge periods
        if len(anomaly_periods) > 1:
            anomaly_periods = adaptive_period_merging(anomaly_periods, config.name)
        
        # Calculate use case characteristics
        characteristics = calculate_use_case_characteristics(ensemble_results, all_period_confidences)
        
        # Apply adaptive confidence filtering
        for period_idx, (start_idx, end_idx, start_time, end_time, duration, avg_confidence) in enumerate(anomaly_periods):
            actual_start_time = start_time - pd.Timedelta(seconds=sensor_delay)
            actual_end_time = end_time
            
            period_mask = (df['Timestamp'] >= actual_start_time) & (df['Timestamp'] <= actual_end_time)
            affected_records = period_mask.sum()
            actual_duration = (actual_end_time - actual_start_time).total_seconds()
            duration_hours = actual_duration / 3600
            
            if affected_records > 0:
                adaptive_threshold = calculate_adaptive_confidence_threshold(duration_hours, characteristics)
                
                if avg_confidence >= adaptive_threshold:
                    df.loc[period_mask, f'{s}_Final_Label'] = config.anomaly_value
                    df.loc[period_mask, f'{s}_Final_Validity'] = 'Invalid'

def run(config):
    """Main function for multi-sensitivity ensemble anomaly detection."""
    print(f"[DETECT] Multi-Sensitivity Ensemble Anomaly Detection: {config.name}")

    # Load data
    df = pd.read_csv(config.sensor_data_path)
    df.columns = df.columns.str.strip()
    df['Timestamp'] = pd.to_datetime(df[config.date_col] + ' ' + df[config.time_col], dayfirst=True)
    df = df.dropna(subset=config.sensor_cols + [config.status_col]).reset_index(drop=True)
    df[config.sensor_cols] = df[config.sensor_cols].interpolate().bfill()
    y_true = df[config.status_col].apply(lambda x: 0 if x == config.normal_value else 1).values

    output_dir = f"output/{config.name}"
    os.makedirs(output_dir, exist_ok=True)

    # Load detector
    from non_vision_subsystem.adaptive_autoencoder import AdaptiveUnsupervisedAutoencoder
    
    detector = AdaptiveUnsupervisedAutoencoder(
        sensor_names=config.sensor_cols,
        case_name=config.name
    )
    
    detector.load_adaptive_model(f"models/{config.name}/adaptive_{config.name}")
    X_data = df[config.sensor_cols].values
    
    dist_path = f"models/{config.name}/{config.name}_error_distributions.json"

    try:
        load_error_distributions(detector, dist_path)
    except FileNotFoundError:
        print("⚙️ Calibrating error distributions (first time)...")
        calibrate_error_distributions(detector, X_data, config.sensor_cols, method="gaussian", save_path=dist_path)


    # # Run multi-sensitivity ensemble
    # ensemble_results = multi_sensitivity_ensemble_detection(detector, X_data, config)
    
    # ensemble_predictions = ensemble_results['ensemble_predictions']
    # ensemble_sensor_confidences = ensemble_results['ensemble_sensor_confidences']
    # ensemble_overall_confidence = ensemble_results['ensemble_overall_confidence']
    
    # Determine detection strategy
    strategy, optimal_sensitivity, expected_accuracy = select_detection_strategy(config)

    if strategy == "ensemble":
        # Run multi-sensitivity ensemble
        ensemble_results = multi_sensitivity_ensemble_detection(detector, X_data, config)
        
        predictions = ensemble_results['ensemble_predictions']
        sensor_confidences = ensemble_results['ensemble_sensor_confidences']
        overall_confidence = ensemble_results['ensemble_overall_confidence']
        
        print(f"Using ensemble approach")
    else:
        # Use single optimal sensitivity
        predictions, insights, drift_info = detector.predict_anomalies_home_optimized(
            X_data, adapt=False, sensitivity=optimal_sensitivity, verbose=False
        )
        
        try:
            sensor_confidences, overall_confidence = calculate_adaptive_confidence_scores(
                detector, X_data, config.sensor_cols
            )
        except Exception:
            sensor_confidences = {}
            for sensor in config.sensor_cols:
                base_conf = 0.3 + 0.4 * predictions
                sensor_confidences[sensor] = np.clip(base_conf, 0.1, 0.9)
            overall_confidence = np.mean([sensor_confidences[s] for s in config.sensor_cols], axis=0)
        
        ensemble_results = {
            'consensus_stats': {'full_agreement': len(predictions), 'partial_agreement': 0, 'no_agreement': 0}
        }
        
        print(f"Using single {optimal_sensitivity.upper()} sensitivity")
    # Initialize DataFrame
    for s in config.sensor_cols:
        df[f'{s}_Final_Label'] = config.normal_value
        df[f'{s}_Final_Validity'] = 'Valid'
        # df[f'{s}_Raw_Anomaly_Flag'] = ensemble_predictions
        # df[f'{s}_Confidence_Score'] = ensemble_sensor_confidences[s]
        df[f'{s}_Raw_Anomaly_Flag'] = predictions
        df[f'{s}_Confidence_Score'] = sensor_confidences[s]

    df['Overall_Confidence_Score'] = overall_confidence
    
    # df['Overall_Confidence_Score'] = ensemble_overall_confidence
    
    # Apply continuous anomaly period detection
    # detect_continuous_anomaly_periods(df, ensemble_predictions, ensemble_sensor_confidences, config, ensemble_results)
    # if config.name.lower() == 'co2':
    #     # Use simple current-record labeling (like no-delay approach)
    #     # Skip detect_continuous_anomaly_periods()
    #     pass
    # else:
    #     # Use full period detection for event-based cases
    # detect_continuous_anomaly_periods(df, ensemble_predictions, ensemble_sensor_confidences, config, ensemble_results)
    detect_continuous_anomaly_periods(df, predictions, sensor_confidences, config, ensemble_results)
    # Calculate overall labels
    label_cols = [f"{s}_Final_Label" for s in config.sensor_cols]
    df['Overall_Final_Label'] = df[label_cols].apply(
        lambda row: config.anomaly_value if config.anomaly_value in row.values else config.normal_value, 
        axis=1
    )
    
    validity_cols = [f"{s}_Final_Validity" for s in config.sensor_cols]
    df['Overall_Final_Validity'] = df[validity_cols].apply(
        lambda row: 'Invalid' if 'Invalid' in row.values else 'Valid', 
        axis=1
    )

    # Evaluation
    # ensemble_report = classification_report(y_true, ensemble_predictions, output_dict=True, zero_division=0)
    # print(f"Ensemble accuracy: {ensemble_report['accuracy']:.1%}")
    detection_report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
    print(f"Detection accuracy: {detection_report['accuracy']:.1%}")
    
    final_predictions = df['Overall_Final_Label'].apply(lambda x: 0 if x == config.normal_value else 1).values
    final_report = classification_report(y_true, final_predictions, output_dict=True, zero_division=0)
    print(f"Final accuracy: {final_report['accuracy']:.1%}")

    # === Apply isotonic calibration if available ===
    calibrator_path = f"models/{config.name}/sensor_confidence_calibration_isotonic.pkl"
    if os.path.exists(calibrator_path):
        try:
            sensor_iso = joblib.load(calibrator_path)
            print(f"✅ Loaded isotonic calibrator from {calibrator_path}")

            # Apply calibration to each sensor confidence
            for s in config.sensor_cols:
                if f"{s}_Confidence_Score" in df.columns:
                    raw_conf = df[f"{s}_Confidence_Score"].values
                    calibrated = sensor_iso.transform(raw_conf)
                    df[f"{s}_Confidence_Score"] = np.clip(calibrated, 0.01, 0.99)

            # Also apply to overall score
            if "Overall_Confidence_Score" in df.columns:
                raw_conf = df["Overall_Confidence_Score"].values
                calibrated = sensor_iso.transform(raw_conf)
                df["Overall_Confidence_Score"] = np.clip(calibrated, 0.01, 0.99)

            print("✅ Sensor confidence scores calibrated")
        except Exception as e:
            print(f"⚠️ Calibration failed: {e}")


    # Save results
    output_file = f"{output_dir}/{config.name}_sensor_labeled_data.csv"
    df.to_csv(output_file, index=False)
    
    # Save ensemble summary
    base_summary = {
        'detection_strategy': strategy,
        'optimal_sensitivity': optimal_sensitivity if strategy == "single_optimal" else "ensemble",
        'consensus_threshold': 1.0,
        'sensitivity_weights': {'high': 1.0, 'medium': 1.5, 'low': 2.0},
        'consensus_stats': {
            'full_agreement': int(ensemble_results['consensus_stats']['full_agreement']),
            'partial_agreement': int(ensemble_results['consensus_stats']['partial_agreement']),
            'no_agreement': int(ensemble_results['consensus_stats']['no_agreement'])
        },
        'final_performance': {
            'anomaly_count': int(np.sum(predictions)),
            'detection_rate': float(np.mean(predictions)),
            'accuracy': float(detection_report['accuracy'])
        }
    }

    # Add individual performance only if using ensemble
    if strategy == "ensemble":
        base_summary['individual_performance'] = {
            sens: {
                'anomaly_count': int(ensemble_results['sensitivity_details'][sens]['anomaly_count']),
                'detection_rate': float(ensemble_results['sensitivity_details'][sens]['detection_rate'])
            }
            for sens in ['high', 'medium', 'low']
        }
    else:
        base_summary['individual_performance'] = {
            optimal_sensitivity: {
                'anomaly_count': int(np.sum(predictions)),
                'detection_rate': float(np.mean(predictions))
            }
        }

    ensemble_summary = base_summary
    ensemble_summary_file = f"{output_dir}/{config.name}_ensemble_summary.json"
    with open(ensemble_summary_file, 'w') as f:
        json.dump(ensemble_summary, f, indent=2)
    
    print(f"Saved results to {output_file}")
    print(f"Saved summary to {ensemble_summary_file}")

    return df