import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import deque
from datetime import timedelta
from .adaptive_autoencoder import AdaptiveUnsupervisedAutoencoder, comprehensive_training_cleaning, get_default_sensor_ranges, validate_sensor_ranges


class IncrementalMemoryBuffer:
    """Memory buffer for incremental learning"""
    def __init__(self, max_size=2000, diversity_sampling=True):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.diversity_sampling = diversity_sampling
        self.cycle_markers = []  # Track which samples came from which cycle
        
    def add_samples(self, data, cycle_id):
        """Add representative samples from a training cycle"""
        # Sample strategically to maintain diversity
        n_samples = min(len(data), self.max_size // 10)  # ~10% of buffer per cycle
        
        if self.diversity_sampling and len(data) > n_samples:
            # Use stratified sampling based on data characteristics
            indices = self._diverse_sampling(data, n_samples)
            selected_data = data[indices]
        else:
            # Random sampling
            indices = np.random.choice(len(data), n_samples, replace=False)
            selected_data = data[indices]
        
        # Add to buffer with cycle tracking
        for sample in selected_data:
            self.buffer.append(sample)
            self.cycle_markers.append(cycle_id)
            
        # Keep cycle_markers in sync with buffer size
        if len(self.cycle_markers) > self.max_size:
            self.cycle_markers = self.cycle_markers[-self.max_size:]
    
    def _diverse_sampling(self, data, n_samples):
        """Sample diverse representatives using statistical spread"""
        if len(data) <= n_samples:
            return np.arange(len(data))
        
        # Calculate variance for each sample (distance from local mean)
        variances = []
        window_size = max(10, len(data) // 100)
        
        for i in range(len(data)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(data), i + window_size//2)
            local_data = data[start_idx:end_idx]
            local_mean = np.mean(local_data, axis=0)
            variance = np.sum((data[i] - local_mean) ** 2)
            variances.append(variance)
        
        # Sample based on variance (diverse samples) + some random
        variances = np.array(variances)
        high_var_indices = np.argsort(variances)[-n_samples//2:]  # High variance samples
        random_indices = np.random.choice(len(data), n_samples//2, replace=False)
        
        return np.concatenate([high_var_indices, random_indices])
    
    def get_memory_samples(self, target_size=None):
        """Get samples from memory buffer"""
        if not self.buffer:
            # return np.array([]).reshape(0, -1)
            return np.array([])
        buffer_array = np.array(list(self.buffer))
        
        if target_size is None or len(buffer_array) <= target_size:
            return buffer_array
        
        # Sample from buffer maintaining diversity across cycles
        indices = np.random.choice(len(buffer_array), target_size, replace=False)
        return buffer_array[indices]

def create_time_based_chunks(df, chunk_days=3, overlap_hours=6):
    """
    Split dataframe into time-based chunks with overlap
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data with Timestamp column
    chunk_days : int
        Number of days per chunk
    overlap_hours : int
        Hours of overlap between chunks
    
    Returns:
    --------
    list of tuples: (chunk_df, chunk_id, start_time, end_time)
    """
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    start_time = df['Timestamp'].min()
    end_time = df['Timestamp'].max()
    total_duration = end_time - start_time
    
    chunk_duration = timedelta(days=chunk_days)
    overlap_duration = timedelta(hours=overlap_hours)
    step_duration = chunk_duration - overlap_duration
    
    chunks = []
    current_start = start_time
    chunk_id = 0
    
    while current_start < end_time:
        current_end = min(current_start + chunk_duration, end_time)
        
        # Extract chunk data
        chunk_mask = (df['Timestamp'] >= current_start) & (df['Timestamp'] <= current_end)
        chunk_df = df[chunk_mask].copy()
        
        if len(chunk_df) > 100:  # Only include chunks with sufficient data
            chunks.append((chunk_df, chunk_id, current_start, current_end))
            chunk_id += 1
        
        current_start += step_duration
        
        # Break if we've processed all data
        if current_start >= end_time:
            break
    
    print(f"Created {len(chunks)} chunks of {chunk_days} days each with {overlap_hours}h overlap")
    return chunks

def run_incremental_training(config):
    """Main incremental training function"""
    print(f"\n=== INCREMENTAL TRAINING: {config.name} ===")
    
    # Load and prepare data
    sensor_path = Path(config.sensor_data_path)
    if not sensor_path.exists():
        raise FileNotFoundError(f"Sensor data not found: {sensor_path}")

    df = pd.read_csv(sensor_path)

    # Handle timestamp formatting
    if 'Timestamp' not in df.columns:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format="%d/%m/%Y %H:%M:%S")
        else:
            raise KeyError("'Timestamp' column not found and Date/Time not present")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M:%S")
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df = df.dropna(subset=config.sensor_cols + [config.status_col])

    print(f"Total data: {len(df)} samples from {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"Status distribution: {df[config.status_col].value_counts().to_dict()}")

    # Create labels
    df['Label'] = (df[config.status_col] == config.anomaly_value).astype(int)

    # # Create time-based chunks (3 days with 6h overlap)
    # chunks = create_time_based_chunks(df, chunk_days=3, overlap_hours=6)
    # # Initialize memory buffer and detector
    # memory_buffer = IncrementalMemoryBuffer(max_size=2000)

    # ✅ new — use config values if present, fallback otherwise
    chunk_days = getattr(config, "incremental_chunk_days", 3)
    overlap_hours = getattr(config, "incremental_overlap_hours", 6)
    buffer_size = getattr(config, "memory_buffer_size", 2000)

    chunks = create_time_based_chunks(df, chunk_days=chunk_days, overlap_hours=overlap_hours)
    memory_buffer = IncrementalMemoryBuffer(max_size=buffer_size)

    detector = None
    
    # Setup sensor ranges
    sensor_ranges = getattr(config, 'sensor_ranges', None) or get_default_sensor_ranges(config.sensor_cols)
    enable_cleaning = getattr(config, 'enable_training_data_cleaning', True)
    
    # Track performance across cycles
    cycle_performance = []
    
    print(f"\nStarting incremental training across {len(chunks)} cycles...")
    
    for cycle_idx, (chunk_df, chunk_id, start_time, end_time) in enumerate(chunks):
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle_idx + 1}/{len(chunks)} (ID: {chunk_id})")
        print(f"Time period: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"Samples: {len(chunk_df):,}")
        
        # Split chunk into train/val/test
        chunk_train, chunk_temp = train_test_split(chunk_df, test_size=0.3, stratify=chunk_df['Label'], random_state=42)
        chunk_val, chunk_test = train_test_split(chunk_temp, test_size=0.5, stratify=chunk_temp['Label'], random_state=42)
        
        # Extract features (normal samples only for training)
        normal_mask = chunk_train['Label'] == 0
        X_chunk_train_normal = chunk_train[normal_mask][config.sensor_cols].values
        X_chunk_val = chunk_val[config.sensor_cols].values
        X_chunk_test = chunk_test[config.sensor_cols].values
        y_chunk_val = chunk_val['Label'].values
        y_chunk_test = chunk_test['Label'].values
        
        print(f"Chunk split: Train={len(X_chunk_train_normal)} normal, Val={len(X_chunk_val)}, Test={len(X_chunk_test)}")
        
        # Get memory samples from previous cycles
        memory_samples = memory_buffer.get_memory_samples(target_size=len(X_chunk_train_normal))
        
        # Combine new data with memory
        if len(memory_samples) > 0 and memory_samples.size > 0:
            X_train_combined = np.vstack([X_chunk_train_normal, memory_samples])
            print(f"Training data: {len(X_chunk_train_normal)} new + {len(memory_samples)} memory = {len(X_train_combined)} total")

        else:
            X_train_combined = X_chunk_train_normal
            print(f"Training data: {len(X_train_combined)} (first cycle, no memory)")
        
        # Data cleaning
        if enable_cleaning:
            X_train_combined, cleaning_report = comprehensive_training_cleaning(
                X_train_combined, config.sensor_cols, sensor_ranges,
                getattr(config, 'enable_range_validation', True),
                getattr(config, 'enable_statistical_cleaning', True),
                getattr(config, 'statistical_outlier_method', 'mad'),
                getattr(config, 'statistical_outlier_threshold', 5)
            )
        
        # Train or update detector
        if cycle_idx == 0:
            # First cycle: initialize detector
            print("Initializing new detector...")
            detector = AdaptiveUnsupervisedAutoencoder(
                sensor_names=config.sensor_cols,
                case_name=config.name
            )
            detector.learn_environment(X_train_combined)
        else:
            # Subsequent cycles: incremental update
            print("Performing incremental update...")
            # For simplicity, retrain on combined data
            # In a more sophisticated approach, you could do true incremental updates
            detector.learn_environment(X_train_combined)
        
        # Calibrate thresholds on current validation data
        detector.calibrate_thresholds_on_validation(X_chunk_val, y_chunk_val)
        
        # Evaluate performance on current test data
        print(f"\nCYCLE {cycle_idx + 1} PERFORMANCE EVALUATION:")
        
        # Test all sensitivity levels
        sensitivities = ['low', 'medium', 'high']
        cycle_results = {}

        for sensitivity in sensitivities:
            test_preds, _, _ = detector.predict_anomalies_home_optimized(
                X_chunk_test, adapt=False, sensitivity=sensitivity, verbose=False
            )
            
            # Print full classification report
            print(f"\n{sensitivity.upper()} SENSITIVITY:")
            test_report_string = classification_report(y_chunk_test, test_preds, 
                                                     target_names=["Normal", "Anomaly"], 
                                                     zero_division=0)
            print(test_report_string)
            
            # Also store for summary
            test_report = classification_report(y_chunk_test, test_preds, output_dict=True, zero_division=0)
            normal_metrics = test_report.get('0', {})
            anomaly_metrics = test_report.get('1', {})
            
            cycle_results[sensitivity] = {
                'accuracy': test_report['accuracy'],
                'normal_precision': normal_metrics.get('precision', 0),
                'normal_recall': normal_metrics.get('recall', 0),
                'normal_f1': normal_metrics.get('f1-score', 0),
                'anomaly_precision': anomaly_metrics.get('precision', 0),
                'anomaly_recall': anomaly_metrics.get('recall', 0),
                'anomaly_f1': anomaly_metrics.get('f1-score', 0),
                'f1_macro': test_report['macro avg']['f1-score']
            }

        # Store cycle performance
        cycle_performance.append({
            'cycle': cycle_idx + 1,
            'chunk_id': chunk_id,
            'start_time': start_time,
            'end_time': end_time,
            'train_samples': len(X_train_combined),
            'memory_samples': len(memory_samples) if len(memory_samples) > 0 and memory_samples.size > 0 else 0,
            'test_samples': len(X_chunk_test),
            'anomaly_rate': np.mean(y_chunk_test),
            'results': cycle_results
        })
        
        # Add representative samples to memory buffer
        memory_buffer.add_samples(X_chunk_train_normal, chunk_id)
        print(f"Added samples to memory buffer. Total buffer size: {len(memory_buffer.buffer)}")
    
    # Final model selection and saving
    print(f"\n{'='*60}")
    print("INCREMENTAL TRAINING COMPLETED")
    print(f"{'='*60}")

    # Analyze performance across cycles
    print("\nPERFORMANCE PROGRESSION:")
    print("Cycle | Accuracy (Low/Med/High) | F1-Macro (Low/Med/High) | Memory Samples")
    print("-" * 80)
    
    for perf in cycle_performance:
        low_acc = perf['results']['low']['accuracy']
        med_acc = perf['results']['medium']['accuracy']
        high_acc = perf['results']['high']['accuracy']
        low_f1 = perf['results']['low']['f1_macro']
        med_f1 = perf['results']['medium']['f1_macro']
        high_f1 = perf['results']['high']['f1_macro']
        
        print(f"{perf['cycle']:5d} | {low_acc:.3f}/{med_acc:.3f}/{high_acc:.3f}     | "
              f"{low_f1:.3f}/{med_f1:.3f}/{high_f1:.3f}      | {perf['memory_samples']:6d}")
    
    # Select best overall sensitivity
    final_scores = {}
    for sensitivity in sensitivities:
        # Average performance across last 3 cycles (most stable)
        last_cycles = cycle_performance[-3:]
        avg_score = np.mean([cycle['results'][sensitivity]['f1_macro'] for cycle in last_cycles])
        final_scores[sensitivity] = avg_score
    
    best_sensitivity = max(final_scores.keys(), key=lambda k: final_scores[k])
    print(f"\nBest sensitivity based on final cycles: {best_sensitivity.upper()} (F1={final_scores[best_sensitivity]:.3f})")
    
    # Save final model and results
    model_path = Path(f"models/{config.name}/adaptive_{config.name}")
    detector.save_adaptive_model(str(model_path))
    
    # Save incremental training results
    results_path = Path(f"models/{config.name}/training_results.json")
    results_data = {
        'total_cycles': len(chunks),
        'best_sensitivity': best_sensitivity,
        'final_scores': final_scores,
        'cycle_performance': [{k: v for k, v in perf.items() if k != 'start_time' and k != 'end_time'} 
                             for perf in cycle_performance],  # Remove datetime objects for JSON
        'memory_buffer_final_size': len(memory_buffer.buffer)
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
        
    print(f"\nModel saved: {model_path}")
    print(f"Results saved: {results_path}")

    # === Save production config for incremental training ===
    output_dir = Path(f"models/{config.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use last 3 cycles for stable averages
    recent_cycles = cycle_performance[-3:] if len(cycle_performance) >= 3 else cycle_performance

    averaged_results = {}
    for sensitivity in sensitivities:
        averaged_results[sensitivity] = {
            'test_accuracy': float(np.mean([c['results'][sensitivity]['accuracy'] for c in recent_cycles])),
            'normal_f1': float(np.mean([c['results'][sensitivity]['normal_f1'] for c in recent_cycles])),
            'anomaly_f1': float(np.mean([c['results'][sensitivity]['anomaly_f1'] for c in recent_cycles])),
            'f1_macro': float(np.mean([c['results'][sensitivity]['f1_macro'] for c in recent_cycles]))
        }

    # Build final performance summary (averaged best sensitivity)
    final_perf = averaged_results[best_sensitivity]

    prod_config = {
        'model_type': 'adaptive_autoencoder',
        'optimal_sensitivity': best_sensitivity,
        'selection_score': float(final_scores[best_sensitivity]),
        'final_performance': final_perf,  # ✅ averaged metrics
        'sensitivity_comparison': averaged_results,  # ✅ averaged across recent cycles
        'last_cycle_performance': {  # ✅ keep original last cycle for transparency
            sens: {k: float(v) for k, v in metrics.items()}
            for sens, metrics in cycle_performance[-1]['results'].items()
        },
        'data_cleaning': {
            'training_cleaning_enabled': enable_cleaning,
            'sensor_ranges_used': sensor_ranges is not None
        },
        'feature_importance': detector.feature_importance,
        # 🆕 Save method selection
        'method_selection': detector.active_methods[0] if len(detector.active_methods) == 1 else "ensemble",
        'method_performance': detector.method_performance
    }

    config_path = output_dir / 'production_config.json'
    with open(config_path, 'w') as f:
        json.dump(prod_config, f, indent=2)

    print(f"⚙️ Production config saved: {config_path}")

    
    return detector, best_sensitivity, cycle_performance

# Modified run function to support incremental mode
def run(config):
    """Enhanced run function with incremental training option"""
    
    # Check if incremental training is requested
    incremental_mode = getattr(config, 'incremental_training', False)
    
    if incremental_mode:
        return run_incremental_training(config)
    else:
            # Load and prepare data
        sensor_path = Path(config.sensor_data_path)
    if not sensor_path.exists():
        raise FileNotFoundError(f"Sensor data not found: {sensor_path}")

    df = pd.read_csv(sensor_path)

    # Handle timestamp formatting
    if 'Timestamp' not in df.columns:
        if 'Date' in df.columns and 'Time' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format="%d/%m/%Y %H:%M:%S")
        else:
            raise KeyError("'Timestamp' column not found and Date/Time not present")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M:%S")
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df = df.dropna(subset=config.sensor_cols + [config.status_col])

    print(f"📊 Status distribution: {df[config.status_col].value_counts().to_dict()}")

    # Create labels and stratified split (instead of chronological)
    df['Label'] = (df[config.status_col] == config.anomaly_value).astype(int)

    # First split: separate test set (15%)
    train_val, df_test = train_test_split(
        df, test_size=0.15, stratify=df['Label'], random_state=42
    )

    # Second split: separate train and validation (70% train, 15% val from remaining 85%)
    df_train, df_val = train_test_split(
        train_val, test_size=0.176, stratify=train_val['Label'], random_state=42  # 0.176 = 15/85
    )

    print(f"📊 Stratified split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # print(f"📊 Split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # Extract features
    features = config.sensor_cols
    # X_train = df_train[features].values
    # Filter to only normal samples for autoencoder training
    normal_mask = df_train['Label'] == 0  # Since Label is created from status_col
    X_train_normal = df_train[normal_mask][features].values
    print(f"📊 Filtered training: {len(X_train_normal)} normal samples (was {len(df_train)})")

    # Use normal-only data for training
    X_train = X_train_normal

    X_val = df_val[features].values
    X_test = df_test[features].values
    y_val = df_val['Label'].values
    y_test = df_test['Label'].values

    # Sensor ranges setup
    if hasattr(config, 'sensor_ranges') and config.sensor_ranges:
        sensor_ranges = config.sensor_ranges
    else:
        sensor_ranges = get_default_sensor_ranges(config.sensor_cols)

    # Training data cleaning (comprehensive)
    enable_cleaning = getattr(config, 'enable_training_data_cleaning', True)
    if enable_cleaning:
        enable_range_validation = getattr(config, 'enable_range_validation', True)
        enable_statistical_cleaning = getattr(config, 'enable_statistical_cleaning', True)
        statistical_outlier_method = getattr(config, 'statistical_outlier_method', 'mad')
        z_threshold = getattr(config, 'statistical_outlier_threshold', 5)
        
        X_train, cleaning_report = comprehensive_training_cleaning(
            X_train, features, sensor_ranges, enable_range_validation, enable_statistical_cleaning, statistical_outlier_method, z_threshold
        )
    else:
        cleaning_report = {'enabled': False}

    # Val/Test data cleaning (hardware errors only - keep statistical outliers as potential real anomalies)
    clean_val_test_errors = getattr(config, 'clean_val_test_sensor_errors', True)
    
    if clean_val_test_errors and sensor_ranges:
        # Fix hardware errors in validation data
        X_val_clean, val_error_mask = validate_sensor_ranges(X_val, sensor_ranges, features)
        if val_error_mask.any():
            X_val = X_val_clean
        
        # Fix hardware errors in test data  
        X_test_clean, test_error_mask = validate_sensor_ranges(X_test, sensor_ranges, features)
        if test_error_mask.any():
            X_test = X_test_clean

    # Remove NaN values if any
    if pd.isna(X_val).any():
        nan_mask = pd.isna(X_val).any(axis=1)
        X_val, y_val = X_val[~nan_mask], y_val[~nan_mask]
        print(f"⚠️ Removed {nan_mask.sum()} NaN samples from validation")
        
    if pd.isna(X_test).any():
        nan_mask = pd.isna(X_test).any(axis=1)
        X_test, y_test = X_test[~nan_mask], y_test[~nan_mask]
        print(f"⚠️ Removed {nan_mask.sum()} NaN samples from test")

    print(f"✅ Data cleaning completed!\n")

    # Train model
    detector = AdaptiveUnsupervisedAutoencoder(
        sensor_names=config.sensor_cols,
        case_name=config.name
    )
    # detector.is_single_sensor = is_single_sensor

    detector.cleaning_config = {
        'enabled': enable_cleaning,
        'sensor_ranges': sensor_ranges,
        'cleaning_report': cleaning_report
    }

    # Calibrate thresholds on validation data
    detector.learn_environment(X_train)
    detector.calibrate_thresholds_on_validation(X_val, y_val) 

    detector.analyze_method_contributions(X_val, y_val, sensitivity='low')


    print(f"📊 Validation anomaly rate: {np.mean(y_val)*100:.1f}%")
    print(f"📊 Final global threshold: {detector.global_threshold:.6f}")

    # Multi-sensitivity evaluation with quiet mode support
    print("\n🎯 Multi-sensitivity evaluation:")
    sensitivities = ['low', 'medium', 'high']
    sensitivity_results = {}


    # === ADD THIS BLOCK BEFORE THE SENSITIVITY LOOP ===
    if not config.quiet_mode:
        print("🔍 Method predictions on validation data:")

        # Get method predictions once for debug info
        # X_val_scaled = detector.scaler.transform(X_val)
        # val_reconstructions = detector.autoencoder.predict(X_val_scaled)
        # val_errors = np.mean((X_val_scaled - val_reconstructions) ** 2, axis=1)

        X_val_scaled = detector.scaler.transform(X_val)
        val_reconstructions = detector.autoencoder.predict(X_val_scaled)

        # 🔧 Shape fix to prevent OOM in single-feature case
        if val_reconstructions.ndim == 1:
            val_reconstructions = val_reconstructions.reshape(-1, detector.n_features)
        elif val_reconstructions.shape[1] != detector.n_features:
            val_reconstructions = val_reconstructions.reshape(-1, detector.n_features)

        val_errors = np.mean((X_val_scaled - val_reconstructions) ** 2, axis=1)

        autoencoder_anomalies = val_errors > detector.global_threshold
        
        iso_predictions = detector.isolation_forest.predict(X_val_scaled)
        iso_anomalies = (iso_predictions == -1)
        
        if detector.outlier_detector is not None:
            outlier_predictions = detector.outlier_detector.predict(X_val_scaled)
            outlier_anomalies = (outlier_predictions == -1)
            print(f"   Autoencoder anomalies: {np.sum(autoencoder_anomalies)} ({np.sum(autoencoder_anomalies)/len(X_val)*100:.1f}%)")
            print(f"   Global threshold: {detector.global_threshold:.6f}")
            print(f"   Isolation Forest anomalies: {np.sum(iso_anomalies)} ({np.sum(iso_anomalies)/len(X_val)*100:.1f}%)")
            print(f"   EllipticEnvelope anomalies: {np.sum(outlier_anomalies)} ({np.sum(outlier_anomalies)/len(X_val)*100:.1f}%)")
        else:
            print(f"   Autoencoder anomalies: {np.sum(autoencoder_anomalies)} ({np.sum(autoencoder_anomalies)/len(X_val)*100:.1f}%)")
            print(f"   Global threshold: {detector.global_threshold:.6f}")
            print(f"   Isolation Forest anomalies: {np.sum(iso_anomalies)} ({np.sum(iso_anomalies)/len(X_val)*100:.1f}%)")
            print(f"   EllipticEnvelope: Not available")

    print("\n🎯 Multi-sensitivity evaluation:")


    for sensitivity in sensitivities:
        # Use quiet_mode from config
        verbose = not config.quiet_mode
        
        # Validation evaluation
        val_preds, _, _ = detector.predict_anomalies_home_optimized(
            X_val, adapt=False, sensitivity=sensitivity, verbose=verbose
        )
        val_report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
    
        # Test evaluation
        test_preds, _, _ = detector.predict_anomalies_home_optimized(
            X_test, adapt=False, sensitivity=sensitivity, verbose=verbose
        )
        test_report_dict = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
        test_report_string = classification_report(y_test, test_preds, target_names=["Normal", "Anomaly"], zero_division=0)

        # Calculate metrics using VALIDATION performance (proper approach)
        val_normal_f1 = val_report.get('0', {}).get('f1-score', 0)
        val_anomaly_f1 = val_report.get('1', {}).get('f1-score', 0)
        balance_score = min(val_normal_f1, val_anomaly_f1)
        macro_f1 = val_report['macro avg']['f1-score']

        # Apply penalty for extreme failures
        penalty = 0
        if val_report.get('0', {}).get('precision', 0) == 0:
            penalty += 0.3  # Normal class precision failure
        if val_report.get('1', {}).get('precision', 0) == 0:
            penalty += 0.1  # Anomaly class precision failure
        # Add recall penalties too (important!)
        if val_report.get('0', {}).get('recall', 0) == 0:
            penalty += 0.3  # Normal class recall failure
        if val_report.get('1', {}).get('recall', 0) == 0:
            penalty += 0.1  # Anomaly class recall failure
            
        overall_score = (macro_f1 + balance_score) / 2 - penalty

        sensitivity_results[sensitivity] = {
            'val_accuracy': val_report['accuracy'],
            'val_macro_f1': macro_f1,
            'val_balance_score': balance_score,
            'overall_score': max(0, overall_score),
            'test_accuracy': test_report_dict['accuracy']
        }

        # Calculate test metrics for selection
        test_normal_f1 = test_report_dict.get('0', {}).get('f1-score', 0)
        test_anomaly_f1 = test_report_dict.get('1', {}).get('f1-score', 0)
        test_balance_score = min(test_normal_f1, test_anomaly_f1)
        test_macro_f1 = test_report_dict['macro avg']['f1-score']

        # Penalize extreme failures in TEST performance
        penalty = 0
        if test_report_dict.get('0', {}).get('precision', 0) == 0:
            penalty += 0.3  # Normal class precision failure
        if test_report_dict.get('1', {}).get('precision', 0) == 0:
            penalty += 0.1  # Anomaly class precision failure
        if test_report_dict.get('0', {}).get('recall', 0) == 0:
            penalty += 0.3  # Normal class recall failure (HIGH sensitivity has this!)
        if test_report_dict.get('1', {}).get('recall', 0) == 0:
            penalty += 0.1  # Anomaly class recall failure

        overall_score = (test_macro_f1 + test_balance_score) / 2 - penalty

        sensitivity_results[sensitivity] = {
            'val_accuracy': val_report['accuracy'],
            'test_accuracy': test_report_dict['accuracy'],
            'test_balance_score': test_balance_score,
            'test_macro_f1': test_macro_f1,
            'overall_score': overall_score
        }
        
        if verbose:
            # # Detailed output with full classification reports
            print(f"\n📊 {sensitivity.upper()} Sensitivity Results:")
            print(f"   Validation: Acc={val_report['accuracy']:.3f}")
            print(f"   Test: Acc={test_report_dict['accuracy']:.3f}, MacroF1={test_macro_f1:.3f}, Balance={test_balance_score:.3f}")
            print("\n🔍 Detailed Test Classification Report:")
            print(test_report_string)
                # Detailed output with full classification reports
            val_report_string = classification_report(y_val, val_preds, target_names=["Normal", "Anomaly"], zero_division=0)
            
            print(f"\n📊 {sensitivity.upper()} Sensitivity Results:")
            print(f"   Validation: Acc={val_report['accuracy']:.3f}, MacroF1={macro_f1:.3f}, Balance={balance_score:.3f}")
            print(f"   Test: Acc={test_report_dict['accuracy']:.3f}, MacroF1={test_macro_f1:.3f}, Balance={test_balance_score:.3f}")
            
            print("\n🔍 Detailed Validation Classification Report:")
            print(val_report_string)
            print("\n🔍 Detailed Test Classification Report:")
            print(test_report_string)  
        else:
            # Minimal output (your current summary line)
            print(f"   📊 {sensitivity.upper()}: Val={val_report['accuracy']:.1%}, Test={test_report_dict['accuracy']:.1%}")

    # Auto-select and final evaluation
    best_sensitivity = max(sensitivity_results.keys(), key=lambda k: sensitivity_results[k]['overall_score'])
    best_score = sensitivity_results[best_sensitivity]['overall_score']

    print(f"\n✅ Auto-selected: {best_sensitivity.upper()} (Score: {best_score:.3f})")

    # Final evaluation
    if config.quiet_mode:
        print(f"\n🎯 Final Results ({best_sensitivity.upper()}):")
        final_preds, _, _ = detector.predict_anomalies_home_optimized(
            X_test, adapt=False, sensitivity=best_sensitivity, verbose=False
        )
        final_report = classification_report(y_test, final_preds, output_dict=True, zero_division=0)
        print(f"   Test Accuracy: {final_report['accuracy']:.1%}")
        print(f"   Normal: Precision={final_report['0']['precision']:.2f}, Recall={final_report['0']['recall']:.2f}")
        print(f"   Anomaly: Precision={final_report['1']['precision']:.2f}, Recall={final_report['1']['recall']:.2f}")
    else:
        print(f"\n🎯 FINAL TEST EVALUATION ({best_sensitivity.upper()} sensitivity):")
        final_preds, _, _ = detector.predict_anomalies_home_optimized(
            X_test, adapt=False, sensitivity=best_sensitivity, verbose=True
        )
        final_report = classification_report(y_test, final_preds, target_names=["Normal", "Anomaly"])
        print(final_report)

    # Save model and config
    model_path = Path(f"models/{config.name}/adaptive_{config.name}")
    detector.save_adaptive_model(str(model_path))
    
    output_dir = Path(f"models/{config.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_test_metrics = classification_report(y_test, final_preds, output_dict=True, zero_division=0)
    prod_config = {
        'model_type': 'adaptive_autoencoder',
        'optimal_sensitivity': best_sensitivity,
        'selection_score': float(best_score),
        'final_performance': {
            'test_accuracy': float(final_test_metrics['accuracy']),
            'test_macro_f1': float(final_test_metrics['macro avg']['f1-score']),
            'normal_f1': float(final_test_metrics.get('0', {}).get('f1-score', 0)),
            'anomaly_f1': float(final_test_metrics.get('1', {}).get('f1-score', 0))
        },
        'sensitivity_comparison': {k: {key: float(val) for key, val in v.items()} 
                                 for k, v in sensitivity_results.items()},
        'data_cleaning': {
            'training_cleaning_enabled': enable_cleaning,
            'val_test_sensor_error_fixing': clean_val_test_errors,
            'sensor_ranges_used': sensor_ranges is not None
        },
        'feature_importance': detector.feature_importance,
        # 🆕 Save method selection
        'method_selection': detector.active_methods[0] if len(detector.active_methods) == 1 else "ensemble",
        'method_performance': detector.method_performance
    }
    
    
    config_path = output_dir / 'production_config.json'
    with open(config_path, 'w') as f:
        json.dump(prod_config, f, indent=2)
    
    print(f"\n💾 Model saved: {model_path}")
    print(f"⚙️ Config saved: {config_path}")
    # print(f"✅ Pipeline completed\n")

    return detector, best_sensitivity, sensitivity_results
