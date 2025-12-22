# deployment/retraining/model_trainer.py

import os
import sys
from datetime import datetime
from pathlib import Path

# Add training modules to path
training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'training'))
if training_path not in sys.path:
    sys.path.insert(0, training_path)

from non_vision_subsystem.adaptive_autoencoder import AdaptiveUnsupervisedAutoencoder


class ModelTrainer:
    """
    Simple model trainer - uses only pickle files, no complex fallback logic.
    """
    
    def __init__(self, config):
        self.config = config
        
    def retrain_model(self, training_data):
        """
        Train new model with collected data, matching latest training logic.
        """
        try:
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
            # Import cleaning config from training/config/config_manager.py
            from training.config.config_manager import get_cleaning_config
            from training.non_vision_subsystem.adaptive_autoencoder import (
                comprehensive_training_cleaning, validate_sensor_ranges, get_default_sensor_ranges
            )

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            print(f"Training model with {len(training_data)} samples...")

            # --- 1. Prepare label column ---
            sensor_cols = self.config.use_case_config.sensor_cols
            # Convert string labels to binary (0 for Normal, 1 for Anomaly)
            if 'Label' in training_data.columns:
                training_data['Label'] = (training_data['Label'] == 'Anomaly').astype(int)
            else:
                training_data['Label'] = 0

            # Debug: Show label value counts before split
            print("[DEBUG] Label value counts before split:", training_data['Label'].value_counts())

            # --- 2. Split train/val so validation can include anomalies ---
            train_df, val_df = train_test_split(training_data, test_size=0.15, stratify=training_data['Label'], random_state=42)
            print(f"[DEBUG] Training set: {len(train_df)} samples, Validation set: {len(val_df)} samples")

            # --- 3. Filter training set to normal only ---
            normal_mask = train_df['Label'] == 0
            X_train = train_df[normal_mask][sensor_cols].values
            print(f"Filtered training to {len(X_train)} normal samples (was {len(train_df)})")

            # --- 4. Validation set can include anomalies ---
            X_val = val_df[sensor_cols].values
            y_val = val_df['Label'].values

            # --- 5. Apply comprehensive cleaning (disable statistical outlier cleaning) ---
            cleaning_config = get_cleaning_config(self.config)
            sensor_ranges = cleaning_config.get('sensor_ranges') or get_default_sensor_ranges(sensor_cols)
            enable_cleaning = cleaning_config.get('enable_training_data_cleaning', True)
            # Force disable statistical outlier cleaning
            enable_statistical_cleaning = False
            if enable_cleaning:
                X_train, cleaning_report = comprehensive_training_cleaning(
                    X_train, sensor_cols, sensor_ranges,
                    cleaning_config.get('enable_range_validation', True),
                    enable_statistical_cleaning,
                    cleaning_config.get('statistical_outlier_method', 'mad'),
                    cleaning_config.get('statistical_outlier_threshold', 5)
                )
            else:
                cleaning_report = {'enabled': False}

            print(f"✅ Data cleaning completed! Shape: {X_train.shape}")

            # --- 4. Train model ---
            new_detector = AdaptiveUnsupervisedAutoencoder(
                sensor_names=sensor_cols,
                case_name=self.config.use_case
            )
            new_detector.cleaning_config = {
                'enabled': enable_cleaning,
                'sensor_ranges': sensor_ranges,
                'cleaning_report': cleaning_report
            }
            new_detector.learn_environment(X_train)


            # --- 5. Calibrate thresholds on validation set ---
            # Always recalculate reconstruction errors with the new model for thresholding/calibration
            print("[DEBUG] Recalculating reconstruction errors with the retrained model...")
            X_val_scaled = new_detector.scaler.transform(X_val)
            reconstructions = new_detector.autoencoder.predict(X_val_scaled)
            if X_val_scaled.shape[1] == 1 and reconstructions.ndim == 1:
                reconstructions = reconstructions.reshape(-1, 1)
            reconstruction_errors = (X_val_scaled - reconstructions) ** 2
            global_errors = np.mean(reconstruction_errors, axis=1)
            print(f"[DEBUG] Validation set reconstruction error stats (after retrain): min={np.min(global_errors):.6f}, max={np.max(global_errors):.6f}, mean={np.mean(global_errors):.6f}")
            # Optionally, print a small sample of errors for manual inspection
            print(f"[DEBUG] Sample of recalculated global_errors: {global_errors[:10]}")
            # Calibrate thresholds using the new errors
            new_detector.calibrate_thresholds_on_validation(X_val, y_val)
            # Print the threshold used (if available)
            if hasattr(new_detector, 'global_threshold'):
                print(f"[DEBUG] Calibrated global threshold: {getattr(new_detector, 'global_threshold', None)}")

            # --- 9b. Update error distributions for confidence estimation ---
            import json
            error_dist_path = Path(f"models/{self.config.use_case}/{self.config.use_case}_error_distributions.json")
            backup_dir = Path(f"models/{self.config.use_case}/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            # Backup old error distribution file if it exists
            if error_dist_path.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = backup_dir / f"{self.config.use_case}_error_distributions_{timestamp}.json"
                error_dist_path.replace(backup_path)
                print(f"[DEBUG] Backed up old error distribution to {backup_path}")

            # Calculate new error stats for each sensor
            sensor_error_stats = {}
            for i, sensor in enumerate(sensor_cols):
                sensor_errors = reconstruction_errors[:, i]
                sensor_error_stats[sensor] = {
                    'type': 'gaussian',
                    'mean': float(np.mean(sensor_errors)),
                    'std': float(np.std(sensor_errors))
                }

            with open(error_dist_path, 'w') as f:
                json.dump(sensor_error_stats, f, indent=4)
            print(f"[DEBUG] Updated error distributions saved to {error_dist_path}")
            print(f"[DEBUG] New error distribution stats:")
            for sensor, stats in sensor_error_stats.items():
                print(f"   {sensor}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")

            # --- 6. Multi-sensitivity evaluation ---
            sensitivities = ['low', 'medium', 'high']
            sensitivity_results = {}
            print("\n🎯 Multi-sensitivity evaluation:")
            for sensitivity in sensitivities:
                val_preds, _, _ = new_detector.predict_anomalies_home_optimized(
                    X_val, adapt=False, sensitivity=sensitivity, verbose=False
                )
                val_report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
                print(f"[DEBUG] val_report keys: {list(val_report.keys())}")
                try:
                    val_normal_f1 = val_report.get('0', {}).get('f1-score', 0)
                    print(f"[DEBUG] val_normal_f1: {val_normal_f1}")
                    val_anomaly_f1 = val_report.get('1', {}).get('f1-score', 0)
                    print(f"[DEBUG] val_anomaly_f1: {val_anomaly_f1}")
                    val_normal_precision = val_report.get('0', {}).get('precision', 0)
                    print(f"[DEBUG] val_normal_precision: {val_normal_precision}")
                    val_anomaly_precision = val_report.get('1', {}).get('precision', 0)
                    print(f"[DEBUG] val_anomaly_precision: {val_anomaly_precision}")
                    val_normal_recall = val_report.get('0', {}).get('recall', 0)
                    print(f"[DEBUG] val_normal_recall: {val_normal_recall}")
                    val_anomaly_recall = val_report.get('1', {}).get('recall', 0)
                    print(f"[DEBUG] val_anomaly_recall: {val_anomaly_recall}")
                    balance_score = min(val_normal_f1, val_anomaly_f1)
                    print(f"[DEBUG] balance_score: {balance_score}")
                    macro_f1 = val_report['macro avg']['f1-score']
                    print(f"[DEBUG] macro_f1: {macro_f1}")
                    overall_score = (macro_f1 + balance_score) / 2
                    print(f"[DEBUG] overall_score: {overall_score}")
                except Exception as debug_e:
                    print(f"[DEBUG] Exception in metric extraction: {debug_e}")
                    raise
                sensitivity_results[sensitivity] = {
                    'val_accuracy': val_report['accuracy'],
                    'val_macro_f1': macro_f1,
                    'val_balance_score': balance_score,
                    'overall_score': overall_score,
                    'val_normal_precision': val_normal_precision,
                    'val_anomaly_precision': val_anomaly_precision,
                    'val_normal_recall': val_normal_recall,
                    'val_anomaly_recall': val_anomaly_recall
                }
                print(f"   {sensitivity.upper()}: Acc={val_report['accuracy']:.3f}, MacroF1={macro_f1:.3f}, Balance={balance_score:.3f}, NormalPrec={val_normal_precision:.3f}, AnomPrec={val_anomaly_precision:.3f}, NormalRec={val_normal_recall:.3f}, AnomRec={val_anomaly_recall:.3f}")

            # --- 7. Auto-select best sensitivity ---
            best_sensitivity = max(sensitivity_results.keys(), key=lambda k: sensitivity_results[k]['overall_score'])
            best_score = sensitivity_results[best_sensitivity]['overall_score']
            print(f"\n✅ Auto-selected: {best_sensitivity.upper()} (Score: {best_score:.3f})")

            # --- 8. Save model consistently ---
            models_dir = Path(f"models/{self.config.use_case}")
            models_dir.mkdir(parents=True, exist_ok=True)
            new_model_path = models_dir / f"retrained_adaptive_{self.config.use_case}"
            new_detector.save_adaptive_model(str(new_model_path))
            if not os.path.exists(f"{new_model_path}.pkl"):
                raise Exception("Model file was not created")
            print(f"💾 Model saved: {new_model_path}")

            # --- 9. Retrain isotonic regression for confidence calibration ---
            try:
                from sklearn.isotonic import IsotonicRegression
                import joblib
                # Use the recalculated global_errors for isotonic regression
                iso_model = IsotonicRegression(out_of_bounds='clip')
                iso_model.fit(global_errors, y_val)

                # Backup old isotonic model if exists
                iso_model_path = Path(f"models/{self.config.use_case}/sensor_isotonic.pkl")
                backup_dir = Path(f"models/{self.config.use_case}/backups")
                backup_dir.mkdir(parents=True, exist_ok=True)
                if iso_model_path.exists():
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_path = backup_dir / f"sensor_isotonic_{timestamp}.pkl"
                    iso_model_path.replace(backup_path)
                    print(f"Backed up old isotonic model to {backup_path}")

                # Save new isotonic model
                joblib.dump(iso_model, iso_model_path)
                print(f"💾 Isotonic regression model saved: {iso_model_path}")
            except Exception as iso_e:
                print(f"⚠️ Isotonic regression retraining failed: {iso_e}")

            # --- 10. Return results ---
            return {
                'success': True,
                'new_model': new_detector,
                'model_path': str(new_model_path),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'best_sensitivity': best_sensitivity,
                'sensitivity_results': sensitivity_results,
                'cleaning_report': cleaning_report
            }

        except Exception as e:
            import traceback
            print(f"Model training failed: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'reason': str(e),
                'training_samples': len(training_data) if training_data is not None else 0
            }
    
    def load_current_model(self):
        """Load current production model"""
        try:
            # Look for adaptive model in the deployment models directory
            model_path = self.config.sensor_model_path  # deployment/models/door
            
            # Try to find the base adaptive model
            base_model_path = os.path.join(model_path, f"adaptive_{self.config.use_case}")
            
            if os.path.exists(f"{base_model_path}.pkl"):
                detector = AdaptiveUnsupervisedAutoencoder(
                    sensor_names=self.config.use_case_config.sensor_cols,
                    case_name=self.config.use_case
                )
                
                detector.load_adaptive_model(base_model_path)
                print(f"Loaded current model from: {base_model_path}")
                return detector
            else:
                print(f"No current model found at: {base_model_path}.pkl")
                return None
            
        except Exception as e:
            print(f"Failed to load current model: {e}")
            return None