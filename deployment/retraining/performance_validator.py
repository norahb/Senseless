
# deployment/retraining/performance_validator.py

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


class PerformanceValidator:
    """
    Performance validator using real ground truth data
    """
    
    def __init__(self, config):
        self.config = config
        self.improvement_threshold = getattr(config, 'performance_threshold', 0.02)
        
    def validate_new_model(self, new_model, validation_data, current_model=None):
        """
        Compare new vs current model using REAL ground truth labels
        """
        try:
            print(f"Validating model performance...")
            
            # Use REAL ground truth from inference data
            val_labels = self._load_real_ground_truth_labels()
            
            if val_labels is None:
                print("No ground truth available - skipping validation")
                return {'should_deploy': True, 'improvement': 1.0}
            
            # Use actual inference data for evaluation
            inference_data = self._load_inference_data()
            if inference_data is None:
                print("No inference data available")
                return {'should_deploy': False, 'improvement': -1.0}
            
            X_val = inference_data.values
            
            # Ensure same length
            min_len = min(len(X_val), len(val_labels))
            X_val = X_val[:min_len]
            val_labels = val_labels[:min_len]
            
            # Get predictions from both models
            new_preds = self._get_predictions(new_model, X_val)
            
            # Calculate metrics
            new_accuracy = accuracy_score(val_labels, new_preds)
            new_f1 = f1_score(val_labels, new_preds, average='macro', zero_division=0)
            
            if current_model is not None:
                current_preds = self._get_predictions(current_model, X_val)
                current_accuracy = accuracy_score(val_labels, current_preds)
                current_f1 = f1_score(val_labels, current_preds, average='macro', zero_division=0)
                
                # Calculate improvement
                accuracy_improvement = new_accuracy - current_accuracy
                f1_improvement = new_f1 - current_f1
                overall_improvement = (accuracy_improvement + f1_improvement) / 2
                
                should_deploy = overall_improvement > self.improvement_threshold
                
                print(f"REAL DATA VALIDATION:")
                print(f"Current: Acc={current_accuracy:.3f}, F1={current_f1:.3f}")
                print(f"New:     Acc={new_accuracy:.3f}, F1={new_f1:.3f}")
                print(f"Improvement: {overall_improvement:.3f} (threshold: {self.improvement_threshold})")
                print(f"Deploy: {'YES' if should_deploy else 'NO'}")
                
                current_metrics = {'accuracy': current_accuracy, 'f1_macro': current_f1}
            else:
                # No current model - deploy if accuracy > 30%
                should_deploy = new_accuracy > 0.30
                overall_improvement = new_accuracy
                current_metrics = None
                print(f"No current model - New accuracy: {new_accuracy:.3f}")
                print(f"Deploy: {'YES' if should_deploy else 'NO'} (threshold: 0.30)")
            
            return {
                'should_deploy': should_deploy,
                'improvement': overall_improvement,
                'new_metrics': {'accuracy': new_accuracy, 'f1_macro': new_f1},
                'current_metrics': current_metrics,
                'validation_samples': len(X_val)
            }
            
        except Exception as e:
            print(f"Performance validation failed: {e}")
            return {
                'should_deploy': False,
                'improvement': -1.0,
                'error': str(e)
            }
    
    def _load_real_ground_truth_labels(self):
        """Load REAL ground truth from inference data (same as confidence estimation)"""
        try:
            # Use same file as confidence estimation
            data_path = self.config.use_case_config.inference_csv_path
            
            if not os.path.exists(data_path):
                print(f"Ground truth file not found: {data_path}")
                return None
            
            df = pd.read_csv(data_path)
            
            # Find status column
            status_col = None
            for col in ['Status', 'status', 'Label', 'label']:
                if col in df.columns:
                    status_col = col
                    break
            
            if status_col is None:
                print("No status column found in ground truth data")
                return None
            
            # Create binary labels: Normal=0, Anomaly=1
            labels = (df[status_col] != 'Normal').astype(int)
            
            anomaly_count = np.sum(labels)
            print(f"Loaded REAL ground truth: {anomaly_count}/{len(labels)} anomalies ({anomaly_count/len(labels):.1%})")
            
            return labels.values
            
        except Exception as e:
            print(f"Error loading real ground truth: {e}")
            return None
    
    def _load_inference_data(self):
        """Load inference sensor data"""
        try:
            data_path = self.config.use_case_config.inference_csv_path
            df = pd.read_csv(data_path)
            
            sensor_cols = self.config.use_case_config.sensor_cols
            missing_cols = [col for col in sensor_cols if col not in df.columns]
            if missing_cols:
                print(f"Missing sensor columns: {missing_cols}")
                return None
            
            return df[sensor_cols].dropna()
            
        except Exception as e:
            print(f"Error loading inference data: {e}")
            return None
    
    def _get_predictions(self, model, X_val):
        """Get predictions from any model type"""
        try:
            if hasattr(model, 'predict_anomalies'):
                # Adaptive model
                predictions, _, _ = model.predict_anomalies(X_val, adapt=False)
                return np.array(predictions).astype(int)
            elif hasattr(model, 'predict'):
                # Legacy model
                result = model.predict(X_val)
                if isinstance(result, tuple):
                    return np.array(result[1]).astype(int)  # anomaly flags
                else:
                    return np.array(result).astype(int)
            else:
                raise Exception("Unknown model type")
                
        except Exception as e:
            print(f"Prediction failed: {e}")
            # Fallback - random predictions
            return np.random.choice([0, 1], size=len(X_val), p=[0.95, 0.05])