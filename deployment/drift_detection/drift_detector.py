# Drift Detection System for Smart Home Anomaly Detection
# 
# PURPOSE: Detect when sensor data patterns change significantly from baseline,
# indicating environment changes that might require model retraining
#
# APPROACH: Compare recent sensor data statistics against stored baseline statistics
# using multiple statistical measures to detect distribution shifts

import os
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class DriftDetector:
    def __init__(self, config):
        self.config = config
        self.drift_history_path = os.path.join("logs", config.use_case, "drift_history.csv")
        self.reference_stats_path = os.path.join("logs", config.use_case, "reference_stats.pkl")
        
        # Drift detection parameters
        self.window_size = 100  # Number of recent samples to compare
        self.drift_threshold = config.drift_threshold  # From config (default 0.3)
        self.min_samples = 50   # Minimum samples before drift detection
        
        # Initialize or load reference statistics
        self.reference_stats = self._load_or_create_reference_stats()
        
    def _load_or_create_reference_stats(self):
        """Load reference statistics or create empty dict if not exists"""
        if os.path.exists(self.reference_stats_path):
            with open(self.reference_stats_path, 'rb') as f:
                return pickle.load(f)
        else:
            os.makedirs(os.path.dirname(self.reference_stats_path), exist_ok=True)
            return {}
    
    def _save_reference_stats(self):
        """Save reference statistics to file"""
        with open(self.reference_stats_path, 'wb') as f:
            pickle.dump(self.reference_stats, f)
    
    def update_reference_stats(self, sensor_data):
        """Update reference statistics with new baseline data"""
        sensor_cols = self.config.use_case_config.sensor_cols
        
        for col in sensor_cols:
            if col in sensor_data.columns:
                clean_data = sensor_data[col].dropna()
                if len(clean_data) > 0:
                    self.reference_stats[col] = {
                        'mean': clean_data.mean(),
                        'std': clean_data.std(),
                        'min': clean_data.min(),
                        'max': clean_data.max(),
                        'percentiles': {
                            '25': clean_data.quantile(0.25),
                            '50': clean_data.quantile(0.50),
                            '75': clean_data.quantile(0.75)
                        }
                    }
        
        self._save_reference_stats()
        print(f"✅ Reference statistics updated for {len(sensor_cols)} sensors")
    
    def detect_statistical_drift(self, recent_data):
        """Detect drift using statistical tests (KS test and distribution comparison)"""
        if len(self.reference_stats) == 0:
            print("⚠️ No reference statistics available. Updating baseline...")
            self.update_reference_stats(recent_data)
            return False, {}
        
        sensor_cols = self.config.use_case_config.sensor_cols
        drift_results = {}
        drift_detected = False
        
        for col in sensor_cols:
            if col not in recent_data.columns or col not in self.reference_stats:
                continue
                
            recent_values = recent_data[col].dropna()
            if len(recent_values) < self.min_samples:
                continue
            
            ref_stats = self.reference_stats[col]
            
            # Statistical drift metrics
            current_mean = recent_values.mean()
            current_std = recent_values.std()
            
            # Mean shift detection
            mean_shift = abs(current_mean - ref_stats['mean']) / (ref_stats['std'] + 1e-8)
            
            # Standard deviation change
            std_change = abs(current_std - ref_stats['std']) / (ref_stats['std'] + 1e-8)
            
            # outlier rate (values outside 2 standard deviations of reference)
            ref_lower = ref_stats['mean'] - 2 * ref_stats['std']
            ref_upper = ref_stats['mean'] + 2 * ref_stats['std']
            outlier_rate = ((recent_values < ref_lower) | (recent_values > ref_upper)).mean()
            
            # Combine metrics into drift score
            drift_score = max(mean_shift, std_change, outlier_rate)
            
            drift_results[col] = {
                'drift_score': drift_score,
                'mean_shift': mean_shift,
                'std_change': std_change,
                'outlier_rate': outlier_rate,
                'drift_detected': drift_score > self.drift_threshold
            }
            
            if drift_score > self.drift_threshold:
                drift_detected = True
                print(f"🚨 Drift detected in {col}: score={drift_score:.3f} (threshold={self.drift_threshold})")
        
        return drift_detected, drift_results
    
    def detect_reconstruction_drift(self, sensor_model, recent_data):
        """Detect drift using reconstruction error from autoencoder"""
        try:
            sensor_cols = self.config.use_case_config.sensor_cols
            clean_data = recent_data[sensor_cols].dropna()
            
            if len(clean_data) < self.min_samples:
                return False, {}
            
            # Get reconstruction errors
            reconstructed = sensor_model.predict(clean_data)
            reconstruction_errors = mean_squared_error(clean_data, reconstructed, multioutput='raw_values')
            avg_reconstruction_error = np.mean(reconstruction_errors)
            
            # Load historical reconstruction error if available
            drift_history = self._load_drift_history()
            if len(drift_history) > 0:
                historical_errors = drift_history['avg_reconstruction_error'].dropna()
                if len(historical_errors) > 10:  # Need some history
                    baseline_error = historical_errors.mean()
                    error_increase = (avg_reconstruction_error - baseline_error) / (baseline_error + 1e-8)
                    
                    reconstruction_drift = error_increase > self.drift_threshold
                    
                    return reconstruction_drift, {
                        'current_error': avg_reconstruction_error,
                        'baseline_error': baseline_error,
                        'error_increase': error_increase,
                        'drift_detected': reconstruction_drift
                    }
            
            return False, {'current_error': avg_reconstruction_error}
            
        except Exception as e:
            print(f"⚠️ Error in reconstruction drift detection: {e}")
            return False, {}
    
    def _load_drift_history(self):
        """Load drift detection history"""
        if os.path.exists(self.drift_history_path):
            return pd.read_csv(self.drift_history_path)
        return pd.DataFrame()
    
    def log_drift_results(self, statistical_results, reconstruction_results):
        """Log drift detection results"""
        timestamp = pd.Timestamp.now()
        
        # Prepare log entry
        log_entry = {
            'timestamp': timestamp,
            'overall_drift_detected': any(r.get('drift_detected', False) for r in statistical_results.values()),
            'reconstruction_drift_detected': reconstruction_results.get('drift_detected', False),
            'avg_reconstruction_error': reconstruction_results.get('current_error', np.nan)
        }
        
        # Add per-sensor drift scores
        for sensor, results in statistical_results.items():
            log_entry[f'{sensor}_drift_score'] = results.get('drift_score', np.nan)
            log_entry[f'{sensor}_drift_detected'] = results.get('drift_detected', False)
        
        # Save to history
        history_df = self._load_drift_history()
        new_row = pd.DataFrame([log_entry])
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        
        # Keep only recent history (last 1000 entries)
        if len(history_df) > 1000:
            history_df = history_df.tail(1000)
        
        os.makedirs(os.path.dirname(self.drift_history_path), exist_ok=True)
        history_df.to_csv(self.drift_history_path, index=False)
    
    def should_retrain(self):
        """Determine if retraining should be triggered based on drift history"""
        history_df = self._load_drift_history()
        if len(history_df) < 1:  # Need some history
            return False
        
        # Check recent drift frequency
        recent_entries = history_df.tail(10)
        drift_rate = recent_entries['overall_drift_detected'].mean()
        
        # Trigger retraining if drift detected frequently
        return drift_rate > 0.3  # 30% of recent checks show drift


def run(config, sensor_model=None, recent_sensor_data=None):
    """Main drift detection function called from deployment pipeline"""
    
    # Load recent sensor data if not provided
    if recent_sensor_data is None:
        if os.path.exists(config.decision_log_path):
            decision_df = pd.read_csv(config.decision_log_path)
            # Get recent data (last 100 entries)
            recent_sensor_data = decision_df.tail(100)
        else:
            print("⚠️ No decision log found for drift detection")
            return {'drift_detected': False, 'should_retrain': False}
    
    detector = DriftDetector(config)
    
    # Detect statistical drift
    statistical_drift, statistical_results = detector.detect_statistical_drift(recent_sensor_data)
    
    # Log results
    detector.log_drift_results(statistical_results, {})
    
    # Check if retraining should be triggered
    should_retrain = detector.should_retrain()
    
    # Summary
    print(f"📊 Drift Detection Summary:")
    print(f"   Statistical drift: {'Yes' if statistical_drift else 'No'}")
    print(f"   Recommend retraining: {'Yes' if should_retrain else 'No'}")
    
    if should_retrain:
        print("🔄 Frequent drift detected - consider retraining models")
    
    return {
        'drift_detected': statistical_drift,
        'should_retrain': should_retrain,
        'statistical_results': statistical_results
    }
