"""
Adaptive Unsupervised Autoencoder for Dynamic Environments
- Fully unsupervised (no labels needed)
- Dynamic threshold adaptation
- Automatic hyperparameter tuning
- Environment drift detection and adaptation
- Real-time deployment ready
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import pickle
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report

from config.config_manager import ConfigManager

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope


def validate_sensor_ranges(data, sensor_ranges, sensor_names):
    """
    Validate sensor readings against realistic ranges
    
    Parameters:
    -----------
    data : numpy.ndarray
        Sensor data
    sensor_ranges : dict
        Dictionary mapping sensor names to (min, max) tuples
    sensor_names : list
        List of sensor names
        
    Returns:
    --------
    clean_data : numpy.ndarray
        Data with sensor errors removed
    error_mask : numpy.ndarray
        Boolean mask indicating sensor errors
    """
    clean_data = data.copy()
    error_mask = np.zeros(data.shape, dtype=bool)
    error_count = 0
    
    for i, sensor in enumerate(sensor_names):
        if sensor in sensor_ranges:
            min_val, max_val = sensor_ranges[sensor]
            
            # Find readings outside realistic range
            sensor_errors = (data[:, i] < min_val) | (data[:, i] > max_val)
            error_mask[:, i] = sensor_errors
            
            if sensor_errors.any():
                error_count += np.sum(sensor_errors)
                # print(f"⚡ Detected {np.sum(sensor_errors)} sensor errors in {sensor}")
                print(f"⚡ Fixed {np.sum(sensor_errors)} sensor errors in {sensor} (Range: {min_val}-{max_val}°C)")
                # print(f"   Range: {min_val}-{max_val}, Errors: {data[sensor_errors, i][:5]}...")
                
                # Replace with interpolated values
                clean_data = clean_data.astype(float) 
                clean_data[sensor_errors, i] = np.nan
    
    # Interpolate missing values
    for i in range(clean_data.shape[1]):
        if np.isnan(clean_data[:, i]).any():
            # Linear interpolation
            valid_indices = ~np.isnan(clean_data[:, i])
            if valid_indices.any():
                clean_data[:, i] = np.interp(
                    np.arange(len(clean_data)),
                    np.where(valid_indices)[0],
                    clean_data[valid_indices, i]
                )
    
    print(f"🧹 Removed {error_count} sensor errors across {len(sensor_names)} sensors")
    return clean_data, error_mask

def statistical_outlier_cleaning(data, sensor_names, z_threshold=5.0, method='mad'):
    """
    Remove statistical outliers using robust methods
    
    Parameters:
    -----------
    data : numpy.ndarray
        Sensor data
    sensor_names : list
        List of sensor names
    z_threshold : float
        Z-score threshold for outlier detection
    method : str
        Method to use ('mad', 'iqr', 'zscore')
        
    Returns:
    --------
    clean_data : numpy.ndarray
        Data with outliers removed
    outlier_mask : numpy.ndarray
        Boolean mask indicating outliers
    """
    clean_data = data.copy()
    outlier_mask = np.zeros(data.shape, dtype=bool)
    total_outliers = 0
    # print("the z_threshold is: ", z_threshold)
    # print(f"📊 Starting statistical outlier cleaning using {method.upper()} method...")

    for i, sensor in enumerate(sensor_names):
        sensor_data = data[:, i]

        if method == 'mad':
            # Median Absolute Deviation (most robust)
            median = np.median(sensor_data)
            mad = np.median(np.abs(sensor_data - median))
            if mad > 0:
                modified_z_scores = 0.6745 * (sensor_data - median) / mad
                outliers = np.abs(modified_z_scores) > z_threshold
            else:
                outliers = np.zeros(len(sensor_data), dtype=bool)
                
        elif method == 'iqr':
            # Interquartile Range
            q1, q3 = np.percentile(sensor_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = (sensor_data < lower_bound) | (sensor_data > upper_bound)
            
        elif method == 'zscore':
            # Standard Z-score (less robust but faster)
            z_scores = np.abs((sensor_data - np.mean(sensor_data)) / (np.std(sensor_data) + 1e-8))
            outliers = z_scores > z_threshold
        
        outlier_mask[:, i] = outliers
        
        if outliers.any():
            n_outliers = np.sum(outliers)
            total_outliers += n_outliers
            print(f"📊 Detected {n_outliers} statistical outliers in {sensor} using {method.upper()}, threshold = {z_threshold}")

            
            # Replace outliers with median
            clean_data[outliers, i] = np.median(sensor_data[~outliers])
    
    print(f"🧹 Removed {total_outliers} statistical outliers using {method.upper()} method")
    return clean_data, outlier_mask

def comprehensive_training_cleaning(data, sensor_names, sensor_ranges=None, 
                                  enable_range_validation=True, 
                                  enable_statistical_cleaning=True,
                                  statistical_outlier_method='mad',
                                  z_threshold=5.0):
    """
    Comprehensive cleaning for training data
    
    Parameters:
    -----------
    data : numpy.ndarray
        Raw training data
    sensor_names : list
        List of sensor names
    sensor_ranges : dict, optional
        Sensor range validation dict
    enable_range_validation : bool
        Whether to perform range validation
    enable_statistical_cleaning : bool
        Whether to perform statistical outlier removal
    z_threshold : float
        Threshold for statistical outlier detection
        
    Returns:
    --------
    clean_data : numpy.ndarray
        Cleaned training data
    cleaning_report : dict
        Report of cleaning operations
    """
    print(f"\n🧹 Data cleaning...")
    # print(f"📊 Input data shape: {data.shape}")
    
    original_data = data.copy()
    current_data = data.copy()
    cleaning_report = {
        'original_samples': data.shape[0],
        'sensor_errors_detected': 0,
        'statistical_outliers_detected': 0,
        'final_samples': 0,
        'data_quality_improvement': 0
    }
    
    # Step 1: Range validation (if enabled and ranges provided)
    if enable_range_validation and sensor_ranges is not None:
        # print("🔍 Step 1: Sensor range validation...")
        current_data, error_mask = validate_sensor_ranges(current_data, sensor_ranges, sensor_names)
        cleaning_report['sensor_errors_detected'] = np.sum(error_mask)
    
    # Step 2: Statistical outlier cleaning
    if enable_statistical_cleaning:
        # print("📊 Step 2: Statistical outlier cleaning...")
        current_data, outlier_mask = statistical_outlier_cleaning(
            current_data, sensor_names, z_threshold=z_threshold, method=statistical_outlier_method
        )
        cleaning_report['statistical_outliers_detected'] = np.sum(outlier_mask)
    
    # Step 3: Remove rows with too many issues (optional)
    # Calculate data quality per row
    if enable_range_validation and sensor_ranges is not None:
        row_error_rate = np.mean(error_mask, axis=1)
        problematic_rows = row_error_rate > 0.5  # More than 50% sensors have errors
        
        if problematic_rows.any():
            print(f"🚫 Removing {np.sum(problematic_rows)} rows with >50% sensor errors")
            current_data = current_data[~problematic_rows]
    
    # Final statistics
    cleaning_report['final_samples'] = current_data.shape[0]
    
    # Calculate data quality improvement
    original_variance = np.mean(np.var(original_data, axis=0))
    cleaned_variance = np.mean(np.var(current_data, axis=0))
    cleaning_report['data_quality_improvement'] = (cleaned_variance / original_variance) if original_variance > 0 else 1.0
    
    # print(f"✅ Data cleaning completed!")
    # print(f"📊 Samples: {cleaning_report['original_samples']} → {cleaning_report['final_samples']}")
    # print(f"⚡ Sensor errors fixed: {cleaning_report['sensor_errors_detected']}")
    # print(f"📈 Statistical outliers cleaned: {cleaning_report['statistical_outliers_detected']}")
    # print(f"🎯 Data quality ratio: {cleaning_report['data_quality_improvement']:.3f}")
    
    return current_data, cleaning_report

def get_default_sensor_ranges(sensor_names):
    """
    Get default sensor ranges for common sensor types
    """
    default_ranges = {
        # Temperature sensors
        'Temperature': (0, 50),
        'Temp': (0, 50),
        'temperature': (0, 50),
        'temp': (0, 50),
        
        # Humidity sensors  
        'Humidity': (0, 100),
        'humidity': (0, 100),
        'RH': (0, 100),
        
        # Pressure sensors
        'Pressure': (95000, 110000),
        'pressure': (95000, 110000),
        'Press': (95000, 110000),
        
        # CO2 sensors
        'CO2': (300, 5000),
        'co2': (300, 5000),
        'Carbon_Dioxide': (300, 5000),
        
        # Light sensors
        'Light': (0, 100000),
        'light': (0, 100000),
        'Lux': (0, 100000),
        'lux': (0, 100000),
        
        # Motion sensors
        'Motion': (0, 1),
        'motion': (0, 1),
        'PIR': (0, 1)
    }
    
    # Return ranges for sensors that match
    ranges = {}
    for sensor in sensor_names:
        if sensor in default_ranges:
            ranges[sensor] = default_ranges[sensor]
            print(f"📏 Default range for {sensor}: {default_ranges[sensor]}")
    
    return ranges

# # Usage example for home care
# def home_care_monitoring_example():
#     """
#     Example of how to use the optimized home care monitoring.
#     """
#     # Initialize for home environment
#     detector = AdaptiveUnsupervisedAutoencoder(
#         sensor_names=['Temperature', 'Humidity', 'CO2', 'Motion'],
#         case_name='elderly_care_home'
#     )
    
#     # Train on normal home data
#     # detector.learn_environment(training_data)
    
#     # Monitor new readings
#     # anomalies, insights, drift = detector.predict_anomalies_home_optimized(
#     #     new_readings, 
#     #     sensitivity='medium'  # Good for most homes
#     # )
    
#     # Example output interpretation
#     example_insights = {
#         'anomaly_explanations': [{
#             'type': 'climate_issue',
#             'message': 'Temperature and humidity levels are unusual',
#             'recommendation': 'Check HVAC system, windows, or room ventilation'
#         }],
#         'severity_levels': ['medium'],
#         'recommendations': ['Check HVAC system, windows, or room ventilation']
#     }
    
#     return example_insights
    
class AdaptiveUnsupervisedAutoencoder:
    def __init__(self, 
                sensor_names,
                adaptation_window=1000,
                min_training_samples=500,
                drift_detection_threshold=0.3,
                auto_retrain_threshold=0.5,
                case_name="adaptive_env"):
        """
        Adaptive unsupervised autoencoder that learns from normal operations
        and adapts to environmental changes without labels.
        
        Parameters:
        -----------
        sensor_names : list
            List of sensor names
        adaptation_window : int
            Number of recent samples to consider for adaptation
        min_training_samples : int
            Minimum samples needed before making predictions
        drift_detection_threshold : float
            Threshold for detecting environment drift (0-1)
        auto_retrain_threshold : float
            Threshold for automatic retraining (0-1)
        case_name : str
            Name for saving/loading models
        """
        self.sensor_names = sensor_names
        self.n_features = len(sensor_names)
        self.adaptation_window = adaptation_window
        self.min_training_samples = min_training_samples
        self.drift_detection_threshold = drift_detection_threshold
        self.auto_retrain_threshold = auto_retrain_threshold
        self.case_name = case_name

        self.config = ConfigManager.get_config(case_name) 
        self.threshold_percentile = self.config.threshold_percentile
        self.threshold_factor = self.config.threshold_factor

        # Dynamic parameters that adapt over time
        self.current_contamination = 0.1  # Will be estimated
        self.contamination = 0.05  # 🔧 ADD THIS LINE for backward compatibility
        self.adaptive_threshold_multiplier = 1.0
        self.confidence_level = 0.95  # For threshold calculation
        
        # Model components
        self.autoencoder = None
        self.isolation_forest = None
        self.outlier_detector = None  # Elliptic Envelope for outlier detection
        self.scaler = None
        self.pca = None  # For dimensionality reduction if needed
        
        # Adaptive thresholds and statistics
        self.global_threshold = None
        self.sensor_thresholds = {}
        self.baseline_stats = {}
        self.recent_errors = deque(maxlen=adaptation_window)
        self.recent_samples = deque(maxlen=adaptation_window)
        
        # Environment monitoring
        self.drift_scores = deque(maxlen=100)
        self.performance_history = deque(maxlen=50)
        self.last_retrain_time = None
        self.training_data_buffer = deque(maxlen=5000)  # Store recent normal data
        
        # Auto-configuration
        self.auto_architecture = True
        self.auto_scaling_method = True
        self.auto_contamination = True
        
        # Deployment state
        self.is_trained = False
        self.samples_seen = 0
        self.anomalies_detected = 0
        
        # Extract use case name from case_name (remove 'adaptive_' prefix if present)
        if case_name.startswith('adaptive_'):
            use_case_name = case_name.replace('adaptive_', '')
        else:
            use_case_name = case_name
        
        # Updated Paths - create models/use_case_name/ directory
        self.base_path = f"models/{use_case_name}"
        os.makedirs(self.base_path, exist_ok=True)
        
        # print(f"🚀 Initialized Adaptive Autoencoder for environment: {case_name}")
        # print(f"📊 Sensors: {sensor_names}")
        # print(f"🔄 Adaptation window: {adaptation_window} samples")
        # print(f"💾 Models will be saved to: {self.base_path}")
        # print(f"⚡ Auto-adaptation enabled")
    def _estimate_contamination(self, X):
        """
        Automatically estimate contamination rate using multiple methods.
        """
        methods_contamination = []
        
        # Method 1: IQR-based outlier detection
        try:
            outlier_counts = []
            for i in range(X.shape[1]):
                q1, q3 = np.percentile(X[:, i], [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = np.sum((X[:, i] < lower_bound) | (X[:, i] > upper_bound))
                outlier_counts.append(outliers)
            
            avg_outlier_rate = np.mean(outlier_counts) / len(X)
            methods_contamination.append(min(0.3, max(0.05, avg_outlier_rate)))
        except:
            pass
        
        # Method 2: Z-score based
        try:
            z_scores = np.abs((X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8))
            outlier_rate = np.mean(np.any(z_scores > 3, axis=1))
            methods_contamination.append(min(0.25, max(0.05, outlier_rate)))
        except:
            pass
        
        # Method 3: Elliptic Envelope
        try:
            ee = EllipticEnvelope(contamination=0.1, random_state=42)
            outlier_pred = ee.fit_predict(X)
            outlier_rate = np.mean(outlier_pred == -1)
            methods_contamination.append(min(0.2, max(0.05, outlier_rate)))
        except:
            pass
        
        # Use median of methods or default
        if methods_contamination:
            estimated = np.median(methods_contamination)
        else:
            estimated = 0.1
            
        # print(f"🔍 Estimated contamination: {estimated:.3f}")
        return estimated

    def _auto_configure_architecture(self):
        """
        Automatically configure architecture based on data characteristics.
        """
        if self.n_features == 1:
            # Single sensor - simple architecture
            hidden_layers = [max(4, self.n_features * 2), 
                           max(2, self.n_features), 
                           max(2, self.n_features), 
                           max(4, self.n_features * 2)]
            solver = 'lbfgs'
            max_iter = 300
        elif self.n_features <= 5:
            # Few sensors - moderate complexity
            hidden_layers = [self.n_features * 4, 
                           self.n_features * 2, 
                           max(2, self.n_features // 2),
                           self.n_features * 2, 
                           self.n_features * 4]
            solver = 'lbfgs'
            max_iter = 400
        elif self.n_features <= 20:
            # Many sensors - higher complexity
            hidden_layers = [self.n_features * 3, 
                           self.n_features * 2, 
                           max(4, self.n_features // 2),
                           self.n_features * 2, 
                           self.n_features * 3]
            solver = 'adam'
            max_iter = 500
        else:
            # Very high dimensional - use PCA + complex architecture
            self.use_pca = True
            pca_components = min(20, self.n_features // 2)
            hidden_layers = [32, 16, 8, 16, 32]
            solver = 'adam'
            max_iter = 600
            
        return hidden_layers, solver, max_iter

    def _choose_scaler(self, X):
        """
        Automatically choose the best scaler based on data characteristics.
        """
        # Calculate skewness and outlier presence
        outlier_ratios = []
        skewness_values = []
        
        for i in range(X.shape[1]):
            # Outlier ratio using IQR
            q1, q3 = np.percentile(X[:, i], [25, 75])
            iqr = q3 - q1
            outliers = np.sum((X[:, i] < q1 - 1.5*iqr) | (X[:, i] > q3 + 1.5*iqr))
            outlier_ratios.append(outliers / len(X))
            
            # Skewness
            mean_val = np.mean(X[:, i])
            std_val = np.std(X[:, i])
            if std_val > 0:
                skew = np.mean(((X[:, i] - mean_val) / std_val) ** 3)
                skewness_values.append(abs(skew))
            else:
                skewness_values.append(0)
        
        avg_outlier_ratio = np.mean(outlier_ratios)
        avg_skewness = np.mean(skewness_values)
        
        # Decision logic
        if avg_outlier_ratio > 0.1 or avg_skewness > 1.5:
            scaler = RobustScaler()
            scaler_name = "RobustScaler"
        else:
            scaler = StandardScaler()
            scaler_name = "StandardScaler"
            
        # print(f"🔧 Auto-selected scaler: {scaler_name} (outliers: {avg_outlier_ratio:.3f}, skewness: {avg_skewness:.3f})")
        return scaler

    # Fix 1: Robust Threshold Calculation
    def calculate_robust_thresholds(self, X_scaled, reconstruction_errors):
        """
        Calculate thresholds with fallback mechanisms for edge cases.
        """
        n_samples = len(reconstruction_errors)

        # Global threshold calculation with safety checks
        if np.std(reconstruction_errors) < 1e-6:
            # Case: Very low variance (like constant data)
            print("⚠️ Low variance in reconstruction errors detected")
            # Fallback case
            self.global_threshold = max(
                np.mean(reconstruction_errors) + 3*np.std(reconstruction_errors), 
                np.percentile(reconstruction_errors, self.threshold_percentile) * self.threshold_factor,
                1e-4  # Minimum threshold
            )
            print(f"🔧 Applied fallback global threshold ({self.threshold_percentile}th percentile * {self.threshold_factor}): {self.global_threshold:.6f}")
        else:
            # Normal case with config
            self.global_threshold = (
                np.percentile(reconstruction_errors, self.threshold_percentile) * 
                self.threshold_factor
            )
            print(f"Calculated global threshold ({self.threshold_percentile}th percentile * {self.threshold_factor}): {self.global_threshold:.6f}")

        # Per-sensor threshold calculation
        # sensor_errors = (X_scaled - self.autoencoder.predict(X_scaled)) ** 2

        recon = self.autoencoder.predict(X_scaled)
        if X_scaled.shape[1] == 1 and recon.ndim == 1:
            recon = recon.reshape(-1, 1)
        sensor_errors = (X_scaled - recon) ** 2

        
        for i, sensor in enumerate(self.sensor_names):
            sensor_reconstruction_errors = sensor_errors[:, i]
            
            if np.std(sensor_reconstruction_errors) < 1e-6:
                # Low variance sensor
                threshold = max(
                    np.mean(sensor_reconstruction_errors) + 3*np.std(sensor_reconstruction_errors),
                    np.percentile(sensor_reconstruction_errors, self.threshold_percentile) * self.threshold_factor,  # ✅ Use config
                    1e-5  # Minimum sensor threshold
                )
                print(f"🔧 Applied fallback threshold for {sensor}: {threshold:.6f}")
            else:
                # Normal sensor - use config values
                threshold = (
                    np.percentile(sensor_reconstruction_errors, self.threshold_percentile) * 
                    self.threshold_factor
                )
                # print(f"✅ Calculated threshold for {sensor}: {threshold:.6f}")
            
            self.sensor_thresholds[sensor] = threshold

    # Fix 2: Adaptive Contamination Parameter
    def estimate_smart_contamination(self, X_scaled):
        """
        Estimate contamination based on data characteristics.
        """
        # Check for constant/near-constant data
        feature_stds = np.std(X_scaled, axis=0)
        low_variance_features = np.sum(feature_stds < 0.1)
        
        if low_variance_features > 0:
            # For datasets with constant features (like ultrasonic sensors)
            contamination = 0.02  # Very conservative
            print(f"🔧 Low variance detected, using conservative contamination: {contamination}")
        else:
            # Normal estimation
            # contamination = min(0.1, max(0.01, self.estimated_contamination))
            contamination = min(0.1, max(0.01, self._estimate_contamination(X_scaled)))
            # print(f"✅ Using estimated contamination: {contamination}")
        
        return contamination


    def calibrate_thresholds_on_validation(self, X_val, y_val):
        """
        Universal enhanced calibration without case-specific logic.
        """
        print("🎯 Calibrating thresholds on validation data...")
        
        # Memory-efficient stratified sampling
        if len(X_val) > 10000:
            try:
                from sklearn.model_selection import train_test_split
                _, X_val_sample, _, y_val_sample = train_test_split(
                    X_val, y_val, test_size=min(0.3, 5000/len(X_val)), 
                    stratify=y_val, random_state=42
                )
                print(f"🔧 Stratified sampling: {len(X_val_sample)} samples")
            except:
                # Fallback to random if stratified fails
                sample_size = min(5000, len(X_val))
                indices = np.random.choice(len(X_val), sample_size, replace=False)
                X_val_sample = X_val[indices]
                y_val_sample = y_val[indices]
                print(f"🔧 Random sampling: {sample_size} samples")
        else:
            X_val_sample = X_val
            y_val_sample = y_val
        
        X_val_scaled = self.scaler.transform(X_val_sample)
        
        # Fix shape issues for all feature counts
        val_reconstructions = self.autoencoder.predict(X_val_scaled)
        if val_reconstructions.shape != X_val_scaled.shape:
            if X_val_scaled.shape[1] == 1 and val_reconstructions.ndim == 1:
                val_reconstructions = val_reconstructions.reshape(-1, 1)
            else:
                val_reconstructions = val_reconstructions.reshape(X_val_scaled.shape)
        
        val_errors = np.mean((X_val_scaled - val_reconstructions) ** 2, axis=1)
        
        # Universal precision-recall optimization
        try:
            from sklearn.metrics import precision_recall_curve
            
            precision, recall, thresholds = precision_recall_curve(y_val_sample, val_errors)
            
            # Calculate multiple scoring methods
            scores = []
            
            for i, thresh in enumerate(thresholds):
                if precision[i] > 0 and recall[i] > 0:
                    # F1 score
                    f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
                    
                    # Balanced accuracy proxy
                    balanced = (precision[i] + recall[i]) / 2
                    
                    # Geometric mean of precision and recall
                    geometric = np.sqrt(precision[i] * recall[i])
                    
                    # Combined score with penalty for extreme values
                    combined = (f1 + balanced + geometric) / 3
                    
                    # Penalty for very low/high detection rates
                    detection_rate = np.sum(val_errors > thresh) / len(val_errors)
                    if detection_rate < 0.01 or detection_rate > 0.9:
                        combined *= 0.5
                    
                    scores.append((combined, thresh, f1, precision[i], recall[i]))
            
            if scores:
                # Select best threshold based on combined score
                best_score, best_threshold, best_f1, best_precision, best_recall = max(scores)
                self.global_threshold = best_threshold
                
                print(f"🎯 Optimized threshold: {self.global_threshold:.6f}")
                print(f"   F1: {best_f1:.3f}, Precision: {best_precision:.3f}, Recall: {best_recall:.3f}")
            else:
                raise Exception("No valid thresholds found")
                
        except Exception as e:
            print(f"⚠️ PR optimization failed ({e}), using adaptive percentile method")
            
            # Adaptive percentile selection based on data characteristics
            anomaly_rate = np.mean(y_val_sample)
            
            # Adjust percentile based on anomaly rate
            if anomaly_rate > 0.15:  # High anomaly rate
                percentile = 85
                factor = 1.1
            elif anomaly_rate < 0.05:  # Low anomaly rate  
                percentile = 95
                factor = 0.9
            else:  # Medium anomaly rate
                percentile = 90
                factor = 1.0
            
            base_threshold = np.percentile(val_errors, percentile)
            self.global_threshold = base_threshold * factor
            
            print(f"🔧 Adaptive threshold: P{percentile} * {factor} = {self.global_threshold:.6f}")
        
        # Validate final threshold
        final_detections = np.sum(val_errors > self.global_threshold)
        detection_rate = final_detections / len(val_errors)
        actual_anomaly_rate = np.mean(y_val_sample)
        
        print(f"📈 Final validation results:")
        print(f"   Detection rate: {detection_rate:.1%}")
        print(f"   Actual anomaly rate: {actual_anomaly_rate:.1%}")
        print(f"   Ratio: {detection_rate/actual_anomaly_rate:.2f}x")
        
        # Calibrate other methods using the same principles
        self._calibrate_ensemble_methods(X_val_scaled, y_val_sample, actual_anomaly_rate)
        
        print("✅ Universal threshold calibration completed!")

        self._select_optimal_methods(X_val, y_val)

    def _calibrate_ensemble_methods(self, X_val_scaled, y_val_sample, actual_anomaly_rate):
        """Calibrate isolation forest and outlier detector generically"""
        
        # Isolation Forest
        if hasattr(self, 'isolation_forest'):
            val_iso_pred = self.isolation_forest.predict(X_val_scaled)
            iso_rate = np.sum(val_iso_pred == -1) / len(val_iso_pred)
            
            # Retrain if detection rate is too far from expected
            if abs(iso_rate - actual_anomaly_rate) > 0.2:
                new_contamination = min(0.5, max(0.01, actual_anomaly_rate * 1.2))
                self.isolation_forest.set_params(contamination=new_contamination)
                self.isolation_forest.fit(X_val_scaled)
        
        # Outlier Detector
        if hasattr(self, 'outlier_detector') and self.outlier_detector is not None:
            try:
                val_outlier_pred = self.outlier_detector.predict(X_val_scaled)
                outlier_rate = np.sum(val_outlier_pred == -1) / len(val_outlier_pred)
                
                if outlier_rate < 0.005 or outlier_rate > 0.8:
                    new_contamination = min(0.5, max(0.01, actual_anomaly_rate * 1.5))
                    from sklearn.covariance import EllipticEnvelope
                    self.outlier_detector = EllipticEnvelope(
                        contamination=new_contamination, random_state=42
                    )
                    # Use smaller sample for memory efficiency
                    train_size = min(2000, len(X_val_scaled))
                    train_indices = np.random.choice(len(X_val_scaled), train_size, replace=False)
                    self.outlier_detector.fit(X_val_scaled[train_indices])
            except Exception as e:
                print(f"⚠️ EllipticEnvelope calibration failed: {e}")
                self.outlier_detector = None

    # def adjust_method_thresholds(self):
    #     """
    #     DEPRECATED: Use calibrate_thresholds_on_validation() instead.
    #     This method caused distribution mismatch by using training data.
    #     """
    #     print("⚠️ Warning: adjust_method_thresholds() is deprecated. Use calibrate_thresholds_on_validation() instead.")
    #     # Keep the original logic as fallback, but warn user
    #     X_train_scaled = self.scaler.transform(self.training_data)
        
    #     train_reconstructions = self.autoencoder.predict(X_train_scaled)
    #     train_errors = np.mean((X_train_scaled - train_reconstructions) ** 2, axis=1)
        
    #     detection_rate = np.sum(train_errors > self.global_threshold) / len(train_errors)
        
    #     if detection_rate > 0.15:
    #         self.global_threshold = np.percentile(train_errors, 98)
    #         print(f"🔧 Reduced sensitivity: new global threshold {self.global_threshold:.6f}")
    #     elif detection_rate < 0.02:
    #         self.global_threshold = np.percentile(train_errors, 90)
    #         print(f"🔧 Increased sensitivity: new global threshold {self.global_threshold:.6f}")
            
    # Fix 4: Complete Updated learn_environment Method
    def learn_environment(self, training_data):
        """
        Updated learn_environment with robust threshold calculation.
        """
        self.training_data = training_data.copy()
        print(f"\n🎓 Learning environment from {len(training_data)} samples...")
        
        # ADD THIS LINE:
        self.scaler = self._choose_scaler(training_data)

        X_scaled = self.scaler.fit_transform(training_data)
        # print(f"🔧 Auto-selected scaler: {type(self.scaler).__name__}")
        
        # Estimate contamination smartly
        self.estimated_contamination = self.estimate_smart_contamination(X_scaled)
        
        #     # Auto-configure architecture
        hidden_layers, solver, max_iter = self._auto_configure_architecture()
        # print(f"🏗️  Architecture: {self.n_features} -> {' -> '.join(map(str, hidden_layers))} -> {self.n_features}")

        # Build and train autoencoder
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layers),
            activation='tanh',
            solver=solver,
            alpha=0.01,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            tol=1e-6,
            verbose=False,
            random_state=42
        )
        
        # Train autoencoder
        try:
            self.autoencoder.fit(X_scaled, X_scaled)
            # print(f"✅ Autoencoder trained in {self.autoencoder.n_iter_} iterations")
        except Exception as e:
            print(f"⚠️  Autoencoder training issue, trying fallback: {e}")
            self.autoencoder.solver = 'adam'
            self.autoencoder.max_iter = 200
            self.autoencoder.fit(X_scaled, X_scaled)
        self.autoencoder.fit(X_scaled, X_scaled)
        # print(f"✅ Autoencoder trained in {self.autoencoder.n_iter_} iterations")
        
        # Train isolation forest with smart contamination
        self.isolation_forest = IsolationForest(
            contamination=self.estimated_contamination,
            random_state=42,
            n_jobs=-1
        )
        self.isolation_forest.fit(X_scaled)
        
        # Train outlier detector with fallback
        try:
            self.outlier_detector = EllipticEnvelope(
                contamination=self.estimated_contamination,
                random_state=42
            )
            self.outlier_detector.fit(X_scaled)
            print("✅ EllipticEnvelope outlier detector trained")
        except ValueError as e:
            print(f"⚠️ EllipticEnvelope failed: {e}")
            print("🔄 Using fallback: IsolationForest as outlier detector")
            self.outlier_detector = None
        
        # Calculate thresholds with robust method
        reconstructions = self.autoencoder.predict(X_scaled)

        # force 2-D when single feature to avoid (n,n) broadcast
        if X_scaled.shape[1] == 1 and reconstructions.ndim == 1:
            reconstructions = reconstructions.reshape(-1, 1)


        reconstruction_errors = np.mean((X_scaled - reconstructions) ** 2, axis=1)
        
        self.calculate_robust_thresholds(X_scaled, reconstruction_errors)

        sensor_errors = (X_scaled - reconstructions) ** 2

        # Feature importance: average per-sensor reconstruction error
        self.feature_importance = {
            sensor: float(np.mean(sensor_errors[:, i]))
            for i, sensor in enumerate(self.sensor_names)
        }

        
        # Calibrate method thresholds
        # self.adjust_method_thresholds()
        # self.calibrate_thresholds_on_validation()
        # print(f"🎯 Environment learning completed!")
        # print(f"📊 Global threshold: {self.global_threshold:.6f}")
        
        self.is_trained = True
        # print(f"🔄 Ready for real-time adaptation")

    # Fix 5: Validation Function
    def validate_thresholds(self, X_test_sample=None):
        """
        Validate that thresholds are reasonable.
        """
        issues = []
        
        if self.global_threshold < 1e-5:
            issues.append(f"Global threshold too low: {self.global_threshold}")
        
        for sensor, threshold in self.sensor_thresholds.items():
            if threshold < 1e-6:
                issues.append(f"Sensor {sensor} threshold too low: {threshold}")
        
        if issues:
            print("⚠️ Threshold validation issues found:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        
        print("✅ Threshold validation passed")
        return True

    def _calculate_adaptive_thresholds(self, X_scaled):
        """
        Calculate adaptive thresholds that can be updated over time.
        """
        # Get reconstruction errors
        reconstructions = self.autoencoder.predict(X_scaled)
        reconstruction_errors = (X_scaled - reconstructions) ** 2
        
        # Global reconstruction error
        global_errors = np.mean(reconstruction_errors, axis=1)
        
        # Store baseline errors
        self.recent_errors.extend(global_errors)
        
        # Calculate robust global threshold
        self._update_global_threshold()
        
        # Calculate per-sensor thresholds
        self.sensor_thresholds = {}
        for i, sensor in enumerate(self.sensor_names):
            sensor_errors = reconstruction_errors[:, i]
            
            # Use robust statistics
            median_error = np.median(sensor_errors)
            mad = np.median(np.abs(sensor_errors - median_error))  # Median Absolute Deviation
            
            # Adaptive threshold based on confidence level
            if mad > 0:
                threshold = median_error + (1.4826 * mad * 2.5)  # ~95% confidence
            else:
                threshold = np.percentile(sensor_errors, 95)
            
            self.sensor_thresholds[sensor] = threshold
        
        print(f"🎯 Calculated adaptive thresholds for {len(self.sensor_names)} sensors")

    def _update_global_threshold(self):
        """
        Update global threshold based on recent errors.
        """
        if len(self.recent_errors) < 50:
            return
            
        errors = list(self.recent_errors)
        
        # Use multiple methods and take conservative estimate
        median_error = np.median(errors)
        mad = np.median(np.abs(errors - median_error))
        
        # IQR method
        q1, q3 = np.percentile(errors, [25, 75])
        iqr_threshold = q3 + 1.5 * (q3 - q1)
        
        # MAD method (more robust)
        mad_threshold = median_error + (1.4826 * mad * 2.5)
        
        # Percentile method
        percentile_threshold = np.percentile(errors, 95)
        
        # Conservative choice
        candidate_thresholds = [mad_threshold, iqr_threshold, percentile_threshold]
        self.global_threshold = np.median(candidate_thresholds) * self.adaptive_threshold_multiplier

    def _store_baseline_stats(self, X_scaled):
        """
        Store baseline statistics for drift detection.
        """
        self.baseline_stats = {
            'mean': np.mean(X_scaled, axis=0),
            'std': np.std(X_scaled, axis=0),
            'median': np.median(X_scaled, axis=0),
            'q25': np.percentile(X_scaled, 25, axis=0),
            'q75': np.percentile(X_scaled, 75, axis=0)
        }

    def detect_drift(self, X_new):
        """
        Detect if the environment has drifted from baseline.
        
        Parameters:
        -----------
        X_new : numpy.ndarray
            New data samples
            
        Returns:
        --------
        drift_score : float
            Drift score (0-1, higher means more drift)
        needs_adaptation : bool
            Whether adaptation is recommended
        """
        if not self.is_trained or len(self.baseline_stats) == 0:
            return 0.0, False
            
        X_new_scaled = self.scaler.transform(X_new)
        
        # Calculate drift using multiple methods
        drift_scores = []
        
        # Method 1: Statistical drift (KL divergence approximation)
        try:
            for i in range(X_new_scaled.shape[1]):
                # Compare distributions using moments
                baseline_mean = self.baseline_stats['mean'][i]
                baseline_std = self.baseline_stats['std'][i]
                
                new_mean = np.mean(X_new_scaled[:, i])
                new_std = np.std(X_new_scaled[:, i])
                
                # Normalized difference in means
                mean_diff = abs(new_mean - baseline_mean) / (baseline_std + 1e-8)
                
                # Ratio of standard deviations
                std_ratio = max(new_std, baseline_std) / (min(new_std, baseline_std) + 1e-8)
                
                sensor_drift = (mean_diff + np.log(std_ratio)) / 2
                drift_scores.append(sensor_drift)
        except:
            pass
        
        # Method 2: Reconstruction error drift
        try:
            reconstructions = self.autoencoder.predict(X_new_scaled)
            new_errors = np.mean((X_new_scaled - reconstructions) ** 2, axis=1)
            
            if len(self.recent_errors) > 0:
                baseline_error_mean = np.mean(list(self.recent_errors))
                baseline_error_std = np.std(list(self.recent_errors))
                
                new_error_mean = np.mean(new_errors)
                error_drift = abs(new_error_mean - baseline_error_mean) / (baseline_error_std + 1e-8)
                drift_scores.append(error_drift)
        except:
            pass
        
        # Aggregate drift score
        if drift_scores:
            drift_score = min(1.0, np.mean(drift_scores))
        else:
            drift_score = 0.0
            
        self.drift_scores.append(drift_score)
        
        # Decision
        needs_adaptation = drift_score > self.drift_detection_threshold
        
        return drift_score, needs_adaptation

    def predict_anomalies(self, X, adapt=True):
        """
        Predict anomalies and adapt thresholds if enabled.
        Updated to handle missing outlier detector.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call learn_environment() first.")
            
        X_scaled = self.scaler.transform(X)
        n_samples = X.shape[0]
        
        # Detect drift
        drift_score, needs_adaptation = self.detect_drift(X)
        
        # Get predictions from all methods
        anomaly_flags = np.zeros(n_samples, dtype=int)
        anomaly_scores = {sensor: [] for sensor in self.sensor_names}
        
        # Autoencoder method
        reconstructions = self.autoencoder.predict(X_scaled)
        reconstruction_errors = (X_scaled - reconstructions) ** 2
        global_errors = np.mean(reconstruction_errors, axis=1)
        
        # Isolation forest method
        iso_predictions = self.isolation_forest.predict(X_scaled)
        iso_anomalies = (iso_predictions == -1)
        
        # Outlier detector method (handle missing detector)
        if self.outlier_detector is not None:
            outlier_predictions = self.outlier_detector.predict(X_scaled)
            outlier_anomalies = (outlier_predictions == -1)
            available_methods = 3
            print("🔍 Using 3 methods: Autoencoder + Isolation Forest + Outlier Detector")
        else:
            outlier_anomalies = np.zeros(n_samples, dtype=bool)
            available_methods = 2
            print("🔍 Using 2 methods: Autoencoder + Isolation Forest (Outlier Detector unavailable)")
        
        # Combine methods with adaptive weighting
        for i in range(n_samples):
            # Global anomaly check
            is_global_anomaly = global_errors[i] > self.global_threshold
            
            # Per-sensor checks
            sensor_anomalies = []
            for j, sensor in enumerate(self.sensor_names):
                sensor_error = reconstruction_errors[i, j]
                threshold = self.sensor_thresholds[sensor]
                
                score = sensor_error / (threshold + 1e-8)
                anomaly_scores[sensor].append(float(score))
                
                sensor_anomalies.append(sensor_error > threshold)
            
            # Decision logic: combine multiple methods (adjusted for available methods)
            methods_agree = sum([
                is_global_anomaly,
                iso_anomalies[i],
                outlier_anomalies[i],
                any(sensor_anomalies)
            ])
            
            # Adjust required methods based on availability
            if available_methods == 3:
                required_methods = 2  # Require at least 2 out of 3 methods
            else:
                required_methods = 2  # Require at least 2 out of 2 available methods
            
            anomaly_flags[i] = int(methods_agree >= required_methods)
        
        # Update statistics if adaptation is enabled
        if adapt:
            self._adaptive_update(X, X_scaled, global_errors, drift_score, needs_adaptation)
        
        # Update counters
        self.samples_seen += n_samples
        self.anomalies_detected += np.sum(anomaly_flags)
        
        # Prepare drift information
        drift_info = {
            'drift_score': drift_score,
            'needs_adaptation': needs_adaptation,
            'samples_seen': self.samples_seen,
            'anomaly_rate': np.sum(anomaly_flags) / n_samples,
            'cumulative_anomaly_rate': self.anomalies_detected / self.samples_seen,
            'available_methods': available_methods
        }
        
        return anomaly_flags, anomaly_scores, drift_info

    # def predict_anomalies_home_optimized(self, X, adapt=True, sensitivity='medium', verbose=True):
    #     """
    #     Optimized anomaly detection for home care environments.
    #     Updated to handle missing outlier detector.
    #     """
    #     if not self.is_trained:
    #         raise ValueError("Model not trained. Call learn_environment() first.")
            
    #     X_scaled = self.scaler.transform(X)
    #     n_samples = X.shape[0]
        
    #     # Detect drift
    #     drift_score, needs_adaptation = self.detect_drift(X)
        
    #     # Get predictions from all methods
    #     # reconstructions = self.autoencoder.predict(X_scaled)
    #     # reconstruction_errors = (X_scaled - reconstructions) ** 2

    #     reconstructions = self.autoencoder.predict(X_scaled)
    #     if X_scaled.shape[1] == 1 and reconstructions.ndim == 1:
    #         reconstructions = reconstructions.reshape(-1, 1)
    #     reconstructions = self.autoencoder.predict(X_scaled)
    #     if X_scaled.shape[1] == 1 and reconstructions.ndim == 1:
    #         reconstructions = reconstructions.reshape(-1, 1)
    #     reconstruction_errors = (X_scaled - reconstructions) ** 2

    #     global_errors = np.mean(reconstruction_errors, axis=1)
        
    #     # Isolation forest
    #     iso_predictions = self.isolation_forest.predict(X_scaled)
    #     iso_anomalies = (iso_predictions == -1)
        
    #     # Outlier detection (handle missing detector)
    #     if self.outlier_detector is not None:
    #         outlier_predictions = self.outlier_detector.predict(X_scaled)
    #         outlier_anomalies = (outlier_predictions == -1)
    #         available_methods = 3
    #     else:
    #         outlier_anomalies = np.zeros(n_samples, dtype=bool)
    #         available_methods = 2
        
    #     # Autoencoder anomalies
    #     autoencoder_anomalies = global_errors > self.global_threshold

    #     # Use the verbose parameter that's already being passed
    #     # if verbose:
    #     #     print(f"   Autoencoder anomalies: {np.sum(autoencoder_anomalies)} ({np.sum(autoencoder_anomalies)/n_samples*100:.1f}%)")
    #     #     print(f"   Global threshold: {self.global_threshold:.6f}")
    #     #     print(f"   Isolation Forest anomalies: {np.sum(iso_anomalies)} ({np.sum(iso_anomalies)/n_samples*100:.1f}%)")
            
    #     #     if self.outlier_detector is not None:
    #     #         print(f"   EllipticEnvelope anomalies: {np.sum(outlier_anomalies)} ({np.sum(outlier_anomalies)/n_samples*100:.1f}%)")
    #     #     else:
    #     #         print(f"   EllipticEnvelope: Not available")
            
    #     #     print(f"   Sensitivity config: {self._get_home_sensitivity_config_adjusted(sensitivity, available_methods)}")
         
    #     # Home-optimized consensus logic
    #     anomaly_flags = np.zeros(n_samples, dtype=int)
    #     home_insights = {
    #         'anomaly_explanations': [],
    #         'sensor_contributions': [],
    #         'severity_levels': [],
    #         'recommendations': []
    #     }
        
    #     # Set sensitivity thresholds (adjusted for available methods)
    #     sensitivity_config = self._get_home_sensitivity_config_adjusted(sensitivity, available_methods)
        
    #     for i in range(n_samples):
    #         # Method predictions
    #         autoencoder_anomaly = global_errors[i] > self.global_threshold
    #         isolation_anomaly = iso_anomalies[i]
    #         outlier_anomaly = outlier_anomalies[i]
            
    #         # Per-sensor analysis for explanations
    #         sensor_analysis = {}
    #         for j, sensor in enumerate(self.sensor_names):
    #             sensor_error = reconstruction_errors[i, j]
    #             threshold = self.sensor_thresholds[sensor]
    #             score = sensor_error / (threshold + 1e-8)
                
    #             sensor_analysis[sensor] = {
    #                 'score': float(score),
    #                 'is_unusual': score > 1.0,
    #                 'severity': 'high' if score > 2.0 else 'medium' if score > 1.5 else 'low'
    #             }
            
    #         # Home-optimized decision logic (adjusted for available methods)
    #         decision_result = self._make_home_care_decision_adjusted(
    #             autoencoder_anomaly, isolation_anomaly, outlier_anomaly,
    #             sensor_analysis, sensitivity_config, available_methods
    #         )
            
    #         anomaly_flags[i] = decision_result['is_anomaly']
            
    #         # Generate home-friendly insights
    #         if decision_result['is_anomaly']:
    #             insight = self._generate_home_insight(sensor_analysis, decision_result)
    #             home_insights['anomaly_explanations'].append(insight)
    #             home_insights['severity_levels'].append(decision_result['severity'])
    #             home_insights['recommendations'].append(insight['recommendation'])
    #         else:
    #             home_insights['anomaly_explanations'].append({'status': 'normal'})
    #             home_insights['severity_levels'].append('none')
    #             home_insights['recommendations'].append('Continue monitoring')
            
    #         home_insights['sensor_contributions'].append(sensor_analysis)
        
    #     # Update adaptive components
    #     if adapt:
    #         self._adaptive_update(X, X_scaled, global_errors, drift_score, needs_adaptation)
        
    #     # Update counters
    #     self.samples_seen += n_samples
    #     self.anomalies_detected += np.sum(anomaly_flags)
        
    #     # Drift information
    #     drift_info = {
    #         'drift_score': drift_score,
    #         'needs_adaptation': needs_adaptation,
    #         'samples_seen': self.samples_seen,
    #         'anomaly_rate': np.sum(anomaly_flags) / n_samples,
    #         'environment_status': self._assess_home_environment_status(drift_score, anomaly_flags),
    #         'available_methods': available_methods
    #     }

    #     return anomaly_flags, home_insights, drift_info

    def predict_anomalies_home_optimized(self, X, adapt=True, sensitivity='medium', verbose=True):
        """
        Optimized anomaly detection for home care environments.
        Updated to handle missing outlier detector and adaptive method selection.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call learn_environment() first.")
            
        X_scaled = self.scaler.transform(X)
        n_samples = X.shape[0]
        
        # Detect drift
        drift_score, needs_adaptation = self.detect_drift(X)
        
        # Get predictions from all methods
        reconstructions = self.autoencoder.predict(X_scaled)
        if X_scaled.shape[1] == 1 and reconstructions.ndim == 1:
            reconstructions = reconstructions.reshape(-1, 1)
        reconstruction_errors = (X_scaled - reconstructions) ** 2
        global_errors = np.mean(reconstruction_errors, axis=1)
        
        # Isolation forest
        iso_predictions = self.isolation_forest.predict(X_scaled)
        iso_anomalies = (iso_predictions == -1)
        
        # Outlier detection (handle missing detector)
        if self.outlier_detector is not None:
            outlier_predictions = self.outlier_detector.predict(X_scaled)
            outlier_anomalies = (outlier_predictions == -1)
            available_methods = 3
        else:
            outlier_anomalies = np.zeros(n_samples, dtype=bool)
            available_methods = 2
        
        # Autoencoder anomalies
        autoencoder_anomalies = global_errors > self.global_threshold
        
        # Home-optimized consensus logic
        anomaly_flags = np.zeros(n_samples, dtype=int)
        home_insights = {
            'anomaly_explanations': [],
            'sensor_contributions': [],
            'severity_levels': [],
            'recommendations': []
        }
        
        for i in range(n_samples):
            # Method predictions
            autoencoder_anomaly = global_errors[i] > self.global_threshold
            isolation_anomaly = iso_anomalies[i]
            outlier_anomaly = outlier_anomalies[i]
            
            # Per-sensor analysis for explanations
            sensor_analysis = {}
            for j, sensor in enumerate(self.sensor_names):
                sensor_error = reconstruction_errors[i, j]
                threshold = self.sensor_thresholds[sensor]
                score = sensor_error / (threshold + 1e-8)
                
                sensor_analysis[sensor] = {
                    'score': float(score),
                    'is_unusual': score > 1.0,
                    'severity': 'high' if score > 2.0 else 'medium' if score > 1.5 else 'low'
                }
            
            # NEW: Adaptive method selection
            if hasattr(self, 'active_methods'):
                method_votes = []
                if 'autoencoder' in self.active_methods:
                    method_votes.append(autoencoder_anomaly)
                if 'isolation' in self.active_methods:
                    method_votes.append(isolation_anomaly)
                if 'outlier' in self.active_methods and self.outlier_detector:
                    method_votes.append(outlier_anomaly)
                
                # For single method, use its decision directly
                if len(self.active_methods) == 1:
                    is_anomaly = any(method_votes)
                else:
                    # For multiple methods, use sensitivity logic
                    required = self._get_required_methods(sensitivity, len(self.active_methods))
                    is_anomaly = sum(method_votes) >= required
                
                anomaly_flags[i] = int(is_anomaly)
                
                # Create decision result for insights
                decision_result = {
                    'is_anomaly': int(is_anomaly),
                    'methods_agreeing': sum(method_votes),
                    'active_methods': len(self.active_methods),
                    'severity': 'high' if sum(method_votes) >= len(self.active_methods) else 'medium' if sum(method_votes) > 1 else 'low'
                }
            else:
                # Fallback to original logic if adaptive selection not available
                sensitivity_config = self._get_home_sensitivity_config_adjusted(sensitivity, available_methods)
                decision_result = self._make_home_care_decision_adjusted(
                    autoencoder_anomaly, isolation_anomaly, outlier_anomaly,
                    sensor_analysis, sensitivity_config, available_methods
                )
                anomaly_flags[i] = decision_result['is_anomaly']
            
            # Generate home-friendly insights
            if decision_result['is_anomaly']:
                insight = self._generate_home_insight(sensor_analysis, decision_result)
                home_insights['anomaly_explanations'].append(insight)
                home_insights['severity_levels'].append(decision_result['severity'])
                home_insights['recommendations'].append(insight['recommendation'])
            else:
                home_insights['anomaly_explanations'].append({'status': 'normal'})
                home_insights['severity_levels'].append('none')
                home_insights['recommendations'].append('Continue monitoring')
            
            home_insights['sensor_contributions'].append(sensor_analysis)
        
        # Update adaptive components
        if adapt:
            self._adaptive_update(X, X_scaled, global_errors, drift_score, needs_adaptation)
        
        # Update counters
        self.samples_seen += n_samples
        self.anomalies_detected += np.sum(anomaly_flags)
        
        # Drift information
        drift_info = {
            'drift_score': drift_score,
            'needs_adaptation': needs_adaptation,
            'samples_seen': self.samples_seen,
            'anomaly_rate': np.sum(anomaly_flags) / n_samples,
            'environment_status': self._assess_home_environment_status(drift_score, anomaly_flags),
            'available_methods': available_methods
        }

        return anomaly_flags, home_insights, drift_info

    def _get_home_sensitivity_config_adjusted(self, sensitivity, available_methods):
        """
        Get sensitivity configuration adjusted for available methods.
        """
        if available_methods == 3:
            # Original 3-method configuration
            configs = {
                'low': {'methods_required': 3, 'description': 'Conservative - all 3 methods must agree'},
                'medium': {'methods_required': 2, 'description': 'Balanced - 2 out of 3 methods must agree'},
                'high': {'methods_required': 1, 'description': 'Sensitive - any method can trigger'},
            }
        else:
            # Adjusted 2-method configuration
            configs = {
                'low': {'methods_required': 2, 'description': 'Conservative - both methods must agree'},
                'medium': {'methods_required': 2, 'description': 'Balanced - both methods must agree'},
                'high': {'methods_required': 1, 'description': 'Sensitive - any method can trigger'},
            }
        

        return configs.get(sensitivity, configs['medium'])

    def _make_home_care_decision_adjusted(self, autoencoder_anomaly, isolation_anomaly, 
                                        outlier_anomaly, sensor_analysis, sensitivity_config, 
                                        available_methods):
        """
        Make anomaly decision adjusted for available methods.
        """
        # Count method agreements (only count available methods)
        if available_methods == 3:
            method_votes = [autoencoder_anomaly, isolation_anomaly, outlier_anomaly]
        else:
            method_votes = [autoencoder_anomaly, isolation_anomaly]
        
        methods_agreeing = sum(method_votes)
        
        # Check for high-severity sensor readings (safety override)
        high_severity_sensors = [s for s, data in sensor_analysis.items() 
                            if data['severity'] == 'high']
        
        # Decision logic
        required_methods = sensitivity_config['methods_required']
        
        # Safety override for critical sensors
        safety_override = False
        if high_severity_sensors:
            # If critical sensors show high anomaly, lower threshold
            critical_sensors = ['Temperature', 'CO2', 'Motion', 'S1_distance', 'S2_distance']
            if any(sensor in critical_sensors for sensor in high_severity_sensors):
                safety_override = True
                required_methods = max(1, required_methods - 1)
        
        # Final decision
        is_anomaly = methods_agreeing >= required_methods
        
        # Determine severity
        if (available_methods == 3 and methods_agreeing == 3) or \
        (available_methods == 2 and methods_agreeing == 2) or \
        safety_override:
            severity = 'high'
        elif methods_agreeing >= (available_methods - 1):
            severity = 'medium' 
        else:
            severity = 'low'
        
        return {
            'is_anomaly': int(is_anomaly),
            'methods_agreeing': methods_agreeing,
            'available_methods': available_methods,
            'required_methods': required_methods,
            'severity': severity,
            'safety_override': safety_override,
            'high_severity_sensors': high_severity_sensors
        }

    def _get_home_sensitivity_config(self, sensitivity):
        """
        Get sensitivity configuration optimized for home environments.
        """
        configs = {
            'low': {
                'methods_required': 3,  # All 3 methods must agree
                'description': 'Conservative - fewer false alarms, might miss subtle issues'
            },
            'medium': {
                'methods_required': 2,  # 2 out of 3 methods must agree
                'description': 'Balanced - good for most home situations'
            },
            'high': {
                'methods_required': 1,  # Any method can trigger
                'description': 'Sensitive - catches more issues, may have false alarms'
            },
            'adaptive': {
                'methods_required': 'dynamic',  # Adjusts based on recent performance
                'description': 'Automatically adjusts sensitivity based on environment'
            }
        }
        return configs.get(sensitivity, configs['medium'])

    def _make_home_care_decision(self, autoencoder_anomaly, isolation_anomaly, 
                            outlier_anomaly, sensor_analysis, sensitivity_config):
        """
        Make anomaly decision optimized for home care context.
        """
        # Count method agreements
        method_votes = [autoencoder_anomaly, isolation_anomaly, outlier_anomaly]
        methods_agreeing = sum(method_votes)
        
        # Check for high-severity sensor readings (safety override)
        high_severity_sensors = [s for s, data in sensor_analysis.items() 
                            if data['severity'] == 'high']
        
        # Decision logic
        if sensitivity_config['methods_required'] == 'dynamic':
            # Adaptive threshold based on recent performance
            required_methods = self._get_adaptive_threshold()
        else:
            required_methods = sensitivity_config['methods_required']
        
        # Safety override for critical sensors
        safety_override = False
        if high_severity_sensors:
            # If critical sensors show high anomaly, lower threshold
            critical_sensors = ['Temperature', 'CO2', 'Motion']  # Configurable
            if any(sensor in critical_sensors for sensor in high_severity_sensors):
                safety_override = True
                required_methods = max(1, required_methods - 1)
        
        # Final decision
        is_anomaly = methods_agreeing >= required_methods
        
        # Determine severity
        if methods_agreeing == 3 or safety_override:
            severity = 'high'
        elif methods_agreeing == 2:
            severity = 'medium' 
        else:
            severity = 'low'
        
        return {
            'is_anomaly': int(is_anomaly),
            'methods_agreeing': methods_agreeing,
            'required_methods': required_methods,
            'severity': severity,
            'safety_override': safety_override,
            'high_severity_sensors': high_severity_sensors
        }

    def _generate_home_insight(self, sensor_analysis, decision_result):
        """
        Generate human-friendly explanations for home care context.
        """
        unusual_sensors = [s for s, data in sensor_analysis.items() 
                        if data['is_unusual']]
        
        if not unusual_sensors:
            return {
                'type': 'general_anomaly',
                'message': 'Unusual pattern detected in overall environment',
                'recommendation': 'Monitor for changes in room conditions'
            }
        
        # Generate specific insights based on sensor combinations
        if 'Temperature' in unusual_sensors and 'Humidity' in unusual_sensors:
            return {
                'type': 'climate_issue',
                'message': 'Temperature and humidity levels are unusual',
                'affected_sensors': unusual_sensors,
                'recommendation': 'Check HVAC system, windows, or room ventilation'
            }
        
        elif 'CO2' in unusual_sensors:
            return {
                'type': 'air_quality',
                'message': 'Air quality indicators show unusual patterns',
                'affected_sensors': unusual_sensors,
                'recommendation': 'Check ventilation or occupancy levels'
            }
        
        elif 'Motion' in unusual_sensors:
            return {
                'type': 'activity_pattern',
                'message': 'Unusual activity or movement patterns detected',
                'affected_sensors': unusual_sensors,
                'recommendation': 'Verify occupant well-being and daily routines'
            }
        
        else:
            return {
                'type': 'sensor_specific',
                'message': f'Unusual readings from: {", ".join(unusual_sensors)}',
                'affected_sensors': unusual_sensors,
                'recommendation': 'Check specific sensor locations and equipment'
            }

    def _assess_home_environment_status(self, drift_score, anomaly_flags):
        """
        Assess overall home environment status.
        """
        anomaly_rate = np.mean(anomaly_flags)
        
        if drift_score > 0.5:
            status = 'environment_changing'
            message = 'Home environment patterns are shifting'
        elif anomaly_rate > 0.2:
            status = 'attention_needed'
            message = 'Multiple unusual readings detected'
        elif anomaly_rate > 0.1:
            status = 'monitoring'
            message = 'Some unusual activity detected'
        else:
            status = 'normal'
            message = 'Home environment appears stable'
        
        return {
            'status': status,
            'message': message,
            'anomaly_rate': anomaly_rate,
            'drift_score': drift_score
        }

    def _adaptive_update(self, X, X_scaled, global_errors, drift_score, needs_adaptation):
        """
        Adaptively update model parameters based on new data.
        """
        # Update recent data buffers
        self.recent_samples.extend(X)
        self.recent_errors.extend(global_errors)
        
        # Update thresholds gradually
        if len(self.recent_errors) >= 100:
            self._update_global_threshold()
            
            # Update per-sensor thresholds
            recent_scaled = self.scaler.transform(list(self.recent_samples)[-500:])
            recent_reconstructions = self.autoencoder.predict(recent_scaled)
            recent_sensor_errors = (recent_scaled - recent_reconstructions) ** 2
            
            for i, sensor in enumerate(self.sensor_names):
                if i < recent_sensor_errors.shape[1]:
                    recent_errors = recent_sensor_errors[:, i]
                    new_threshold = np.percentile(recent_errors, 95)
                    
                    # Gradual update (moving average)
                    alpha = 0.1  # Learning rate
                    old_threshold = self.sensor_thresholds[sensor]
                    self.sensor_thresholds[sensor] = (1 - alpha) * old_threshold + alpha * new_threshold
        
        # Trigger retraining if needed
        if (needs_adaptation and 
            drift_score > self.auto_retrain_threshold and
            len(self.recent_samples) >= self.min_training_samples):
            
            time_since_retrain = datetime.now() - (self.last_retrain_time or datetime.min)
            if time_since_retrain > timedelta(hours=1):  # Don't retrain too frequently
                print(f"🔄 Triggering automatic retraining (drift: {drift_score:.3f})")
                self._retrain_on_recent_data()

    def _retrain_on_recent_data(self):
        """
        Retrain the model on recent data.
        """
        try:
            # Use recent data for retraining
            recent_data = np.array(list(self.recent_samples))
            
            if len(recent_data) >= self.min_training_samples:
                # print(f"🔄 Retraining on {len(recent_data)} recent samples...")
                self.learn_environment(recent_data, retrain=True)
                print("✅ Retraining completed")
            else:
                print(f"⚠️  Not enough recent data for retraining ({len(recent_data)} < {self.min_training_samples})")
                
        except Exception as e:
            print(f"❌ Retraining failed: {e}")

    def save_adaptive_model(self, filepath=None):
        """
        Save the adaptive model with all its state.
        """
        if filepath is None:
            filepath = os.path.join(self.base_path,"adaptive_model")
            
        # Prepare model data
        model_data = {
            'autoencoder': self.autoencoder,
            'isolation_forest': self.isolation_forest,
            'outlier_detector': self.outlier_detector,
            'scaler': self.scaler,
            'sensor_names': self.sensor_names,
            'global_threshold': self.global_threshold,
            'sensor_thresholds': self.sensor_thresholds,
            'baseline_stats': self.baseline_stats,
            'current_contamination': self.current_contamination,
            'adaptive_threshold_multiplier': self.adaptive_threshold_multiplier,
            'is_trained': self.is_trained,
            'samples_seen': self.samples_seen,
            'anomalies_detected': self.anomalies_detected,
            'last_retrain_time': self.last_retrain_time,
            'feature_importance': self.feature_importance
        }
        
        # Save with joblib
        joblib.dump(model_data, f"{filepath}.pkl")
        
        # Save configuration
        config = {
            'sensor_names': self.sensor_names,
            'adaptation_window': self.adaptation_window,
            'min_training_samples': self.min_training_samples,
            'drift_detection_threshold': self.drift_detection_threshold,
            'auto_retrain_threshold': self.auto_retrain_threshold,
            'case_name': self.case_name
        }
        
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(config, f, indent=2, default=str)
            
        # print(f"✅ Adaptive model saved to {filepath}")
        return self

    def load_adaptive_model(self, filepath=None):
        """
        Load the adaptive model with all its state.
        """
        if filepath is None:
            filepath = os.path.join(self.base_path, "adaptive_model")
            
        # Load model data
        model_data = joblib.load(f"{filepath}.pkl")
        
        # Restore model components
        self.autoencoder = model_data['autoencoder']
        self.isolation_forest = model_data['isolation_forest']
        self.outlier_detector = model_data['outlier_detector']
        self.scaler = model_data['scaler']
        self.sensor_names = model_data['sensor_names']
        self.global_threshold = model_data['global_threshold']
        self.sensor_thresholds = model_data['sensor_thresholds']
        self.baseline_stats = model_data['baseline_stats']
        self.current_contamination = model_data['current_contamination']
        self.adaptive_threshold_multiplier = model_data['adaptive_threshold_multiplier']
        self.is_trained = model_data['is_trained']
        self.samples_seen = model_data['samples_seen']
        self.anomalies_detected = model_data['anomalies_detected']
        self.last_retrain_time = model_data['last_retrain_time']
        
        # print(f"✅ Adaptive model loaded from {filepath}")
        # print(f"📊 Model state: Trained={self.is_trained}, Samples={self.samples_seen}")
        return self

    def get_status_report(self):
        """
        Get a comprehensive status report of the adaptive system.
        """
        if not self.is_trained:
            return {"status": "not_trained", "message": "Model needs training"}
            
        recent_drift = np.mean(list(self.drift_scores)[-10:]) if self.drift_scores else 0
        
        report = {
            "status": "operational",
            "environment": self.case_name,
            "sensors": self.sensor_names,
            "samples_processed": self.samples_seen,
            "anomalies_detected": self.anomalies_detected,
            "overall_anomaly_rate": self.anomalies_detected / max(1, self.samples_seen),
            "current_contamination": self.current_contamination,
            "global_threshold": self.global_threshold,
            "recent_drift_score": recent_drift,
            "last_retrain": self.last_retrain_time,
            "model_components": {
                "autoencoder_trained": self.autoencoder is not None,
                "isolation_forest_trained": self.isolation_forest is not None,
                "outlier_detector_trained": self.outlier_detector is not None
            },
            "thresholds": self.sensor_thresholds,
            "drift_status": {
                "score": recent_drift,
                "needs_attention": recent_drift > self.drift_detection_threshold,
                "auto_retrain_enabled": recent_drift > self.auto_retrain_threshold
            }
        }
        
        return report

    def recalibrate_threshold(self, recent_data, target_anomaly_rate=0.05):
        """Adjust threshold based on recent data distribution"""
        X_scaled = self.scaler.transform(recent_data)
        reconstructed = self.autoencoder.predict(X_scaled)
        errors = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        new_threshold = np.percentile(errors, (1 - target_anomaly_rate) * 100)
        self.detection_threshold = new_threshold
        return new_threshold

    def analyze_method_contributions(self, X_val, y_val, sensitivity='medium'):
        """
        Analyze per-method contributions on validation data.
        Reports precision, recall, F1 for Autoencoder, Isolation Forest, and EllipticEnvelope.
        """
        print(f"\n🔍 Analyzing method contributions on validation set (sensitivity={sensitivity.upper()})")
        if not self.is_trained:
            raise ValueError("Model not trained. Call learn_environment() first.")

        X_scaled = self.scaler.transform(X_val)

        # === Autoencoder predictions ===
        reconstructions = self.autoencoder.predict(X_scaled)

        # 🔧 Memory/shape fix
        if reconstructions.ndim == 1:
            reconstructions = reconstructions.reshape(-1, self.n_features)
        elif reconstructions.shape[1] != self.n_features:
            reconstructions = reconstructions.reshape(-1, self.n_features)

        recon_errors = np.mean((X_scaled - reconstructions) ** 2, axis=1)
        ae_preds = (recon_errors > self.global_threshold).astype(int)
        ae_report = classification_report(y_val, ae_preds, output_dict=True, zero_division=0)

        # === Isolation Forest predictions ===
        iso_preds = (self.isolation_forest.predict(X_scaled) == -1).astype(int)
        iso_report = classification_report(y_val, iso_preds, output_dict=True, zero_division=0)

        # === Elliptic Envelope predictions ===
        if self.outlier_detector is not None:
            ee_preds = (self.outlier_detector.predict(X_scaled) == -1).astype(int)
            ee_report = classification_report(y_val, ee_preds, output_dict=True, zero_division=0)
        else:
            ee_report = None

        # === Print results ===
        print(f"\n🔍 Contribution Analysis (Validation, sensitivity={sensitivity.upper()}):")
        def print_metrics(name, report):
            print(f"  {name:18} | "
                f"Precision={report['1']['precision']:.2f} "
                f"Recall={report['1']['recall']:.2f} "
                f"F1={report['1']['f1-score']:.2f}")

        print_metrics("Autoencoder", ae_report)
        print_metrics("Isolation Forest", iso_report)
        if ee_report:
            print_metrics("Elliptic Envelope", ee_report)
        else:
            print("  Elliptic Envelope   | Not available")

        print("\n  Ensemble (current logic): run normal predict_anomalies_home_optimized() to compare.")

    def _select_optimal_methods(self, X_val, y_val):
        """
        Determine which methods to use based on validation performance.
        Sets self.active_methods and self.method_weights.
        """
        from sklearn.metrics import classification_report
        
        X_val_scaled = self.scaler.transform(X_val)
        
        # Test each method individually
        methods_performance = {}
        
        # Autoencoder
        reconstructions = self.autoencoder.predict(X_val_scaled)
        if reconstructions.shape != X_val_scaled.shape:
            reconstructions = reconstructions.reshape(X_val_scaled.shape)
        recon_errors = np.mean((X_val_scaled - reconstructions) ** 2, axis=1)
        ae_preds = (recon_errors > self.global_threshold).astype(int)
        ae_report = classification_report(y_val, ae_preds, output_dict=True, zero_division=0)
        methods_performance['autoencoder'] = ae_report['1']['f1-score']
        
        # Isolation Forest
        iso_preds = (self.isolation_forest.predict(X_val_scaled) == -1).astype(int)
        iso_report = classification_report(y_val, iso_preds, output_dict=True, zero_division=0)
        methods_performance['isolation'] = iso_report['1']['f1-score']
        
        # Elliptic Envelope
        if self.outlier_detector is not None:
            ee_preds = (self.outlier_detector.predict(X_val_scaled) == -1).astype(int)
            ee_report = classification_report(y_val, ee_preds, output_dict=True, zero_division=0)
            methods_performance['outlier'] = ee_report['1']['f1-score']
        
        # Find best method
        best_method = max(methods_performance.keys(), key=lambda k: methods_performance[k])
        best_f1 = methods_performance[best_method]
        
        # Decision logic: use best single method if it's significantly better
        if best_f1 > 0.1:  # Minimum threshold for usable method
            self.active_methods = [best_method]
            print(f"Selected single best method: {best_method} (F1={best_f1:.3f})")
        else:
            # Fallback to all methods if all perform poorly
            self.active_methods = list(methods_performance.keys())
            print(f"Using all methods (best F1={best_f1:.3f} too low)")
        
        # Store for reference
        self.method_performance = methods_performance

    def _get_required_methods(self, sensitivity, total_methods):
        """Get required number of methods based on sensitivity and available methods"""
        if total_methods == 1:
            return 1
        elif sensitivity == 'low':
            return total_methods  # All must agree
        elif sensitivity == 'medium':
            return max(1, total_methods - 1)  # N-1 must agree
        else:  # high
            return 1  # Any can trigger
