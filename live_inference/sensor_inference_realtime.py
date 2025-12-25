"""
Real-Time Sensor Inference Module

Wraps deployment sensor inference for streaming single-sample predictions.
Loads model once at startup and processes individual sensor readings.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib

# Add training path for model imports
training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training'))
if training_path not in sys.path:
    sys.path.insert(0, training_path)

try:
    from non_vision_subsystem.adaptive_autoencoder import AdaptiveUnsupervisedAutoencoder
except ImportError:
    print("⚠️ Warning: Could not import AdaptiveUnsupervisedAutoencoder")

logger = logging.getLogger(__name__)


class RealtimeSensorInference:
    """
    Real-time sensor inference wrapper.
    
    Features:
    - Single-sample streaming predictions
    - Model loaded once at startup
    - Handles both adaptive and legacy models
    - Outlier detection with physical bounds
    - Supports all use cases (sensor-based or rule-based)
    """
    
    def __init__(self, config):
        """
        Initialize real-time sensor inference.
        
        Args:
            config: LiveInferenceConfig instance
        """
        self.config = config
        self.use_case = config.use_case
        self.sensor_cols = config.use_case_config.sensor_cols
        self.sensor_metadata = config.use_case_config.sensor_metadata
        self.error_thresholds = config.use_case_config.sensor_error_thresholds
        self.mild_outlier_zscore = config.use_case_config.mild_outlier_zscore
        self.extreme_corruption_zscore = config.use_case_config.extreme_corruption_zscore
        
        self.model = None
        self.model_type = None
        self.production_config = None
        self.inference_count = 0
        self.total_outliers_detected = 0
        self.total_sensor_errors = 0
        self.sensor_isotonic = None
        
        # Load model at startup
        self._load_model()
    
    def _load_model(self):
        """Load model once at startup."""
        try:
            # Log environment versions for debugging pickle compatibility
            try:
                import numpy as _np
                import joblib as _jb
                logger.info(f"🔢 numpy={getattr(_np,'__version__','?')}, joblib={getattr(_jb,'__version__','?')}")
            except Exception:
                pass

            model_dir = Path(self.config.sensor_model_path)
            production_config_path = model_dir / 'production_config.json'
            
            if not production_config_path.exists():
                logger.warning(f"⚠️ Production config not found: {production_config_path}")
                self.model_type = 'legacy'
            else:
                with open(production_config_path, 'r') as f:
                    self.production_config = json.load(f)
                self.model_type = self.production_config.get('model_type', 'legacy')
            
            # Load adaptive model
            if self.model_type == 'adaptive_autoencoder':
                logger.info(f"🧠 Loading adaptive autoencoder for {self.use_case}...")
                self.model = AdaptiveUnsupervisedAutoencoder(
                    sensor_names=self.sensor_cols,
                    case_name=self.use_case
                )
                model_path = model_dir / f'adaptive_{self.use_case}'
                try:
                    self.model.load_adaptive_model(str(model_path))
                    logger.info(f"✅ Loaded adaptive model from {model_path}")
                    # Load mandatory isotonic calibration model
                    iso_path = model_dir / 'sensor_isotonic.pkl'
                    if not iso_path.exists():
                        logger.error(f"❌ Required isotonic calibration missing: {iso_path}")
                        self.sensor_isotonic = None
                    else:
                        try:
                            self.sensor_isotonic = joblib.load(str(iso_path))
                            logger.info(f"✅ Loaded isotonic calibration from {iso_path}")
                        except Exception as iso_err:
                            logger.error(f"❌ Failed to load isotonic calibration: {iso_err}")
                            self.sensor_isotonic = None
                except Exception as e:
                    logger.error(f"❌ Failed to load model: {e}")
                    # Common cause: numpy version mismatch for pickled BitGenerator
                    logger.warning(
                        "⚠️ Adaptive model load failed. If error mentions MT19937/BitGenerator, "
                        "please update the environment's numpy (>=1.17) and joblib to match the model's pickled version."
                    )
                    # Fallback to simple inference until environment is fixed
                    self.model = None
                    self.model_type = 'adaptive_fallback'
            else:
                logger.info(f"🧠 Loading legacy model for {self.use_case}...")
                model_path = model_dir / f'{self.use_case}_model.pkl'
                if not model_path.exists():
                    # Try alternative names
                    model_path = model_dir / f'{self.use_case}.pkl'
                    if not model_path.exists():
                        raise FileNotFoundError(f"Model not found in {model_dir}")
                
                self.model = joblib.load(str(model_path))
                logger.info(f"✅ Loaded legacy model from {model_path}")
        
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _detect_sensor_error(self, sensor_name: str, value: float) -> bool:
        """
        Check if sensor reading violates physical bounds.
        
        Args:
            sensor_name: Name of sensor
            value: Reading value
            
        Returns:
            True if sensor error detected, False otherwise
        """
        if sensor_name not in self.error_thresholds:
            return False
        
        bounds = self.error_thresholds[sensor_name]
        min_val = bounds.get("min", float('-inf'))
        max_val = bounds.get("max", float('inf'))
        
        return value < min_val or value > max_val
    
    def _detect_outlier(self, sensor_values: np.ndarray) -> Tuple[bool, bool]:
        """
        Detect outliers using Z-score (uses MAD for robustness).
        
        Args:
            sensor_values: Array of single sensor's recent history
            
        Returns:
            (is_mild_outlier, is_extreme_outlier)
        """
        if len(sensor_values) < 2:
            return False, False
        
        # Use Median Absolute Deviation (MAD) for robustness
        median = np.median(sensor_values)
        mad = np.median(np.abs(sensor_values - median))
        
        if mad == 0:
            return False, False
        
        z_score = 0.6745 * (sensor_values[-1] - median) / mad
        
        is_mild = abs(z_score) > self.mild_outlier_zscore
        is_extreme = abs(z_score) > self.extreme_corruption_zscore
        
        return is_mild, is_extreme
    
    def infer_sample(
        self,
        sensor_data: Dict[str, float],
        sensor_history: Optional[Dict[str, list]] = None
    ) -> Dict:
        """
        Run inference on a single sensor sample.
        
        Args:
            sensor_data: Dict mapping sensor name -> value
            sensor_history: Optional rolling history for outlier detection
            
        Returns:
            Dict with:
                - status: 'Normal' | 'Anomaly' | 'Sensor_Error'
                - confidence: float [0-1]
                - reconstruction_error: float or None
                - details: str
                - sensor_validation: Dict of validation flags
        """
        self.inference_count += 1
        
        result = {
            "status": "Normal",
            "confidence": 1.0,
            "reconstruction_error": None,
            "details": "",
            "sensor_validation": {}
        }
        
        try:
            # === Validate sensor readings ===
            for sensor in self.sensor_cols:
                if sensor not in sensor_data:
                    logger.warning(f"⚠️ Missing sensor {sensor}")
                    result["sensor_validation"][sensor] = "Missing"
                    result["status"] = "Sensor_Error"
                    self.total_sensor_errors += 1
                    return result
                
                value = sensor_data[sensor]
                
                # Check physical bounds
                if self._detect_sensor_error(sensor, value):
                    logger.warning(f"⚠️ Sensor {sensor}={value} violates physical bounds")
                    result["sensor_validation"][sensor] = "Out_of_Bounds"
                    result["status"] = "Sensor_Error"
                    self.total_sensor_errors += 1
                    return result
                
                # Check for outliers (if history available)
                if sensor_history and sensor in sensor_history:
                    history = sensor_history[sensor]
                    is_mild, is_extreme = self._detect_outlier(np.array(history + [value]))
                    
                    if is_extreme:
                        logger.warning(f"⚠️ Sensor {sensor}={value} is extreme outlier")
                        result["sensor_validation"][sensor] = "Extreme_Outlier"
                        result["status"] = "Sensor_Error"
                        self.total_sensor_errors += 1
                        self.total_outliers_detected += 1
                        return result
                    
                    elif is_mild:
                        logger.debug(f"⚠️ Sensor {sensor}={value} is mild outlier")
                        result["sensor_validation"][sensor] = "Mild_Outlier"
                    else:
                        result["sensor_validation"][sensor] = "Valid"
                else:
                    result["sensor_validation"][sensor] = "Valid"
            
            # === Run inference ===
            if self.model_type == 'adaptive_autoencoder':
                result = self._infer_adaptive(sensor_data, result)
            elif self.model_type == 'adaptive_fallback':
                result = self._infer_simple(sensor_data, result, sensor_history)
            else:
                result = self._infer_legacy(sensor_data, result)
            
            logger.debug(
                f"📊 Inference #{self.inference_count}: "
                f"{result['status']} (conf={result['confidence']:.3f})"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            result["status"] = "Sensor_Error"
            result["details"] = str(e)
            self.total_sensor_errors += 1
            return result
    
    def _infer_adaptive(self, sensor_data: Dict[str, float], result: Dict) -> Dict:
        """Run adaptive autoencoder inference."""
        try:
            # Prepare input
            X = np.array([[sensor_data[col] for col in self.sensor_cols]])
            X_scaled = self.model.scaler.transform(X)
            
            # Reconstruct (sklearn MLPRegressor.predict has no verbose arg)
            X_reconstructed = self.model.autoencoder.predict(X_scaled)
            
            # Calculate reconstruction error
            reconstruction_error = np.mean(np.power(X_scaled - X_reconstructed, 2))
            result["reconstruction_error"] = float(reconstruction_error)
            
            # Get threshold (from production config or default)
            optimal_sensitivity = self.production_config.get('optimal_sensitivity', 'medium')
            thresholds = {
                'low': 0.05,
                'medium': 0.02,
                'high': 0.01
            }
            threshold = thresholds.get(optimal_sensitivity, 0.02)
            
            # Determine anomaly
            if reconstruction_error > threshold:
                result["status"] = "Anomaly"
                result["confidence"] = min(1.0, reconstruction_error / threshold)
            else:
                result["status"] = "Normal"
                result["confidence"] = 1.0 - (reconstruction_error / threshold)

            # Apply mandatory isotonic calibration
            if self.sensor_isotonic is None:
                raise RuntimeError("Isotonic calibration model missing (required)")
            try:
                calibrated = float(self.sensor_isotonic.transform(np.array([result["confidence"]]))[0])
                # Clip to [0,1]
                result["confidence"] = max(0.0, min(1.0, calibrated))
            except Exception as iso_e:
                raise RuntimeError(f"Isotonic calibration failed: {iso_e}")
            
            result["details"] = f"recon_error={reconstruction_error:.6f}, threshold={threshold:.6f}"
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Adaptive inference failed: {e}")
            result["status"] = "Sensor_Error"
            result["details"] = str(e)
            self.total_sensor_errors += 1
            return result
    
    def _infer_legacy(self, sensor_data: Dict[str, float], result: Dict) -> Dict:
        """Run legacy model inference."""
        try:
            # Prepare input
            X = np.array([[sensor_data[col] for col in self.sensor_cols]])
            
            # Check if model is isolation forest or other
            if hasattr(self.model, 'predict'):
                # For Isolation Forest and similar
                prediction = self.model.predict(X)[0]
                # predict returns -1 for anomaly, 1 for normal
                result["status"] = "Anomaly" if prediction == -1 else "Normal"
                result["confidence"] = 0.7  # Legacy models don't provide confidence
            else:
                # Fallback
                result["status"] = "Normal"
                result["confidence"] = 0.5
            
            result["details"] = "legacy_model_prediction"
            
            return result
        except Exception as e:
            logger.error(f"❌ Legacy inference failed: {e}")
            result["status"] = "Sensor_Error"
            result["details"] = str(e)
            self.total_sensor_errors += 1
            return result

    def _infer_simple(self, sensor_data: Dict[str, float], result: Dict, sensor_history: Optional[Dict[str, list]]) -> Dict:
        """Rule-based fallback inference when adaptive model cannot be loaded."""
        try:
            # If any sensor was marked error earlier, keep status
            if result.get("status") == "Sensor_Error":
                result["confidence"] = 1.0
                result["details"] = "rule_based_sensor_error"
                return result

            # Use mild outlier check if history is present on any sensor
            outlier_flags = []
            for sensor in self.sensor_cols:
                val = sensor_data.get(sensor)
                hist = (sensor_history or {}).get(sensor, [])
                if hist:
                    is_mild, is_extreme = self._detect_outlier(np.array(hist + [val]))
                    outlier_flags.append((is_mild, is_extreme))

            if any(ext for _, ext in outlier_flags):
                result["status"] = "Sensor_Error"
                result["confidence"] = 1.0
                result["details"] = "rule_based_extreme_outlier"
            elif any(mild for mild, _ in outlier_flags):
                result["status"] = "Anomaly"
                result["confidence"] = 0.6
                result["details"] = "rule_based_mild_outlier"
            else:
                result["status"] = "Normal"
                result["confidence"] = 0.8
                result["details"] = "rule_based_normal"

            return result
        except Exception as e:
            logger.error(f"❌ Simple inference failed: {e}")
            result["status"] = "Sensor_Error"
            result["confidence"] = 1.0
            result["details"] = str(e)
            return result
    
    def get_stats(self) -> Dict:
        """Return inference statistics."""
        return {
            "inference_count": self.inference_count,
            "total_outliers_detected": self.total_outliers_detected,
            "total_sensor_errors": self.total_sensor_errors,
            "error_rate": (
                self.total_sensor_errors / self.inference_count
                if self.inference_count > 0 else 0
            ),
        }
