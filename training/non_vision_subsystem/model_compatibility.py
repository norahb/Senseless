# non_vision_subsystem/model_compatibility.py
"""
Model Compatibility Module
Handles loading and conversion between legacy and adaptive autoencoder formats
"""

import os
import joblib
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional

class ModelCompatibilityManager:
    """
    Manages compatibility between legacy and adaptive autoencoder models.
    Provides unified interface for loading models regardless of format.
    """
    
    def __init__(self, config):
        self.config = config
        self.model_dir = f"models/{config.name}"
        self.model_format = None  # Will be detected automatically
        
    def detect_model_format(self) -> str:
        """
        Detect which model format is available.
        
        Returns:
        --------
        str : 'adaptive', 'legacy', or 'none'
        """
        adaptive_path = os.path.join(self.model_dir, f"adaptive_{self.config.name}.pkl")
        legacy_autoencoder_path = os.path.join(self.model_dir, f"{self.config.name}_enhanced_autoencoder.joblib")
        
        if os.path.exists(adaptive_path):
            return 'adaptive'
        elif os.path.exists(legacy_autoencoder_path):
            return 'legacy'
        else:
            return 'none'

    def load_model_components(self):
        """Simple fix for loading adaptive models"""
        try:
            # Check for adaptive model first
            adaptive_path = f"models/{self.config.name}/adaptive_{self.config.name}.pkl"
            if os.path.exists(adaptive_path):
                print("🔄 Loading adaptive autoencoder model...")
                
                # Load the model data directly
                model_data = joblib.load(adaptive_path)
                
                # Create simple wrapper to preserve components
                class SimpleModelWrapper:
                    def __init__(self, data):
                        self.autoencoder = data['autoencoder']
                        self.isolation_forest = data['isolation_forest'] 
                        self.outlier_detector = data['outlier_detector']
                        self.scaler = data['scaler']
                        self.global_threshold = data['global_threshold']
                        self.sensor_thresholds = data['sensor_thresholds']
                        self.feature_importance = data['feature_importance']
                    
                    def predict(self, X):
                        return self.autoencoder.predict(X)
                
                model = SimpleModelWrapper(model_data)
                scaler = model_data['scaler']
                thresholds = {
                    'global_threshold': model_data['global_threshold'],
                    'sensor_thresholds': model_data['sensor_thresholds']
                }
                feature_importance = model_data['feature_importance']
                
                self.model_format = "adaptive"
                print("✅ Successfully loaded adaptive model components")
                return model, scaler, thresholds, feature_importance
                
        except Exception as e:
            print(f"❌ Failed to load adaptive model: {e}")
        
        # Your existing fallback code here...
        return self._load_legacy_format()    
    # def load_model_components(self) -> Tuple[Any, Any, Dict, Dict]:
    #     """
    #     Load model components in a unified format.
        
    #     Returns:
    #     --------
    #     tuple : (model, scaler, thresholds, feature_importance)
    #         - model: The autoencoder model (with .autoencoder attribute)
    #         - scaler: The data scaler
    #         - thresholds: Dictionary of sensor thresholds
    #         - feature_importance: Dictionary of feature importance scores
    #     """
    #     self.model_format = self.detect_model_format()
        
    #     if self.model_format == 'none':
    #         raise FileNotFoundError(f"No model found in {self.model_dir}")
        
    #     if self.model_format == 'adaptive':
    #         return self._load_adaptive_model()
    #     else:
    #         return self._load_legacy_model()
    
    def _load_adaptive_model(self) -> Tuple[Any, Any, Dict, Dict]:
        """Load adaptive autoencoder model."""
        print("🔄 Loading adaptive autoencoder model...")
        
        adaptive_path = os.path.join(self.model_dir, f"adaptive_{self.config.name}.pkl")
        model_data = joblib.load(adaptive_path)
        
        # Extract components
        autoencoder = model_data['autoencoder']
        scaler = model_data['scaler']
        
        # Get thresholds
        if 'sensor_thresholds' in model_data:
            thresholds = model_data['sensor_thresholds']
        else:
            # Fallback to global threshold
            global_threshold = model_data.get('global_threshold', 0.01)
            thresholds = {sensor: global_threshold for sensor in self.config.sensor_cols}
        
        # Get or create feature importance
        if 'feature_importance' in model_data:
            feature_importance = model_data['feature_importance']
        else:
            feature_importance = self._create_default_feature_importance()
        
        # Create wrapper for compatibility
        class AdaptiveModelWrapper:
            def __init__(self, autoencoder):
                self.autoencoder = autoencoder
        
        model = AdaptiveModelWrapper(autoencoder)
        
        print("✅ Successfully loaded adaptive model components")
        return model, scaler, thresholds, feature_importance
    
    def _load_legacy_model(self) -> Tuple[Any, Any, Dict, Dict]:
        """Load legacy autoencoder model."""
        print("🔄 Loading legacy autoencoder model...")
        
        # Load legacy components
        model_path = os.path.join(self.model_dir, f"{self.config.name}_enhanced_autoencoder.joblib")
        scaler_path = os.path.join(self.model_dir, f"{self.config.name}_scaler.joblib")
        threshold_path = os.path.join(self.model_dir, f"{self.config.name}_thresholds.json")
        feature_importance_path = os.path.join(self.model_dir, f"{self.config.name}_feature_importance.json")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(threshold_path, 'r') as f:
            thresholds = json.load(f)
        
        # Load feature importance or create default
        if os.path.exists(feature_importance_path):
            with open(feature_importance_path, 'r') as f:
                feature_importance = json.load(f)
        else:
            print("⚠️ Feature importance not found, creating default...")
            feature_importance = self._create_default_feature_importance()
            # Save for future use
            with open(feature_importance_path, 'w') as f:
                json.dump(feature_importance, f, indent=2)
        
        print("✅ Successfully loaded legacy model components")
        return model, scaler, thresholds, feature_importance
    
    def _create_default_feature_importance(self) -> Dict[str, float]:
        """Create default feature importance based on sensor metadata."""
        feature_importance = {}
        
        for sensor in self.config.sensor_cols:
            sensor_type = self.config.sensor_metadata.get(sensor, '').upper()
            if sensor_type == 'IRS':
                feature_importance[sensor] = 1.0  # IRS sensors get full weight
            else:
                feature_importance[sensor] = 0.8  # DRS sensors get slightly lower weight
        
        # Normalize to sum to number of sensors (for weighted average)
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance * len(self.config.sensor_cols) 
                                for k, v in feature_importance.items()}
        else:
            # Equal weights if all zero
            feature_importance = {sensor: 1.0 for sensor in self.config.sensor_cols}
        
        print(f"📊 Created default feature importance: {feature_importance}")
        return feature_importance
    
    def load_delays(self) -> Dict[str, Any]:
        """
        Load delays with fallback hierarchy.
        
        Returns:
        --------
        dict : Delay configuration with all required keys
        """
        # Try calibrated delays first
        calibrated_path = os.path.join(self.model_dir, f"{self.config.name}_calibrated_delays.json")
        if os.path.exists(calibrated_path):
            with open(calibrated_path, 'r') as f:
                delays = json.load(f)
            print("✅ Loaded calibrated delays")
            return delays
        
        # Try lab delays
        lab_path = os.path.join(self.model_dir, f"{self.config.name}_delays.json")
        if os.path.exists(lab_path):
            with open(lab_path, 'r') as f:
                lab_delays = json.load(f)
            
            # Create calibrated delays structure with lab delays
            delays = {
                "lab_baseline_delays": lab_delays,
                "measured_onsite_delays": {s: 0.0 for s in self.config.sensor_cols},
                "lab+onsite_delays": lab_delays,
                "final_calibrated_delays_with_reference_sensor": lab_delays,
                "human_adjusted_delays": {}
            }
            print("✅ Using lab delays as fallback")
            return delays
        
        # Default zero delays
        print("⚠️ No delays found, using zero delays...")
        delays = {
            "lab_baseline_delays": {s: 0.0 for s in self.config.sensor_cols},
            "measured_onsite_delays": {s: 0.0 for s in self.config.sensor_cols},
            "lab+onsite_delays": {s: 0.0 for s in self.config.sensor_cols},
            "final_calibrated_delays_with_reference_sensor": {s: 0.0 for s in self.config.sensor_cols},
            "human_adjusted_delays": {}
        }
        return delays
    
    def get_delay_for_sensor(self, delays: Dict, sensor: str) -> float:
        """
        Get delay for a specific sensor with fallback hierarchy.
        
        Parameters:
        -----------
        delays : dict
            Delay configuration
        sensor : str
            Sensor name
            
        Returns:
        --------
        float : Delay value in seconds
        """
        # Try different delay sources in order of preference
        delay_sources = [
            "final_calibrated_delays_with_reference_sensor",
            "lab+onsite_delays", 
            "lab_baseline_delays",
            "measured_onsite_delays"
        ]
        
        for source in delay_sources:
            if source in delays and sensor in delays[source]:
                delay_val = delays[source][sensor]
                if delay_val is not None:
                    return float(delay_val)
        
        print(f"⚠️ No delay found for sensor '{sensor}', using 0.0")
        return 0.0
    
    def get_threshold_for_sensor(self, thresholds: Dict, sensor: str) -> float:
        """
        Get threshold for a specific sensor.
        
        Parameters:
        -----------
        thresholds : dict or float
            Threshold configuration
        sensor : str
            Sensor name
            
        Returns:
        --------
        float : Threshold value
        """
        if isinstance(thresholds, dict):
            return thresholds.get(sensor, 0.01)  # Default fallback
        else:
            return float(thresholds)  # Single threshold value
    
    def save_adaptive_format(self, model, scaler, thresholds, feature_importance, 
                           additional_data: Optional[Dict] = None):
        """
        Save model in adaptive format for future use.
        
        Parameters:
        -----------
        model : object
            Model with .autoencoder attribute
        scaler : object
            Data scaler
        thresholds : dict
            Sensor thresholds
        feature_importance : dict
            Feature importance scores
        additional_data : dict, optional
            Additional data to save
        """
        # Prepare model data
        model_data = {
            'autoencoder': model.autoencoder,
            'scaler': scaler,
            'sensor_thresholds': thresholds,
            'feature_importance': feature_importance,
            'sensor_names': self.config.sensor_cols,
            'is_trained': True,
            'samples_seen': 0,
            'anomalies_detected': 0,
        }
        
        # Add additional data if provided
        if additional_data:
            model_data.update(additional_data)
        
        # Save adaptive format
        adaptive_path = os.path.join(self.model_dir, f"adaptive_{self.config.name}.pkl")
        joblib.dump(model_data, adaptive_path)
        
        print(f"✅ Saved model in adaptive format: {adaptive_path}")
    
    def convert_legacy_to_adaptive(self):
        """
        Convert legacy model format to adaptive format.
        """
        if self.detect_model_format() != 'legacy':
            print("⚠️ No legacy model found to convert")
            return False
        
        print("🔄 Converting legacy model to adaptive format...")
        
        try:
            # Load legacy components
            model, scaler, thresholds, feature_importance = self.load_model_components()
            
            # Save in adaptive format
            self.save_adaptive_format(model, scaler, thresholds, feature_importance)
            
            print("✅ Successfully converted legacy model to adaptive format")
            return True
            
        except Exception as e:
            print(f"❌ Failed to convert legacy model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about available models.
        
        Returns:
        --------
        dict : Model information
        """
        format_detected = self.detect_model_format()
        
        info = {
            'model_format': format_detected,
            'model_dir': self.model_dir,
            'available_files': [],
            'recommendations': []
        }
        
        # List available files
        if os.path.exists(self.model_dir):
            info['available_files'] = [f for f in os.listdir(self.model_dir) 
                                     if f.endswith(('.pkl', '.joblib', '.json'))]
        
        # Add recommendations
        if format_detected == 'none':
            info['recommendations'].append("Train a model first using train_adaptive_autoencoder")
        elif format_detected == 'legacy':
            info['recommendations'].append("Consider converting to adaptive format using convert_legacy_to_adaptive()")
        elif format_detected == 'adaptive':
            info['recommendations'].append("Model is in optimal adaptive format")
        
        return info


def load_unified_model(config):
    """
    Convenience function to load model components in unified format.
    
    Parameters:
    -----------
    config : object
        Configuration object
        
    Returns:
    --------
    tuple : (model, scaler, thresholds, feature_importance)
    """
    manager = ModelCompatibilityManager(config)
    return manager.load_model_components()


def load_unified_delays(config):
    """
    Convenience function to load delays in unified format.
    
    Parameters:
    -----------
    config : object
        Configuration object
        
    Returns:
    --------
    dict : Delay configuration
    """
    manager = ModelCompatibilityManager(config)
    return manager.load_delays()


# Example usage
if __name__ == "__main__":
    from config.config_manager import ConfigManager
    
    # Example: Check model status for all use cases
    use_cases = ["door", "co2", "appliance", "abnormal_object"]
    
    for use_case in use_cases:
        print(f"\n--- {use_case.upper()} ---")
        try:
            config = ConfigManager.get_config(use_case)
            manager = ModelCompatibilityManager(config)
            info = manager.get_model_info()
            
            print(f"Format: {info['model_format']}")
            print(f"Files: {info['available_files']}")
            for rec in info['recommendations']:
                print(f"💡 {rec}")
                
        except Exception as e:
            print(f"❌ Error: {e}")