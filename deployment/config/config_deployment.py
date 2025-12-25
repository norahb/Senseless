# deployment/config/config_deployment.py

import os
import pathlib as Path
import json
from deployment.config.use_case_configs import USE_CASE_CONFIGS
try:
    from training.config.config_manager import get_cleaning_config
except ImportError:
    get_cleaning_config = lambda config: {}

class DeploymentConfig:
    def __init__(self, use_case_name: str):
        if use_case_name not in USE_CASE_CONFIGS:
            raise ValueError(f"Unknown use case: {use_case_name}")
        
        self.use_case = use_case_name
        self.use_case_config = USE_CASE_CONFIGS[use_case_name]

        # Get the deployment directory (parent of config directory)
        config_dir = os.path.dirname(__file__)  # deployment/config/
        deployment_dir = os.path.dirname(config_dir)  # deployment/
        
        # === System Modes ===
        self.system_mode = "live"
        self.enable_sensor_inference = True
        self.enable_confidence_estimation = True
        self.confidence_method = "entropy"
        self.entropy_threshold = 0.5
        self.enable_vision_fallback = True
        self.enable_human_review = False

        # === Model Paths (absolute paths to avoid redundancy) ===
        # Always load models from deployment folder
        self.sensor_model_path = os.path.abspath(os.path.join(deployment_dir, "models", use_case_name))
        vision_filenames = {
            "door": "door_mobilenetv2.pth",
            "appliance": "appliance_mobilenetv2.pth",
            "abnormal_object": "abnormal_object_mobilenetv2.pth",
            "co2": "teacher.pt",
        }
        default_vision_name = vision_filenames.get(use_case_name, f"{use_case_name}_mobilenetv2.pth")
        self.vision_model_path = os.path.abspath(os.path.join(deployment_dir, "models", use_case_name, default_vision_name))

        # === Data Paths ===
        self.incoming_sensor_data_path = self.use_case_config.inference_csv_path
        self.image_folder_path = self.use_case_config.inference_image_folder
        self.image_timestamp_csv = self.use_case_config.inference_image_csv_path

        # === Output Paths ===
        self.decision_log_path = os.path.join("logs", self.use_case, "decisions.csv")
        self.alert_output_dir = os.path.join("alerts")

        # === Visualization ===
        self.enable_visual_window = True
        self.visual_window_duration = 60

        # === Baseline Management ===
        self.update_baseline_before_drift = True
        self.baseline_data_source = "decision_log"
        self.baseline_validation = True

        # === Drift Detection ===
        self.enable_drift_detection = True
        self.drift_threshold = 0.3
        self.drift_window_size = 100
        self.drift_check_frequency = "daily"

        # === Adaptation ===
        self.enable_adaptation = True
        self.adaptation_frequency = "weekly"

        # === Retraining ===
        # Point to ACTUAL training data
        self.original_training_data_path = self.use_case_config.training_csv_path

        # Temporal windows
        self.recent_window_days = 14
        self.min_samples_for_retraining = 200  # More realistic

        # Sampling ratios
        self.recent_data_ratio = 0.5
        self.historical_data_ratio = 0.3
        self.vision_data_ratio = 0.1
        self.replay_buffer_ratio = 0.1
        self.performance_threshold = 0.01

        # Enable automatic retraining
        self.enable_auto_retraining = True  # Enable automatic retraining by default

        # Load cleaning config from training/config/config_manager.py
        self.cleaning_config = get_cleaning_config(self)

    def get_model_type(self):
        """Determine if using adaptive or legacy model"""
        model_dir = Path(self.sensor_model_path).parent
        production_config_path = model_dir / 'production_config.json'
        
        if production_config_path.exists():
            try:
                with open(production_config_path, 'r') as f:
                    config = json.load(f)
                return config.get('model_type', 'legacy')
            except:
                return 'legacy'
        return 'legacy'