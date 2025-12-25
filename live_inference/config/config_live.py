"""
Live Inference Configuration Module

Extends DeploymentConfig with real-time specific settings:
  - Arduino serial connection parameters
  - Image capture settings
  - Per-use-case persistence flags
  - Buffer and retraining data logging
"""

import os
from pathlib import Path
from deployment.config.config_deployment import DeploymentConfig
import platform


class LiveInferenceConfig(DeploymentConfig):
    """
    Configuration for real-time live inference pipeline.
    
    Extends deployment config with:
    - Arduino/serial communication settings
    - Image capture parameters
    - Persistence control (whether to keep captured images)
    - Retraining data logging
    """
    
    def __init__(self, use_case_name: str, arduino_port: str = None):
        """
        Initialize live inference configuration.
        
        Args:
            use_case_name: Use case identifier (door, co2, appliance, abnormal_object)
            arduino_port: Serial port for Arduino (e.g., '/dev/rfcomm0' or 'COM3')
                          If None, uses configured default
        """
        super().__init__(use_case_name)
        
        # Get the live_inference directory (parent of this config file)
        config_dir = os.path.dirname(__file__)  # live_inference/config/
        live_inference_dir = os.path.dirname(config_dir)  # live_inference/
        # Project root is the parent of live_inference (don't step up twice)
        root_dir = os.path.dirname(live_inference_dir)
        
        # === Arduino Serial Communication ===
        self.arduino_port = arduino_port or self._get_default_arduino_port()
        self.arduino_baudrate = 9600  # HC-05 Bluetooth standard
        self.arduino_timeout = 5  # seconds
        self.arduino_max_retry_delay = 30  # seconds
        self.arduino_heartbeat_interval = 5  # seconds
        
        # === Image Capture Settings ===
        self.enable_image_capture = True
        self.image_capture_resolution = (256, 256)
        self.image_capture_format = "jpg"  # jpg or png
        self.image_capture_quality = 90  # JPEG quality (1-100)
        self.image_capture_timeout = 5  # seconds max to capture
        self.image_capture_max_time_window = 60  # seconds (match images ±60s from sensor)
        
        # === Image Persistence (Per-Use-Case) ===
        # Control whether to save fallback images or discard after inference
        persistence_defaults = {
            "door": False,        # Don't save images (vision only for fallback)
            "co2": True,          # Save images for authorized person detection
            "appliance": False,   # Don't save images (vision only for fallback)
            "abnormal_object": False  # Don't save images (vision only for fallback)
        }
        self.save_fallback_images = persistence_defaults.get(use_case_name, False)
        self.save_anomaly_images_only = True  # Only save if anomaly detected
        
        # === Retraining Data Logging ===
        self.log_incoming_sensor_data = True  # Always log for retraining
        self.sensor_data_log_dir = os.path.join(
            root_dir, "live_inference", "logs", use_case_name
        )
        self.sensor_data_log_path = os.path.join(
            self.sensor_data_log_dir, "incoming_sensor_data.csv"
        )
        self.sensor_data_buffer_dir = os.path.join(
            root_dir, "live_inference", "buffers", use_case_name
        )
        
        # === Live Inference Paths ===
        self.live_inference_logs_dir = os.path.join(
            root_dir, "live_inference", "logs", use_case_name
        )
        self.live_inference_alerts_dir = os.path.join(
            root_dir, "live_inference", "alerts", use_case_name
        )
        self.live_fallback_images_dir = os.path.join(
            root_dir, "live_inference", "fallback_images", use_case_name
        )
        
        # Create directories if they don't exist
        for dir_path in [
            self.sensor_data_log_dir,
            self.sensor_data_buffer_dir,
            self.live_inference_logs_dir,
            self.live_inference_alerts_dir,
            self.live_fallback_images_dir
        ]:
            os.makedirs(dir_path, exist_ok=True)
        
        # === Processing Settings ===
        self.batch_size = 1  # Process single samples (streaming mode)
        self.confidence_threshold_for_vision_fallback = 0.3  # Use vision if < 30% confidence
        
        # === Monitoring & Logging ===
        self.enable_performance_logging = True
        self.log_inference_latency = True
        self.max_samples_in_buffer = 1000  # Keep rolling buffer for retraining

        # === Camera (Laptop defaults) ===
        # Use OpenCV webcam on non-Pi environments
        self.camera_backend = "opencv"
        self.camera_source = 0
        # Try these indices if the primary source fails
        self.camera_fallback_indices = [0, 1]
        # Optionally show a short preview window during capture
        self.preview_capture = False
        self.preview_delay_ms = 500
        # Compact output (minimal per-sample summary)
        self.compact_output = False
        
    def _get_default_arduino_port(self) -> str:
        """
        Detect Arduino port based on platform.
        
        Returns:
            Port string (e.g., '/dev/rfcomm0' on Linux, 'COM3' on Windows)
        """
        import platform
        import sys
        
        system = platform.system()
        
        # Linux/Raspberry Pi defaults (Bluetooth)
        if system == "Linux":
            return "/dev/rfcomm0"
        
        # macOS defaults
        elif system == "Darwin":
            return "/dev/tty.HC-05"
        
        # Windows defaults
        elif system == "Windows":
            # Try to find COM ports
            for port_num in range(1, 10):
                try:
                    import serial
                    port = f"COM{port_num}"
                    s = serial.Serial(port, timeout=0.1)
                    s.close()
                    return port
                except:
                    continue
            return "COM3"  # Fallback
        
        return "/dev/ttyUSB0"  # Fallback for unknown systems
    
    def override_arduino_port(self, port: str):
        """Override detected Arduino port."""
        self.arduino_port = port
    
    def override_image_persistence(self, save: bool, anomalies_only: bool = True):
        """Override image persistence settings."""
        self.save_fallback_images = save
        self.save_anomaly_images_only = save and anomalies_only
    
    def get_live_config_summary(self) -> dict:
        """Return summary of live inference configuration."""
        # Resolve model directories and default paths explicitly for visibility
        sensor_model_dir = Path(self.sensor_model_path)
        vision_model_default = Path(self.vision_model_path)

        return {
            "use_case": self.use_case,
            "arduino": {
                "port": self.arduino_port,
                "baudrate": self.arduino_baudrate,
                "timeout": self.arduino_timeout,
                "heartbeat_interval": self.arduino_heartbeat_interval,
            },
            "image_capture": {
                "enabled": self.enable_image_capture,
                "resolution": self.image_capture_resolution,
                "format": self.image_capture_format,
                "save_images": self.save_fallback_images,
                "save_anomalies_only": self.save_anomaly_images_only,
            },
            "inference": {
                "batch_size": self.batch_size,
                "confidence_threshold_for_vision_fallback": self.confidence_threshold_for_vision_fallback,
            },
            "logging": {
                "sensor_data_log": self.sensor_data_log_path,
                "inference_logs": self.live_inference_logs_dir,
                "alerts_dir": self.live_inference_alerts_dir,
                "fallback_images_dir": self.live_fallback_images_dir,
            },
            "models": {
                "sensor_model_dir": str(sensor_model_dir),
                "vision_model_path": str(vision_model_default),
            },
            "camera": {
                "backend": getattr(self, "camera_backend", "opencv"),
                "source": getattr(self, "camera_source", 0),
                "fallback_indices": getattr(self, "camera_fallback_indices", [0, 1]),
                "preview_capture": getattr(self, "preview_capture", False),
                "preview_delay_ms": getattr(self, "preview_delay_ms", 500),
            },
            "output": {
                "compact": getattr(self, "compact_output", False),
            },
            "performance": {
                "enable_performance_logging": self.enable_performance_logging,
                "log_inference_latency": self.log_inference_latency,
                "max_samples_in_buffer": self.max_samples_in_buffer,
            },
        }
