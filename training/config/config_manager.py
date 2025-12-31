

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path

@dataclass
class UseCaseConfig:
    # Basic info
    name: str
    sensor_data_path: str
    image_data_path: str
    image_folder_path: str
    image_ssl_folder_path: str
    image_training_folder_path: str
    image_validation_folder_path: str 
    image_evaluation_folder_path: str
    inference_csv_path: str = None
    inference_image_folder: str = None
    sensor_cols: List[str] = field(default_factory=list)
    sensor_metadata: Dict[str, str] = field(default_factory=dict)
    is_single_sensor: bool = False  # True for single-sensor cases like CO2

    # Column names in the CSV
    date_col: str = "Date"
    time_col: str = "Time"
    status_col: str = "Status"
    anomaly_value: str = "Anomaly"
    normal_value: str = "Normal"
    image_column_name: str = "Image_Name"


    # model_type: str = "sklearn"   # options: "sklearn", "tf", "lstm"
    quiet_mode: bool = False  # Set to True for minimal output
    skip_training: bool = True

    # === SUBSYSTEM FLAGS ===
    train_autoencoder: bool = False
    train_adaptive_autoencoder: bool = False
    calculate_delays: bool = False
    detect_anomalies: bool = False
    align_images: bool = True
    train_ssl: bool = False
    human_labeling: bool = True
    enable_image_classification_split: bool = True
    train_image_classifier: bool = False
    finetune_image_model: bool = False
    run_image_inference: bool = False

    # Training parameters
    image_classifier_epochs: int = 10
    early_stopping_patience: int = 3
    threshold_percentile: int = 75
    threshold_factor: float = 1.2
    max_detection_window: int = 120
    alignment_window_sec: int = 5  
    refinement_confidence_threshold: float = 0.60
    confidence_level = 0.95  # or adjust as needed

    
    # LSTM AE parameters
    # window_size: int = 60
    # noise_std: float = 0.05
    # dropout_rate: float = 0.2
    # stride: int = 5
    # max_train_windows: int = 50000
    # max_outlier_windows: int = 20000
    
    # === DATA CLEANING CONFIGURATION ===
    enable_training_data_cleaning: bool = True
    clean_val_test_sensor_errors: bool = True
    enable_range_validation: bool = True
    enable_statistical_cleaning: bool = True
    statistical_outlier_method: str = 'mad'    
    statistical_outlier_threshold: float = 5
    sensor_ranges: Optional[Dict[str, tuple]] = field(default_factory=dict)

    # === INCREMENTAL TRAINING CONFIGURATION ===
    incremental_training: bool = True
    incremental_chunk_days: int = 3
    incremental_overlap_hours: int = 6
    memory_buffer_size: int = 2000

class ConfigManager:
    base_path = Path(r".\training\data")

    # base_path = folder where main_training.py (and your data) lives
    # base_path = Path(__file__).resolve().parent.parent   # goes up to project_root

    configs = {
        "appliance": UseCaseConfig(
            name="appliance",
            sensor_data_path=base_path / "sensor_data" / "appliance" / "Appliance_June_2025_training_split.csv", 
            image_data_path=base_path / "sensor_data" / "appliance" / "Appliance_images_June2025.csv",
            image_folder_path=base_path / "images" / "appliance" / "image_full",
            image_ssl_folder_path=base_path / "images" / "appliance" / "image_training_ssl_splitted_noclasses",
            image_training_folder_path=base_path / "images" / "appliance" / "image_training_dataset" / "train",
            image_validation_folder_path=base_path / "images" / "appliance" / "image_training_dataset" / "val",
            image_evaluation_folder_path=base_path / "images" / "appliance" / "image_training_dataset" / "test",
            inference_csv_path=base_path / "sensor_data" / "appliance" / "Appliance_images_June2025.csv",
            inference_image_folder=base_path / "images" / "appliance" / "image_full",
            sensor_cols=["Temperature", "Humidity", "CO2"],
            sensor_metadata={"Temperature": "DRS", "Humidity": "DRS", "CO2": "DRS"},
            is_single_sensor=False,
            threshold_percentile = 75,
            threshold_factor = 1.5,
            refinement_confidence_threshold = 0.5,
            # Cleaning config for appliance environment
            enable_training_data_cleaning=True,
            enable_range_validation=True,
            enable_statistical_cleaning=False,
            statistical_outlier_method='iqr',
            statistical_outlier_threshold=5,
            # Incremental training config
            incremental_training = False,
            incremental_chunk_days = 5,     # from 3 → 5 days
            incremental_overlap_hours = 12, # from 6 → 12h,
            memory_buffer_size = 2500,  
            sensor_ranges={
                'Temperature': (12, 70),    
                'Humidity': (12, 90),       
                'CO2': (400, 1000)          
            }
        ),
        "co2": UseCaseConfig(
            name="co2",
            sensor_data_path=base_path / "sensor_data" / "co2" / "CO2_Data_June2025_training_split.csv", 
            image_data_path=base_path / "sensor_data" / "co2" / "CO2_images_June2025.csv",
            image_folder_path=base_path / "images" / "co2" / "image_full",
            image_ssl_folder_path=base_path / "images" / "co2" / "image_training_ssl_splitted_noclasses",
            image_training_folder_path=base_path / "images" / "co2" / "image_training_dataset" / "train",
            image_validation_folder_path=base_path / "images" / "co2" / "image_training_dataset" / "val",
            image_evaluation_folder_path=base_path / "images" / "co2" / "image_training_dataset" / "test",
            inference_csv_path=base_path / "sensor_data" / "co2" / "CO2_images_1sec_v2.csv",
            inference_image_folder=base_path / "images" / "co2" / "image_full",
            sensor_cols=["CO2"],
            sensor_metadata={"CO2": "DRS"},
            is_single_sensor=True,
            threshold_percentile = 60,
            threshold_factor = 0.85,
            refinement_confidence_threshold = 0.5,
            # Cleaning config for CO2 environment
            enable_training_data_cleaning=True,
            enable_range_validation=True,
            enable_statistical_cleaning=True,
            statistical_outlier_method='mad',
            statistical_outlier_threshold=3,  
            # Incremental training config for CO2
            incremental_training = True,
            incremental_chunk_days = 5,     # from 3 → 5 days
            incremental_overlap_hours = 12, # from 6 → 12h,
            memory_buffer_size = 2500,  
            sensor_ranges={
                'CO2': (300, 3000)  
            }
        ),

        "door": UseCaseConfig(
            name="door",
            sensor_data_path=base_path  / "sensor_data" / "Door" / "Door_Data_June2025_training_split.csv", 
            image_data_path=base_path / "sensor_data" / "door" / "Door_images_June2025.csv",
            image_folder_path=base_path / "images" / "door" / "image_full",
            image_ssl_folder_path=base_path / "images" / "door" / "image_training_ssl_splitted_noclasses",
            image_training_folder_path=base_path / "images" / "door" / "image_training_dataset" / "train",
            image_validation_folder_path=base_path / "images" / "door" / "image_training_dataset" / "val",
            image_evaluation_folder_path=base_path / "images" / "door" / "image_training_dataset" / "test",
            inference_csv_path=base_path / "sensor_data" / "door" / "Door_images_June2025.csv",
            inference_image_folder=base_path / "images" / "door" / "image_full",
            sensor_cols=["Temperature", "Humidity", "Pressure"],
            sensor_metadata={"Temperature": "DRS", "Humidity": "DRS", "Pressure": "DRS"},
            is_single_sensor=False,
            threshold_percentile = 65,
            threshold_factor = 0.80,
            refinement_confidence_threshold = 0.78,  
            # Cleaning config for door environment
            enable_training_data_cleaning=True,
            enable_range_validation=True,
            enable_statistical_cleaning=False,
            statistical_outlier_method='mad',
            statistical_outlier_threshold=5,
             # Incremental training config
            incremental_training = False,
            incremental_chunk_days = 4,     
            incremental_overlap_hours = 12, # from 6 → 12h,
            memory_buffer_size = 3000,
            sensor_ranges={
                'Temperature': (10, 45),    
                'Humidity': (10, 90),       
                'Pressure': (99000, 102000) 
            }
        ),
        
        "abnormal_object": UseCaseConfig(
            name="abnormal_object",
            sensor_data_path=base_path / "sensor_data" / "abnormal_object" / "Abnormalobj_June_2025_training_split.csv",
            image_data_path=base_path / "sensor_data" / "abnormal_object" / "Abnormal_object_images_June2025.csv",
            image_folder_path=base_path / "images" / "abnormal_object" / "image_full",
            image_ssl_folder_path=base_path / "images" / "abnormal_object" / "image_training_ssl_splitted_noclasses",
            image_training_folder_path=base_path / "images" / "abnormal_object" / "image_training_dataset" / "train",
            image_validation_folder_path=base_path / "images" / "abnormal_object" / "image_training_dataset" / "val",
            image_evaluation_folder_path=base_path / "images" / "abnormal_object" / "image_training_dataset" / "test",
            inference_csv_path=base_path / "sensor_data" / "co2" / "Abnormalobj_images_1sec.csv",
            inference_image_folder=base_path / "images" / "abnormal_object" / "image_full",
            sensor_cols=["S1_distance", "S2_distance"],
            sensor_metadata={"S1_distance": "IRS", "S2_distance": "IRS"},
            is_single_sensor=False,
            threshold_percentile = 60,
            threshold_factor = 0.8,
            refinement_confidence_threshold = 0.73,
            # Cleaning config for abnormal object environment
            enable_training_data_cleaning=True,
            enable_range_validation=True,
            enable_statistical_cleaning=False,
            statistical_outlier_method='iqr',
            statistical_outlier_threshold=5.0,  
            sensor_ranges={
                'S1_distance': (2, 315),    # Typical ultrasonic sensor range (cm)
                'S2_distance': (2, 315),    # Typical ultrasonic sensor range (cm)
            }
        )
    }
    
    @staticmethod
    def get_config(use_case: str) -> UseCaseConfig:
        if use_case not in ConfigManager.configs:
            raise ValueError(f"Unknown use case: {use_case}. Available: {list(ConfigManager.configs.keys())}")
        return ConfigManager.configs[use_case]
    
    @staticmethod
    def disable_cleaning_for_config(use_case: str):
        """
        Utility method to disable cleaning for a specific use case (for testing)
        """
        if use_case in ConfigManager.configs:
            config = ConfigManager.configs[use_case]
            config.enable_training_data_cleaning = False
            config.enable_range_validation = False
            config.enable_statistical_cleaning = False
            print(f"🚫 Disabled data cleaning for {use_case}")
        else:
            raise ValueError(f"Unknown use case: {use_case}")
    
    @staticmethod
    def get_cleaning_summary():
        """
        Get a summary of cleaning configurations for all use cases
        """
        summary = {}
        for name, config in ConfigManager.configs.items():
            summary[name] = {
                'cleaning_enabled': config.enable_training_data_cleaning,
                'range_validation': config.enable_range_validation,
                'statistical_cleaning': config.enable_statistical_cleaning,
                'outlier_threshold': config.statistical_outlier_threshold,
                'sensor_ranges': config.sensor_ranges
            }
        return summary
    

# --- Cleaning config accessor for deployment ---
def get_cleaning_config(config):
    """
    Extract cleaning configuration from a config object (UseCaseConfig or DeploymentConfig).
    """
    return {
        'enable_training_data_cleaning': getattr(config, 'enable_training_data_cleaning', True),
        'enable_range_validation': getattr(config, 'enable_range_validation', True),
        'enable_statistical_cleaning': getattr(config, 'enable_statistical_cleaning', False),
        'statistical_outlier_method': getattr(config, 'statistical_outlier_method', 'mad'),
        'statistical_outlier_threshold': getattr(config, 'statistical_outlier_threshold', 5),
        'sensor_ranges': getattr(config, 'sensor_ranges', None)
    }