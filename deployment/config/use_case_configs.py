from dataclasses import dataclass
import os

@dataclass
class UseCaseConfig:
    name: str
    # sensor_data_path: str
    # image_data_path: str
    # image_folder_path: str
    # image_training_folder_path: str
    inference_csv_path: str
    training_csv_path: str
    inference_image_folder: str
    inference_image_csv_path: str
    sensor_cols: list
    sensor_metadata: dict
    sensor_model_path: str
    vision_model_path: str
    # Add anything else you want per use case
    sensor_error_thresholds: dict
    mild_outlier_zscore: float = 3.5
    extreme_corruption_zscore: float = 5.0

# base_path = "stream_input"
# base_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'training'))

base_path = "stream_input"
# 🔧 FIX: Check for both 'data' and 'datta' directories
base_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'training'))

# Debug: Print the actual path being used
# print(f"🔍 DEBUG: base_model_path = {base_model_path}")
data_path = os.path.join(base_model_path, 'data')
datta_path = os.path.join(base_model_path, 'datta')

if os.path.exists(data_path):
    image_base_path = data_path
    # print(f"✅ Using 'data' directory: {data_path}")
elif os.path.exists(datta_path):
    image_base_path = datta_path
    print(f"✅ Using 'datta' directory: {datta_path}")
else:
    image_base_path = data_path  # Default fallback
    print(f"⚠️ Neither 'data' nor 'datta' directory found, using default: {data_path}")

USE_CASE_CONFIGS = {
    "door": UseCaseConfig(
        name="door",
        inference_csv_path=os.path.join(base_path, "door", "Door_Data_June2025_inference_split.csv"),
        training_csv_path=os.path.join(data_path, "sensor_data","door", "Door_Data_June2025_training_split.csv"),
        inference_image_csv_path=os.path.join(base_path, "door", "Door_images_June2025_inference_split.csv"),
        # 🔧 FIX: Use the detected image base path
        inference_image_folder=os.path.join(base_path, "door", "images"),
        sensor_cols=["Temperature", "Humidity", "Pressure"],
        sensor_metadata={"Temperature": "DRS", "Humidity": "DRS", "Pressure": "DRS"},
        sensor_model_path=os.path.join(base_model_path, "models", "door"), 
        vision_model_path=os.path.join(base_model_path, "models", "door", "door_mobilenetv2.pth"),
        # NEW: Door-specific thresholds
        sensor_error_thresholds={
            "Temperature": {"min": 10, "max": 40},
            "Humidity": {"min": 20, "max": 95},
            "Pressure": {"min": 99000,"max": 102000}
        },
        mild_outlier_zscore=4.0,
        extreme_corruption_zscore=6.0
    ),
    
        "co2": UseCaseConfig(
        name="co2",
        inference_csv_path=os.path.join(base_path, "co2", "CO2_Data_June2025_inference_split.csv"),
        training_csv_path=os.path.join(data_path, "sensor_data","co2", "CO2_Data_June2025_training_split.csv"),
        inference_image_csv_path=os.path.join(base_path, "co2", "CO2_images_June2025_inference_split.csv"),
        inference_image_folder=os.path.join(base_path, "co2", "images","images"),
        sensor_cols=["CO2"],
        sensor_metadata={"CO2": "DRS"},
        sensor_model_path=os.path.join(base_model_path, "models", "co2"),  # Just the directory
        vision_model_path=os.path.join(base_model_path, "models", "co2", "effcc_distilled_main", "teacher.pt"),
        # NEW: CO2-specific thresholds
        sensor_error_thresholds={
            "CO2": {"min": 300, "max": 1000}
        },
        mild_outlier_zscore=3.5,
        extreme_corruption_zscore=5.0
    ),

        "appliance": UseCaseConfig(
        name="appliance",
        inference_csv_path=os.path.join(base_path, "appliance", "Appliance_June_2025_inference_split.csv"),
        training_csv_path=os.path.join(data_path, "sensor_data","appliance", "Appliance_June_2025_training_split.csv"),
        inference_image_csv_path=os.path.join(base_path, "appliance", "Appliance_images_June2025_inference_split.csv"),
        inference_image_folder=os.path.join(base_path, "appliance", "images"),
        sensor_cols=["Temperature", "Humidity", "CO2"],
        sensor_metadata={"Temperature": "DRS", "Humidity": "DRS", "CO2": "DRS"},
        sensor_model_path=os.path.join(base_model_path, "models", "appliance"),
        vision_model_path=os.path.join(base_model_path, "models", "appliance", "appliance_mobilenetv2.pth"),
        sensor_error_thresholds={
            "Temperature": {"min": 12, "max": 70},  # Allow high temps for appliance anomalies
            "Humidity": {"min": 12, "max": 90},
            "CO2": {"min": 300, "max": 1000}
        },
        mild_outlier_zscore=5.0,        
        extreme_corruption_zscore=7.0   
    ),
        "abnormal_object": UseCaseConfig(
        name="abnormal_object",
        inference_csv_path=os.path.join(base_path, "abnormal_object", "Abnormalobj_June_2025_inference_split.csv"),
        training_csv_path=os.path.join(data_path, "sensor_data","abnormal_object", "Abnormalobj_June_2025_training_split.csv"),
        inference_image_csv_path=os.path.join(base_path, "abnormal_object", "Abnormalobj_images_June2025_inference_split.csv"),
        inference_image_folder=os.path.join(base_path, "abnormal_object", "images"),
        sensor_cols=["S1_distance", "S2_distance"],
        sensor_metadata={"S1_distance": "IRS", "S2_distance": "IRS"},
        sensor_model_path=os.path.join(base_model_path, "models", "abnormal_object"),
        vision_model_path=os.path.join(base_model_path, "models", "abnormal_object", "abnormal_object_mobilenetv2.pth"),
        # NEW: Distance sensor thresholds
        sensor_error_thresholds={
            "S1_distance": {"min": 2, "max": 315},  # cm range
            "S2_distance": {"min": 2, "max": 315}   # cm range
        },
        mild_outlier_zscore=3.5,
        extreme_corruption_zscore=5.0
    )
    }