
#     return decision_df

# 1. First, let's add debug logging to understand the data flow
import os
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from datetime import datetime
from PIL import Image
import warnings
import sys
from .alert_manager import generate_alert_for_vision_fallback
import time


sys.path.append(
    r"C:\Users\c21034189\OneDrive - Cardiff University\PhD files\Paper 05\Codes\anomaly_pipeline\training\models\co2\effcc_distilled_main"
)
from timmML2.models.factory import create_model

# === FIXED: Shared image preprocessing constants ===
corp_size = 256
MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([
    transforms.Resize((corp_size, corp_size)),  # Add explicit resize
    transforms.ToTensor(),
    transforms.Normalize(MEAN_STD[0], MEAN_STD[1])
])

def run(config):
    print("\n 🧠 Triggering Vision Fallback for Low Confidence + Sensor Error Cases...")
    
    start_time = time.time()

    decision_df = pd.read_csv(config.decision_log_path)

    image_df = pd.read_csv(config.image_timestamp_csv)


    # Identify fallback cases: low confidence or sensor error
    fallback_df = decision_df[
        (decision_df["Low_Confidence"] == True) | (decision_df["Sensor_Status"] == "Sensor_Error")
    ].copy()
    # fallback_df.reset_index(drop=True, inplace=True)  # 🔥 Ensure clean fallback_df index
    # assert not fallback_df.index.duplicated().any(), "❌ Duplicate index in fallback_df!"


    print(f"🔍 DEBUG: Fallback cases breakdown:")
    print(f"   Total decision records: {len(decision_df)}")
    print(f"   Low confidence cases: {(decision_df.get('Low_Confidence', False) == True).sum()}")
    print(f"   Sensor error cases: {(decision_df.get('Sensor_Status', '') == 'Sensor_Error').sum()}")
    print(f"   Combined fallback cases: {len(fallback_df)}")

    if fallback_df.empty:
        print("✅ No fallback-triggering records found.")
        return
    
    # Filter out non-image files (keep only .jpg, .png, etc.)
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    before_filter = len(image_df)
    
    image_df = image_df[
        image_df['Image_Name'].str.lower().str.endswith(tuple(valid_extensions))
    ].copy()
    
    after_filter = len(image_df)
    print(f"🔧 FIX: Filtered image dataset: {before_filter} -> {after_filter} (removed {before_filter - after_filter} non-image entries)")
    
    # Ensure we're using the correct path (no 'daata' typos)
    corrected_path = config.image_folder_path.replace('daata', 'data').replace('datta', 'data')
    if corrected_path != config.image_folder_path:
        print(f"🔧 FIX: Corrected path typo: {config.image_folder_path} -> {corrected_path}")
        config.image_folder_path = corrected_path
    
    print(f"🔍 DEBUG: Final image folder path: {config.image_folder_path}")
    print(f"🔍 DEBUG: Image inferece csv file path: {config.image_timestamp_csv}")
    
    # List some files in the directory to verify
    if os.path.exists(config.image_folder_path):
        try:
            files_in_dir = os.listdir(config.image_folder_path)[:5]  # First 5 files
            # print(f"🔍 DEBUG: Sample files in image directory: {files_in_dir}")
        except Exception as e:
            print(f"⚠️ Error listing directory contents: {e}")

    
    image_df["Timestamp"] = pd.to_datetime(image_df["Timestamp"], dayfirst=True, errors="coerce")
    fallback_df["Sensor_Timestamp"] = pd.to_datetime(fallback_df["Sensor_Timestamp"], errors="coerce")
   
    # Add "Used" flag to track which images have been matched
    image_df["Used"] = False

    print(f"📊 Parsed timestamps - Images: {image_df['Timestamp'].notna().sum()}/{len(image_df)} valid")
    print(f"📊 Parsed timestamps - Sensors: {fallback_df['Sensor_Timestamp'].notna().sum()}/{len(fallback_df)} valid")
    
    # Check for timestamp overlap between datasets
    image_ts_set = set(image_df[image_df['Timestamp'].notna()]['Timestamp'])
    fallback_ts_set = set(fallback_df[fallback_df['Sensor_Timestamp'].notna()]['Sensor_Timestamp'])
    
    exact_timestamp_matches = image_ts_set.intersection(fallback_ts_set)
    print(f"🔍 DEBUG: Exact timestamp matches available: {len(exact_timestamp_matches)}")
    
    if len(exact_timestamp_matches) < 100:
        print("⚠️ WARNING: Very few exact timestamp matches!")
        print("   This might explain the low match rate.")
        print("   Consider increasing the time window or checking data alignment.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"🔍 DEBUG: Using device: {device}")

    # Load model based on use case
    try:
        if config.use_case.lower() == "co2":
            model = create_model("efficientnet_lite2")
            model.load_state_dict(torch.load(config.vision_model_path, map_location=device, weights_only=True))
            print("✅ Loaded CO2 EfficientNet model")
        else:
            model = models.mobilenet_v2(num_classes=2)
            model.load_state_dict(torch.load(config.vision_model_path, map_location=device, weights_only=True))
            print("✅ Loaded MobileNetV2 model")

        model.eval().to(device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Fallback output containers
    vision_preds, matched_timestamps, confidences, matched_image_names = [], [], [], []
    max_time_diff = pd.Timedelta(seconds=60)

    total_fallback_cases = len(fallback_df)
    successful_matches = 0
    processing_errors = 0
    inference_times = []

    print(f"\n🔄 Processing {total_fallback_cases} fallback cases...")
    
    for idx, row in fallback_df.iterrows():
        if idx % 10000 == 0:  # Progress indicator
            print(f"   Processed {idx}/{total_fallback_cases} cases...")
            
        sensor_ts = row["Sensor_Timestamp"]

        if pd.isna(sensor_ts):
            vision_preds.append("Missing")
            matched_timestamps.append("NaT")
            confidences.append(float("nan"))
            matched_image_names.append("NaN")
            continue

        # Calculate time differences
        image_df["Time_Diff"] = (image_df["Timestamp"] - sensor_ts).abs()
        
        # Find valid matches within time window that haven't been used
        valid_matches = image_df[
            (image_df["Time_Diff"] <= max_time_diff) & 
            (~image_df["Used"]) &
            (image_df["Timestamp"].notna())
        ]

        if valid_matches.empty:
            vision_preds.append("Missing")
            matched_timestamps.append("NaT")
            confidences.append(float("nan"))
            matched_image_names.append("NaN")
            continue

        # Get the best match
        best_match = valid_matches.loc[valid_matches["Time_Diff"].idxmin()]
        matched_image_name = best_match["Image_Name"]
        matched_image_ts = best_match["Timestamp"]
        matched_image_path = os.path.join(config.image_folder_path, matched_image_name)
        
        # Mark this image as used
        image_df.at[best_match.name, "Used"] = True
        successful_matches += 1

        if os.path.exists(matched_image_path):
            try:
                img = Image.open(matched_image_path).convert("RGB")                
                input_tensor = img_transform(img).unsqueeze(0).to(device)

                inference_start = time.time()
                with torch.no_grad():
                    output = model(input_tensor)

                    if config.use_case.lower() == "co2":
                        if isinstance(output, tuple):
                            output = output[0]
                        raw_count = output.sum().item()
                        count = int(raw_count) if raw_count >= 1 else 0
                        label = "Anomaly" if count > 0 else "Normal"
                        confidence = min(raw_count / 10.0, 1.0)
                        
                        if successful_matches <= 5:  # Only show first few for debugging
                            print(f"🔍 [CO2] Raw count: {raw_count:.2f} → Final count: {count} → Label: {label} (Conf: {confidence:.2f})")
                    else:
                        probs = torch.softmax(output, dim=1)
                        pred_class = probs.argmax(dim=1).item()
                        label = "Anomaly" if pred_class == 1 else "Normal"
                        confidence = probs[0, pred_class].item()
                        
                        if successful_matches <= 5:  # Only show first few for debugging
                            print(f"🔍 [CLS] Label: {label} (Class {pred_class}) — Confidence: {confidence:.2f}")

                inference_time = time.time() - inference_start
                inference_times.append(inference_time)

                vision_preds.append(label)
                matched_timestamps.append(matched_image_ts)
                confidences.append(confidence)
                matched_image_names.append(matched_image_name)

            except Exception as e:
                processing_errors += 1
                if processing_errors <= 5:  # Only show first few errors
                    print(f"⚠️ Error processing image {matched_image_path}: {e}")
                
                vision_preds.append("Error")
                matched_timestamps.append("NaT")
                confidences.append(float("nan"))
                matched_image_names.append(matched_image_name)
        else:
            vision_preds.append("Missing")
            matched_timestamps.append("NaT")
            confidences.append(float("nan"))
            matched_image_names.append(matched_image_name)

    # Update results and print comprehensive statistics
    print(f"\n📊 Comprehensive Matching Statistics:")
    print(f"   Total decision records: {len(decision_df)}")
    print(f"   Fallback cases triggered: {total_fallback_cases} ({total_fallback_cases/len(decision_df)*100:.1f}%)")
    print(f"   Images available: {len(image_df)}")
    print(f"   Exact timestamp matches available: {len(exact_timestamp_matches)}")
    print(f"   Successful matches: {successful_matches}")
    print(f"   Processing errors: {processing_errors}")
    print(f"   Match rate: {successful_matches/total_fallback_cases*100:.1f}%")

    # Update the decision dataframe
    decision_df.loc[fallback_df.index, "Vision_Prediction"] = vision_preds
    decision_df.loc[fallback_df.index, "Matched_Image_Timestamp"] = matched_timestamps
    decision_df.loc[fallback_df.index, "Matched_Image_Name"] = matched_image_names
    decision_df.loc[fallback_df.index, "Vision_Confidence"] = confidences

    # Sensor health status
    decision_df.to_csv(config.decision_log_path, index=False)
    output_dir = os.path.dirname(config.decision_log_path)

    # Debug: Show vision confidence distribution FIRST
    if "Vision_Confidence" in decision_df.columns:
        vision_confs = decision_df[decision_df["Vision_Prediction"] == "Anomaly"]["Vision_Confidence"].dropna()
        if len(vision_confs) > 0:
            print(f"🔍 Vision anomaly confidences: min={vision_confs.min():.3f}, max={vision_confs.max():.3f}")
                    
            # Clean creation of vision_anomalies
            vision_anomalies = decision_df[
                (decision_df["Vision_Prediction"] == "Anomaly") & 
                (decision_df["Vision_Confidence"] >= 0.75)
            ].copy()

            # Rename columns for alert manager
            vision_anomalies = vision_anomalies.rename(columns={
                'Vision_Confidence': 'Confidence',
                'Vision_Prediction': 'Sensor_Status'
            })

            # ✅ Force proper new index — this is bulletproof
            vision_anomalies.reset_index(drop=True, inplace=True)

            # ✅ Double-check
            assert not vision_anomalies.index.duplicated().any(), "❌ Still duplicate index in vision_anomalies!"

            # Call alert
            generate_alert_for_vision_fallback(vision_anomalies, config.use_case)
    
    # Check and log sensor error rates
    if "Sensor_Error" in decision_df["Sensor_Status"].values:
        print("🩺 Saving sensor health log...")
        sensor_error_df = decision_df[decision_df["Sensor_Status"] == "Sensor_Error"]
        sensor_names = getattr(config, "sensor_cols", [])

        error_counts = {}
        for sensor in sensor_names:
            if sensor in decision_df.columns:
                error_rate = sensor_error_df["Corrupt_Sensors"].apply(lambda x: sensor in x.split(",")).sum() / len(sensor_error_df)
                error_counts[sensor] = error_rate

        sensor_error_rate = pd.DataFrame(list(error_counts.items()), columns=["Sensor_Name", "Error_Rate"])
        sensor_error_rate.to_csv(os.path.join(output_dir, "sensor_health_log.csv"), index=False)

    # Human-in-the-loop review file
    print("🧍 Saving sensor error review file...")
    sensor_error_rows = decision_df[decision_df["Sensor_Status"] == "Sensor_Error"]
    sensor_error_rows.to_csv(os.path.join(output_dir, "sensor_error_review.csv"), index=False)

    # Summary
    match_summary = pd.Series(vision_preds).value_counts()
    matched_total = match_summary.get("Normal", 0) + match_summary.get("Anomaly", 0)
    missing_total = match_summary.get("Missing", 0) + match_summary.get("Error", 0)

    # print(f"📊 Vision fallback summary — Matched: {matched_total}, Missing/Error: {missing_total}")
    # print(f"✅ Vision fallback predictions updated in log: {config.decision_log_path}")

    total_time = time.time() - start_time
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    
    print(f"\n⏱️ TIMING SUMMARY:")
    print(f"   Total vision fallback time: {total_time:.2f}s")
    print(f"   Average inference per image: {avg_inference_time*1000:.1f}ms")
    print(f"   Total images processed: {len(inference_times)}")

    return decision_df