import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import cv2


# def train_calibration_models(config):
def train_calibration_models(config, gt_file, ssl_file, nonvision_file, model_dir):

    # Load CSVs
    gt = pd.read_csv(gt_file)
    ssl = pd.read_csv(ssl_file)
    sensor = pd.read_csv(nonvision_file)

    gt.columns = gt.columns.str.strip()
    ssl.columns = ssl.columns.str.strip()
    sensor.columns = sensor.columns.str.strip()

    # SSL CSV schema: Image_Name, Cluster, Label_ssl, Confidence_Score
    ssl.rename(columns={"Confidence_Score": "SSL_Conf"}, inplace=True)

    # Sensor schema: Image_Name, Label, Confidence_Score_nonvision, True_Label
    sensor.rename(columns={"Label": "Sensor_Label",
                           "Confidence_Score_nonvision": "Sensor_Conf"}, inplace=True)

    # Merge
    df = gt.merge(ssl[["Image_Name", "Label_ssl", "SSL_Conf"]], on="Image_Name", how="inner")
    df = df.merge(sensor[["Image_Name", "Sensor_Label", "Sensor_Conf", "True_Label"]],
                  on="Image_Name", how="inner")

    # Correctness flags
    df["SSL_Correct"] = (df["Label_ssl"].str.lower() == df["True_Label"].str.lower()).astype(int)
    df["Sensor_Correct"] = (df["Sensor_Label"].str.lower() == df["True_Label"].str.lower()).astype(int)

    # Fit isotonic models
    ssl_iso = IsotonicRegression(out_of_bounds="clip").fit(df["SSL_Conf"], df["SSL_Correct"])
    sensor_iso = IsotonicRegression(out_of_bounds="clip").fit(df["Sensor_Conf"], df["Sensor_Correct"])

    # Find optimal cutoff
    def find_optimal_cutoff(calibrated_conf, correct, target_acc=90):
        thresholds = np.linspace(0, 1, 50)
        optimal_thresh, best_score = None, -1
        for t in thresholds:
            mask = calibrated_conf >= t
            if mask.sum() > 0:
                acc = correct[mask].mean() * 100
                cov = mask.mean() * 100
                score = (acc / 100.0) * (cov / 100.0)
                if acc >= target_acc and score > best_score:
                    best_score, optimal_thresh = score, t
        return optimal_thresh

    calibrated_ssl = ssl_iso.transform(df["SSL_Conf"])
    ssl_cutoff = find_optimal_cutoff(calibrated_ssl, df["SSL_Correct"])
    calibrated_sensor = sensor_iso.transform(df["Sensor_Conf"])
    sensor_cutoff = find_optimal_cutoff(calibrated_sensor, df["Sensor_Correct"])

    # Safe save
    model_dir = os.path.join("models", config.name)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(ssl_iso, os.path.join(model_dir, "ssl_isotonic.pkl"))
    joblib.dump(sensor_iso, os.path.join(model_dir, "sensor_isotonic.pkl"))

    thresholds = {
        "ssl_cutoff": float(ssl_cutoff) if ssl_cutoff is not None else 0.5,
        "sensor_cutoff": float(sensor_cutoff) if sensor_cutoff is not None else 0.5,
    }
    with open(os.path.join(model_dir, "calibration_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"✅ Calibration models and thresholds saved to {model_dir}")
    print(f"   → SSL cutoff: {thresholds['ssl_cutoff']}")
    print(f"   → Sensor cutoff: {thresholds['sensor_cutoff']}")

    return ssl_iso, sensor_iso, thresholds

def apply_dynamic_refinement(ssl_df, nonvision_df, thresholds):
    ssl_thresh, nonvision_thresh = thresholds

    merged_df = pd.merge(
        ssl_df, nonvision_df, on=["Image_Name"], how="outer", suffixes=("_ssl", "_nonvision")
    )

    merged_df["Label_ssl"] = merged_df["Label_ssl"].fillna("Unknown")
    merged_df["Label_nonvision"] = merged_df["Label_nonvision"].fillna("Unknown")
    merged_df["Confidence_Score_ssl"] = merged_df["Confidence_Score_ssl"].fillna(0)
    merged_df["Confidence_Score_nonvision"] = merged_df["Confidence_Score_nonvision"].fillna(0)
    merged_df["Timestamp"] = merged_df.get("Image_Timestamp", "Unknown")
    merged_df["True_Label"] = merged_df["True_Label"].fillna("Unknown")

    decisions = []

    def decide_label(row):
        ssl_ok = row["Confidence_Score_ssl"] >= ssl_thresh
        sensor_ok = row["Confidence_Score_nonvision"] >= nonvision_thresh

        # Case 1: both below thresholds
        if not ssl_ok and not sensor_ok:
            return "Unknown", "Unknown", 0.0, "Disagree"

        # Case 2: both agree and at least one above threshold
        if row["Label_ssl"] == row["Label_nonvision"] and (ssl_ok or sensor_ok):
            conf = max(row["Confidence_Score_ssl"], row["Confidence_Score_nonvision"])
            return row["Label_ssl"], "Agree", conf, "Agree"

        # Case 3: SSL passes threshold and has higher confidence
        if ssl_ok and (not sensor_ok or row["Confidence_Score_ssl"] >= row["Confidence_Score_nonvision"]):
            return row["Label_ssl"], "SSL", row["Confidence_Score_ssl"], "Disagree"

        # Case 4: Sensor passes threshold and has higher confidence
        if sensor_ok and (not ssl_ok or row["Confidence_Score_nonvision"] > row["Confidence_Score_ssl"]):
            return row["Label_nonvision"], "NonVision", row["Confidence_Score_nonvision"], "Disagree"

        # Fallback
        return "Unknown", "Unknown", 0.0, "Disagree"

    merged_df[["Refined_Label", "Source", "Confidence", "Match_Type"]] = pd.DataFrame(
        merged_df.apply(decide_label, axis=1).tolist(),
        index=merged_df.index
    )

    # --- Logging summary ---
    print("\n[Refinement Source Breakdown]")
    for src in ["Agree", "SSL", "NonVision", "Unknown"]:
        subset = merged_df[merged_df["Source"] == src]
        if not subset.empty:
            acc = (
                (subset["Refined_Label"].str.lower() == subset["True_Label"].str.lower())
                .mean() * 100
            )
            print(f"{src:9s}: {len(subset):5d} samples | Accuracy={acc:.2f}%")

    return merged_df[
        [
            "Image_Name",
            "Timestamp",
            "Label_ssl",
            "Confidence_Score_ssl",
            "Label_nonvision",
            "Confidence_Score_nonvision",
            "Refined_Label",
            "Source",
            "Confidence",
            "Match_Type",
            "True_Label",
        ]
    ]

def load_and_prepare_data(ssl_file, nonvision_file):
    # ssl_df = pd.read_csv(config.ssl_labels_path)
    # nonvision_df = pd.read_csv(config.sensor_labels_path)
    ssl_df = pd.read_csv(ssl_file)
    nonvision_df = pd.read_csv(nonvision_file)

    if "Predicted_Label" in ssl_df.columns:
        ssl_df.rename(columns={"Predicted_Label": "Label_ssl"}, inplace=True)
    if "Confidence_Score" in ssl_df.columns:
        ssl_df.rename(columns={"Confidence_Score": "Confidence_Score_ssl"}, inplace=True)

    if "Label" in nonvision_df.columns:
        nonvision_df.rename(columns={"Label": "Label_nonvision"}, inplace=True)
    if "Confidence_Score" in nonvision_df.columns:
        nonvision_df.rename(columns={"Confidence_Score": "Confidence_Score_nonvision"}, inplace=True)

    return ssl_df, nonvision_df

def interactive_human_labeling(df, config):
    image_dir = config.image_folder_path
    temp_csv = os.path.join("output", config.name, "human_labels_temp.csv")

    if os.path.exists(temp_csv):
        existing_labels = pd.read_csv(temp_csv)
    else:
        existing_labels = pd.DataFrame(columns=["Image_Name", "Human_Label"])

    all_labels = existing_labels.dropna(subset=["Human_Label"])
    df = pd.merge(df, all_labels, on="Image_Name", how="left")
    df.loc[df["Human_Label"].notna(), "Refined_Label"] = df["Human_Label"]
    df.loc[df["Human_Label"].notna(), "Source"] = "Human"
    df.drop(columns=["Human_Label"], inplace=True)

    labeled_images = set(all_labels["Image_Name"])
    unknown_df = df[(df["Refined_Label"] == "Unknown") & (~df["Image_Name"].isin(labeled_images))].copy()

    if unknown_df.empty:
        print("✅ No unknowns to label manually.")
    else:
        print("🔍 Starting manual labeling. Press 'n' for Normal, 'a' for Anomaly, or 'q' to quit.")
        new_entries = []

        for _, row in unknown_df.iterrows():
            image_name = row["Image_Name"]
            image_path = os.path.join(image_dir, image_name)

            if not os.path.exists(image_path):
                print(f"❌ Missing image: {image_name}")
                continue

            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Could not read image: {image_path}")
                continue

            cv2.imshow("Label This Image - Press 'n' for Normal, 'a' for Anomaly, or 'q' to quit", img)
            key = cv2.waitKey(0)
            label = None
            if key == ord("n"):
                label = "Normal"
            elif key == ord("a"):
                label = "Anomaly"
            elif key == ord("q"):
                print("❌ Labeling aborted by user.")
                break
            else:
                print("⚠️ Invalid key pressed. Skipping image.")

            cv2.destroyAllWindows()

            if label:
                new_entries.append({"Image_Name": image_name, "Human_Label": label})
                pd.DataFrame([{"Image_Name": image_name, "Human_Label": label}]).to_csv(
                    temp_csv, index=False, mode="a", header=not os.path.exists(temp_csv)
                )

        print(f"✅ Manually labeled {len(new_entries)} images this session.")

        all_labels = pd.read_csv(temp_csv).dropna(subset=["Human_Label"])
        df = pd.merge(df, all_labels, on="Image_Name", how="left")
        df.loc[df["Human_Label"].notna(), "Refined_Label"] = df["Human_Label"]
        df.loc[df["Human_Label"].notna(), "Source"] = "Human"
        df.drop(columns=["Human_Label"], inplace=True)

    return df

def run(config):
    print(f"\n🔧 Calibration and label refinement for use case: {config.name}")

    
    # Define file paths once
    dir_path = os.path.join("output", config.name)
    model_dir = os.path.join("models", config.name)
    gt_file = config.image_data_path
    ssl_file = os.path.join(dir_path, f"{config.name}_ssl_images_labels.csv")
    nonvision_file = os.path.join(dir_path, f"{config.name}_sensors_images_labels.csv")

    model_dir = os.path.join("models", config.name)
    if not (os.path.exists(os.path.join(model_dir, "ssl_isotonic.pkl")) and
            os.path.exists(os.path.join(model_dir, "sensor_isotonic.pkl")) and
            os.path.exists(os.path.join(model_dir, "calibration_thresholds.json"))):
        # train_calibration_models(config)
        train_calibration_models(config, gt_file, ssl_file, nonvision_file, model_dir)


    # Load models + thresholds
    ssl_iso = joblib.load(os.path.join(model_dir, "ssl_isotonic.pkl"))
    sensor_iso = joblib.load(os.path.join(model_dir, "sensor_isotonic.pkl"))
    with open(os.path.join(model_dir, "calibration_thresholds.json")) as f:
        thresholds = json.load(f)

    ssl_cutoff = thresholds.get("ssl_cutoff")
    sensor_cutoff = thresholds.get("sensor_cutoff")

    # Safe fallback
    if ssl_cutoff is None:
        print("⚠️ SSL cutoff missing, defaulting to 0.5")
        ssl_cutoff = 0.5
    if sensor_cutoff is None:
        print("⚠️ Sensor cutoff missing, defaulting to 0.5")
        sensor_cutoff = 0.5

    print(f"📏 Using cutoffs → SSL: {ssl_cutoff}, Sensor: {sensor_cutoff}")

    # # Load and calibrate data
    # ssl_df, nonvision_df = load_and_prepare_data(config)
    # ssl_df["Confidence_Score_ssl"] = ssl_iso.predict(ssl_df["Confidence_Score_ssl"].values)
    # nonvision_df["Confidence_Score_nonvision"] = sensor_iso.predict(
    #     nonvision_df["Confidence_Score_nonvision"].values
    # )
        # Load and calibrate data
    ssl_df, nonvision_df = load_and_prepare_data(ssl_file, nonvision_file)
    ssl_df["Confidence_Score_ssl"] = ssl_iso.predict(ssl_df["Confidence_Score_ssl"].values)
    nonvision_df["Confidence_Score_nonvision"] = sensor_iso.predict(
        nonvision_df["Confidence_Score_nonvision"].values
    )

    # Apply refinement
    refined_df = apply_dynamic_refinement(ssl_df, nonvision_df, (ssl_cutoff, sensor_cutoff))

    # Automated results
    valid_df_auto = refined_df[refined_df["Refined_Label"] != "Unknown"]
    accuracy_auto = (
        (valid_df_auto["Refined_Label"].str.strip().str.lower()
         == valid_df_auto["True_Label"].str.strip().str.lower()).mean()
        * 100
        if not valid_df_auto.empty
        else 0
    )
    coverage_auto = len(valid_df_auto) / len(refined_df) * 100

    print(f"🤖 Automated Labeling Results:")
    print(f"   Coverage: {coverage_auto:.1f}% ({len(valid_df_auto)}/{len(refined_df)} images)")
    print(f"   Accuracy: {accuracy_auto:.2f}% (excluding Unknown)")

    # Human-in-loop
    if getattr(config, "human_labeling", False):
        print(f"🚀 Label Refinement with Human-in-the-Loop...")
        refined_df = interactive_human_labeling(refined_df, config)

        valid_df_full = refined_df[refined_df["Refined_Label"] != "Unknown"]
        accuracy_full = (
            (valid_df_full["Refined_Label"].str.strip().str.lower()
             == valid_df_full["True_Label"].str.strip().str.lower()).mean()
            * 100
            if not valid_df_full.empty
            else 0
        )
        coverage_full = len(valid_df_full) / len(refined_df) * 100
        human_labeled = len(refined_df[refined_df["Source"] == "Human"])

        print(f"👥 Full Coverage Results (with Human Labeling):")
        print(f"   Coverage: {coverage_full:.1f}% ({len(valid_df_full)}/{len(refined_df)} images)")
        print(f"   Accuracy: {accuracy_full:.2f}% (including human labels)")
        print(f"   Human Labels: {human_labeled} images ({human_labeled/len(refined_df)*100:.1f}%)")
    else:
        print(f"⚠️ Human labeling disabled - some images remain Unknown")

    # Save results
    output_path = os.path.join("output", config.name, f"{config.name}_final_labels.csv")
    refined_df.to_csv(output_path, index=False)
    print(f"✅ Final labels saved to: {output_path}")

    if "Source" in refined_df.columns:
        source_counts = refined_df["Source"].value_counts()
        print(f"📊 Label Sources:")
        for source, count in source_counts.items():
            percentage = count / len(refined_df) * 100
            print(f"   {source}: {count} ({percentage:.1f}%)")

    return refined_df
