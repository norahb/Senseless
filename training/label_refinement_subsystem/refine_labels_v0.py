
import pandas as pd
import numpy as np
import os
from config.config_manager import ConfigManager
import matplotlib.pyplot as plt
import cv2

def apply_dynamic_refinement(ssl_df, nonvision_df, threshold):
    merged_df = pd.merge(ssl_df, nonvision_df, on=["Image_Name"], how="outer", suffixes=("_ssl", "_nonvision"))

    merged_df["Label_ssl"] = merged_df["Label_ssl"].fillna("Unknown")
    merged_df["Label_nonvision"] = merged_df["Label_nonvision"].fillna("Unknown")
    merged_df["Confidence_Score_ssl"] = merged_df["Confidence_Score_ssl"].fillna(0)
    merged_df["Confidence_Score_nonvision"] = merged_df["Confidence_Score_nonvision"].fillna(0)
    # merged_df["Timestamp"] = merged_df["Timestamp"].fillna("Unknown")
    merged_df["Timestamp"] = merged_df["Image_Timestamp"].fillna("Unknown")
    merged_df["True_Label"] = merged_df["True_Label"].fillna("Unknown")

    merged_df["ssl_weight"] = 1 / (1 + np.exp(-10 * (merged_df["Confidence_Score_ssl"] - 0.5)))
    merged_df["nonvision_weight"] = 1 - merged_df["ssl_weight"]

    sources = []
    def decide_label(row):
        if row['Confidence_Score_ssl'] < threshold and row['Confidence_Score_nonvision'] < threshold:
            sources.append("Unknown")
            return "Unknown"
        if row['Label_ssl'] == row['Label_nonvision']:
            sources.append("Agree")
            return row['Label_ssl']
        if row['ssl_weight'] >= row['nonvision_weight'] and row['Label_ssl'] != 'Unknown':
            sources.append("SSL")
            return row['Label_ssl']
        elif row['Label_nonvision'] != 'Unknown':
            sources.append("NonVision")
            return row['Label_nonvision']
        else:
            sources.append("Unknown")
            return 'Unknown'

    merged_df["Refined_Label"] = merged_df.apply(decide_label, axis=1)
    merged_df["Source"] = sources
    merged_df["Confidence"] = merged_df.apply(lambda row: max(row['Confidence_Score_ssl'], row['Confidence_Score_nonvision'])
                                               if row['Refined_Label'] != "Unknown" else 0, axis=1)

    merged_df["Match_Type"] = merged_df.apply(lambda row: "Agree" if row['Label_ssl'] == row['Label_nonvision']
                                               else "Disagree", axis=1)


    return merged_df[[
        "Image_Name", "Timestamp", 
        "Label_ssl", "Confidence_Score_ssl", 
        "Label_nonvision", "Confidence_Score_nonvision", 
        "Refined_Label", "Source", "Confidence", 
        "Match_Type", "True_Label", "Correct"]]

def interactive_human_labeling(df, config):
    image_dir = config.image_folder_path
    temp_csv = os.path.join("output", config.name, "human_labels_temp.csv")

    if os.path.exists(temp_csv):
        existing_labels = pd.read_csv(temp_csv)
    else:
        existing_labels = pd.DataFrame(columns=["Image_Name", "Human_Label"])

    # Reuse existing human labels
    all_labels = existing_labels.dropna(subset=["Human_Label"])
    df = pd.merge(df, all_labels, on="Image_Name", how="left")
    df.loc[df["Human_Label"].notna(), "Refined_Label"] = df["Human_Label"]
    df.loc[df["Human_Label"].notna(), "Source"] = "Human"
    df.drop(columns=["Human_Label"], inplace=True)

    # Identify unknown and unlabeled images
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
            if key == ord('n'):
                label = "Normal"
            elif key == ord('a'):
                label = "Anomaly"
            elif key == ord('q'):
                print("❌ Labeling aborted by user.")
                break
            else:
                print("⚠️ Invalid key pressed. Skipping image.")

            cv2.destroyAllWindows()

            if label:
                new_entries.append({"Image_Name": image_name, "Human_Label": label})
                pd.DataFrame([{"Image_Name": image_name, "Human_Label": label}]).to_csv(
                    temp_csv, index=False, mode='a', header=not os.path.exists(temp_csv)
                )

        print(f"✅ Manually labeled {len(new_entries)} images this session.")

        # Reload merged labels and apply them
        all_labels = pd.read_csv(temp_csv).dropna(subset=["Human_Label"])
        df = pd.merge(df, all_labels, on="Image_Name", how="left")
        df.loc[df["Human_Label"].notna(), "Refined_Label"] = df["Human_Label"]
        df.loc[df["Human_Label"].notna(), "Source"] = "Human"
        df.drop(columns=["Human_Label"], inplace=True)

    # Correct and confidence columns
    df["Correct"] = df.apply(
        lambda row: "Yes" if str(row['Refined_Label']).strip().lower() == str(row['True_Label']).strip().lower() else "No",
        axis=1)

    df["Confidence"] = df.apply(
        lambda row: 1.0 if row["Source"] == "Human"
        else max(row['Confidence_Score_ssl'], row['Confidence_Score_nonvision'])
        if row['Refined_Label'] != "Unknown" else 0,
        axis=1
    )

    return df

def load_and_prepare_data(dir_path, use_case):
    ssl_file = os.path.join(dir_path, f"{use_case}_ssl_images_labels.csv")
    nonvision_file = os.path.join(dir_path, f"{use_case}_sensors_images_labels.csv")

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

# def run(config):
#     dir_path = os.path.join("output", config.name)
#     print(f"\n🔧 Refining labels for use case: {config.name}")

#     ssl_df, nonvision_df = load_and_prepare_data(dir_path, config.name)
#     threshold = getattr(config, "refinement_confidence_threshold", 0.4)
#     refined_df = apply_dynamic_refinement(ssl_df, nonvision_df, threshold)

#     if getattr(config, "human_labeling", False):
#         refined_df = interactive_human_labeling(refined_df, config)

#     output_path = os.path.join(dir_path, f"{config.name}_final_labels.csv")
#     refined_df.to_csv(output_path, index=False)

#     valid_df = refined_df[refined_df["Refined_Label"] != "Unknown"]
#     accuracy = (valid_df["Refined_Label"].str.strip().str.lower() == valid_df["True_Label"].str.strip().str.lower()).mean() * 100 if not valid_df.empty else 0

#     print(f"✅ Final labels saved to: {output_path}")
#     print(f"🎯 Final Labeling Accuracy (excluding Unknown): {accuracy:.2f}%")
#     print(f"📦 Total Samples: {len(refined_df)} | Unknowns: {len(refined_df) - len(valid_df)}")

def run(config):
    dir_path = os.path.join("output", config.name)
    print(f"\n🔧 Refining labels for use case: {config.name}")

    ssl_df, nonvision_df = load_and_prepare_data(dir_path, config.name)
    threshold = getattr(config, "refinement_confidence_threshold", 0.4)
    refined_df = apply_dynamic_refinement(ssl_df, nonvision_df, threshold)

    # Calculate accuracy BEFORE human labeling (automated only)
    valid_df_auto = refined_df[refined_df["Refined_Label"] != "Unknown"]
    accuracy_auto = (valid_df_auto["Refined_Label"].str.strip().str.lower() == 
                    valid_df_auto["True_Label"].str.strip().str.lower()).mean() * 100 if not valid_df_auto.empty else 0
    coverage_auto = len(valid_df_auto) / len(refined_df) * 100
    
    print(f"🤖 Automated Labeling Results:")
    print(f"   Coverage: {coverage_auto:.1f}% ({len(valid_df_auto)}/{len(refined_df)} images)")
    print(f"   Accuracy: {accuracy_auto:.2f}% (excluding Unknown)")

    if getattr(config, "human_labeling", False):
        print(f"🚀 Label Refinement...")
        refined_df = interactive_human_labeling(refined_df, config)
        
        # Calculate accuracy AFTER human labeling (full coverage)
        valid_df_full = refined_df[refined_df["Refined_Label"] != "Unknown"]
        accuracy_full = (valid_df_full["Refined_Label"].str.strip().str.lower() == 
                        valid_df_full["True_Label"].str.strip().str.lower()).mean() * 100 if not valid_df_full.empty else 0
        coverage_full = len(valid_df_full) / len(refined_df) * 100
        
        # Count human-labeled images
        human_labeled = len(refined_df[refined_df["Source"] == "Human"])
        
        print(f"👥 Full Coverage Results (with Human Labeling):")
        print(f"   Coverage: {coverage_full:.1f}% ({len(valid_df_full)}/{len(refined_df)} images)")
        print(f"   Accuracy: {accuracy_full:.2f}% (including human labels)")
        print(f"   Human Labels: {human_labeled} images ({human_labeled/len(refined_df)*100:.1f}% of total)")
        
        # Show improvement
        coverage_gain = coverage_full - coverage_auto
        accuracy_change = accuracy_full - accuracy_auto
        print(f"📈 Human Labeling Impact:")
        print(f"   Coverage Gain: +{coverage_gain:.1f}%")
        print(f"   Accuracy Change: {accuracy_change:+.2f}%")
    
    else:
        print(f"⚠️  Human labeling disabled - some images remain Unknown")
        accuracy_full = accuracy_auto
        coverage_full = coverage_auto

    output_path = os.path.join(dir_path, f"{config.name}_final_labels.csv")
    refined_df.to_csv(output_path, index=False)

    print(f"✅ Final labels saved to: {output_path}")
    print(f"🎯 Final Labeling Accuracy (excluding Unknown): {accuracy_auto:.2f}%")
    print(f"📦 Total Samples: {len(refined_df)} | Unknowns: {len(refined_df) - len(valid_df_auto)}")
    
    # Print source breakdown
    if "Source" in refined_df.columns:
        source_counts = refined_df["Source"].value_counts()
        print(f"📊 Label Sources:")
        for source, count in source_counts.items():
            percentage = count / len(refined_df) * 100
            print(f"   {source}: {count} ({percentage:.1f}%)")
            
    return refined_df