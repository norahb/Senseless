import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

def create_image_classification_dataset(config):
    LABEL_CSV_PATH = os.path.join("output", config.name, f"{config.name}_final_labels.csv")
    IMAGE_SOURCE_DIR = config.image_folder_path
    OUTPUT_BASE_DIR = os.path.join("data","images", config.name, "image_training_dataset")
    SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test

    # Load and filter labels
    df = pd.read_csv(LABEL_CSV_PATH)
    df = df[df["Refined_Label"] != "Unknown"]

    # Split
    train_val_df, test_df = train_test_split(
        df, test_size=SPLIT_RATIOS[2], stratify=df["Refined_Label"], random_state=42)
    train_df, val_df = train_test_split(
        train_val_df, test_size=SPLIT_RATIOS[1]/(SPLIT_RATIOS[0]+SPLIT_RATIOS[1]), stratify=train_val_df["Refined_Label"], random_state=42)

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    # Copy images
    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            label = row["Refined_Label"]
            image_name = row["Image_Name"]

            src_path = os.path.join(IMAGE_SOURCE_DIR, image_name)
            dst_dir = os.path.join(OUTPUT_BASE_DIR, split_name, label)
            dst_path = os.path.join(dst_dir, image_name)

            os.makedirs(dst_dir, exist_ok=True)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"⚠️ Image not found: {src_path}")

    print(f"✅ {config.name}: Images organized for classification at {OUTPUT_BASE_DIR}")


# Optional hook for pipeline integration
def run(config):
    if getattr(config, "enable_image_classification_split", False):
        print("Organizing images for classification training...")
        create_image_classification_dataset(config)
