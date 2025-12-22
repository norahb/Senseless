# # vision_subsystem/image_inference.py

import os
import json
import torch
import pandas as pd
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
from sklearn.metrics import classification_report

def run(config):
    print(f"\n[INFER] 🖼️ Running inference for: {config.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class map
    class_map_path = f"models/{config.name}/{config.name}_class_map.json"
    with open(class_map_path, 'r') as f:
        class_names = json.load(f)

    # Load model
    model_path = f"models/{config.name}/{config.name}_mobilenetv2.pth"
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Define transforms (same as validation/test)
    transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load CSV with image names
    df = pd.read_csv(config.inference_csv_path)

    # Prepare predictions
    predictions = []
    logits_list = []  

    for img_name in df[config.image_column_name]:
        img_path = os.path.join(config.inference_image_folder, img_name)
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            predictions.append("Missing")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred_class = output.argmax(dim=1).item()
                predictions.append(class_names[pred_class])
                logits_list.append(output.cpu())  # ADD THIS

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            predictions.append("Error")

    df['Predicted_Label'] = predictions
    logits_tensor = torch.cat(logits_list, dim=0)
    df['logits'] = logits_tensor.tolist()  # Optional: store raw logits

    # Compare with ground truth if 'Status' column exists 
    # Change 'Status' to the actual column name in your CSV if different
    if 'Status' in df.columns:
        print("\n📊 Classification Report vs Ground Truth (Status):")
        y_true = df['Status']
        y_pred = df['Predicted_Label']
        print(classification_report(y_true, y_pred))

    # Save results
    result_path = f"output/{config.name}/{config.name}_image_inference_results_anomaly_detection.csv"
    df.to_csv(result_path, index=False)
    print(f"✅ Saved inference results to {result_path}")
