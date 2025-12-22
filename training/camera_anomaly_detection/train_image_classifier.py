
# # vision_subsystem/train_image_classifier.py

# import os
# import json
# import joblib
# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
# from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report
# from tqdm import tqdm

# def run(config):
#     print(f"\n[TRAIN] 🗉 Training image classifier (MobileNetV2) for: {config.name}")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Transforms with augmentation for training
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#         transforms.RandomRotation(15),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # No augmentation for validation and test
#     val_test_transform = transforms.Compose([
#         transforms.Resize(232),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     # Load datasets
#     # data_dir = config.image_training_folder_path
#     # train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
#     # val_data =  datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_test_transform)
#     # test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)

#     train_data = datasets.ImageFolder(config.image_training_folder_path, transform=train_transform)
#     val_data =  datasets.ImageFolder(config.image_validation_folder_path, transform=val_test_transform)
#     test_data = datasets.ImageFolder(config.image_evaluation_folder_path, transform=val_test_transform)

#     train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
#     test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

#     # Load pretrained model
#     weights = MobileNet_V2_Weights.DEFAULT
#     model = mobilenet_v2(weights=weights)
#     num_classes = len(train_data.classes)
#     model.classifier[1] = nn.Linear(model.last_channel, num_classes)

#     # Freeze or fine-tune based on config
#     if not config.finetune_image_model:
#         for param in model.features.parameters():
#             param.requires_grad = False
#         print("🔐 Feature extractor frozen. Only classifier will be trained.")
#     else:
#         print("🔓 Fine-tuning all layers.")

#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

#     # Training with early stopping
#     best_val_loss = float('inf')
#     patience = config.early_stopping_patience if hasattr(config, 'early_stopping_patience') else 3
#     num_epochs = config.image_classifier_epochs if hasattr(config, 'image_classifier_epochs') else 3
#     wait = 0

#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         model.train()
#         for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#         avg_train_loss = running_loss / len(train_loader)
#         print(f"📉 Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

#         # Validation loss
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#         avg_val_loss = val_loss / len(val_loader)
#         print(f"📋 Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

#         # Early stopping logic
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             wait = 0
#             torch.save(model.state_dict(), "temp_best_model.pth")
#         else:
#             wait += 1
#             print(f"⏳ No improvement. Early stopping patience: {wait}/{patience}")
#             if wait >= patience:
#                 print("🛑 Early stopping triggered.")
#                 break

#     # Load best model
#     model.load_state_dict(torch.load("temp_best_model.pth", weights_only=True))
#     os.remove("temp_best_model.pth")

#     # Save model and class names
#     model_dir = f"models/{config.name}"
#     os.makedirs(model_dir, exist_ok=True)
#     model_path = f"{model_dir}/{config.name}_mobilenetv2.pth"
#     torch.save(model.state_dict(), model_path)
#     print(f"✅ Saved model to {model_path}")

#     class_map_path = f"{model_dir}/{config.name}_class_map.json"
#     with open(class_map_path, 'w') as f:
#         json.dump(train_data.classes, f)
#     print(f"✅ Saved class names to {class_map_path}")

#     # Test performance
#     all_preds, all_labels = [], []
#     model.eval()
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.numpy())

#     print("\n📊 Test Classification Report:")
#     print(classification_report(all_labels, all_preds, target_names=train_data.classes))

import os
import json
import joblib
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
from pathlib import Path
import time

def normalize_class_names(class_data):
    """
    Normalize class names to consistent list format.
    Handles both list and dict formats from different training versions.
    """
    if isinstance(class_data, list):
        print(f"   ✅ Standard list format detected")
        return class_data
    elif isinstance(class_data, dict):
        print(f"   🔄 Dictionary format detected, converting to list")
        sorted_items = sorted(class_data.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
        normalized = [value for key, value in sorted_items]
        print(f"   ✅ Converted: {class_data} → {normalized}")
        return normalized
    else:
        raise ValueError(f"Unsupported class data format: {type(class_data)}")

def check_class_mapping_source(class_map_path):
    """Analyze the class mapping file to understand its source."""
    print(f"🔍 Analyzing class mapping file: {class_map_path}")
    
    if os.path.exists(class_map_path):
        file_size = os.path.getsize(class_map_path)
        mod_time = os.path.getmtime(class_map_path)
        mod_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
        
        print(f"   📁 File size: {file_size} bytes")
        print(f"   📅 Last modified: {mod_date}")
        
        with open(class_map_path, 'r') as f:
            content = f.read().strip()
        
        if content.startswith('{"') and content.endswith('"}'):
            print(f"   🔍 Format: Dictionary (possibly from older PyTorch or manual edit)")
        elif content.startswith('[') and content.endswith(']'):
            print(f"   🔍 Format: List (standard ImageFolder format)")
        else:
            print(f"   ⚠️ Format: Unknown")
    
    return True

def load_existing_model(config, device):
    """Check if a trained model exists and load it."""
    model_dir = Path(f"models/{config.name}")
    model_path = model_dir / f"{config.name}_mobilenetv2.pth"
    class_map_path = model_dir / f"{config.name}_class_map.json"
    
    # Check if both model and class map exist
    if not (model_path.exists() and class_map_path.exists()):
        print(f"❌ Model files not found:")
        if not model_path.exists():
            print(f"   Missing: {model_path}")
        if not class_map_path.exists():
            print(f"   Missing: {class_map_path}")
        return None, None, False
    
    try:
        # Analyze the class mapping file
        check_class_mapping_source(class_map_path)
        
        # Load class names
        with open(class_map_path, 'r') as f:
            raw_class_data = json.load(f)
        
        print(f"🔍 Raw class data from file: {raw_class_data} (type: {type(raw_class_data)})")
        
        # Normalize class names to list format
        class_names = normalize_class_names(raw_class_data)
        print(f"🔄 Normalized class names: {class_names}")
        
        # Create model architecture
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=None)  # Don't load pretrained weights
        num_classes = len(class_names)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        
        print(f"✅ Successfully loaded existing model from {model_path}")
        print(f"📋 Model trained for {num_classes} classes: {class_names}")
        
        return model, class_names, True
        
    except Exception as e:
        print(f"❌ Error loading existing model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

def evaluate_model(model, test_loader, class_names, device):
    """Evaluate the model on test data."""
    all_preds, all_labels = [], []
    correct, total = 0, 0
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    classification_report_str = classification_report(all_labels, all_preds, target_names=class_names)
    
    return test_accuracy, classification_report_str

def train_new_model(config, device, train_loader, val_loader, num_classes):
    """Train a new model from scratch."""
    print("🏗️ Creating and training new model...")
    
    # Load pretrained model
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    # Freeze or fine-tune based on config
    if not getattr(config, 'finetune_image_model', False):
        for param in model.features.parameters():
            param.requires_grad = False
        print("🔒 Feature extractor frozen. Only classifier will be trained.")
    else:
        print("🔓 Fine-tuning all layers.")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Training with early stopping
    best_val_loss = float('inf')
    patience = getattr(config, 'early_stopping_patience', 3)
    num_epochs = getattr(config, 'image_classifier_epochs', 10)
    wait = 0

    for epoch in range(num_epochs):
        # Training phase
        running_loss = 0.0
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"📉 Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        print(f"📋 Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), "temp_best_model.pth")
            print(f"💾 New best model saved (Val Loss: {avg_val_loss:.4f})")
        else:
            wait += 1
            print(f"⏳ No improvement. Early stopping patience: {wait}/{patience}")
            if wait >= patience:
                print("🛑 Early stopping triggered.")
                break

    # Load best model
    if os.path.exists("temp_best_model.pth"):
        model.load_state_dict(torch.load("temp_best_model.pth", weights_only=True))
        os.remove("temp_best_model.pth")
        print("✅ Loaded best model from training")

    return model

def save_model_and_classes(model, class_names, config):
    """Save the trained model and class names."""
    model_dir = Path(f"models/{config.name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f"{config.name}_mobilenetv2.pth"
    class_map_path = model_dir / f"{config.name}_class_map.json"
    
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved model to {model_path}")

    with open(class_map_path, 'w') as f:
        json.dump(class_names, f)
    print(f"✅ Saved class names to {class_map_path}")

def run(config):
    print(f"\n[TRAIN] 🖼️ Image classifier pipeline for: {config.name}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    try:
        train_data = datasets.ImageFolder(config.image_training_folder_path, transform=train_transform)
        val_data = datasets.ImageFolder(config.image_validation_folder_path, transform=val_test_transform)
        test_data = datasets.ImageFolder(config.image_evaluation_folder_path, transform=val_test_transform)
        
        print(f"📊 Dataset loaded successfully:")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Validation samples: {len(val_data)}")
        print(f"   Test samples: {len(test_data)}")
        print(f"   Classes: {train_data.classes}")
        
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # === CHECK FOR EXISTING MODEL ===
    print("\n🔍 Checking for existing trained model...")
    existing_model, existing_classes, model_exists = load_existing_model(config, device)
    
    if model_exists:
        # Verify class compatibility
        print(f"🔍 Class comparison:")
        print(f"   Existing (normalized): {existing_classes}")
        print(f"   Current dataset:       {train_data.classes}")
        
        # Enhanced compatibility check
        classes_match = (
            len(existing_classes) == len(train_data.classes) and
            set(existing_classes) == set(train_data.classes)
        )
        
        if classes_match:
            print("✅ Existing model classes match current dataset")
            
            # Evaluate existing model
            print("\n📊 Evaluating existing model on test data...")
            test_accuracy, test_report = evaluate_model(existing_model, test_loader, existing_classes, device)
            
            print(f"🎯 Existing Model Test Accuracy: {test_accuracy:.2f}%")
            print("\n📋 Existing Model Test Classification Report:")
            print(test_report)
            
            # Check if we should retrain
            retrain_decision = getattr(config, 'force_retrain', False)
            
            if not retrain_decision:
                print("\n✅ Using existing model. Set config.force_retrain=True to retrain.")
                return {
                    'model_path': f"models/{config.name}/{config.name}_mobilenetv2.pth",
                    'test_accuracy': test_accuracy,
                    'class_names': existing_classes,
                    'retrained': False
                }
            else:
                print("\n🔄 force_retrain=True, proceeding with retraining...")
        
        else:
            print("⚠️ Existing model classes don't match current dataset")
            print(f"   Class count - Existing: {len(existing_classes)}, Current: {len(train_data.classes)}")
            print(f"   Missing from existing: {set(train_data.classes) - set(existing_classes)}")
            print(f"   Extra in existing: {set(existing_classes) - set(train_data.classes)}")
            print("🔄 Will train new model...")
    
    else:
        print("❌ No existing model found. Training new model...")

    # === TRAIN NEW MODEL ===
    print("\n🏋️ Starting model training...")
    model = train_new_model(config, device, train_loader, val_loader, len(train_data.classes))
    
    # === SAVE MODEL ===
    save_model_and_classes(model, train_data.classes, config)

    # === FINAL EVALUATION ===
    print("\n🎯 Final evaluation on test data...")
    test_accuracy, test_report = evaluate_model(model, test_loader, train_data.classes, device)
    
    print(f"\n✅ FINAL RESULTS:")
    print(f"🎯 Test Accuracy: {test_accuracy:.2f}%")
    print(f"\n📋 Test Classification Report:")
    print(test_report)
    
    print("=" * 60)
    print("✅ Image classifier training completed!")
    
    return {
        'model_path': f"models/{config.name}/{config.name}_mobilenetv2.pth",
        'test_accuracy': test_accuracy,
        'class_names': train_data.classes,
        'retrained': True
    }