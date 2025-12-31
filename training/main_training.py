 # To run the pipeline, you can use the following command:
# python main_training.py --use_case door

import argparse
import os
import sys
from pathlib import Path
import time

# Get current and project root directories
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent

# Add to system path
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

# === IMPORT MODULES ===
try:
    from config.config_manager import ConfigManager
    from non_vision_subsystem import train_adaptive_autoencoder, delay_calculation, dynamic_delay_calibration, detect_anomalies, align_images, rule_based_detector
    from ssl_subsystem import ssl_train_cluster
    from label_refinement_subsystem import refine_labels
    from camera_anomaly_detection import split_dataset, train_image_classifier, image_inference
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# === VALIDATION FUNCTION ===
def validate_prerequisites(config):
    print("🔍 Validating pipeline prerequisites...")
    issues = []

    if not os.path.exists(config.sensor_data_path):
        issues.append(f"❌ Sensor data file not found: {config.sensor_data_path}")
    else:
        print(f"✅ Sensor data file found")

    if getattr(config, 'align_images', False):
        if not os.path.exists(getattr(config, 'image_data_path', '')):
            issues.append(f"❌ Image data file not found: {config.image_data_path}")
        else:
            print(f"✅ Image data file found")

    os.makedirs(f"models/{config.name}", exist_ok=True)
    os.makedirs(f"output/{config.name}", exist_ok=True)
    print(f"✅ Output/model directories created/verified")

    if issues:
        print("\n⚠️ Validation issues:")
        for issue in issues:
            print(f"   {issue}")
        return False
    return True

def run_step(step_name, condition, function, config):
    if not condition or function is None:
        return

    print(f"\n🚀 {step_name}...")
    start_time = time.time()

    try:
        result = function.run(config)
        duration = time.time() - start_time

        if isinstance(result, dict) and not result.get("success", True):
            print(f"❌ {step_name} failed: {result.get('error')}")
        else:
            print(f"✅ {step_name} completed in {duration:.2f} seconds")
            if isinstance(result, dict):
                if 'best_f1' in result:
                    print(f"📊 Best F1-Score: {result['best_f1']:.3f}")
                if 'optimal_sensitivity' in result:
                    print(f"🎯 Optimal Sensitivity: {result['optimal_sensitivity']}")
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ {step_name} failed after {duration:.2f} seconds: {e}")
        import traceback
        traceback.print_exc()

# === MAIN PIPELINE ===
def main(use_case_name):
    print(f"\n🚀 Running pipeline for: {use_case_name.upper()}\n")

    config = ConfigManager.get_config(use_case_name)

    # === SPECIAL HANDLING for rule-based case ===
    if use_case_name.lower() == "abnormal_object":
        rule_based_detector.run(config)

    # === CONFIG FLAGS ===
    config.train_adaptive_autoencoder = False
    config.dynamic_delay = False
    config.detect_anomalies = False
    config.align_images = False
    config.train_ssl = False
    config.refine_labels = True
    config.enable_image_classification_split = False
    config.train_image_classifier = False
    # config.run_image_inference = False

    # === SUBSYSTEM STEPS ===
    steps = [
        ("Training Adaptive Autoencoder", config.train_adaptive_autoencoder, train_adaptive_autoencoder),
        ("Sensor Delay Calculation", config.calculate_delays, delay_calculation),
        ("Dynamic Delay Calibration", config.dynamic_delay, dynamic_delay_calibration),
        ("Anomaly Detection", config.detect_anomalies, detect_anomalies),
        ("Image-Sensor Alignment", config.align_images, align_images),
        ("SSL Training", config.train_ssl, ssl_train_cluster),
        ("Label Refinement", config.refine_labels, refine_labels)
    ]

    image_steps = [
        ("Dataset Splitting", config.enable_image_classification_split, split_dataset),
        ("Image Classifier Training", config.train_image_classifier, train_image_classifier),
        ("Image Inference", config.run_image_inference, image_inference)
    ]

    for step_name, condition, func in steps:
        run_step(step_name, condition, func, config)

    if any(s[1] for s in image_steps):
        print("\n📷 IMAGE CLASSIFICATION SUBSYSTEM")
        for step_name, condition, func in image_steps:
            run_step(step_name, condition, func, config)

    return True

# === ENTRY POINT ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection pipeline")
    parser.add_argument("--use_case", type=str, required=True, choices=["door", "appliance", "co2", "abnormal_object"], help="Specify use case")
    parser.add_argument("--validate_only", action="store_true", help="Only validate prerequisites")
    parser.add_argument("--skip_validation", action="store_true", help="Skip validation")
    args = parser.parse_args()

    try:
        config = ConfigManager.get_config(args.use_case)
    except Exception as e:
        print(f"❌ Configuration load failed: {e}")
        sys.exit(1)

    if not args.skip_validation and not validate_prerequisites(config):
        sys.exit(1)

    if args.validate_only:
        print("\n✅ Validation complete.")
        sys.exit(0)

    try:
        success = main(args.use_case)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
