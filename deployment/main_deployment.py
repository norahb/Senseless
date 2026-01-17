# # main_deployment.py
# # To run the deployment pipeline, use:
# # python main_deployment.py --use_case door

import argparse
import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from deployment.config.config_deployment import DeploymentConfig
from deployment.sensor_inference import run_sensor_model
from deployment.uncertainty.confidence_estimation import run as estimate_confidence
from deployment.uncertainty.fallback_manager import run as vision_fallback
from deployment.drift_detection.baseline_manager import update_baseline
from deployment.drift_detection.drift_detector import run as detect_drift
from deployment.retraining.retraining_manager import RetrainingManager


def main(use_case_name):
    print(f"\n🚀 Running deployment pipeline for: {use_case_name.upper()}\n")

    config = DeploymentConfig(use_case_name)
    
    # Configuration settings
    config.enable_sensor_inference = True  
    config.enable_confidence_estimation = True  
    config.enable_vision_fallback = True
    config.update_baseline_before_drift = False
    config.enable_drift_detection = False
    config.enable_auto_retraining = False


    # 1. Sensor inference 
    if config.enable_sensor_inference:
        print("🔍 Running sensor model inference...")
        sensor_df, sensor_logits = run_sensor_model(config)
    else:
        sensor_df = pd.read_csv(config.decision_log_path)
        sensor_logits = None
    
    # 2. Confidence estimation 
    if config.enable_confidence_estimation:
        sensor_df = estimate_confidence(config, sensor_logits, sensor_df)

    # 3. Vision fallback 
    if config.enable_vision_fallback:
        vision_fallback(config)
    
    # 4. Drift detection + 5. Automatic retraining 
    if config.enable_drift_detection:
        if config.update_baseline_before_drift:
            print("\n🔄 Updating baseline before drift detection...")
            update_baseline(config.use_case, config.baseline_data_source)
            
        print("🔍 Running drift detection...")
        drift_results = detect_drift(config)
        
        # Automatic retraining
        if drift_results['should_retrain'] and config.enable_auto_retraining:
            print("\n🔄 Drift detected - triggering automatic retraining...")
            
            try:
                retraining_manager = RetrainingManager(config)
                retrain_results = retraining_manager.trigger_retraining(drift_results)

                if retrain_results['success']:
                    print(f"\n✅ AUTOMATIC RETRAINING SUCCESSFUL!")
                    print(f"   Performance improvement: {retrain_results['performance_improvement']:.3f}")
                    print(f"   Training samples: {retrain_results['training_samples']}")
                    print(f"   Data quality: {retrain_results['data_quality_score']:.2f}")
                    print(f"   Backup created: {retrain_results['backup_created']}")
                    

                else:
                    print(f"\n⚠️  Automatic retraining failed: {retrain_results.get('reason', 'Unknown')}")
                    
            except Exception as e:
                # print(f"\n❌ Retraining system error: {e}")
                print(f"   Data quality: {retrain_results.get('data_quality_score', 'N/A')}")

        
        elif drift_results['should_retrain']:
            print("\n🚨 ALERT: Model retraining recommended but auto-retraining disabled!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deployment pipeline")
    parser.add_argument("--use_case", type=str, required=True,
                        choices=["door", "appliance", "co2", "abnormal_object"],
                        help="Specify the use case to deploy")
    args = parser.parse_args()
    main(args.use_case)
