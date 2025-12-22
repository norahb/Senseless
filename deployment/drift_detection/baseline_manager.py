import os
import sys
import pandas as pd
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from deployment.config.config_deployment import DeploymentConfig
from drift_detection.drift_detector import DriftDetector

def update_baseline(use_case_name, data_source='sensor_data'):
    """Update drift detection baseline with current clean data"""
    print(f"\n🔄 Updating drift detection baseline for: {use_case_name.upper()}")
    print(f"📁 Data source: {data_source}")  # Add this debug line
    
    config = DeploymentConfig(use_case_name)
    
    if data_source == 'sensor_data':
        print(f"🔍 Looking for sensor data at: {config.incoming_sensor_data_path}")  # Add this
        if os.path.exists(config.incoming_sensor_data_path):
            sensor_data = pd.read_csv(config.incoming_sensor_data_path)
            print(f"📊 Loaded {len(sensor_data)} samples from sensor data")  # Add this
            
            detector = DriftDetector(config)
            detector.update_reference_stats(sensor_data)
            print(f"✅ Baseline updated with {len(sensor_data)} sensor samples")
        else:
            print(f"⚠️ Sensor data not found: {config.incoming_sensor_data_path}")
            return False
    
    if data_source == 'decision_log':
        if os.path.exists(config.decision_log_path):
            recent_data = pd.read_csv(config.decision_log_path)
            clean_data = recent_data[recent_data.get('Sensor_Status', 'Normal') != 'Sensor_Error']
            
            # VALIDATION CODE HERE
            if len(clean_data) < 100:
                print(f"⚠️ WARNING: Only {len(clean_data)} samples for baseline - recommend at least 100")
            
            error_rate = (len(recent_data) - len(clean_data)) / len(recent_data)
            if error_rate > 0.1:
                print(f"⚠️ WARNING: {error_rate:.1%} of data contains sensor errors")
            
            if len(clean_data) > 0:
                detector = DriftDetector(config)
                detector.update_reference_stats(clean_data)
                print(f"✅ Baseline updated with {len(clean_data)} clean samples")
            else:
                print("⚠️ No clean data available for baseline update")
                return False
        else:
            print(f"⚠️ Decision log not found: {config.decision_log_path}")
            return False
    
    elif data_source == 'sensor_data':
        if os.path.exists(config.incoming_sensor_data_path):
            sensor_data = pd.read_csv(config.incoming_sensor_data_path)
            
            # VALIDATION CODE HERE
            if len(sensor_data) < 100:
                print(f"⚠️ WARNING: Only {len(sensor_data)} samples for baseline - recommend at least 100")
            
            # Check for missing sensor columns
            missing_cols = [col for col in config.use_case_config.sensor_cols if col not in sensor_data.columns]
            if missing_cols:
                print(f"⚠️ WARNING: Missing sensor columns: {missing_cols}")
                return False
            
            detector = DriftDetector(config)
            detector.update_reference_stats(sensor_data)
            print(f"✅ Baseline updated with {len(sensor_data)} sensor samples")
        else:
            print(f"⚠️ Sensor data not found: {config.incoming_sensor_data_path}")
            return False
    
    return True

def reset_baseline(use_case_name):
    """Reset baseline statistics (delete existing baseline)"""
    print(f"\n🗑️ Resetting drift detection baseline for: {use_case_name.upper()}")
    
    config = DeploymentConfig(use_case_name)
    detector = DriftDetector(config)
    
    if os.path.exists(detector.reference_stats_path):
        os.remove(detector.reference_stats_path)
        print("✅ Baseline reset successfully")
    else:
        print("⚠️ No existing baseline found")

def view_baseline(use_case_name):
    """View current baseline statistics"""
    print(f"\n📊 Current baseline statistics for: {use_case_name.upper()}")
    
    config = DeploymentConfig(use_case_name)
    detector = DriftDetector(config)
    
    if len(detector.reference_stats) == 0:
        print("⚠️ No baseline statistics found")
        return
    
    for sensor, stats in detector.reference_stats.items():
        print(f"\n{sensor}:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std: {stats['std']:.3f}")
        print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
        print(f"  Percentiles: 25%={stats['percentiles']['25']:.3f}, 50%={stats['percentiles']['50']:.3f}, 75%={stats['percentiles']['75']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage drift detection baseline")
    parser.add_argument("--use_case", type=str, required=True,
                        choices=["door", "appliance", "co2", "abnormal_object"],
                        help="Specify the use case")
    parser.add_argument("--action", type=str, required=True,
                        choices=["update", "reset", "view"],
                        help="Action to perform")
    parser.add_argument("--source", type=str, choices=["decision_log", "sensor_data"],
                        default="decision_log",
                        help="Data source for baseline update")
    
    args = parser.parse_args()
    
    if args.action == "update":
        update_baseline(args.use_case, args.source)
    elif args.action == "reset":
        reset_baseline(args.use_case)
    elif args.action == "view":
        view_baseline(args.use_case)