# migration_utility.py
"""
Migration Utility for Model Formats
Helps transition from legacy to adaptive autoencoder models
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.config_manager import ConfigManager
from non_vision_subsystem.model_compatibility import ModelCompatibilityManager

def check_all_models():
    """Check status of all models across use cases."""
    print("🔍 CHECKING ALL MODEL FORMATS")
    print("=" * 60)
    
    use_cases = ["door", "co2", "appliance", "abnormal_object"]
    results = {}
    
    for use_case in use_cases:
        print(f"\n--- {use_case.upper()} ---")
        try:
            config = ConfigManager.get_config(use_case)
            manager = ModelCompatibilityManager(config)
            info = manager.get_model_info()
            
            results[use_case] = info
            
            print(f"📁 Model Directory: {info['model_dir']}")
            print(f"📊 Format Detected: {info['model_format']}")
            
            if info['available_files']:
                print(f"📄 Available Files: {len(info['available_files'])}")
                for file in sorted(info['available_files'])[:5]:  # Show first 5
                    print(f"   • {file}")
                if len(info['available_files']) > 5:
                    print(f"   ... and {len(info['available_files']) - 5} more")
            else:
                print("📄 No model files found")
            
            for rec in info['recommendations']:
                print(f"💡 {rec}")
                
        except Exception as e:
            print(f"❌ Error checking {use_case}: {e}")
            results[use_case] = {'error': str(e)}
    
    return results

def migrate_use_case(use_case: str, force: bool = False):
    """Migrate a specific use case to adaptive format."""
    print(f"\n🔄 MIGRATING {use_case.upper()} TO ADAPTIVE FORMAT")
    print("=" * 60)
    
    try:
        config = ConfigManager.get_config(use_case)
        manager = ModelCompatibilityManager(config)
        
        # Check current format
        current_format = manager.detect_model_format()
        print(f"📊 Current format: {current_format}")
        
        if current_format == 'none':
            print("❌ No model found to migrate. Train a model first.")
            return False
        
        if current_format == 'adaptive' and not force:
            print("✅ Model is already in adaptive format")
            return True
        
        if current_format == 'adaptive' and force:
            print("🔄 Force migration requested, will reload adaptive format...")
        
        # Perform migration
        success = manager.convert_legacy_to_adaptive()
        
        if success:
            print("✅ Migration completed successfully!")
            
            # Verify migration
            new_format = manager.detect_model_format()
            print(f"📊 New format: {new_format}")
            
            # Test loading
            try:
                model, scaler, thresholds, feature_importance = manager.load_model_components()
                print("✅ Model loading test passed")
                
                delays = manager.load_delays()
                print("✅ Delay loading test passed")
                
                return True
                
            except Exception as e:
                print(f"❌ Model loading test failed: {e}")
                return False
        
        else:
            print("❌ Migration failed")
            return False
            
    except Exception as e:
        print(f"❌ Migration error: {e}")
        return False

def migrate_all_models(force: bool = False):
    """Migrate all use cases to adaptive format."""
    print("\n🚀 MIGRATING ALL MODELS TO ADAPTIVE FORMAT")
    print("=" * 60)
    
    use_cases = ["door", "co2", "appliance", "abnormal_object"]
    results = {}
    
    for use_case in use_cases:
        print(f"\n{'='*20} {use_case.upper()} {'='*20}")
        try:
            success = migrate_use_case(use_case, force)
            results[use_case] = {'success': success}
        except Exception as e:
            print(f"❌ Failed to migrate {use_case}: {e}")
            results[use_case] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 MIGRATION SUMMARY")
    print("=" * 60)
    
    successful = []
    failed = []
    
    for use_case, result in results.items():
        if result.get('success', False):
            successful.append(use_case)
            print(f"✅ {use_case.upper()}: Success")
        else:
            failed.append(use_case)
            error = result.get('error', 'Unknown error')
            print(f"❌ {use_case.upper()}: Failed - {error}")
    
    print(f"\n🎯 Results: {len(successful)} successful, {len(failed)} failed")
    
    if successful:
        print(f"✅ Successfully migrated: {', '.join(successful)}")
    
    if failed:
        print(f"❌ Failed to migrate: {', '.join(failed)}")
        print("\n💡 For failed migrations:")
        print("   1. Make sure the use case has been trained")
        print("   2. Check that all required files exist")
        print("   3. Run with --verbose for more details")
    
    return len(failed) == 0

def cleanup_legacy_files(use_case: str, confirm: bool = False):
    """Clean up legacy model files after successful migration."""
    print(f"\n🧹 CLEANING UP LEGACY FILES FOR {use_case.upper()}")
    print("=" * 60)
    
    try:
        config = ConfigManager.get_config(use_case)
        model_dir = f"models/{config.name}"
        
        if not os.path.exists(model_dir):
            print(f"❌ Model directory not found: {model_dir}")
            return False
        
        # Check if adaptive model exists
        adaptive_path = os.path.join(model_dir, f"adaptive_{config.name}.pkl")
        if not os.path.exists(adaptive_path):
            print("❌ No adaptive model found. Migration must be completed first.")
            return False
        
        # Legacy files to potentially remove
        legacy_files = [
            f"{config.name}_enhanced_autoencoder.joblib",
            f"{config.name}_scaler.joblib", 
            f"{config.name}_thresholds.json",
            f"{config.name}_feature_importance.json"
        ]
        
        # Find existing legacy files
        existing_legacy = []
        for file in legacy_files:
            file_path = os.path.join(model_dir, file)
            if os.path.exists(file_path):
                existing_legacy.append(file_path)
        
        if not existing_legacy:
            print("✅ No legacy files found to clean up")
            return True
        
        print(f"📄 Found {len(existing_legacy)} legacy files:")
        for file_path in existing_legacy:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"   • {os.path.basename(file_path)} ({file_size:.2f} MB)")
        
        if not confirm:
            print("\n⚠️  Use --confirm to actually delete these files")
            print("💡 These files are no longer needed after successful migration")
            return False
        
        # Delete legacy files
        deleted_count = 0
        for file_path in existing_legacy:
            try:
                os.remove(file_path)
                print(f"🗑️  Deleted: {os.path.basename(file_path)}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to delete {os.path.basename(file_path)}: {e}")
        
        print(f"\n✅ Successfully deleted {deleted_count}/{len(existing_legacy)} legacy files")
        
        # Verify adaptive model still works
        try:
            manager = ModelCompatibilityManager(config)
            model, scaler, thresholds, feature_importance = manager.load_model_components()
            print("✅ Adaptive model verification passed")
            return True
        except Exception as e:
            print(f"❌ Adaptive model verification failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Cleanup error: {e}")
        return False

def validate_pipeline_compatibility(use_case: str):
    """Validate that the pipeline can run with current model format."""
    print(f"\n🔍 VALIDATING PIPELINE COMPATIBILITY FOR {use_case.upper()}")
    print("=" * 60)
    
    try:
        config = ConfigManager.get_config(use_case)
        manager = ModelCompatibilityManager(config)
        
        # Test model loading
        print("1️⃣ Testing model loading...")
        model, scaler, thresholds, feature_importance = manager.load_model_components()
        print(f"   ✅ Model format: {manager.model_format}")
        print(f"   ✅ Sensors: {len(config.sensor_cols)}")
        print(f"   ✅ Thresholds: {len(thresholds)} configured")
        print(f"   ✅ Feature importance: {len(feature_importance)} configured")
        
        # Test delay loading
        print("\n2️⃣ Testing delay loading...")
        delays = manager.load_delays()
        print(f"   ✅ Delay structure validated")
        for sensor in config.sensor_cols:
            delay_val = manager.get_delay_for_sensor(delays, sensor)
            print(f"   ✅ {sensor}: {delay_val:.2f} sec")
        
        # Test data compatibility
        print("\n3️⃣ Testing data compatibility...")
        if os.path.exists(config.sensor_data_path):
            import pandas as pd
            df = pd.read_csv(config.sensor_data_path)
            print(f"   ✅ Sensor data loaded: {df.shape}")
            
            # Quick prediction test
            sample_data = df[config.sensor_cols].head(10).values
            X_scaled = scaler.transform(sample_data)
            predictions = model.autoencoder.predict(X_scaled)
            print(f"   ✅ Prediction test passed: {predictions.shape}")
        else:
            print(f"   ⚠️  Sensor data file not found: {config.sensor_data_path}")
        
        print("\n✅ All compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Model Migration Utility")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check status of all models')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate models to adaptive format')
    migrate_parser.add_argument('--use_case', choices=['door', 'co2', 'appliance', 'abnormal_object', 'all'], 
                               default='all', help='Use case to migrate')
    migrate_parser.add_argument('--force', action='store_true', help='Force migration even if already adaptive')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up legacy files after migration')
    cleanup_parser.add_argument('use_case', choices=['door', 'co2', 'appliance', 'abnormal_object'], 
                                help='Use case to clean up')
    cleanup_parser.add_argument('--confirm', action='store_true', help='Actually delete the files')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate pipeline compatibility')
    validate_parser.add_argument('use_case', choices=['door', 'co2', 'appliance', 'abnormal_object'], 
                                 help='Use case to validate')
    
    args = parser.parse_args()
    
    if args.command == 'check':
        check_all_models()
        
    elif args.command == 'migrate':
        if args.use_case == 'all':
            success = migrate_all_models(args.force)
            sys.exit(0 if success else 1)
        else:
            success = migrate_use_case(args.use_case, args.force)
            sys.exit(0 if success else 1)
            
    elif args.command == 'cleanup':
        success = cleanup_legacy_files(args.use_case, args.confirm)
        sys.exit(0 if success else 1)
        
    elif args.command == 'validate':
        success = validate_pipeline_compatibility(args.use_case)
        sys.exit(0 if success else 1)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()