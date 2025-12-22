
# deployment/retraining/deployment_manager.py

import os
import shutil
import json
from datetime import datetime
from pathlib import Path


class DeploymentManager:
    """
    Simple deployment manager - just copies model files with basic backup.
    """
    
    def __init__(self, config):
        self.config = config
        self.current_model_path = config.sensor_model_path
        self.backup_dir = Path(f"models/{config.use_case}/backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def deploy_new_model(self, new_model, new_model_path):
        """
        Deploy new model by replacing base model and cleaning up.
        """
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            print(f"Starting deployment...")
            
            # Define target paths
            target_file = os.path.join(self.current_model_path, f"adaptive_{self.config.use_case}.pkl")
            target_config = os.path.join(self.current_model_path, f"adaptive_{self.config.use_case}_config.json")

            # Backup current model if it exists
            backup_path = self._backup_current_model(deployment_id, target_file, target_config)
            
            # Copy new model to production location
            source_file = f"{new_model_path}.pkl"
            source_config = f"{new_model_path}_config.json"
            
            if not os.path.exists(source_file):
                return {'success': False, 'reason': f'Source model not found: {source_file}'}
            
            # Copy files to production location
            shutil.copy2(source_file, target_file)
            if os.path.exists(source_config):
                shutil.copy2(source_config, target_config)
            
            # Verify deployment
            if not os.path.exists(target_file):
                return {'success': False, 'reason': 'Deployment verification failed'}
            
            # Clean up old retrained files
            self._cleanup_old_retrained_files()
            
            print(f"Model deployed successfully")
            
            return {
                'success': True,
                'deployment_id': deployment_id,
                'backup_created': backup_path,
                'deployed_path': target_file
            }
            
        except Exception as e:
            print(f"Deployment failed: {e}")
            return {'success': False, 'reason': str(e)}
    
    def _backup_current_model(self, deployment_id, target_file, target_config):
        """Create backup of current production model"""
        try:
            if not os.path.exists(target_file):
                print(f"No current model to backup")
                return None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"backup_{deployment_id}_{timestamp}"
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(exist_ok=True)
            
            # Copy current production model to backup
            shutil.copy2(target_file, backup_path / f"adaptive_{self.config.use_case}.pkl")
            if os.path.exists(target_config):
                shutil.copy2(target_config, backup_path / f"adaptive_{self.config.use_case}_config.json")
            
            print(f"Backed up current model")
            return str(backup_path)
            
        except Exception as e:
            print(f"Backup failed: {e}")
            return None
    
    def _cleanup_old_retrained_files(self):
        """Clean up old retrained model files"""
        try:
            model_dir = Path(self.current_model_path)
            
            # Find all retrained files
            retrained_files = list(model_dir.glob("retrained_*.pkl"))
            retrained_configs = list(model_dir.glob("retrained_*_config.json"))
            
            # Remove them
            for file in retrained_files + retrained_configs:
                file.unlink()
                print(f"Cleaned up: {file.name}")
            
            if retrained_files or retrained_configs:
                print(f"Cleaned up {len(retrained_files)} retrained models and {len(retrained_configs)} configs")
            
        except Exception as e:
            print(f"Cleanup failed: {e}")