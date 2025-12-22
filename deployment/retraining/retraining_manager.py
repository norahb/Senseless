

# deployment/retraining/retraining_manager.py

import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from .data_collector import DataCollector
from .model_trainer import ModelTrainer
from .performance_validator import PerformanceValidator
from .deployment_manager import DeploymentManager


class RetrainingManager:
    """
    Simple retraining workflow coordinator.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize components
        self.data_collector = DataCollector(config)
        self.model_trainer = ModelTrainer(config)
        self.validator = PerformanceValidator(config)
        self.deployment_manager = DeploymentManager(config)
        
        # Settings
        self.min_samples = getattr(config, 'min_samples_for_retraining', 500)
        self.performance_threshold = getattr(config, 'performance_threshold', 0.02)
        
        # Log path
        self.retraining_log_path = f"logs/{config.use_case}/retraining_history.csv"
        os.makedirs(os.path.dirname(self.retraining_log_path), exist_ok=True)
        
    def trigger_retraining(self, drift_results):
        """
        Main retraining workflow.
        """
        print(f"RETRAINING TRIGGERED for {self.config.use_case.upper()}")
        
        retrain_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        try:
            # Step 1: Collect training data
            print("Collecting training data...")
            data_result = self.data_collector.collect_retraining_data()
            
            if not data_result['success']:
                return self._log_failure(retrain_id, f"Data collection failed: {data_result['reason']}")
                        
            # Step 2: Train new model (with cleaning, calibration, evaluation)
            training_result = self.model_trainer.retrain_model(data_result['data'])
            if not training_result['success']:
                return self._log_failure(retrain_id, f"Training failed: {training_result['reason']}")

            print(f"Model trained on {training_result['training_samples']} samples")
            print(f"Validation samples: {training_result.get('validation_samples', 0)}")
            print(f"Best sensitivity: {training_result.get('best_sensitivity', 'N/A')}")
            print(f"Sensitivity results:")
            for sens, metrics in training_result.get('sensitivity_results', {}).items():
                print(f"  {sens}: Acc={metrics['val_accuracy']:.3f}, MacroF1={metrics['val_macro_f1']:.3f}, Balance={metrics['val_balance_score']:.3f}, Score={metrics['overall_score']:.3f}")
            print(f"Cleaning report: {training_result.get('cleaning_report', {})}")

            # Step 3: Validate performance (optional, can be skipped if not needed)
            # current_model = self.model_trainer.load_current_model()
            # validation_result = self.validator.validate_new_model(
            #     training_result['new_model'], 
            #     data_result['validation_data'],
            #     current_model
            # )

            # Step 4: Deploy new model (always deploy if retraining succeeds)
            print("Deploying new model...")
            deployment_result = self.deployment_manager.deploy_new_model(
                training_result['new_model'],
                training_result['model_path']
            )
            if deployment_result['success']:
                total_time = (datetime.now() - start_time).total_seconds()
                success_result = {
                    'success': True,
                    'retrain_id': retrain_id,
                    'action': 'deployed',
                    'training_samples': training_result['training_samples'],
                    'validation_samples': training_result.get('validation_samples', 0),
                    'best_sensitivity': training_result.get('best_sensitivity', 'N/A'),
                    'sensitivity_results': training_result.get('sensitivity_results', {}),
                    'cleaning_report': training_result.get('cleaning_report', {}),
                    'total_time_seconds': total_time,
                    'backup_created': deployment_result['backup_created']
                }
                self._log_event(success_result)
                print(f"RETRAINING SUCCESS! Model deployed.")
                return success_result
            else:
                return self._log_failure(retrain_id, f"Deployment failed: {deployment_result['reason']}")
                
        except Exception as e:
            print(f"RETRAINING ERROR: {e}")
            return self._log_failure(retrain_id, f"Unexpected error: {str(e)}")
    
    def _log_failure(self, retrain_id, reason):
        """Log failure and return result"""
        result = {
            'success': False,
            'retrain_id': retrain_id,
            'action': 'failed',
            'reason': reason
        }
        
        self._log_event(result)
        print(f"RETRAINING FAILED: {reason}")
        return result
    
    def _log_event(self, result):
        """Simple event logging"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'retrain_id': result['retrain_id'],
                'action': result['action'],
                'success': result['success'],
                'reason': result.get('reason', ''),
                'performance_improvement': result.get('performance_improvement', 0.0),
                'training_samples': result.get('training_samples', 0)
            }
            
            # Load existing log
            if os.path.exists(self.retraining_log_path):
                history = pd.read_csv(self.retraining_log_path)
            else:
                history = pd.DataFrame()
            
            # Append new entry
            new_row = pd.DataFrame([log_entry])
            history = pd.concat([history, new_row], ignore_index=True)
            
            # Keep last 50 entries
            if len(history) > 50:
                history = history.tail(50)
            
            # Save
            history.to_csv(self.retraining_log_path, index=False)
            
        except Exception as e:
            print(f"Failed to log event: {e}")
    
    def get_retraining_status(self):
        """Get simple retraining status"""
        try:
            if not os.path.exists(self.retraining_log_path):
                return {'status': 'ready', 'total_retrains': 0}
            
            history = pd.read_csv(self.retraining_log_path)
            
            if len(history) == 0:
                return {'status': 'ready', 'total_retrains': 0}
            
            last_entry = history.iloc[-1]
            successful_retrains = len(history[history['success'] == True])
            
            return {
                'status': 'active' if last_entry['success'] else 'last_failed',
                'total_retrains': len(history),
                'successful_retrains': successful_retrains,
                'last_retrain': {
                    'timestamp': last_entry['timestamp'],
                    'action': last_entry['action'],
                    'success': last_entry['success'],
                    'improvement': last_entry.get('performance_improvement', 0.0)
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}