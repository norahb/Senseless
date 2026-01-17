
# deployment/retraining/data_collector.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

from .replay_buffer import ReplayBuffer

class DataCollector:
    """
     data collector with dynamic sampling and proper label harmonization.
    """
    
    def __init__(self, config):
        self.config = config
        self.min_samples = getattr(config, 'min_samples_for_retraining', 200)
        self.validation_split = getattr(config, 'validation_split', 0.2)
        self.replay_buffer = ReplayBuffer(config)
        
    def _calculate_dynamic_sample_size(self, available_data_sizes):
        """Calculate total target sample size based on data availability."""
        total_available = sum(available_data_sizes.values())
        
        percentage = 0.2  # 2% of available data
        dynamic_size = int(total_available * percentage)
        dynamic_size = max(dynamic_size, self.min_samples)  

        print(f"   📊 Dynamic sampling calculation:")
        print(f"      Total available: {total_available}")
        print(f"      Using {percentage*100}% = {dynamic_size} samples")
        
        return dynamic_size
        
    def collect_retraining_data(self):
        """Collect training data with dynamic sampling and proper ratios."""
        try:
            # First pass: check available data sizes
            available_sizes = self._get_available_data_sizes()
            
            # Calculate dynamic total size
            total_target_size = self._calculate_dynamic_sample_size(available_sizes)
            
            # Calculate per-source target sizes using config ratios
            source_targets = {
                'recent': int(total_target_size * self.config.recent_data_ratio),
                'historical': int(total_target_size * self.config.historical_data_ratio),
                'replay': int(total_target_size * self.config.replay_buffer_ratio)
            }
            
            print(f"   🎯 Target samples per source: {source_targets}")
            
            # Collect from sources with calculated targets
            sensor_data = self._collect_sensor_data(source_targets['recent'])
            replay_data = self._collect_replay_samples(source_targets['replay'])
            original_data = self._collect_original_data(source_targets['historical'])

            # Combine all sources
            combined_data = self._combine_data_sources(sensor_data, replay_data, original_data)

            if combined_data is None or len(combined_data) < self.min_samples:
                return {
                    'success': False,
                    'reason': f'Insufficient data: {len(combined_data) if combined_data is not None else 0} < {self.min_samples}'
                }

            # Update replay buffer
            self._update_replay_buffer(sensor_data)

            return {
                'success': True,
                'data': combined_data,
                'validation_data': None,
                'data_quality': {'quality_score': 0.8, 'total_samples': len(combined_data)},
                'sources': {
                    'sensor_samples': len(sensor_data),
                    'replay_samples': len(replay_data),
                    'original_samples': len(original_data),
                    'total_samples': len(combined_data)
                }
            }

        except Exception as e:
            return {'success': False, 'reason': f'Data collection error: {str(e)}'}
    
    def _get_available_data_sizes(self):
        """Get available data sizes from all sources for planning."""
        sizes = {'recent': 0, 'historical': 0, 'replay': 0}
        
        try:
            # Check recent data
            if os.path.exists(self.config.decision_log_path):
                decisions = pd.read_csv(self.config.decision_log_path)
                clean_data = decisions[decisions.get('Sensor_Status', 'Normal') != 'Sensor_Error']
                sizes['recent'] = len(clean_data)
            
            # Check historical data
            training_path = self.config.use_case_config.training_csv_path
            if training_path and os.path.exists(training_path):
                original = pd.read_csv(training_path)
                sizes['historical'] = len(original)
            
            # Check replay buffer
            sizes['replay'] = len(self.replay_buffer.samples)
            
        except Exception as e:
            print(f"   ⚠️  Error checking data sizes: {e}")
        
        return sizes
    
    def _collect_sensor_data(self, target_size):
        """Collect sensor data with 80/20 ratio and target size."""
        try:
            if not os.path.exists(self.config.decision_log_path):
                print(f"   ⚠️  Decision log not found")
                return pd.DataFrame()

            decisions = pd.read_csv(self.config.decision_log_path)
            clean_data = decisions[decisions.get('Sensor_Status', 'Normal') != 'Sensor_Error']

            sensor_cols = self.config.use_case_config.sensor_cols
            required_cols = sensor_cols + ['Sensor_Timestamp', 'Sensor_Status']
            available_cols = [col for col in required_cols if col in clean_data.columns]

            if len([col for col in sensor_cols if col in available_cols]) == 0:
                print(f"   ⚠️  No sensor columns found")
                return pd.DataFrame()

            # Get recent data
            recent_data = clean_data.tail(min(target_size * 2, len(clean_data)))[available_cols].dropna()
            
            # Apply 80/20 sampling
            normal_target = int(target_size * 0.8)
            anomaly_target = int(target_size * 0.2)
            
            normal_data = recent_data[recent_data['Sensor_Status'] == 'Normal']
            anomaly_data = recent_data[recent_data['Sensor_Status'] != 'Normal']
            
            # Sample with available limits
            normal_sample = normal_data.sample(n=min(normal_target, len(normal_data)), random_state=42) if len(normal_data) > 0 else pd.DataFrame()
            anomaly_sample = anomaly_data.sample(n=min(anomaly_target, len(anomaly_data)), random_state=42) if len(anomaly_data) > 0 else pd.DataFrame()
            
            sampled_data = pd.concat([normal_sample, anomaly_sample], ignore_index=True)
            
            print(f"   ✅ Recent data: {len(normal_sample)} normal + {len(anomaly_sample)} anomaly = {len(sampled_data)}")
            
            # Standardize columns
            if 'Sensor_Timestamp' in sampled_data.columns:
                sampled_data = sampled_data.rename(columns={'Sensor_Timestamp': 'Timestamp'})
                sampled_data['Timestamp'] = sampled_data['Timestamp'].astype(str)
            if 'Sensor_Status' in sampled_data.columns:
                sampled_data = sampled_data.rename(columns={'Sensor_Status': 'Label'})

            return sampled_data

        except Exception as e:
            print(f"   ❌ Error collecting sensor data: {e}")
            return pd.DataFrame()
    
    def _collect_replay_samples(self, target_size):
        """Get replay buffer samples with target size."""
        try:
            replay_data = self.replay_buffer.get_replay_samples(max_samples=target_size)

            if len(replay_data) > 0:
                sensor_cols = self.config.use_case_config.sensor_cols
                
                missing_cols = [col for col in sensor_cols if col not in replay_data.columns]
                if missing_cols:
                    print(f"   ⚠️  Replay buffer missing sensor columns: {missing_cols}")
                    return pd.DataFrame()

                # Add missing columns
                if 'Timestamp' not in replay_data.columns:
                    replay_data['Timestamp'] = pd.Timestamp.now().strftime('%d/%m/%Y %H:%M:%S')
                else:
                    replay_data['Timestamp'] = replay_data['Timestamp'].astype(str)
                
                if 'Label' not in replay_data.columns:
                    replay_data['Label'] = 'Normal'

                print(f"   ✅ Retrieved {len(replay_data)} replay buffer samples")
            else:
                print(f"   ⚠️  No replay buffer samples")

            return replay_data

        except Exception as e:
            print(f"   ❌ Error accessing replay buffer: {e}")
            return pd.DataFrame()

    def _collect_original_data(self, target_size):
        """Collect original training data with 80/20 ratio and target size."""
        try:
            training_path = self.config.use_case_config.training_csv_path

            if not training_path or not os.path.exists(training_path):
                print(f"   ❌ Training data not found")
                return pd.DataFrame()

            original = pd.read_csv(training_path)
            sensor_cols = self.config.use_case_config.sensor_cols
            
            required_cols = sensor_cols + ['Timestamp', 'Status']
            available_cols = [col for col in required_cols if col in original.columns]

            if len([col for col in sensor_cols if col in available_cols]) == 0:
                print(f"   ⚠️  No sensor columns in training data")
                return pd.DataFrame()
                
            # Apply 80/20 sampling with target size
            normal_target = int(target_size * 0.8)
            anomaly_target = int(target_size * 0.2)
            
            normal_data = original[original['Status'] == 'Normal'][available_cols]
            anomaly_data = original[original['Status'] != 'Normal'][available_cols]
            
            # Sample with available limits
            normal_sample = normal_data.sample(n=min(normal_target, len(normal_data)), random_state=42) if len(normal_data) > 0 else pd.DataFrame()
            anomaly_sample = anomaly_data.sample(n=min(anomaly_target, len(anomaly_data)), random_state=42) if len(anomaly_data) > 0 else pd.DataFrame()
            
            original_data = pd.concat([normal_sample, anomaly_sample], ignore_index=True).dropna()
            
            print(f"   ✅ Historical data: {len(normal_sample)} normal + {len(anomaly_sample)} anomaly = {len(original_data)}")

            # Standardize columns
            if 'Status' in original_data.columns:
                original_data = original_data.rename(columns={'Status': 'Label'})
            if 'Timestamp' in original_data.columns:
                original_data['Timestamp'] = original_data['Timestamp'].astype(str)

            return original_data
        
        except Exception as e:
            print(f"   ❌ Error collecting original data: {e}")
            return pd.DataFrame()
    
    def _combine_data_sources(self, sensor_data, replay_data, original_data):
        """Combine sources with proper timestamp sorting and deduplication."""
        print("\n   🔄 Starting data combination process...")
        datasets = []
        source_info = []

        print("   📊 Initial data sizes:")
        print(f"      Sensor data: {len(sensor_data)} samples")
        print(f"      Replay data: {len(replay_data)} samples")
        print(f"      Original data: {len(original_data)} samples")

        # Collect non-empty datasets
        for df, name in zip([sensor_data, replay_data, original_data], ['sensor', 'replay', 'original']):
            if not df.empty:
                sensor_cols = self.config.use_case_config.sensor_cols
                if all(col in df.columns for col in sensor_cols + ['Timestamp', 'Label']):
                    datasets.append(df.copy())
                    source_info.append(f"{name}({len(df)})")
                else:
                    missing = [col for col in sensor_cols + ['Timestamp', 'Label'] if col not in df.columns]
                    print(f"   ⚠️  {name} data missing columns: {missing}")

        if not datasets:
            print(f"   ❌ No valid data sources")
            return None

        print(f"   📊 Combining sources: {', '.join(source_info)}")
        
        # Combine datasets
        combined = pd.concat(datasets, ignore_index=True, sort=False)

        # Handle timestamps
        if 'Timestamp' in combined.columns:
            def parse_timestamp(ts_str):
                try:
                    return pd.to_datetime(ts_str, format='%d/%m/%Y %H:%M:%S')
                except:
                    try:
                        return pd.to_datetime(ts_str, errors='coerce')
                    except:
                        return pd.NaT
            
            combined['Timestamp'] = combined['Timestamp'].apply(parse_timestamp)
            
            # Remove invalid timestamps
            before_invalid = len(combined)
            combined = combined.dropna(subset=['Timestamp'])
            after_invalid = len(combined)
            if before_invalid > after_invalid:
                print(f"   🧹 Removed {before_invalid - after_invalid} invalid timestamps")
            
            # Sort and remove duplicates
            combined = combined.sort_values('Timestamp').reset_index(drop=True)
            before_dedup = len(combined)
            combined = combined.drop_duplicates(subset=['Timestamp'], keep='first')
            after_dedup = len(combined)
            
            if before_dedup > after_dedup:
                print(f"   🧹 Removed {before_dedup - after_dedup} duplicate timestamps")

        final_size = len(combined)

        # Show final distribution
        if 'Label' in combined.columns:
            label_counts = combined['Label'].value_counts()
            print(f"   📊 Final label distribution: {dict(label_counts)}")
            
            anomaly_count = len(combined[combined['Label'] != 'Normal'])
            if anomaly_count == 0:
                print(f"   ⚠️  WARNING: No anomaly samples in final dataset!")

        print(f"   ✅ Final dataset: {final_size} samples, sorted by timestamp")
        return combined
    
    def _update_replay_buffer(self, sensor_data):
        """Update replay buffer with new sensor data."""
        try:
            if len(sensor_data) > 0:
                sample_size = min(100, len(sensor_data))
                sensor_sample = sensor_data.sample(n=sample_size, random_state=42)
                self.replay_buffer.add_samples(
                    sensor_sample, 
                    source="deployment", 
                    label_type="mixed"
                )
                
        except Exception as e:
            print(f"   ⚠️  Failed to update replay buffer: {e}")