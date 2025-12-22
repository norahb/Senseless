# deployment/retraining/replay_buffer.py

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from collections import deque
from sklearn.cluster import KMeans


class ReplayBuffer:
    """
    Simple replay buffer to maintain diverse historical data for stable retraining.
    Prevents catastrophic forgetting by keeping representative samples from different periods.
    """
    
    def __init__(self, config, buffer_size=1500):
        self.config = config
        self.buffer_size = buffer_size
        self.buffer_path = os.path.join("logs", config.use_case, "replay_buffer.pkl")
        
        # Initialize buffer storage
        self.samples = []  # List of {data, timestamp, source, label_type}
        
        # Diversity settings
        self.max_normal_ratio = 0.6  # Max 70% normal samples
        self.max_anomaly_ratio = 0.3  # Max 20% sensor anomalies  
        self.max_vision_ratio = 0.1   # Max 10% vision-labeled
        
        # Load existing buffer
        self._load_buffer()
        
    def add_samples(self, data, source="sensor", label_type="normal"):
        """
        Add new samples to replay buffer with diversity management.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            New sensor data samples
        source : str
            Data source ("sensor", "vision_labeled", "original")
        label_type : str
            Type of samples ("normal", "anomaly", "vision_labeled")
        """
        timestamp = datetime.now()
        
        # Convert each row to buffer entry
        new_entries = []
        for _, row in data.iterrows():
            entry = {
                'data': row.to_dict(),
                'timestamp': timestamp,
                'source': source,
                'label_type': label_type,
                'features': row.values  # For diversity calculation
            }
            new_entries.append(entry)
        
        # Add to buffer with diversity management
        for entry in new_entries:
            self._add_single_sample(entry)
        
        # Save updated buffer
        self._save_buffer()
        
        # print(f"📦 Replay buffer: Added {len(new_entries)} {label_type} samples from {source}")
        # print(f"    Current size: {len(self.samples)}/{self.buffer_size}")
        self._print_buffer_stats()
    
    def _add_single_sample(self, entry):
        """Add single sample with diversity constraints"""
        
        # If buffer not full, just add
        if len(self.samples) < self.buffer_size:
            self.samples.append(entry)
            return
        
        # Buffer is full - need to manage diversity
        current_stats = self._get_buffer_stats()
        
        # Check if we can add this type of sample
        if entry['label_type'] == 'normal' and current_stats['normal_ratio'] >= self.max_normal_ratio:
            # Too many normal samples - replace oldest normal sample
            self._replace_oldest_by_type('normal', entry)
        elif entry['label_type'] in ['anomaly', 'sensor_anomaly'] and current_stats['anomaly_ratio'] >= self.max_anomaly_ratio:
            # Too many anomaly samples - replace oldest anomaly
            self._replace_oldest_by_type(['anomaly', 'sensor_anomaly'], entry)
        elif entry['label_type'] == 'vision_labeled' and current_stats['vision_ratio'] >= self.max_vision_ratio:
            # Too many vision samples - replace oldest vision
            self._replace_oldest_by_type('vision_labeled', entry)
        else:
            # We can add this type - replace least diverse sample
            self._replace_least_diverse(entry)
    
    def _replace_oldest_by_type(self, target_types, new_entry):
        """Replace oldest sample of specified type(s)"""
        if isinstance(target_types, str):
            target_types = [target_types]
        
        # Find oldest sample of target type
        oldest_idx = -1
        oldest_time = datetime.now()
        
        for i, sample in enumerate(self.samples):
            if sample['label_type'] in target_types:
                if sample['timestamp'] < oldest_time:
                    oldest_time = sample['timestamp']
                    oldest_idx = i
        
        if oldest_idx >= 0:
            self.samples[oldest_idx] = new_entry
    
    def _replace_least_diverse(self, new_entry):
        """Replace the least diverse sample (closest to existing samples)"""
        if len(self.samples) == 0:
            self.samples.append(new_entry)
            return
        
        # Calculate diversity scores for existing samples
        new_features = np.array(new_entry['features']).reshape(1, -1)
        
        min_diversity_idx = 0
        min_diversity_score = float('inf')
        
        for i, sample in enumerate(self.samples):
            # Calculate similarity to new sample
            sample_features = np.array(sample['features']).reshape(1, -1)
            
            try:
                # Use Euclidean distance as diversity measure
                distance = np.linalg.norm(new_features - sample_features)
                
                # Lower distance = less diverse = candidate for replacement
                if distance < min_diversity_score:
                    min_diversity_score = distance
                    min_diversity_idx = i
            except:
                continue
        
        # Replace least diverse sample
        self.samples[min_diversity_idx] = new_entry
    
    def get_replay_samples(self, max_samples=None):
        """
        Get samples from replay buffer for retraining.
        
        Parameters:
        -----------
        max_samples : int, optional
            Maximum number of samples to return
            
        Returns:
        --------
        pandas.DataFrame : Replay buffer samples
        """
        if len(self.samples) == 0:
            return pd.DataFrame()
        
        # Determine sample size
        if max_samples is None:
            max_samples = len(self.samples)
        else:
            max_samples = min(max_samples, len(self.samples))
        
        # Stratified sampling to maintain diversity
        sampled_entries = self._stratified_sample(max_samples)
        
        # Convert to DataFrame
        replay_data = []
        for entry in sampled_entries:
            replay_data.append(entry['data'])
        
        replay_df = pd.DataFrame(replay_data)
        
        # print(f"📦 Retrieved {len(replay_df)} diverse samples from replay buffer")
        return replay_df
    
    def _stratified_sample(self, n_samples):
        """Sample maintaining label type proportions"""
        if n_samples >= len(self.samples):
            return self.samples.copy()
        
        # Group by label type
        type_groups = {}
        for sample in self.samples:
            label_type = sample['label_type']
            if label_type not in type_groups:
                type_groups[label_type] = []
            type_groups[label_type].append(sample)
        
        # Calculate proportional samples for each type
        sampled = []
        remaining_samples = n_samples
        
        for label_type, group in type_groups.items():
            if remaining_samples <= 0:
                break
            
            # Calculate proportion
            proportion = len(group) / len(self.samples)
            type_sample_count = max(1, int(proportion * n_samples))
            type_sample_count = min(type_sample_count, remaining_samples, len(group))
            
            # Random sample from this type
            indices = np.random.choice(len(group), type_sample_count, replace=False)
            for idx in indices:
                sampled.append(group[idx])
            
            remaining_samples -= type_sample_count
        
        return sampled
    
    def _get_buffer_stats(self):
        """Get current buffer composition statistics"""
        if len(self.samples) == 0:
            return {'normal_ratio': 0, 'anomaly_ratio': 0, 'vision_ratio': 0}
        
        type_counts = {}
        for sample in self.samples:
            label_type = sample['label_type']
            type_counts[label_type] = type_counts.get(label_type, 0) + 1
        
        total = len(self.samples)
        return {
            'normal_ratio': type_counts.get('normal', 0) / total,
            'anomaly_ratio': (type_counts.get('anomaly', 0) + type_counts.get('sensor_anomaly', 0)) / total,
            'vision_ratio': type_counts.get('vision_labeled', 0) / total,
            'type_counts': type_counts,
            'total_samples': total
        }
    
    def _print_buffer_stats(self):
        """Print current buffer statistics"""
        stats = self._get_buffer_stats()
        # print(f"    Composition: Normal={stats['normal_ratio']:.1%}, "
        #       f"Anomaly={stats['anomaly_ratio']:.1%}, Vision={stats['vision_ratio']:.1%}")
    
    def _save_buffer(self):
        """Save replay buffer to disk"""
        try:
            os.makedirs(os.path.dirname(self.buffer_path), exist_ok=True)
            with open(self.buffer_path, 'wb') as f:
                pickle.dump(self.samples, f)
        except Exception as e:
            print(f"⚠️  Failed to save replay buffer: {e}")
    
    def _load_buffer(self):
        """Load replay buffer from disk"""
        try:
            if os.path.exists(self.buffer_path):
                with open(self.buffer_path, 'rb') as f:
                    self.samples = pickle.load(f)
                # print(f"📦 Loaded replay buffer: {len(self.samples)} samples")
                self._print_buffer_stats()
            else:
                print(f"📦 Initialized empty replay buffer")
        except Exception as e:
            print(f"⚠️  Failed to load replay buffer: {e}")
            self.samples = []
    
    def clear_buffer(self):
        """Clear the replay buffer"""
        self.samples = []
        if os.path.exists(self.buffer_path):
            os.remove(self.buffer_path)
        print("📦 Replay buffer cleared")
    
    def get_buffer_info(self):
        """Get buffer information for monitoring"""
        stats = self._get_buffer_stats()
        
        # Calculate temporal coverage
        if len(self.samples) > 0:
            timestamps = [sample['timestamp'] for sample in self.samples]
            oldest = min(timestamps)
            newest = max(timestamps)
            temporal_span = (newest - oldest).days
        else:
            temporal_span = 0
        
        return {
            'size': len(self.samples),
            'max_size': self.buffer_size,
            'utilization': len(self.samples) / self.buffer_size,
            'temporal_span_days': temporal_span,
            'composition': stats['type_counts'],
            'diversity_ratios': {
                'normal': stats['normal_ratio'],
                'anomaly': stats['anomaly_ratio'], 
                'vision': stats['vision_ratio']
            }
        }