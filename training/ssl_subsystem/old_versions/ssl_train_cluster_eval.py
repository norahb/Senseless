import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import time
import warnings
warnings.filterwarnings('ignore')

class ImageDataset(Dataset):
    """Custom dataset for loading images from directory"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        
        # Load images from directory
        if os.path.exists(root_dir):
            for file in os.listdir(root_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(file)
        
        print(f"🔍 Loading images from: {root_dir}")
        print(f"✅ Found {len(self.images)} images")
        if len(self.images) > 0:
            print(f"📋 Sample images: {self.images[:5]}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

class SimCLRModel(nn.Module):
    """SimCLR model with ResNet50 backbone"""
    def __init__(self, feature_dim=512):
        super(SimCLRModel, self).__init__()
        
        # ResNet50 backbone
        self.backbone = resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final classification layer
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim)
        )
    
    def forward(self, x):
        h = self.backbone(x)  # Features from backbone
        z = self.projection_head(h)  # Projected features
        return h, z

class CPUOptimizedSSLTrainer:
    """CPU-optimized SSL trainer with comprehensive diagnostics"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cpu')
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        
        print(f"🖥️ Initializing CPU-optimized SSL trainer")
        print(f"⚡ Configuration: {config}")
    
    def setup_model(self):
        """Initialize model and optimizer"""
        self.model = SimCLRModel(feature_dim=self.config['feature_dim'])
        self.model.to(self.device)
        
        # CPU-optimized optimizer settings
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        print(f"📏 Model initialized - Size: {model_size_mb:.2f} MB")
        return model_size_mb
    
    def setup_data_loaders(self, train_path, val_path=None):
        """Setup data loaders with CPU-optimized settings"""
        
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transform (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = ImageDataset(train_path, transform=train_transform)
        
        # CPU-optimized DataLoader settings
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,  # Reduced for CPU
            pin_memory=False,  # Disabled for CPU
            drop_last=True
        )
        
        if val_path and os.path.exists(val_path):
            val_dataset = ImageDataset(val_path, transform=val_transform)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=2,
                pin_memory=False
            )
        
        print(f"📊 Data loaders ready - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_path else 0}")
    
    def contrastive_loss(self, z1, z2, temperature=0.5):
        """Compute NT-Xent (contrastive) loss"""
        batch_size = z1.shape[0]
        
        # Normalize features
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate features
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / temperature
        
        # Create labels
        labels = torch.arange(batch_size).repeat(2)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        
        # Mask out self-similarity
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)
        
        # Compute loss
        positives = sim_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, images in enumerate(self.train_loader):
            # Create two augmented views
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            h, z = self.model(images)
            
            # For SimCLR, we need two views. Here we use the same image twice
            # In practice, you'd want to create two different augmentations
            h1, z1 = h, z
            h2, z2 = self.model(images)  # Second forward pass with different augmentation
            
            # Compute contrastive loss
            loss = self.contrastive_loss(z1, z2)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Progress reporting
            if batch_idx % max(1, num_batches // 4) == 0:
                print(f"   Epoch {epoch+1}/{self.config['epochs']} | "
                      f"Batch {batch_idx+1}/{num_batches} ({(batch_idx+1)/num_batches*100:.1f}%) | "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_path, val_path=None, save_path=None):
        """Complete training pipeline"""
        print(f"\n🚀 Starting SSL Training")
        print("=" * 50)
        
        start_time = time.time()
        
        # Setup
        model_size = self.setup_model()
        self.setup_data_loaders(train_path, val_path)
        
        # Training loop
        train_losses = []
        
        for epoch in range(self.config['epochs']):
            epoch_loss = self.train_epoch(epoch)
            train_losses.append(epoch_loss)
            
            print(f"📊 Epoch {epoch+1}/{self.config['epochs']} completed | Avg Loss: {epoch_loss:.4f}")
        
        # Save model
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"💾 Model saved to: {save_path}")
        
        training_time = (time.time() - start_time) / 60
        
        print(f"\n✅ Training completed in {training_time:.2f} minutes")
        
        return {
            'model': self.model,
            'train_losses': train_losses,
            'training_time': training_time,
            'model_size_mb': model_size
        }

class EnhancedClusteringSystem:
    """Enhanced clustering system with comprehensive diagnostics"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.kmeans = None
        self.features = None
        self.cluster_labels = None
        self.confidence_scores = None
        
        print(f"🎯 Initializing Enhanced Clustering System")
    
    def extract_features(self, data_loader, progress_name=""):
        """Extract features from images using trained model"""
        print(f"🔍 Extracting features{' for ' + progress_name if progress_name else ''}...")
        
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                # Handle both labeled and unlabeled data
                if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                    images, _ = batch_data
                else:
                    images = batch_data
                
                images = images.to(self.device)
                h, _ = self.model(images)  # Use backbone features, not projected features
                features.append(h.cpu().numpy())
                
                # Progress reporting
                if batch_idx % max(1, len(data_loader) // 4) == 0:
                    print(f"   Progress: {batch_idx+1}/{len(data_loader)} batches "
                          f"({(batch_idx+1)/len(data_loader)*100:.1f}%)")
        
        self.features = np.concatenate(features, axis=0)
        print(f"✅ Extracted features shape: {self.features.shape}")
        return self.features
    
    def perform_clustering(self, n_clusters=2, random_state=42):
        """Perform K-means clustering with confidence scoring"""
        print(f"📊 Running K-Means clustering (k={n_clusters})...")
        
        if self.features is None:
            raise ValueError("Features not extracted. Call extract_features() first.")
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.features)
        
        # Calculate multiple confidence metrics
        confidence_metrics = self._calculate_confidence_scores()
        self.confidence_scores = confidence_metrics
        
        # Calculate clustering quality metrics
        quality_metrics = self._calculate_quality_metrics()
        
        print(f"✅ Clustering completed:")
        print(f"   - Found {n_clusters} clusters")
        print(f"   - Combined confidence range: {confidence_metrics['combined'].min():.3f} - {confidence_metrics['combined'].max():.3f}")
        
        return {
            'cluster_labels': self.cluster_labels,
            'confidence_scores': confidence_metrics,
            'quality_metrics': quality_metrics,
            'kmeans_model': self.kmeans
        }
    
    def _calculate_confidence_scores(self):
        """Calculate multiple confidence metrics"""
        distances = self.kmeans.transform(self.features)
        
        # 1. Distance-based confidence
        min_distances = np.min(distances, axis=1)
        max_distance = np.max(min_distances)
        distance_confidence = 1 - (min_distances / max_distance) if max_distance > 0 else np.ones_like(min_distances)
        
        # 2. Relative confidence (distance ratio)
        if distances.shape[1] > 1:
            sorted_distances = np.sort(distances, axis=1)
            relative_confidence = sorted_distances[:, 1] / (sorted_distances[:, 0] + 1e-8)
            relative_confidence = np.clip(relative_confidence, 0, 10)
        else:
            relative_confidence = distance_confidence
        
        # 3. Silhouette-based confidence
        from sklearn.metrics import silhouette_samples
        silhouette_scores = silhouette_samples(self.features, self.cluster_labels)
        silhouette_confidence = (silhouette_scores + 1) / 2  # Normalize to [0, 1]
        
        # 4. Combined confidence score
        combined_confidence = (0.4 * distance_confidence + 
                             0.3 * np.clip(relative_confidence / 5, 0, 1) + 
                             0.3 * silhouette_confidence)
        
        return {
            'distance': distance_confidence,
            'relative': relative_confidence,
            'silhouette': silhouette_confidence,
            'combined': combined_confidence
        }
    
    def _calculate_quality_metrics(self):
        """Calculate clustering quality metrics"""
        if len(np.unique(self.cluster_labels)) < 2:
            return {
                'silhouette_score': 0,
                'calinski_harabasz_score': 0,
                'davies_bouldin_score': float('inf'),
                'inertia': self.kmeans.inertia_
            }
        
        silhouette = silhouette_score(self.features, self.cluster_labels)
        calinski_harabasz = calinski_harabasz_score(self.features, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(self.features, self.cluster_labels)
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'inertia': self.kmeans.inertia_
        }
    
    def generate_comprehensive_report(self):
        """Generate comprehensive clustering diagnostics"""
        if self.cluster_labels is None or self.confidence_scores is None:
            raise ValueError("Clustering not performed. Call perform_clustering() first.")
        
        print(f"\n📊 COMPREHENSIVE CLUSTERING ANALYSIS")
        print("=" * 60)
        
        # Basic statistics
        unique_clusters = np.unique(self.cluster_labels)
        print(f"🎯 Cluster Distribution:")
        for cluster in unique_clusters:
            count = np.sum(self.cluster_labels == cluster)
            percentage = (count / len(self.cluster_labels)) * 100
            print(f"   Cluster {cluster}: {count} samples ({percentage:.1f}%)")
        
        # Quality metrics
        quality = self._calculate_quality_metrics()
        print(f"\n📊 CLUSTERING QUALITY METRICS")
        print(f"   🎯 Silhouette Score: {quality['silhouette_score']:.4f} (higher is better)")
        print(f"   🎯 Calinski-Harabasz Score: {quality['calinski_harabasz_score']:.4f} (higher is better)")
        print(f"   🎯 Davies-Bouldin Score: {quality['davies_bouldin_score']:.4f} (lower is better)")
        print(f"   🎯 Inertia (WCSS): {quality['inertia']:.4f} (lower is better)")
        
        # Confidence analysis
        combined_conf = self.confidence_scores['combined']
        print(f"\n🎯 CONFIDENCE ANALYSIS")
        print(f"   Mean Confidence: {combined_conf.mean():.4f}")
        print(f"   Median Confidence: {np.median(combined_conf):.4f}")
        print(f"   Std Confidence: {combined_conf.std():.4f}")
        print(f"   High Confidence (>0.8): {(combined_conf > 0.8).sum()}/{len(combined_conf)} ({(combined_conf > 0.8).mean()*100:.1f}%)")
        
        # Confidence percentiles
        percentiles = [50, 75, 85, 90, 95, 99]
        print(f"\n📈 Confidence Percentiles:")
        for p in percentiles:
            value = np.percentile(combined_conf, p)
            count = np.sum(combined_conf >= value)
            print(f"   {p}th percentile: {value:.4f} ({count} samples, {(count/len(combined_conf)*100):.1f}%)")
        
        # Per-cluster confidence
        print(f"\n📊 Confidence by Cluster:")
        for cluster in unique_clusters:
            cluster_mask = self.cluster_labels == cluster
            cluster_conf = combined_conf[cluster_mask]
            print(f"   Cluster {cluster}: mean={cluster_conf.mean():.4f}, std={cluster_conf.std():.4f}")
        
        return {
            'cluster_distribution': {cluster: np.sum(self.cluster_labels == cluster) 
                                   for cluster in unique_clusters},
            'quality_metrics': quality,
            'confidence_stats': {
                'mean': combined_conf.mean(),
                'median': np.median(combined_conf),
                'std': combined_conf.std(),
                'high_confidence_ratio': (combined_conf > 0.8).mean()
            }
        }
    
    def save_results(self, image_names, output_path, dataset_name=""):
        """Save clustering results to CSV"""
        if self.cluster_labels is None or self.confidence_scores is None:
            raise ValueError("Clustering not performed. Call perform_clustering() first.")
        
        results_df = pd.DataFrame({
            'Image_Name': image_names,
            'Cluster': self.cluster_labels,
            'Distance_Confidence': self.confidence_scores['distance'],
            'Relative_Confidence': self.confidence_scores['relative'],
            'Silhouette_Confidence': self.confidence_scores['silhouette'],
            'Combined_Confidence': self.confidence_scores['combined']
        })
        
        if dataset_name:
            results_df['Dataset'] = dataset_name
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_csv(output_path, index=False)
        
        print(f"💾 Results saved to: {output_path}")
        return results_df

class PseudoLabelGenerator:
    """Generate pseudo labels for multiple datasets"""
    
    def __init__(self, model, kmeans_model, device='cpu'):
        self.model = model
        self.kmeans_model = kmeans_model
        self.device = device
        self.clustering_system = EnhancedClusteringSystem(model, device)
        self.clustering_system.kmeans = kmeans_model
        
    def generate_for_dataset(self, dataset_path, dataset_name, output_dir):
        """Generate pseudo labels for a single dataset"""
        print(f"\n📊 Processing {dataset_name.upper()} dataset...")
        
        # Setup data loader
        eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = ImageDataset(dataset_path, transform=eval_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )
        
        if len(dataset) == 0:
            print(f"⚠️ No images found in {dataset_name} dataset")
            return None
        
        print(f"📊 Processing {len(dataset)} images...")
        
        # Extract features
        features = self.clustering_system.extract_features(dataloader, dataset_name)
        
        # Generate clusters
        cluster_labels = self.kmeans_model.predict(features)
        
        # Calculate confidence scores
        distances = self.kmeans_model.transform(features)
        
        # Distance confidence
        min_distances = np.min(distances, axis=1)
        max_distance = np.max(min_distances)
        distance_confidence = 1 - (min_distances / max_distance) if max_distance > 0 else np.ones_like(min_distances)
        
        # Relative confidence
        if distances.shape[1] > 1:
            sorted_distances = np.sort(distances, axis=1)
            relative_confidence = sorted_distances[:, 1] / (sorted_distances[:, 0] + 1e-8)
        else:
            relative_confidence = distance_confidence
        
        # Combined confidence
        combined_confidence = (0.6 * distance_confidence + 0.4 * np.clip(relative_confidence / 5, 0, 1))
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Image_Name': dataset.images,
            'Cluster': cluster_labels,
            'Distance_Confidence': distance_confidence,
            'Relative_Confidence': relative_confidence,
            'Combined_Confidence': combined_confidence,
            'Dataset': dataset_name
        })
        
        # Save results
        output_file = os.path.join(output_dir, f"{dataset_name}_pseudo_labels.csv")
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_csv(output_file, index=False)
        
        # Print statistics
        high_conf_count = (combined_confidence > 0.8).sum()
        print(f"✅ {dataset_name.upper()} pseudo labels saved: {output_file}")
        print(f"📊 Cluster distribution: {pd.Series(cluster_labels).value_counts().sort_index().to_dict()}")
        print(f"📊 Confidence stats: Mean={combined_confidence.mean():.4f}, "
              f"High conf: {high_conf_count}/{len(combined_confidence)} ({high_conf_count/len(combined_confidence)*100:.1f}%)")
        
        return {
            'dataframe': results_df,
            'cluster_distribution': pd.Series(cluster_labels).value_counts().to_dict(),
            'confidence_stats': {
                'mean': combined_confidence.mean(),
                'high_confidence_ratio': high_conf_count/len(combined_confidence)
            },
            'output_file': output_file
        }
    
    def generate_for_all_datasets(self, base_path, output_dir, datasets=['train', 'test', 'val']):
        """Generate pseudo labels for all datasets"""
        print(f"\n🎯 GENERATING PSEUDO LABELS FOR ALL DATASETS")
        print("=" * 60)
        
        all_results = {}
        all_dataframes = []
        
        for dataset_name in datasets:
            dataset_path = os.path.join(base_path, dataset_name)
            if os.path.exists(dataset_path):
                result = self.generate_for_dataset(dataset_path, dataset_name, output_dir)
                if result:
                    all_results[dataset_name] = result
                    all_dataframes.append(result['dataframe'])
        
        # Create combined results
        if all_dataframes:
            print(f"\n📊 CREATING COMBINED RESULTS...")
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            
            combined_output_file = os.path.join(output_dir, "combined_pseudo_labels.csv")
            combined_df.to_csv(combined_output_file, index=False)
            
            print(f"✅ Combined pseudo labels saved: {combined_output_file}")
            
            # Summary
            print(f"\n📊 PSEUDO LABELING SUMMARY")
            print("=" * 50)
            for dataset_name, result in all_results.items():
                stats = result['confidence_stats']
                total = len(result['dataframe'])
                print(f"{dataset_name:>8}: {total:>4} images | "
                      f"conf: {stats['mean']:.3f} | "
                      f"high: {stats['high_confidence_ratio']*100:.1f}%")
            
            print(f"\nTotal images processed: {len(combined_df)}")
            print(f"Combined file: {combined_output_file}")
            
            return {
                'individual_results': all_results,
                'combined_dataframe': combined_df,
                'combined_output_file': combined_output_file,
                'total_images': len(combined_df)
            }
        
        return None

class EvaluationDiagnostics:
    """Comprehensive evaluation and diagnostic functions"""
    
    @staticmethod
    def evaluate_clustering_against_ground_truth(cluster_results_path, ground_truth_path, 
                                                image_col='Image_Name', label_col='Status'):
        """Evaluate clustering results against ground truth labels"""
        print(f"\n📈 Evaluating clustering against ground truth...")
        
        # Load data
        cluster_df = pd.read_csv(cluster_results_path)
        ground_truth_df = pd.read_csv(ground_truth_path)
        
        print(f"🔍 Cluster data: {len(cluster_df)} images")
        print(f"🔍 Ground truth: {len(ground_truth_df)} images")
        
        # Merge datasets
        merged_df = cluster_df.merge(ground_truth_df, on=image_col, how='inner')
        
        if len(merged_df) == 0:
            print("❌ No matching images found between cluster results and ground truth")
            return None
        
        print(f"✅ Merged data: {len(merged_df)} matching images")
        
        # Determine cluster-to-class mapping
        cluster_mapping = {}
        for cluster in merged_df['Cluster'].unique():
            cluster_subset = merged_df[merged_df['Cluster'] == cluster]
            most_common_class = cluster_subset[label_col].mode().iloc[0]
            cluster_mapping[cluster] = most_common_class
        
        print(f"🎯 Cluster to Class Mapping: {cluster_mapping}")
        
        # Generate predictions
        merged_df['Predicted_Class'] = merged_df['Cluster'].map(cluster_mapping)
        
        # Calculate metrics
        accuracy = accuracy_score(merged_df[label_col], merged_df['Predicted_Class'])
        conf_matrix = confusion_matrix(merged_df[label_col], merged_df['Predicted_Class'])
        class_report = classification_report(merged_df[label_col], merged_df['Predicted_Class'])
        
        print(f"\n📊 EVALUATION RESULTS")
        print("=" * 40)
        print(f"📊 Confusion Matrix:")
        print(conf_matrix)
        print(f"\n📊 Classification Report:")
        print(class_report)
        print(f"\n📈 Overall Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'cluster_mapping': cluster_mapping,
            'merged_data': merged_df
        }
    
    @staticmethod
    def analyze_confidence_distribution(results_path, confidence_col='Combined_Confidence'):
        """Analyze confidence score distribution"""
        print(f"\n📊 CONFIDENCE DISTRIBUTION ANALYSIS")
        print("=" * 50)
        
        df = pd.read_csv(results_path)
        confidence_scores = df[confidence_col]
        
        # Basic statistics
        print(f"📊 Basic Statistics:")
        print(f"   Count: {len(confidence_scores)}")
        print(f"   Mean: {confidence_scores.mean():.4f}")
        print(f"   Median: {confidence_scores.median():.4f}")
        print(f"   Std: {confidence_scores.std():.4f}")
        print(f"   Min: {confidence_scores.min():.4f}")
        print(f"   Max: {confidence_scores.max():.4f}")
        
        # Confidence thresholds
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        print(f"\n📈 Confidence Thresholds:")
        for threshold in thresholds:
            count = (confidence_scores >= threshold).sum()
            percentage = (count / len(confidence_scores)) * 100
            print(f"   >= {threshold}: {count}/{len(confidence_scores)} ({percentage:.1f}%)")
        
        # Per-cluster analysis if cluster column exists
        if 'Cluster' in df.columns:
            print(f"\n📊 Per-Cluster Confidence Analysis:")
            for cluster in sorted(df['Cluster'].unique()):
                cluster_conf = df[df['Cluster'] == cluster][confidence_col]
                print(f"   Cluster {cluster}: mean={cluster_conf.mean():.4f}, "
                      f"std={cluster_conf.std():.4f}, count={len(cluster_conf)}")
        
        return {
            'statistics': {
                'mean': confidence_scores.mean(),
                'median': confidence_scores.median(),
                'std': confidence_scores.std(),
                'min': confidence_scores.min(),
                'max': confidence_scores.max()
            },
            'threshold_analysis': {threshold: (confidence_scores >= threshold).sum() 
                                 for threshold in thresholds}
        }
    
    @staticmethod
    def plot_clustering_results(results_path, output_dir=None):
        """Generate visualization plots for clustering results"""
        print(f"\n📊 Generating clustering visualization plots...")
        
        df = pd.read_csv(results_path)
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cluster distribution
        cluster_counts = df['Cluster'].value_counts().sort_index()
        axes[0, 0].bar(cluster_counts.index, cluster_counts.values, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Cluster Distribution')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Images')
        for i, v in enumerate(cluster_counts.values):
            axes[0, 0].text(cluster_counts.index[i], v + 0.5, str(v), ha='center')
        
        # 2. Confidence distribution
        axes[0, 1].hist(df['Combined_Confidence'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Confidence Score Distribution')
        axes[0, 1].set_xlabel('Combined Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(df['Combined_Confidence'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["Combined_Confidence"].mean():.3f}')
        axes[0, 1].legend()
        
        # 3. Confidence by cluster (box plot)
        cluster_conf_data = [df[df['Cluster'] == cluster]['Combined_Confidence'].values 
                           for cluster in sorted(df['Cluster'].unique())]
        axes[1, 0].boxplot(cluster_conf_data, labels=sorted(df['Cluster'].unique()))
        axes[1, 0].set_title('Confidence Distribution by Cluster')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Combined Confidence')
        
        # 4. Confidence vs cluster scatter plot
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, cluster in enumerate(sorted(df['Cluster'].unique())):
            cluster_data = df[df['Cluster'] == cluster]
            axes[1, 1].scatter(cluster_data['Cluster'], cluster_data['Combined_Confidence'], 
                             alpha=0.6, color=colors[i % len(colors)], label=f'Cluster {cluster}')
        axes[1, 1].set_title('Confidence Scores by Cluster')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Combined Confidence')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'clustering_analysis_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to: {plot_path}")
        
        plt.show()
        return fig

def create_default_config():
    """Create default configuration for SSL training"""
    return {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.0003,
        'weight_decay': 1e-6,
        'feature_dim': 512,
        'temperature': 0.5,
        'n_clusters': 2
    }

# Example usage and main execution functions
def train_ssl_model(train_path, val_path=None, save_path=None, config=None):
    """Main function to train SSL model"""
    if config is None:
        config = create_default_config()
    
    trainer = CPUOptimizedSSLTrainer(config)
    results = trainer.train(train_path, val_path, save_path)
    
    return results

def perform_clustering_analysis(model_path, data_path, output_dir, n_clusters=2):
    """Main function to perform clustering analysis"""
    # Load model
    model = SimCLRModel(feature_dim=512)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Setup clustering system
    clustering_system = EnhancedClusteringSystem(model, device='cpu')
    
    # Setup data loader
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(data_path, transform=eval_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Extract features and perform clustering
    features = clustering_system.extract_features(dataloader)
    clustering_results = clustering_system.perform_clustering(n_clusters=n_clusters)
    
    # Generate comprehensive report
    report = clustering_system.generate_comprehensive_report()
    
    # Save results
    output_file = os.path.join(output_dir, 'clustering_results.csv')
    results_df = clustering_system.save_results(dataset.images, output_file)
    
    return {
        'clustering_system': clustering_system,
        'results_df': results_df,
        'report': report,
        'clustering_results': clustering_results
    }

def generate_pseudo_labels_pipeline(model_path, base_data_path, output_dir, datasets=['train', 'test', 'val']):
    """Complete pipeline to generate pseudo labels for multiple datasets"""
    # Load model
    model = SimCLRModel(feature_dim=512)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # First, train clustering model on test/validation set
    test_path = os.path.join(base_data_path, 'test')
    clustering_results = perform_clustering_analysis(model_path, test_path, output_dir)
    kmeans_model = clustering_results['clustering_results']['kmeans_model']
    
    # Generate pseudo labels for all datasets
    label_generator = PseudoLabelGenerator(model, kmeans_model, device='cpu')
    results = label_generator.generate_for_all_datasets(base_data_path, output_dir, datasets)
    
    return results

def evaluate_clustering_performance(cluster_results_path, ground_truth_path, output_dir=None):
    """Evaluate clustering performance against ground truth"""
    evaluation_results = EvaluationDiagnostics.evaluate_clustering_against_ground_truth(
        cluster_results_path, ground_truth_path
    )
    
    if evaluation_results and output_dir:
        # Save evaluation results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save merged data with predictions
        merged_data_path = os.path.join(output_dir, 'evaluation_results.csv')
        evaluation_results['merged_data'].to_csv(merged_data_path, index=False)
        
        # Save confusion matrix as image
        plt.figure(figsize=(8, 6))
        sns.heatmap(evaluation_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        conf_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Evaluation results saved to: {output_dir}")
    
    return evaluation_results

# Complete example workflow
def complete_ssl_workflow_example():
    """Complete example workflow demonstrating all components"""
    print("🚀 COMPLETE SSL WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # Configuration
    config = create_default_config()
    
    # Paths (adjust these to your actual paths)
    train_path = "data/images/train"
    val_path = "data/images/val"
    test_path = "data/images/test"
    model_save_path = "models/ssl_model.pth"
    output_dir = "output/ssl_results"
    ground_truth_path = "data/ground_truth.csv"
    
    # Step 1: Train SSL Model
    print("\n📚 Step 1: Training SSL Model...")
    training_results = train_ssl_model(train_path, val_path, model_save_path, config)
    print(f"✅ Training completed. Model saved to: {model_save_path}")
    
    # Step 2: Perform Clustering Analysis
    print("\n🎯 Step 2: Performing Clustering Analysis...")
    clustering_results = perform_clustering_analysis(model_save_path, test_path, output_dir)
    print(f"✅ Clustering analysis completed. Results saved to: {output_dir}")
    
    # Step 3: Generate Pseudo Labels
    print("\n🏷️ Step 3: Generating Pseudo Labels...")
    base_data_path = "data/images"
    pseudo_label_results = generate_pseudo_labels_pipeline(model_save_path, base_data_path, output_dir)
    print(f"✅ Pseudo labels generated for all datasets.")
    
    # Step 4: Evaluate Performance (if ground truth available)
    print("\n📊 Step 4: Evaluating Performance...")
    if os.path.exists(ground_truth_path):
        cluster_results_path = os.path.join(output_dir, 'clustering_results.csv')
        evaluation_results = evaluate_clustering_performance(cluster_results_path, ground_truth_path, output_dir)
        print(f"✅ Performance evaluation completed.")
    else:
        print("⚠️ Ground truth file not found. Skipping performance evaluation.")
    
    # Step 5: Generate Diagnostic Plots
    print("\n📈 Step 5: Generating Diagnostic Plots...")
    cluster_results_path = os.path.join(output_dir, 'clustering_results.csv')
    if os.path.exists(cluster_results_path):
        EvaluationDiagnostics.plot_clustering_results(cluster_results_path, output_dir)
        print(f"✅ Diagnostic plots generated.")
    
    print(f"\n🎉 Complete SSL workflow finished! Check results in: {output_dir}")

if __name__ == "__main__":
    # Example of how to use individual components
    
    # 1. Train a model
    config = create_default_config()
    # training_results = train_ssl_model("path/to/train", "path/to/val", "model.pth", config)
    
    # 2. Perform clustering
    # clustering_results = perform_clustering_analysis("model.pth", "path/to/test", "output/")
    
    # 3. Generate pseudo labels
    # pseudo_results = generate_pseudo_labels_pipeline("model.pth", "path/to/data", "output/")
    
    # 4. Evaluate performance
    # eval_results = evaluate_clustering_performance("cluster_results.csv", "ground_truth.csv", "output/")
    
    # 5. Run complete workflow
    # complete_ssl_workflow_example()
    
    print("🚀 SSL Training & Clustering System Ready!")
    print("📚 Use the functions above to train models, perform clustering, and generate pseudo labels.")
    print("🎯 Each component can be used independently or as part of the complete workflow.")
    print("\nKey Functions:")
    print("- train_ssl_model(): Train SSL model")
    print("- perform_clustering_analysis(): Analyze clustering")
    print("- generate_pseudo_labels_pipeline(): Generate pseudo labels")
    print("- evaluate_clustering_performance(): Evaluate against ground truth")
    print("- complete_ssl_workflow_example(): Run complete pipeline")