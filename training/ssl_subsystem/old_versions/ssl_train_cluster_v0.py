# ### ✅ ssl_subsystem/ssl_train_cluster.py (Now with clustering evaluation)

import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score, calinski_harabasz_score, davies_bouldin_score
import time
import json
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Define device - FORCE GPU when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"🖥️ Using device: {device}")
if torch.cuda.is_available():
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# SimCLR model (GPU optimized)
class SimCLR(nn.Module):
    def __init__(self, base_model='resnet18', proj_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = models.__dict__[base_model](weights=None)
        self.encoder.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.inplanes, self.encoder.inplanes),
            nn.ReLU(),
            nn.Linear(self.encoder.inplanes, proj_dim)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

# Updated ImageDataset for flat directory structure (no subdirectories)
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, include_labels=False):
        self.root_dir = root_dir
        self.transform = transform
        self.include_labels = include_labels
        self.images = []
        self.image_labels = {}  # For compatibility
        
        # print(f"🔍 Loading images from flat directory: {root_dir}")
        
        # Get all image files directly from the root directory
        all_files = os.listdir(root_dir)
        self.images = [f for f in all_files 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff', '.webp'))]
        
        # Initialize labels as unknown since we don't have subdirectory structure
        for img in self.images:
            self.image_labels[img] = "unknown"
        
        # print(f"✅ Found {len(self.images)} images in {root_dir}")
        
        if len(self.images) == 0:
            print(f"❌ ERROR: No images found in {root_dir}")
            print(f"Directory contents: {all_files[:10]}...")  # Show first 10 files
        
        # Show sample filenames for debugging
        # sample_count = min(5, len(self.images))
        # if sample_count > 0:
        #     print(f"📋 Sample images: {self.images[:sample_count]}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            if self.include_labels:
                return image, self.image_labels[self.images[idx]]
            
            return image
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            # Return a dummy image in case of error
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            
            if self.include_labels:
                return dummy_image, "unknown"
            return dummy_image
    
    def get_image_labels(self):
        """Return dictionary mapping image names to their labels (for compatibility)"""
        return self.image_labels.copy()

# Data augmentation for SimCLR training (GPU optimized)
class SimCLRDataTransform:
    def __init__(self, size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)

# NT-Xent loss (GPU optimized)
def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    
    sim_i_j = torch.diag(sim, N)
    sim_j_i = torch.diag(sim, -N)
    
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2*N, 1)
    mask = torch.eye(N, dtype=torch.bool).to(z.device)
    mask = mask.repeat(2, 2)
    negative_samples = sim[~mask].reshape(2*N, -1)
    
    labels = torch.zeros(2*N).to(z.device).long()
    logits = torch.cat([positive_samples, negative_samples], dim=1)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

# GPU-optimized training function
def train_simclr(model, train_loader, optimizer, epochs):
    model.train()
    start_time = time.time()
    print(f"🚀 Training SimCLR model on GPU for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, images in enumerate(train_loader):
            images = torch.cat(images, dim=0).to(device, non_blocking=True)
            batch_size = images.shape[0] // 2
            
            _, z = model(images)
            loss = nt_xent_loss(z[:batch_size], z[batch_size:])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Progress logging
            if batch_idx % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{epochs}] completed, Average Loss: {avg_loss:.4f}")
    
    end_time = time.time()
    training_duration = (end_time - start_time) / 60
    print(f"✅ GPU Training completed in {training_duration:.2f} minutes")
    return training_duration

# ENHANCED: KMeans clustering with comprehensive confidence scores
def kmeans_cluster_with_confidence(features, n_clusters=2):
    """
    Enhanced clustering with multiple confidence score types
    """
    print(f"🎯 Running K-Means clustering with confidence scoring...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Calculate distances to all cluster centers
    distances = kmeans.transform(features)
    
    # CONFIDENCE SCORE 1: Distance-based confidence (0-1, higher = more confident)
    min_distances = np.min(distances, axis=1)
    max_distance = np.max(min_distances)
    distance_confidence = 1 - (min_distances / max_distance) if max_distance > 0 else np.ones_like(min_distances)
    
    # CONFIDENCE SCORE 2: Relative confidence (distance ratio)
    sorted_distances = np.sort(distances, axis=1)
    if distances.shape[1] > 1:
        # Ratio of distance to closest vs second closest cluster
        # relative_confidence = sorted_distances[:, 1] / (sorted_distances[:, 0] + 1e-8)
        # relative_confidence = np.clip(relative_confidence, 0, 10)  # Cap extreme values

        # Use: 
        margin = sorted_distances[:, 1] - sorted_distances[:, 0]
        relative_confidence = 1 / (1 + np.exp(-margin))  # Sigmoid transform for 0-1 range
    else:
        relative_confidence = distance_confidence
    
    # CONFIDENCE SCORE 3: Silhouette-based confidence
    from sklearn.metrics import silhouette_samples
    try:
        silhouette_scores = silhouette_samples(features, cluster_labels)
        # Normalize to 0-1 range
        silhouette_confidence = (silhouette_scores + 1) / 2
    except:
        silhouette_confidence = distance_confidence
    
    # CONFIDENCE SCORE 4: Combined confidence (weighted average)
    combined_confidence = (
        0.4 * distance_confidence + 
        0.3 * np.clip(relative_confidence / 5, 0, 1) + 
        0.3 * silhouette_confidence
    )
    
    print(f"✅ Clustering completed")
    print(f"   - Found {n_clusters} clusters")
    print(f"   - Distance confidence range: {distance_confidence.min():.3f} - {distance_confidence.max():.3f}")
    print(f"   - Relative confidence range: {relative_confidence.min():.3f} - {relative_confidence.max():.3f}")
    print(f"   - Silhouette confidence range: {silhouette_confidence.min():.3f} - {silhouette_confidence.max():.3f}")
    print(f"   - Combined confidence range: {combined_confidence.min():.3f} - {combined_confidence.max():.3f}")
    
    return cluster_labels, {
        'distance_confidence': distance_confidence,
        'relative_confidence': relative_confidence,
        'silhouette_confidence': silhouette_confidence,
        'combined_confidence': combined_confidence
    }, kmeans

# Function to generate pseudo labels for all datasets (train, test, val)
def generate_pseudo_labels_for_all_datasets(model, kmeans_model, config, device, cluster_mapping=None):
    """
    Generate pseudo labels for train, test, and val datasets
    """
    print(f"\n🎯 GENERATING PSEUDO LABELS FOR ALL DATASETS")
    print("=" * 60)
    
    # Define evaluation transform (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_results = {}
    dataset_stats = {}

    dataset_paths = {
        'train': os.path.join(config.image_ssl_folder_path, "train"),
        'test': os.path.join(config.image_ssl_folder_path, "test"),
    }

    # Check if val dataset exists in SSL folder
    val_path = os.path.join(config.image_ssl_folder_path, "val")
    if os.path.exists(val_path):
        dataset_paths['val'] = val_path

    model.eval()
    
    for dataset_name, dataset_path in dataset_paths.items():
        try:
            # Load dataset FIRST
            dataset = ImageDataset(root_dir=dataset_path, transform=eval_transform)
            
            # NOW print with the correct length
            print(f"\n📊 Processing {dataset_name.upper()} dataset ({len(dataset)} images)...")
            
            dataloader = DataLoader(
                dataset, 
                batch_size=32, 
                shuffle=False, 
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )
            
            if len(dataset) == 0:
                print(f"⚠️ No images found in {dataset_name} dataset, skipping...")
                continue
            
            # Extract features
            print(f"🔍 Extracting features from {len(dataset)} images...")
            features = []
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(dataloader):
                    # Handle both cases: with and without labels
                    if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                        images, labels = batch_data
                    else:
                        images = batch_data
                    
                    images = images.to(device, non_blocking=True)
                    h, _ = model(images)  # Extract features
                    features.append(h.cpu().numpy())
                    
                    # if batch_idx % 10 == 0:
                    #     print(f"   Processed batch {batch_idx+1}/{len(dataloader)}")
                    if batch_idx % max(1, len(dataloader) // 4) == 0:
                        print(f"   Progress: {batch_idx+1}/{len(dataloader)} batches ({(batch_idx+1)/len(dataloader)*100:.1f}%)")
                    
            features = np.concatenate(features, axis=0)
            print(f"✅ Extracted features shape: {features.shape}")
            
            # Generate clusters using the trained k-means model
            cluster_labels = kmeans_model.predict(features)
            
            # Calculate confidence scores
            distances = kmeans_model.transform(features)
            min_distances = np.min(distances, axis=1)
            max_distance = np.max(min_distances)
            distance_confidence = 1 - (min_distances / max_distance) if max_distance > 0 else np.ones_like(min_distances)
            
            # Relative confidence (if multiple clusters)
            if distances.shape[1] > 1:
                sorted_distances = np.sort(distances, axis=1)
                relative_confidence = sorted_distances[:, 1] / (sorted_distances[:, 0] + 1e-8)
                relative_confidence = np.clip(relative_confidence, 0, 10)
            else:
                relative_confidence = distance_confidence
            
            # Combined confidence
            combined_confidence = (0.6 * distance_confidence + 0.4 * np.clip(relative_confidence / 5, 0, 1))
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Image_Name': dataset.images,
                'Cluster': cluster_labels,
                'Distance_Confidence': distance_confidence,
                'Relative_Confidence': relative_confidence,
                'Confidence_Score': combined_confidence,
                'Dataset': dataset_name
            })

            # Calculate stats for summary
            high_conf_count = len(results_df[results_df['Confidence_Score'] > 0.8])
            dataset_stats[dataset_name] = {
                'total': len(results_df),
                'mean_conf': results_df['Confidence_Score'].mean(),
                'high_conf_ratio': (high_conf_count / len(results_df)) * 100
            }
            
            # Save results
            output_dir = os.path.join("output", config.name)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{config.name}_{dataset_name}_pseudo_labels.csv")
            results_df.to_csv(output_file, index=False)
            
            print(f"✅ {dataset_name.upper()} pseudo labels saved: {output_file}")
            print(f"📊 {dataset_name.upper()} cluster distribution:")
            print(results_df['Cluster'].value_counts().sort_index())
            print(f"📊 {dataset_name.upper()} confidence stats:")
            print(f"   Mean confidence: {combined_confidence.mean():.4f}")
            print(f"   High confidence (>0.8): {(combined_confidence > 0.8).sum()}/{len(combined_confidence)} ({(combined_confidence > 0.8).mean()*100:.1f}%)")
            
            all_results[dataset_name] = {
                'dataframe': results_df,
                'cluster_distribution': results_df['Cluster'].value_counts().to_dict(),
                'mean_confidence': combined_confidence.mean(),
                'high_confidence_count': (combined_confidence > 0.8).sum(),
                'total_images': len(results_df),
                'output_file': output_file
            }
            
        except Exception as e:
            print(f"❌ Error processing {dataset_name} dataset: {e}")
            import traceback
            traceback.print_exc()
    
    # Create combined results file
    if all_results:
        print(f"\n📊 CREATING COMBINED RESULTS...")
        
        # Combine all datasets
        combined_df = pd.concat([result['dataframe'] for result in all_results.values()], 
                               ignore_index=True)
        
        # cluster_mapping = {0: 'Anomaly', 1: 'Normal'}
        # combined_df['Predicted_Label'] = combined_df['Cluster'].map(cluster_mapping)

        # ADD CLUSTER MAPPING HERE
        if cluster_mapping is not None:
            print(f"🎯 Applying cluster mapping: {cluster_mapping}")
            combined_df['Predicted_Label'] = combined_df['Cluster'].map(cluster_mapping)
        else:
            # Default mapping (adjust based on your domain knowledge)
            print(f"⚠️ No cluster mapping provided, using default mapping")
            default_mapping = {0: 'Anomaly', 1: 'Normal'}  # Adjust as needed
            combined_df['Predicted_Label'] = combined_df['Cluster'].map(default_mapping)
            print(f"🎯 Using default mapping: {default_mapping}")
        
        # Save combined results with predicted labels
        combined_output_file = os.path.join("output", config.name, f"{config.name}_ssl_images_labels.csv")
        combined_df.to_csv(combined_output_file, index=False)
        
        print(f"✅ Combined pseudo labels saved: {combined_output_file}")

        # # Save combined results with predicted labels
        # combined_output_file = os.path.join("output", config.name, f"{config.name}_ssl_images_labels.csv")
        # combined_df.to_csv(combined_output_file, index=False)

        print(f"✅ Combined pseudo labels saved: {combined_output_file}")
        # print(f"\n📊 OVERALL SUMMARY:")
        # print(f"   Total images processed: {len(combined_df)}")
        # print(f"   Overall cluster distribution:")
        # print(combined_df['Cluster'].value_counts().sort_index())
        # print(f"   Dataset breakdown:")
        # print(combined_df['Dataset'].value_counts())

        print(f"\n📊 PSEUDO LABELING SUMMARY")
        print("=" * 50)
        for dataset_name, stats in dataset_stats.items():
            print(f"{dataset_name:>8}: {stats['total']:>4} images | conf: {stats['mean_conf']:.3f} | high: {stats['high_conf_ratio']:.1f}%")
        print(f"\nTotal images processed: {len(combined_df)}")
        print(f"Combined file: {combined_output_file}")
        
        # Calculate overall statistics
        overall_stats = {
            'total_images': len(combined_df),
            'cluster_distribution': combined_df['Cluster'].value_counts().to_dict(),
            'dataset_breakdown': combined_df['Dataset'].value_counts().to_dict(),
            'mean_confidence': combined_df['Confidence_Score'].mean(),
            'high_confidence_ratio': (combined_df['Confidence_Score'] > 0.8).mean(),
            'combined_output_file': combined_output_file,
            'individual_results': all_results
        }
        
        return overall_stats
    
    else:
        print("❌ No datasets were successfully processed")
        return None

# ENHANCED: Confidence score analysis and threshold optimization
def analyze_confidence_thresholds(df, confidence_col='Confidence_Score'):
    """
    Analyze confidence scores to find optimal thresholds for high-confidence predictions
    """
    print(f"\n🎯 CONFIDENCE SCORE ANALYSIS")
    print("=" * 50)
    
    confidence_scores = df[confidence_col]
    
    # # Basic statistics
    print(f"📊 Confidence Statistics:")
    print(f"   Mean: {confidence_scores.mean():.4f}")
    print(f"   Median: {confidence_scores.median():.4f}")
    print(f"   Std: {confidence_scores.std():.4f}")
    print(f"   Min: {confidence_scores.min():.4f}")
    print(f"   Max: {confidence_scores.max():.4f}")
    
    # Percentile analysis
    percentiles = [50, 75, 85, 90, 95, 99]
    print(f"\n📈 Confidence Percentiles:")
    for p in percentiles:
        threshold = np.percentile(confidence_scores, p)
        count = (confidence_scores >= threshold).sum()
        percentage = count / len(confidence_scores) * 100
        print(f"   {p}th percentile: {threshold:.4f} ({count} images, {percentage:.1f}%)")
    
    # Cluster-wise confidence analysis
    if 'Cluster' in df.columns:
        print(f"\n📊 Confidence by Cluster:")
        for cluster in sorted(df['Cluster'].unique()):
            cluster_conf = df[df['Cluster'] == cluster][confidence_col]
            print(f"   Cluster {cluster}: mean={cluster_conf.mean():.4f}, std={cluster_conf.std():.4f}")
    
    return confidence_scores.describe()

# ENHANCED: Clustering quality metrics
def calculate_clustering_metrics(features, cluster_labels, confidence_scores):
    """
    Calculate comprehensive clustering quality metrics
    """
    # print(f"\n📊 CLUSTERING QUALITY METRICS")
    print("=" * 50)
    
    from sklearn.metrics import (
        silhouette_score, calinski_harabasz_score, davies_bouldin_score
    )
    
    metrics = {}
    
    # Silhouette Score (higher is better, range: -1 to 1)
    silhouette = silhouette_score(features, cluster_labels)
    metrics['silhouette_score'] = silhouette
    # print(f"🎯 Silhouette Score: {silhouette:.4f} (higher is better)")
    
    # Calinski-Harabasz Score (higher is better)
    try:
        calinski = calinski_harabasz_score(features, cluster_labels)
        metrics['calinski_harabasz_score'] = calinski
        # print(f"🎯 Calinski-Harabasz Score: {calinski:.4f} (higher is better)")
    except:
        metrics['calinski_harabasz_score'] = None
    
    # Davies-Bouldin Score (lower is better)
    try:
        davies_bouldin = davies_bouldin_score(features, cluster_labels)
        metrics['davies_bouldin_score'] = davies_bouldin
        # print(f"🎯 Davies-Bouldin Score: {davies_bouldin:.4f} (lower is better)")
    except:
        metrics['davies_bouldin_score'] = None
    
    # Inertia (within-cluster sum of squares)
    from sklearn.cluster import KMeans
    kmeans_temp = KMeans(n_clusters=len(np.unique(cluster_labels)), random_state=42)
    kmeans_temp.fit(features)
    metrics['inertia'] = kmeans_temp.inertia_
    # print(f"🎯 Inertia (WCSS): {kmeans_temp.inertia_:.4f} (lower is better)")
    
    # Confidence-based metrics
    metrics['mean_confidence'] = np.mean(confidence_scores['combined_confidence'])
    metrics['high_confidence_ratio'] = (confidence_scores['combined_confidence'] > 0.8).mean()
    # print(f"🎯 Mean Confidence: {metrics['mean_confidence']:.4f}")
    # print(f"🎯 High Confidence Ratio (>0.8): {metrics['high_confidence_ratio']:.4f}")
    
    return metrics

# Function to calculate model size in MB
def calculate_model_size(model):
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024**2)
    print(f"📏 Model Size: {model_size:.2f} MB")
    return model_size

# ENHANCED: Better merging with debugging
def merge_with_ground_truth_robust(cluster_df, gt_df, config):
    """
    Enhanced merging that handles different naming patterns with extensive debugging
    """
    print(f"🔍 Attempting to merge clustering results with ground truth...")
    print(f"   Cluster data: {len(cluster_df)} images")
    print(f"   Ground truth: {len(gt_df)} images")
    
    # Auto-detect image column in ground truth
    possible_image_cols = ['Image_Name', 'Image Name', 'image_name', 'filename', 'file', 'Image', 'image']
    actual_image_col = None
    
    for col in possible_image_cols:
        if col in gt_df.columns:
            actual_image_col = col
            break
    
    if actual_image_col is None:
        print(f"❌ No image column found in ground truth. Available columns: {list(gt_df.columns)}")
        return pd.DataFrame()
    
    print(f"✅ Using ground truth image column: '{actual_image_col}'")
    
    # # Show sample data for debugging
    # print(f"📊 Sample cluster data (first 3):")
    # for i in range(min(3, len(cluster_df))):
    #     img_name = cluster_df.iloc[i]['Image_Name']
    #     cluster = cluster_df.iloc[i]['Cluster']
    #     if 'Confidence_Score' in cluster_df.columns:
    #         confidence = cluster_df.iloc[i]['Confidence_Score']
    #         print(f"   {i+1}. '{img_name}' → Cluster {cluster} (conf: {confidence:.3f})")
    #     else:
    #         print(f"   {i+1}. '{img_name}' → Cluster {cluster}")
    
    # print(f"📊 Sample ground truth data (first 3):")
    # for i in range(min(3, len(gt_df))):
    #     gt_name = gt_df.iloc[i][actual_image_col]
    #     if config.status_col in gt_df.columns:
    #         status = gt_df.iloc[i][config.status_col]
    #         print(f"   {i+1}. '{gt_name}' → Status: {status}")
    #     else:
    #         print(f"   {i+1}. '{gt_name}'")
    
    # Method 1: Direct merge (exact match)
    merged = pd.merge(cluster_df, gt_df, left_on='Image_Name', right_on=actual_image_col, how='inner')
    
    if len(merged) > 0:
        print(f"✅ Direct merge successful: {len(merged)} matches")
        return merged
    
    print("❌ Direct merge failed, trying basename matching...")
    
    # Method 2: Basename matching (remove directory paths)
    cluster_df_copy = cluster_df.copy()
    cluster_df_copy['Image_Basename'] = cluster_df_copy['Image_Name'].apply(os.path.basename)
    gt_df_copy = gt_df.copy()
    gt_df_copy['GT_Basename'] = gt_df_copy[actual_image_col].apply(os.path.basename)
    
    print(f"📊 Sample basename matching:")
    for i in range(min(3, len(cluster_df_copy))):
        original = cluster_df_copy.iloc[i]['Image_Name']
        basename = cluster_df_copy.iloc[i]['Image_Basename']
        print(f"   Cluster: '{original}' → '{basename}'")
    
    for i in range(min(3, len(gt_df_copy))):
        original = gt_df_copy.iloc[i][actual_image_col]
        basename = gt_df_copy.iloc[i]['GT_Basename']
        print(f"   GT: '{original}' → '{basename}'")
    
    merged = pd.merge(cluster_df_copy, gt_df_copy, left_on='Image_Basename', right_on='GT_Basename', how='inner')
    
    if len(merged) > 0:
        print(f"✅ Basename merge successful: {len(merged)} matches")
        return merged.drop(['Image_Basename', 'GT_Basename'], axis=1)
    
    print("❌ Basename merge failed, trying stem matching...")
    
    # Method 3: Stem matching (no extensions)
    cluster_df_copy['Image_Stem'] = cluster_df_copy['Image_Name'].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    gt_df_copy['GT_Stem'] = gt_df_copy[actual_image_col].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    
    print(f"📊 Sample stem matching:")
    for i in range(min(3, len(cluster_df_copy))):
        original = cluster_df_copy.iloc[i]['Image_Name']
        stem = cluster_df_copy.iloc[i]['Image_Stem']
        print(f"   Cluster: '{original}' → '{stem}'")
    
    merged = pd.merge(cluster_df_copy, gt_df_copy, left_on='Image_Stem', right_on='GT_Stem', how='inner')
    
    if len(merged) > 0:
        print(f"✅ Stem merge successful: {len(merged)} matches")
        return merged.drop(['Image_Basename', 'GT_Basename', 'Image_Stem', 'GT_Stem'], axis=1)
    
    print("❌ All merge attempts failed!")
    print("🔍 Debugging merge failure:")
    print(f"   Unique cluster image patterns (first 5): {cluster_df['Image_Name'].head().tolist()}")
    print(f"   Unique GT image patterns (first 5): {gt_df[actual_image_col].head().tolist()}")
    
    return pd.DataFrame()

# FIXED: Safe mapping function
def map_clusters_to_classes_safe(merged_df, status_col):
    """
    Safe cluster-to-class mapping with error handling
    """
    if len(merged_df) == 0:
        print("⚠️ No data to map clusters to classes")
        return {}
    
    if status_col not in merged_df.columns:
        print(f"❌ Status column '{status_col}' not found. Available: {list(merged_df.columns)}")
        return {}
    
    print(f"📊 Cluster distribution:")
    cluster_dist = merged_df['Cluster'].value_counts().sort_index()
    print(cluster_dist)
    
    print(f"📊 Status distribution:")
    status_dist = merged_df[status_col].value_counts()
    print(status_dist)
    
    # Create cluster-to-class mapping using majority vote
    cluster_to_class_mapping = merged_df.groupby('Cluster')[status_col].agg(
        lambda x: x.value_counts().idxmax()
    )
    
    print("🎯 Cluster to Class Mapping:")
    print(cluster_to_class_mapping)
    
    return cluster_to_class_mapping.to_dict()

# FIXED: Safe evaluation function
def evaluate_clusters_safe(merged_df, cluster_to_class, status_col, method_name="KMeans"):
    """
    Safe cluster evaluation with proper error handling
    """
    print(f"📈 Evaluating clusters for {method_name}...")
    
    if len(merged_df) == 0:
        print("⚠️ No merged data available for evaluation")
        return None
    
    if len(cluster_to_class) == 0:
        print("⚠️ No cluster mapping available for evaluation")
        return None
    
    if status_col not in merged_df.columns:
        print(f"❌ Status column '{status_col}' not found")
        return None
    
    # Get true and predicted labels
    y_true = merged_df[status_col].astype(str)
    y_pred = merged_df['Cluster'].map(cluster_to_class).astype(str)
    
    # Remove NaN values
    valid_indices = y_true.notna() & y_pred.notna()
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]
    
    if len(y_true_filtered) == 0:
        print("⚠️ No valid predictions after filtering")
        return None
    
    print(f"✅ Evaluating {len(y_true_filtered)} valid predictions...")
    
    try:
        # Get unique labels for confusion matrix
        unique_labels = sorted(list(set(y_true_filtered) | set(y_pred_filtered)))
        
        if len(unique_labels) < 2:
            print("⚠️ Less than 2 unique labels found, cannot create meaningful confusion matrix")
            return None
        
        # Compute confusion matrix
        conf_mat = confusion_matrix(y_true_filtered, y_pred_filtered, labels=unique_labels)
        print("📊 Confusion Matrix:")
        print(conf_mat)
        
        # Compute classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_true_filtered, y_pred_filtered, zero_division=0))
        
        # Calculate additional metrics
        accuracy = (y_true_filtered == y_pred_filtered).mean()
        print(f"\n📈 Overall Accuracy: {accuracy:.4f}")
        
        return conf_mat
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None

# ENHANCED: Feature space visualization and analysis
def visualize_feature_space(features, cluster_labels, confidence_scores, output_dir, config_name):
    """
    Create comprehensive visualizations of the feature space
    """
    print(f"\n📊 FEATURE SPACE VISUALIZATION")
    print("=" * 50)
    
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    os.makedirs(output_dir, exist_ok=True)
    
    # PCA visualization
    print("🔍 Computing PCA...")
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # Create visualization plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: PCA by clusters
    scatter1 = axes[0].scatter(features_pca[:, 0], features_pca[:, 1], 
                               c=cluster_labels, cmap='viridis', alpha=0.7, s=30)
    axes[0].set_title(f'PCA by Clusters\n(Explained Variance: {pca.explained_variance_ratio_.sum():.3f})')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot 2: PCA by confidence
    scatter2 = axes[1].scatter(features_pca[:, 0], features_pca[:, 1], 
                               c=confidence_scores['combined_confidence'], 
                               cmap='plasma', alpha=0.7, s=30)
    axes[1].set_title('PCA by Confidence Score')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.colorbar(scatter2, ax=axes[1])
    
    # Plot 3: Confidence distribution
    axes[2].hist(confidence_scores['combined_confidence'], bins=30, alpha=0.7, edgecolor='black')
    axes[2].axvline(confidence_scores['combined_confidence'].mean(), 
                     color='red', linestyle='--', label=f'Mean: {confidence_scores["combined_confidence"].mean():.3f}')
    axes[2].set_title('Confidence Score Distribution')
    axes[2].set_xlabel('Combined Confidence Score')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = os.path.join(output_dir, f"{config_name}_feature_space_analysis.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ Feature space visualization saved to: {viz_path}")
    plt.close()
    
    return {
        'pca_explained_variance': pca.explained_variance_ratio_.sum(),
        'visualization_path': viz_path
    }

# NEW: Extract high confidence samples for analysis
def extract_high_confidence_samples(df, confidence_threshold=0.8, output_dir=None, config_name=None):
    """
    Extract high confidence samples for further analysis
    """
    print(f"\n🎯 HIGH CONFIDENCE SAMPLE EXTRACTION")
    print("=" * 50)
    
    high_conf_mask = df['Confidence_Score'] >= confidence_threshold
    high_conf_samples = df[high_conf_mask].copy()
    
    print(f"📊 High confidence samples (≥{confidence_threshold}):")
    print(f"   Total: {len(high_conf_samples)}/{len(df)} ({len(high_conf_samples)/len(df)*100:.1f}%)")
    print(f"   By cluster: {high_conf_samples['Cluster'].value_counts().to_dict()}")
    
    if output_dir and config_name:
        high_conf_file = os.path.join(output_dir, f"{config_name}_high_confidence_samples.csv")
        high_conf_samples.to_csv(high_conf_file, index=False)
        print(f"✅ High confidence samples saved: {high_conf_file}")
    
    return high_conf_samples

# NEW: Assess cluster quality and stability
def assess_cluster_quality(features, cluster_labels, n_bootstrap=10):
    """
    Assess cluster quality using bootstrap sampling
    """
    print(f"\n🔍 CLUSTER QUALITY ASSESSMENT")
    print("=" * 50)
    
    from sklearn.utils import resample
    from sklearn.metrics import adjusted_rand_score
    
    # Bootstrap stability test
    ari_scores = []
    silhouette_scores = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        boot_indices = resample(range(len(features)), random_state=i)
        boot_features = features[boot_indices]
        boot_labels = cluster_labels[boot_indices]
        
        # Re-cluster bootstrap sample
        kmeans_boot = KMeans(n_clusters=2, random_state=42)
        boot_pred_labels = kmeans_boot.fit_predict(boot_features)
        
        # Calculate stability metrics
        ari = adjusted_rand_score(boot_labels, boot_pred_labels)
        sil = silhouette_score(boot_features, boot_pred_labels)
        
        ari_scores.append(ari)
        silhouette_scores.append(sil)
    
    print(f"📊 Cluster Stability (Bootstrap n={n_bootstrap}):")
    print(f"   ARI mean: {np.mean(ari_scores):.4f} ± {np.std(ari_scores):.4f}")
    print(f"   Silhouette mean: {np.mean(silhouette_scores):.4f} ± {np.std(silhouette_scores):.4f}")
    
    return {
        'ari_mean': np.mean(ari_scores),
        'ari_std': np.std(ari_scores),
        'silhouette_mean': np.mean(silhouette_scores),
        'silhouette_std': np.std(silhouette_scores)
    }

def gmm_cluster_with_confidence(embeddings, n_components=2, pca_components=32, 
                                reg_covar=1e-2, temperature=2.0, smooth_alpha=0.2, random_state=42):

    # === PCA ===
    if pca_components is not None and pca_components < embeddings.shape[1]:
        pca = PCA(n_components=pca_components, random_state=random_state)
        X = pca.fit_transform(embeddings)
    else:
        pca = None
        X = embeddings

    # === Bayesian GMM training ===
    gmm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type="tied",
        reg_covar=reg_covar,
        weight_concentration_prior_type="dirichlet_distribution",
        random_state=random_state
    )
    gmm.fit(X)

    # === Posterior probs ===
    probs = gmm.predict_proba(X)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    labels = np.argmax(probs, axis=1)

    # === Temperature scaling in log-space ===
    _, log_resp = gmm._estimate_log_prob_resp(X)   # (N,K)
    scaled_resp = log_resp / temperature
    probs_T = np.exp(scaled_resp - scaled_resp.max(axis=1, keepdims=True))
    probs_T = probs_T / np.sum(probs_T, axis=1, keepdims=True)

    # === Label smoothing ===
    K = probs.shape[1]
    probs = (1 - smooth_alpha) * probs + smooth_alpha * (1.0 / K)
    probs_T = (1 - smooth_alpha) * probs_T + smooth_alpha * (1.0 / K)

    # === Distance-based confidence ===
    dists = cdist(X, gmm.means_)
    min_d = np.min(dists, axis=1)
    max_d = np.max(dists, axis=1)
    dist_conf = 1 - (min_d / (max_d + 1e-8))

    # === Confidence variants ===
    def conf_variants(P):
        if P.shape[1] == 1:
            max_prob = np.ones(P.shape[0])
            margin = np.zeros(P.shape[0])
            hybrid = max_prob
        else:
            top2 = np.partition(P, -2, axis=1)[:, -2:]
            max_prob = np.max(P, axis=1)
            margin = top2[:, 1] - top2[:, 0]
            hybrid = (margin + max_prob) / 2.0
        return margin, max_prob, hybrid

    margin, max_prob, hybrid = conf_variants(probs)
    margin_T, max_prob_T, hybrid_T = conf_variants(probs_T)

    # === Entropy-based confidence ===
    eps = 1e-12
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    max_ent = np.log(K) if K > 1 else 1.0
    entropy_conf = 1 - (ent / max_ent)

    # === Hybrid confidence ===
    hybrid_conf = 0.5 * (max_prob + dist_conf)

    # === Build conf dict (force flat arrays) ===
    conf_dict = {
        "margin": np.asarray(margin).flatten(),
        "max": np.asarray(max_prob).flatten(),
        "hybrid": np.asarray(hybrid).flatten(),
        "margin_T": np.asarray(margin_T).flatten(),
        "max_T": np.asarray(max_prob_T).flatten(),
        "hybrid_T": np.asarray(hybrid_T).flatten(),
        "entropy": np.asarray(entropy_conf).flatten(),
        "distance": np.asarray(dist_conf).flatten(),
        "hybrid_conf": np.asarray(hybrid_conf).flatten(),
        "combined_confidence": np.asarray(hybrid_conf).flatten()  # always 1D, length = N
    }

    # === Debug print ===
    for i in range(min(5, len(labels))):
        print(f"Sample {i}: margin={conf_dict['margin'][i]:.3f}, "
              f"max={conf_dict['max'][i]:.3f}, "
              f"hybrid={conf_dict['hybrid'][i]:.3f}, "
              f"margin_T={conf_dict['margin_T'][i]:.3f}, "
              f"max_T={conf_dict['max_T'][i]:.3f}, "
              f"hybrid_T={conf_dict['hybrid_T'][i]:.3f}, "
              f"entropy={conf_dict['entropy'][i]:.3f}, "
              f"dist_conf={conf_dict['distance'][i]:.3f}, "
              f"hybrid_conf={conf_dict['hybrid_conf'][i]:.3f}")

    # === Return with identity scaler for compatibility ===
    scaler = FunctionTransformer(lambda x: x, validate=False)
    return labels, conf_dict, gmm, pca, scaler

def validate_confidence_scores(confidence_scores, method_name):
    """
    Validate that confidence scores have meaningful spread
    """
    print(f"\n{method_name} Confidence Score Validation:")
    print("=" * 50)
    
    scores = confidence_scores if isinstance(confidence_scores, np.ndarray) else confidence_scores['combined_confidence']
    
    print(f"Range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
    print(f"Mean: {np.mean(scores):.3f}")
    print(f"Std: {np.std(scores):.3f}")
    print(f"Median: {np.median(scores):.3f}")
    
    # Distribution analysis
    high_conf = (scores > 0.8).mean() * 100
    med_conf = ((scores > 0.4) & (scores <= 0.8)).mean() * 100
    low_conf = (scores <= 0.4).mean() * 100
    
    print(f"Distribution:")
    print(f"  High confidence (>0.8): {high_conf:.1f}%")
    print(f"  Medium confidence (0.4-0.8): {med_conf:.1f}%")
    print(f"  Low confidence (≤0.4): {low_conf:.1f}%")
    
    # Warning flags
    if np.std(scores) < 0.05:
        print("WARNING: Confidence scores are collapsed (std < 0.05)")
        return False
    elif (scores > 0.95).mean() > 0.9:
        print("WARNING: 90%+ scores are > 0.95 (overconfident)")
        return False
    elif (scores < 0.05).mean() > 0.9:
        print("WARNING: 90%+ scores are < 0.05 (underconfident)")  
        return False
    else:
        print("SUCCESS: Confidence scores have good spread")
        return True

def evaluate_clustering_method(model_components, test_features, test_labels, test_names, method_type):
    """
    Evaluate a clustering method on test data
    """
    if method_type == "kmeans":
        model = model_components
        test_preds = model.predict(test_features)
        
        # Calculate confidence for test set
        distances = model.transform(test_features)
        min_distances = np.min(distances, axis=1)
        max_distance = np.max(min_distances)
        test_confidence = 1 - (min_distances / max_distance) if max_distance > 0 else np.ones_like(min_distances)
        
    elif method_type == "gmm":
        model, pca, scaler = model_components
        test_reduced = pca.transform(test_features)
        test_scaled = scaler.transform(test_reduced)
        
        test_preds = model.predict(test_scaled)
        test_probs = model.predict_proba(test_scaled)
        
        # Use entropy confidence
        entropy = -np.sum(test_probs * np.log(test_probs + 1e-12), axis=1)
        max_entropy = np.log(test_probs.shape[1])
        test_confidence = 1 - (entropy / max_entropy)
    
    # Map clusters to ground truth classes
    mapping = {}
    for cluster in np.unique(test_preds):
        mask = test_preds == cluster
        if mask.sum() > 0:
            mapping[cluster] = pd.Series(test_labels[mask]).mode()[0]
    
    mapped_preds = [mapping.get(c, "Normal") for c in test_preds]
    accuracy = (np.array(test_labels) == np.array(mapped_preds)).mean()
    
    print(f"{method_type.upper()} Test Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Cluster mapping: {mapping}")
    print(classification_report(test_labels, mapped_preds, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'mapping': mapping,
        'predictions': test_preds,
        'confidence': test_confidence
    }

def generate_full_labels_kmeans(model, all_features, all_names, mapping):
    """Generate labels for full dataset using KMeans"""
    all_preds = model.predict(all_features)
    distances = model.transform(all_features)
    min_distances = np.min(distances, axis=1)
    max_distance = np.max(min_distances)
    all_confidence = 1 - (min_distances / max_distance) if max_distance > 0 else np.ones_like(min_distances)
    
    return all_preds, all_confidence, mapping

def generate_full_labels_gmm(gmm_model, pca, scaler, all_features, all_names, mapping, variant="hybrid_conf"):
    """Generate full dataset labels + confidence for GMM"""
    # Apply PCA if present
    if pca is not None:
        all_features = pca.transform(all_features)

    # Predict on full dataset
    probs = gmm_model.predict_proba(all_features)
    labels = np.argmax(probs, axis=1)

    # Distance-based confidence
    from scipy.spatial.distance import cdist
    dists = cdist(all_features, gmm_model.means_)
    min_d = np.min(dists, axis=1)
    max_d = np.max(dists, axis=1)
    dist_conf = 1 - (min_d / (max_d + 1e-8))

    # Variants
    def conf_variants(P):
        if P.shape[1] == 1:
            max_prob = np.ones(P.shape[0])
            margin = np.zeros(P.shape[0])
            hybrid = max_prob
        else:
            top2 = np.partition(P, -2, axis=1)[:, -2:]
            max_prob = np.max(P, axis=1)
            margin = top2[:, 1] - top2[:, 0]
            hybrid = (margin + max_prob) / 2.0
        return margin, max_prob, hybrid

    margin, max_prob, hybrid = conf_variants(probs)

    eps = 1e-12
    ent = -np.sum(probs * np.log(probs + eps), axis=1)
    max_ent = np.log(probs.shape[1]) if probs.shape[1] > 1 else 1.0
    entropy_conf = 1 - (ent / max_ent)

    hybrid_conf = 0.5 * (max_prob + dist_conf)

    # Select variant
    variants = {
        "margin": margin,
        "max": max_prob,
        "hybrid": hybrid,
        "entropy": entropy_conf,
        "distance": dist_conf,
        "hybrid_conf": hybrid_conf
    }
    confidence = variants.get(variant, hybrid_conf)  # default hybrid_conf

    return labels, confidence, mapping

def save_final_results(all_names, labels, confidence, mapping, config, method_used):
    """Save final results to CSV with required schema"""
    mapped_labels = [mapping.get(c, "Normal") for c in labels]
    
    df_out = pd.DataFrame({
        "Image_Name": all_names,
        "Cluster": labels,
        "Label_ssl": mapped_labels,
        "Confidence_Score": confidence
    })
    
    output_dir = os.path.join("output", config.name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{config.name}_ssl_images_labels.csv")
    df_out.to_csv(output_file, index=False)
    
    print(f"\nFINAL RESULTS SAVED:")
    print(f"  Method used: {method_used.upper()}")
    print(f"  Output file: {output_file}")
    print(f"  Total images: {len(df_out)}")
    print(f"  Confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"  Confidence std: {confidence.std():.3f}")

def run_clustering_comparison(train_features, train_labels, test_features, test_labels, 
                            test_names, all_features, all_names, config):
    """
    Compare KMeans vs GMM clustering with proper confidence validation
    """
    print("\n" + "="*60)
    print("CLUSTERING METHOD COMPARISON")
    print("="*60)
    
    results = {}
    
    # Method 1: Enhanced KMeans (your existing implementation)
    print("\n1. TESTING KMEANS CLUSTERING")
    print("-" * 40)
    
    kmeans_labels, kmeans_conf_dict, kmeans_model = kmeans_cluster_with_confidence(
        train_features, n_clusters=2
    )
    
    # Validate KMeans confidence
    kmeans_valid = validate_confidence_scores(
        kmeans_conf_dict['combined_confidence'], "KMeans"
    )
    
    # Evaluate KMeans on test set
    kmeans_test_results = evaluate_clustering_method(
        kmeans_model, test_features, test_labels, test_names, "kmeans"
    )
    
    results['kmeans'] = {
        'test_accuracy': kmeans_test_results['accuracy'],
        'confidence_valid': kmeans_valid,
        'confidence_stats': {
            'mean': np.mean(kmeans_conf_dict['combined_confidence']),
            'std': np.std(kmeans_conf_dict['combined_confidence']),
            'range': [np.min(kmeans_conf_dict['combined_confidence']), 
                     np.max(kmeans_conf_dict['combined_confidence'])]
        }
    }
    
    # Method 2: Robust GMM
    print("\n2. TESTING ROBUST GMM CLUSTERING")  
    print("-" * 40)
    
    # gmm_labels, gmm_conf_dict, gmm_model, pca, scaler = gmm_cluster_with_confidence(
    #     train_features, n_components=4, pca_components=32
    # )


    # Train GMM on train set only
    gmm_labels, gmm_conf_dict, gmm_model, pca, scaler = gmm_cluster_with_confidence(
        train_features, n_components=4, pca_components=32
    )

    # Validate GMM confidence
    gmm_valid = validate_confidence_scores(
        gmm_conf_dict['combined_confidence'], "Robust GMM"
    )

    # Evaluate GMM on test set
    gmm_test_results = evaluate_clustering_method(
        (gmm_model, pca, scaler), test_features, test_labels, test_names, "gmm"
    )


    results['gmm'] = {
        'test_accuracy': gmm_test_results['accuracy'],
        'confidence_valid': gmm_valid,
        'confidence_stats': {
            'mean': np.mean(gmm_conf_dict['combined_confidence']),
            'std': np.std(gmm_conf_dict['combined_confidence']),
            'range': [np.min(gmm_conf_dict['combined_confidence']), 
                     np.max(gmm_conf_dict['combined_confidence'])]
        }
    }

    # Select best method based on accuracy AND confidence validity
    print("\n3. METHOD SELECTION")
    print("-" * 40)
    
    if results['kmeans']['confidence_valid'] and results['gmm']['confidence_valid']:
        # Both have valid confidence - choose by accuracy
        best_method = 'gmm' if results['gmm']['test_accuracy'] > results['kmeans']['test_accuracy'] else 'kmeans'
        print(f"Both methods have valid confidence scores")
        print(f"Selected {best_method.upper()} based on accuracy: "
              f"KMeans={results['kmeans']['test_accuracy']:.3f}, "
              f"GMM={results['gmm']['test_accuracy']:.3f}")
    elif results['kmeans']['confidence_valid']:
        best_method = 'kmeans'
        print("Selected KMEANS (GMM confidence collapsed)")
    elif results['gmm']['confidence_valid']:
        best_method = 'gmm'  
        print("Selected GMM (KMeans confidence issues)")
    else:
        best_method = 'kmeans'  # Default fallback
        print("WARNING: Both methods have confidence issues, defaulting to KMeans")
    
    # Generate full dataset labels with best method
    if best_method == 'kmeans':
        final_labels, final_conf, final_mapping = generate_full_labels_kmeans(
            kmeans_model, all_features, all_names, kmeans_test_results['mapping']
        )
    else:
        # final_labels, final_conf, final_mapping = generate_full_labels_gmm(
        #     gmm_model, pca, scaler, all_features, all_names, gmm_test_results['mapping']
        # )
        # Generate full dataset labels (with recomputed confidence)
        final_labels, final_conf, final_mapping = generate_full_labels_gmm(
            gmm_model, pca, scaler, all_features, all_names, gmm_test_results['mapping'],
            variant="hybrid_conf"   # use the same default as combined_confidence
        )

    # Save final results
    save_final_results(all_names, final_labels, final_conf, final_mapping, config, best_method)
    
    return results, best_method

def analyze_data_distribution(gt_df, status_col):
    """Analyze ground truth distribution"""
    print("Ground Truth Distribution:")
    dist = gt_df[status_col].value_counts()
    print(dist)
    print(f"Normal: {dist.get('Normal', 0)} ({dist.get('Normal', 0)/len(gt_df)*100:.1f}%)")
    print(f"Anomaly: {dist.get('Anomaly', 0)} ({dist.get('Anomaly', 0)/len(gt_df)*100:.1f}%)")
    return dist

def run(config):
    # === Setup paths ===
    image_csv = config.image_data_path
    models_dir = os.path.join("models", config.name)
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{config.name}_simclr_model.pth")

    # === Hyperparameters ===
    batch_size = getattr(config, "simclr_batch_size", 256) if torch.cuda.is_available() else 32
    epochs = getattr(config, "simclr_epochs", 100)
    learning_rate = getattr(config, "simclr_lr", 3e-4)

    # === Transforms ===
    train_transform = SimCLRDataTransform(size=256)
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # === Decide paths ===
    training_required = not os.path.exists(model_path)
    if training_required:
        print("🧪 No trained model found. Training SimCLR from scratch...")
        train_dataset_path = os.path.join(config.image_ssl_folder_path, "train")
        eval_dataset_path = os.path.join(config.image_ssl_folder_path, "val")
    else:
        print("✅ Found trained SimCLR model. Using full dataset for inference.")
        train_dataset_path = None
        eval_dataset_path = config.image_folder_path

    # === Load datasets ===
    if training_required:
        train_dataset = ImageDataset(root_dir=train_dataset_path, transform=train_transform, include_labels=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4 if torch.cuda.is_available() else 2,
                                  pin_memory=torch.cuda.is_available(), drop_last=True)
    else:
        train_loader = None

    eval_dataset = ImageDataset(root_dir=eval_dataset_path, transform=eval_transform, include_labels=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4 if torch.cuda.is_available() else 2,
                             pin_memory=torch.cuda.is_available())

    # === Load or train model ===
    model = SimCLR().to(device)
    if training_required:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        training_duration = train_simclr(model, train_loader, optimizer, epochs)
        torch.save(model.state_dict(), model_path)
        print(f"✅ SimCLR trained and saved to {model_path}")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        training_duration = 0

    model_size = calculate_model_size(model)

    # === Feature extraction ===
    print("🔍 Extracting features...")
    model.eval()
    features = []
    with torch.no_grad():
        for batch_data in eval_loader:
            images = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
            images = images.to(device, non_blocking=True)
            h, _ = model(images)
            features.append(h.cpu().numpy())
    features = np.concatenate(features, axis=0)
    print(f"Extracted features shape: {features.shape}")

    # Save embeddings
    embeddings_path = os.path.join("output", config.name, f"{config.name}_ssl_embeddings.npy")
    np.save(embeddings_path, features)
    print(f"💾 Saved embeddings to {embeddings_path}")

    # === Load ground truth ===
    gt_df = pd.read_csv(image_csv)
    gt_df.columns = gt_df.columns.str.strip()
    ground_truth_labels = gt_df[config.status_col].values

    # === Split train/test with filenames ===
    features_train, features_test, labels_train, labels_test, names_train, names_test = train_test_split(
        features, ground_truth_labels, eval_dataset.images,
        test_size=0.3, random_state=42, stratify=ground_truth_labels
    )
    print(f"📊 Split embeddings: Train={len(features_train)}, Test={len(features_test)}")

    # Add this before clustering in your run() function
    print("DIAGNOSING DATA DISTRIBUTION:")
    dist_analysis = analyze_data_distribution(gt_df, config.status_col)
    print(f"Distribution analysis:", dist_analysis)
    # Check if your features are actually separable
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features[:1000])  # Sample for speed

    import matplotlib.pyplot as plt
    colors = ['red' if label == 'Anomaly' else 'blue' for label in ground_truth_labels[:1000]]
    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.6)
    plt.title("t-SNE visualization of SimCLR features")
    plt.legend(['Normal', 'Anomaly'])
    plt.savefig(f"output/{config.name}/feature_visualization.png")
    plt.close()
    print("Saved feature visualization - check if classes are separable")
    
    # === RUN CLUSTERING COMPARISON ===
    comparison_results, best_method = run_clustering_comparison(
        features_train, labels_train, features_test, labels_test, names_test,
        features, eval_dataset.images, config
    )
    
    # === RETURN RESULTS ===
    return {
        "model_size_mb": model_size,
        "training_duration_min": training_duration,
        "best_clustering_method": best_method,
        "kmeans_results": comparison_results['kmeans'],
        "gmm_results": comparison_results['gmm'],
        "total_images": len(eval_dataset.images)
    }

# def run(config):
#     # Configuration
#     image_csv = config.image_data_path
#     models_dir = os.path.join("models", config.name)
#     os.makedirs(models_dir, exist_ok=True)
#     model_path = os.path.join(models_dir, f"{config.name}_simclr_model.pth")

#     # Hyperparameters
#     batch_size = getattr(config, "simclr_batch_size", 256) if torch.cuda.is_available() else 32
#     epochs = getattr(config, "simclr_epochs", 100)
#     learning_rate = getattr(config, "simclr_lr", 3e-4)

#     # Data transforms
#     train_transform = SimCLRDataTransform(size=256)
#     eval_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Decide paths based on model status
#     training_required = not os.path.exists(model_path)

#     if training_required:
#         print("🧪 No trained model found. Using SSL dataset for training and validation.")
#         train_dataset_path = os.path.join(config.image_ssl_folder_path, "train")
#         eval_dataset_path = os.path.join(config.image_ssl_folder_path, "val")
#     else:
#         print("✅ Found trained model. Using full image dataset for inference.")
#         train_dataset_path = None  # not used
#         eval_dataset_path = config.image_folder_path

#     # Load datasets
#     if training_required:
#         train_dataset = ImageDataset(root_dir=train_dataset_path, transform=train_transform, include_labels=False)
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=4 if torch.cuda.is_available() else 2,
#             pin_memory=torch.cuda.is_available(),
#             drop_last=True
#         )
#     else:
#         train_loader = None

#     eval_dataset = ImageDataset(root_dir=eval_dataset_path, transform=eval_transform, include_labels=False)
#     eval_loader = DataLoader(
#         eval_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4 if torch.cuda.is_available() else 2,
#         pin_memory=torch.cuda.is_available()
#     )

#     # Load or train model
#     model = SimCLR().to(device)

#     if not training_required:
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         training_duration = 0
#     else:
#         print("🚀 Training new SimCLR model...")
#         print(f"⚡ Hyperparameters:")
#         print(f"   Device: {device}")
#         print(f"   Batch size: {batch_size}")
#         print(f"   Epochs: {epochs}")
#         print(f"   Learning rate: {learning_rate}")
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         training_duration = train_simclr(model, train_loader, optimizer, epochs)
#         torch.save(model.state_dict(), model_path)
#         print(f"✅ SimCLR model trained and saved to: {model_path}")

#     # Calculate model size
#     model_size = calculate_model_size(model)

#     # Feature extraction
#     print("🔍 Extracting features...")
#     model.eval()
#     features = []

#     with torch.no_grad():
#         for batch_data in eval_loader:
#             if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
#                 images, _ = batch_data
#             else:
#                 images = batch_data
#             images = images.to(device, non_blocking=True)
#             h, _ = model(images)
#             features.append(h.cpu().numpy())

#     features = np.concatenate(features, axis=0)
#     print(f"Extracted features shape: {features.shape}")

#     # # KMeans clustering
#     # # print("📊 Running enhanced KMeans clustering...")
#     # cluster_labels, confidence_scores, kmeans_model = kmeans_cluster_with_confidence(features, n_clusters=2)
#     # clustering_metrics = calculate_clustering_metrics(features, cluster_labels, confidence_scores)
#     # silhouette_avg = clustering_metrics['silhouette_score']

#     # Save results
#     # df = pd.DataFrame({
#     #     'Image_Name': eval_dataset.images,
#     #     'Cluster': cluster_labels,
#     #     'Distance_Confidence': confidence_scores['distance_confidence'],
#     #     'Relative_Confidence': confidence_scores['relative_confidence'],
#     #     'Silhouette_Confidence': confidence_scores['silhouette_confidence'],
#     #     'Confidence_Score': confidence_scores['combined_confidence']
#     # })

#     # === CHOOSE CLUSTERING METHOD ===
#     # You can toggle between enhanced KMeans (with heuristic confidences)
#     # and Gaussian Mixture Models (with probability-based confidences).
#     n_clusters = 2
#     use_gmm = True  # 🔧 toggle between KMeans and GMM

#     if not use_gmm:
#         # KMeans option
#         print("🔵 Using enhanced KMeans clustering...")
#         cluster_labels, confidence_scores, kmeans_model = kmeans_cluster_with_confidence(
#             features, n_clusters=n_clusters
#         )
#         labels = cluster_labels
#         conf_scores = confidence_scores['combined_confidence']

#     else:
#         # GMM option
#         labels, conf_scores, gmm_model, best_k = gmm_cluster_with_confidence(features)
#         print(f"ℹ️ GMM finished with {best_k} clusters")

#     # === SAVE CONSISTENT OUTPUT SCHEMA ===
#     df = pd.DataFrame({
#         "Image_Name": eval_dataset.images,
#         "Cluster": labels,
#         "Label_ssl": labels,
#         "Confidence_Score": conf_scores
#     })

#     # Confidence analysis
#     # confidence_analysis = analyze_confidence_thresholds(df)

#     # Generate pseudo labels for ALL datasets only if training was done
#     all_dataset_results = None
#     if training_required:
#         all_dataset_results = generate_pseudo_labels_for_all_datasets(
#             model=model, 
#             kmeans_model=kmeans_model, 
#             config=config, 
#             device=device
#         )
#     else:
#         print("⚠️ Skipping pseudo-label generation for SSL training datasets (already trained model)")

#     # Ground truth evaluation
#     conf_mat = None
#     accuracy = 0.0
#     cluster_to_class = {}

#     try:
#         print("📈 Evaluating clustering against ground truth...")
#         gt_df = pd.read_csv(image_csv)
#         gt_df.columns = gt_df.columns.str.strip()
#         merged = merge_with_ground_truth_robust(df, gt_df, config)

#         if len(merged) > 0:
#             cluster_to_class = map_clusters_to_classes_safe(merged, config.status_col)
#             if len(cluster_to_class) > 0:
#                 merged['Predicted_Label'] = merged['Cluster'].map(cluster_to_class)
#                 conf_mat = evaluate_clusters_safe(merged, cluster_to_class, config.status_col)
#                 if config.status_col in merged.columns:
#                     y_true = merged[config.status_col]
#                     y_pred = merged['Predicted_Label']
#                     accuracy = (y_true == y_pred).mean()

#     except Exception as e:
#         print(f"⚠️ Ground truth evaluation failed: {e}")
    
#     output_dir = os.path.join("output", config.name)
#     os.makedirs(output_dir, exist_ok=True)

#     if len(cluster_to_class) > 0:
#         df['Predicted_Label'] = df['Cluster'].map(cluster_to_class)

#         test_output_file = os.path.join(output_dir, f"{config.name}_ssl_images_labels.csv")  # ✅ Move here


#         if not config.quiet_mode:
#             print(f"✅ Updated test results with predicted labels saved to: {test_output_file}")

#         # Save class mapping to JSON
#         # class_map_file = os.path.join(output_dir, f"{config.name}_class_map.json")
#         class_map_file = os.path.join(models_dir, f"{config.name}_class_map.json")

#         with open(class_map_file, "w") as f:
#             json.dump(cluster_to_class, f, indent=4)
#         if not config.quiet_mode:
#             print(f"✅ Class map saved to: {class_map_file}")

#     # test_output_file = os.path.join(output_dir, f"{config.name}_ssl_images_labels.csv")
#     df.to_csv(test_output_file, index=False)
#     print(f"✅ Test results saved to: {test_output_file}")

#     # Summary
#     # print(f"\n📊 SUMMARY:")
#     # print(f"   Model size: {model_size:.2f} MB")
#     # print(f"   Training duration: {training_duration:.2f} minutes")
#     # print(f"   Silhouette score: {silhouette_avg:.4f}")
#     # print(f"   Test accuracy: {accuracy:.4f}")
#     # print(f"   Cluster mapping: {cluster_to_class}")
#     if all_dataset_results:
#         print(f"   Total images with pseudo labels: {all_dataset_results['total_images']}")

#     return {
#         # 'silhouette_score': silhouette_avg,
#         # 'clustering_metrics': clustering_metrics,
#         'confusion_matrix': conf_mat.tolist() if conf_mat is not None else None,
#         'cluster_mapping': cluster_to_class,
#         'model_size_mb': model_size,
#         'training_duration_min': training_duration,
#         'test_accuracy': accuracy,
#         'total_test_images': len(df),
#         'all_datasets_results': all_dataset_results
#     }
