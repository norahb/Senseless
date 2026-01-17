
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
from sklearn.metrics import silhouette_samples


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
        self.encoder.fc = nn.Identity()  
        
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

# KMeans clustering with comprehensive confidence scores
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

# Function to calculate model size in MB
def calculate_model_size(model):
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024**2)
    print(f"📏 Model Size: {model_size:.2f} MB")
    return model_size

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
        "combined_confidence": np.asarray(hybrid_conf).flatten()  
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

    print("✅ Clustering completed")
    print(f"   - Found {probs.shape[1]} clusters")
    print(f"   - Distance confidence range: {dist_conf.min():.3f} - {dist_conf.max():.3f}")
    print(f"   - Relative confidence range: {margin.min():.3f} - {margin.max():.3f}")  # using margin as relative measure
    try:
        sil_scores = silhouette_samples(X, labels)
        sil_range = f"{sil_scores.min():.3f} - {sil_scores.max():.3f}"
    except Exception:
        sil_range = "N/A"
    print(f"   - Silhouette confidence range: {sil_range}")
    print(f"   - Combined confidence range: {hybrid_conf.min():.3f} - {hybrid_conf.max():.3f}")

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

def evaluate_clustering_method(model_components, test_features, test_labels, test_names, method_type, mapping=None):
    """
    Evaluate a clustering method on test data using a provided mapping
    """
    if method_type == "kmeans":
        model = model_components
        test_preds = model.predict(test_features)
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
        entropy = -np.sum(test_probs * np.log(test_probs + 1e-12), axis=1)
        max_entropy = np.log(test_probs.shape[1])
        test_confidence = 1 - (entropy / max_entropy)

    # ✅ Use mapping from training set
    if mapping is None:
        raise ValueError("Cluster→class mapping must be provided from training")
    
    print(f"✅ Using training-derived mapping for evaluation: {mapping}")

    mapped_preds = [mapping.get(c, "Normal") for c in test_preds]
    accuracy = (np.array(test_labels) == np.array(mapped_preds)).mean()

    print(f"{method_type.upper()} Test Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Cluster mapping (from training): {mapping}")
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

    
def run_clustering_comparison(train_features, train_labels, test_features, test_labels,
                              test_names, all_features, all_names, config, gt_df):
    """
    Compare KMeans vs GMM clustering with proper confidence validation
    """
    print("\n" + "="*60)
    print("CLUSTERING METHOD COMPARISON")
    print("="*60)

    results = {}

    # --- KMeans ---
    print("\n1. TESTING KMEANS CLUSTERING")
    print("-" * 40)

    kmeans_labels, kmeans_conf_dict, kmeans_model = kmeans_cluster_with_confidence(
        train_features, n_clusters=2
    )

    kmeans_valid = validate_confidence_scores(
        kmeans_conf_dict['combined_confidence'], "KMeans"
    )

    kmeans_mapping = build_mapping_from_training(kmeans_model, train_features, train_labels, "kmeans")
    kmeans_test_results = evaluate_clustering_method(
        kmeans_model, test_features, test_labels, test_names, "kmeans", mapping=kmeans_mapping
    )

    kmeans_test_results['mapping'] = kmeans_mapping  # override with training mapping

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

    # --- GMM ---
    print("\n2. TESTING ROBUST GMM CLUSTERING")
    print("-" * 40)

    gmm_labels, gmm_conf_dict, gmm_model, pca, scaler = gmm_cluster_with_confidence(
        train_features, n_components=2, pca_components=32
    )

    gmm_valid = validate_confidence_scores(
        gmm_conf_dict['combined_confidence'], "Robust GMM"
    )

    # ✅ Build mapping from training set
    gmm_mapping = build_mapping_from_training(gmm_model, train_features, train_labels, "gmm", pca, scaler)

    # Evaluate on test set using fixed mapping
    gmm_test_results = evaluate_clustering_method(
        (gmm_model, pca, scaler), test_features, test_labels, test_names, "gmm", mapping=gmm_mapping
    )
    gmm_test_results['mapping'] = gmm_mapping  # override with training mapping

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

    # --- Select Best Method ---
    print("\n3. METHOD SELECTION")
    print("-" * 40)
    # For Evaluation
    # if results['kmeans']['confidence_valid'] and results['gmm']['confidence_valid']:
    #     best_method = 'gmm' if results['gmm']['test_accuracy'] > results['kmeans']['test_accuracy'] else 'kmeans'
    #     print(f"Both methods valid → Selected {best_method.upper()}")
    # elif results['kmeans']['confidence_valid']:
    #     best_method = 'kmeans'
    #     print("Selected KMEANS (GMM invalid)")
    # elif results['gmm']['confidence_valid']:
    #     best_method = 'gmm'
    #     print("Selected GMM (KMeans invalid)")
    # else:
    #     best_method = 'kmeans'
    #     print("WARNING: Both invalid, defaulting to KMeans")

    best_method = 'gmm'
    print(f"Selected {best_method.upper()}")

    # --- Final Labels ---
    if best_method == 'kmeans':
        final_labels, final_conf, final_mapping = generate_full_labels_kmeans(
            kmeans_model, all_features, all_names, kmeans_mapping
        )
    else:
        final_labels, final_conf, final_mapping = generate_full_labels_gmm(
            gmm_model, pca, scaler, all_features, all_names, gmm_mapping, variant="hybrid_conf"
        )

    output_dir = os.path.join("output", config.name)
    os.makedirs(output_dir, exist_ok=True)

    # Save raw clusters (true pseudo-labels) ===
    df_clusters = pd.DataFrame({
        "Image_Name": all_names,
        "Cluster": final_labels,
        "Confidence_Score": final_conf
    })
    clusters_file = os.path.join(output_dir, f"{config.name}_ssl_clusters.csv")
    df_clusters.to_csv(clusters_file, index=False)
    print(f"✅ Raw cluster assignments saved: {clusters_file}")

    # === Save mapped labels (evaluation only) ===
    mapped_labels = [final_mapping.get(c, "Unknown") for c in final_labels]
    df_labels = pd.DataFrame({
        "Image_Name": all_names,
        "Cluster": final_labels,
        "Label_ssl": mapped_labels,
        "Confidence_Score": final_conf
    })
    labels_file = os.path.join(output_dir, f"{config.name}_ssl_images_labels.csv")
    df_labels.to_csv(labels_file, index=False)
    print(f"✅ Mapped SSL labels saved: {labels_file}")

    # Save mapping JSON (convert keys to str to avoid np.int64 issues)
    class_map_file = os.path.join(output_dir, f"{config.name}_class_map.json")
    safe_mapping = {str(int(k)): v for k, v in final_mapping.items()}
    with open(class_map_file, "w") as f:
        json.dump(safe_mapping, f, indent=4)
    print(f"✅ Class map saved: {class_map_file}")

    # Simple full dataset evaluation
    gt_labels = [gt_df[gt_df['Image_Name'] == name][config.status_col].iloc[0] 
                for name in all_names if name in gt_df['Image_Name'].values]
    pred_labels = [final_mapping.get(c, "Unknown") for c in final_labels]

    accuracy = sum(p == t for p, t in zip(pred_labels, gt_labels)) / len(gt_labels)
    print(f"\nFull dataset accuracy: {accuracy:.1%}")

    return results, best_method

def analyze_data_distribution(gt_df, status_col):
    """Analyze ground truth distribution"""
    print("Ground Truth Distribution:")
    dist = gt_df[status_col].value_counts()
    print(dist)
    print(f"Normal: {dist.get('Normal', 0)} ({dist.get('Normal', 0)/len(gt_df)*100:.1f}%)")
    print(f"Anomaly: {dist.get('Anomaly', 0)} ({dist.get('Anomaly', 0)/len(gt_df)*100:.1f}%)")
    return dist

def build_mapping_from_training(model, train_features, train_labels, method_type="kmeans", pca=None, scaler=None):
    """Build cluster→class mapping using majority vote on training labels only"""
    if method_type == "kmeans":
        preds = model.predict(train_features)
    else:  # gmm
        reduced = pca.transform(train_features) if pca is not None else train_features
        scaled = scaler.transform(reduced) if scaler is not None else reduced
        preds = model.predict(scaled)

    mapping = {}
    for cluster in np.unique(preds):
        mask = preds == cluster
        if mask.sum() > 0:
            mapping[cluster] = pd.Series(train_labels[mask]).mode()[0]

    print(f"🎯 Training-set cluster→class mapping ({method_type}): {mapping}")
    for cluster, label in mapping.items():
        count = (preds == cluster).sum()
        print(f"   Cluster {cluster}: {count} samples → mapped to {label}")
    return mapping

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

    # === Feature extraction or load from cache ===
    embeddings_path = os.path.join("output", config.name, f"{config.name}_ssl_embeddings.npy")

    if os.path.exists(embeddings_path):
        print(f"💾 Found saved embeddings: {embeddings_path}")
        features = np.load(embeddings_path)
        print(f"Loaded embeddings shape: {features.shape}")
    else:
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

    print("DIAGNOSING DATA DISTRIBUTION:")
    analyze_data_distribution(gt_df, config.status_col)

    # Check if features are actually separable
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features[:1000])  

    import matplotlib.pyplot as plt
    colors = ['red' if label == 'Anomaly' else 'blue' for label in ground_truth_labels[:1000]]
    plt.figure(figsize=(10, 8))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.6)
    plt.title("t-SNE visualization of SimCLR features")
    plt.legend(['Normal', 'Anomaly'])
    plt.savefig(f"output/{config.name}/feature_visualization.png")
    plt.close()
    print("Saved feature visualization - check if classes are separable")
    

    comparison_results, best_method = run_clustering_comparison(
    features_train, labels_train, features_test, labels_test, names_test,
    features, eval_dataset.images, config, gt_df)

    # === RETURN RESULTS ===
    return {
        "model_size_mb": model_size,
        "training_duration_min": training_duration,
        "best_clustering_method": best_method,
        "kmeans_results": comparison_results['kmeans'],
        "gmm_results": comparison_results['gmm'],
        "total_images": len(eval_dataset.images)
    }
