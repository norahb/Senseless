import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os
from PIL import Image
from scipy.spatial.distance import cdist
import joblib
from tqdm import tqdm

class ImageDataset(Dataset):
    """Custom dataset for loading images"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, image_path
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                return torch.zeros(3, 224, 224), image_path
            else:
                return Image.new('RGB', (224, 224)), image_path

class SimCLRModel(nn.Module):
    """SimCLR model for feature extraction"""
    def __init__(self, base_model='mobilenet_v2', projection_dim=128):
        super(SimCLRModel, self).__init__()
        
        # Base encoder
        if base_model == 'mobilenet_v2':
            self.encoder = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
            self.encoder.classifier = nn.Identity()
            encoder_dim = 1280
        elif base_model == 'resnet50':
            self.encoder = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
            self.encoder.fc = nn.Identity()
            encoder_dim = 2048
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z

def load_trained_model(model_path, base_model='mobilenet_v2', device='cpu'):
    """Load trained SSL model"""
    print(f"🔄 Loading trained model from: {model_path}")
    
    model = SimCLRModel(base_model=base_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model

def extract_features_from_model(model, data_loader, device):
    """Extract features using trained model"""
    print("🔍 Extracting features from images...")
    
    model.eval()
    features = []
    image_paths = []
    
    with torch.no_grad():
        for images, paths in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            h, _ = model(images)  # Use encoder features
            features.append(h.cpu().numpy())
            image_paths.extend(paths)
    
    features = np.concatenate(features, axis=0)
    print(f"✅ Extracted features shape: {features.shape}")
    
    return features, image_paths

def calculate_confidence_scores(features, cluster_labels, kmeans_model):
    """Calculate confidence scores for cluster assignments"""
    print("📊 Calculating confidence scores...")
    
    # Get cluster centers
    cluster_centers = kmeans_model.cluster_centers_
    
    # Calculate distances to all cluster centers
    distances_to_centers = cdist(features, cluster_centers, metric='euclidean')
    
    # Calculate confidence scores using multiple methods
    confidence_scores = {}
    
    # Method 1: Distance-based confidence (closer to center = higher confidence)
    assigned_distances = []
    for i, label in enumerate(cluster_labels):
        assigned_distances.append(distances_to_centers[i, label])
    assigned_distances = np.array(assigned_distances)
    
    # Normalize distances to 0-1 range (1 = high confidence, 0 = low confidence)
    max_distance = np.max(assigned_distances)
    min_distance = np.min(assigned_distances)
    distance_confidence = 1 - (assigned_distances - min_distance) / (max_distance - min_distance + 1e-8)
    
    # Method 2: Relative distance confidence (how much closer to assigned vs other center)
    relative_confidence = []
    for i, label in enumerate(cluster_labels):
        assigned_dist = distances_to_centers[i, label]
        other_dists = [distances_to_centers[i, j] for j in range(len(cluster_centers)) if j != label]
        if other_dists:
            min_other_dist = min(other_dists)
            # Higher confidence when assigned distance is much smaller than others
            rel_conf = min_other_dist / (assigned_dist + 1e-8)
            relative_confidence.append(min(rel_conf, 10))  # Cap at 10 for normalization
        else:
            relative_confidence.append(1.0)
    
    relative_confidence = np.array(relative_confidence)
    # Normalize to 0-1
    max_rel = np.max(relative_confidence)
    min_rel = np.min(relative_confidence)
    relative_confidence = (relative_confidence - min_rel) / (max_rel - min_rel + 1e-8)
    
    # Method 3: Silhouette-like confidence
    silhouette_confidence = []
    for i, label in enumerate(cluster_labels):
        # Distance to own cluster center
        own_dist = distances_to_centers[i, label]
        
        # Distance to nearest other cluster center
        other_dists = [distances_to_centers[i, j] for j in range(len(cluster_centers)) if j != label]
        if other_dists:
            nearest_other_dist = min(other_dists)
            # Silhouette-like score: (b - a) / max(a, b)
            silh_score = (nearest_other_dist - own_dist) / max(nearest_other_dist, own_dist)
            # Convert to 0-1 range
            silh_conf = (silh_score + 1) / 2
        else:
            silh_conf = 1.0
        silhouette_confidence.append(silh_conf)
    
    silhouette_confidence = np.array(silhouette_confidence)
    
    # Method 4: Combined confidence score
    combined_confidence = (distance_confidence + relative_confidence + silhouette_confidence) / 3
    
    confidence_scores = {
        'distance_confidence': distance_confidence,
        'relative_confidence': relative_confidence, 
        'silhouette_confidence': silhouette_confidence,
        'combined_confidence': combined_confidence,
        'distance_to_center': assigned_distances
    }
    
    print("✅ Confidence scores calculated")
    return confidence_scores

def assign_class_labels(cluster_labels, cluster_to_class_mapping):
    """Map cluster labels to class labels"""
    class_labels = []
    for cluster in cluster_labels:
        if cluster in cluster_to_class_mapping:
            class_labels.append(cluster_to_class_mapping[cluster])
        else:
            class_labels.append('Unknown')
    
    return np.array(class_labels)

def create_confidence_categories(confidence_scores, thresholds=None):
    """Create categorical confidence levels"""
    if thresholds is None:
        thresholds = {
            'high': 0.7,
            'medium': 0.4,
            'low': 0.0
        }
    
    categories = []
    for score in confidence_scores:
        if score >= thresholds['high']:
            categories.append('High')
        elif score >= thresholds['medium']:
            categories.append('Medium')
        else:
            categories.append('Low')
    
    return np.array(categories)

def generate_labels_with_confidence(
    model_path,
    images_directory,
    cluster_to_class_mapping,
    output_file,
    base_model='mobilenet_v2',
    device='auto',
    batch_size=32,
    image_size=224,
    confidence_threshold=0.5,
    save_kmeans_model=True
):
    """
    Generate labels for full dataset with confidence scores
    
    Args:
        model_path: Path to trained SSL model
        images_directory: Directory containing images to label
        cluster_to_class_mapping: Dict mapping cluster indices to class names
                                  e.g., {0: 'Normal', 1: 'Anomaly'}
        output_file: Path to save results CSV
        base_model: Base model architecture used
        device: Device to use ('auto', 'cuda', 'cpu')
        batch_size: Batch size for inference
        image_size: Image size for preprocessing
        confidence_threshold: Threshold for high-confidence predictions
        save_kmeans_model: Whether to save the KMeans model
    """
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"🖥️ Using device: {device}")
    print(f"🔄 Processing images in: {images_directory}")
    
    # Load trained model
    model = load_trained_model(model_path, base_model, device)
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get all image paths
    print("📂 Scanning for images...")
    image_paths = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    for root, dirs, files in os.walk(images_directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_paths.append(os.path.join(root, file))
    
    print(f"✅ Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("❌ No images found!")
        return
    
    # Create dataset and loader
    dataset = ImageDataset(image_paths, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Extract features
    features, processed_paths = extract_features_from_model(model, data_loader, device)
    
    # Perform clustering
    print("📊 Performing KMeans clustering...")
    n_clusters = len(cluster_to_class_mapping)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    print(f"✅ Clustering completed. Found {len(np.unique(cluster_labels))} clusters")
    
    # Calculate confidence scores
    confidence_scores = calculate_confidence_scores(features, cluster_labels, kmeans)
    
    # Assign class labels
    class_labels = assign_class_labels(cluster_labels, cluster_to_class_mapping)
    
    # Create confidence categories
    confidence_categories = create_confidence_categories(
        confidence_scores['combined_confidence'],
        thresholds={'high': 0.7, 'medium': 0.4, 'low': 0.0}
    )
    
    # Create results dataframe
    print("📊 Creating results dataframe...")
    results_df = pd.DataFrame({
        'Image_Path': processed_paths,
        'Image_Name': [os.path.basename(path) for path in processed_paths],
        'Directory': [os.path.dirname(path) for path in processed_paths],
        'Cluster_ID': cluster_labels,
        'Predicted_Class': class_labels,
        'Distance_Confidence': confidence_scores['distance_confidence'],
        'Relative_Confidence': confidence_scores['relative_confidence'],
        'Silhouette_Confidence': confidence_scores['silhouette_confidence'],
        'Combined_Confidence': confidence_scores['combined_confidence'],
        'Confidence_Category': confidence_categories,
        'Distance_to_Center': confidence_scores['distance_to_center'],
        'High_Confidence': confidence_scores['combined_confidence'] >= confidence_threshold
    })
    
    # Add cluster statistics
    cluster_stats = []
    for i, row in results_df.iterrows():
        cluster_id = row['Cluster_ID']
        cluster_mask = cluster_labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        cluster_stats.append(cluster_size)
    
    results_df['Cluster_Size'] = cluster_stats
    
    # Sort by confidence (highest first)
    results_df = results_df.sort_values('Combined_Confidence', ascending=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📊 LABELING SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(results_df)}")
    print(f"Cluster distribution:")
    for cluster_id, class_name in cluster_to_class_mapping.items():
        count = np.sum(cluster_labels == cluster_id)
        percentage = (count / len(cluster_labels)) * 100
        print(f"  Cluster {cluster_id} ({class_name}): {count} images ({percentage:.1f}%)")
    
    print(f"\nConfidence distribution:")
    conf_dist = results_df['Confidence_Category'].value_counts()
    for category in ['High', 'Medium', 'Low']:
        if category in conf_dist:
            count = conf_dist[category]
            percentage = (count / len(results_df)) * 100
            print(f"  {category} confidence: {count} images ({percentage:.1f}%)")
    
    high_conf_count = np.sum(results_df['High_Confidence'])
    print(f"\nHigh confidence predictions (≥{confidence_threshold}): {high_conf_count} ({high_conf_count/len(results_df)*100:.1f}%)")
    
    print(f"\nConfidence score statistics:")
    print(f"  Mean: {results_df['Combined_Confidence'].mean():.3f}")
    print(f"  Median: {results_df['Combined_Confidence'].median():.3f}")
    print(f"  Std: {results_df['Combined_Confidence'].std():.3f}")
    print(f"  Min: {results_df['Combined_Confidence'].min():.3f}")
    print(f"  Max: {results_df['Combined_Confidence'].max():.3f}")
    
    # Save results
    print(f"\n💾 Saving results to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    
    # Save KMeans model if requested
    if save_kmeans_model:
        kmeans_path = output_file.replace('.csv', '_kmeans_model.pkl')
        joblib.dump(kmeans, kmeans_path)
        print(f"💾 KMeans model saved to: {kmeans_path}")
    
    # Save cluster mapping
    mapping_path = output_file.replace('.csv', '_cluster_mapping.txt')
    with open(mapping_path, 'w') as f:
        f.write("Cluster to Class Mapping:\n")
        for cluster_id, class_name in cluster_to_class_mapping.items():
            f.write(f"Cluster {cluster_id}: {class_name}\n")
    print(f"💾 Cluster mapping saved to: {mapping_path}")
    
    # Create summary report
    summary_path = output_file.replace('.csv', '_summary_report.txt')
    with open(summary_path, 'w') as f:
        f.write("ANOMALY DETECTION LABELING REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model used: {model_path}\n")
        f.write(f"Images directory: {images_directory}\n")
        f.write(f"Total images processed: {len(results_df)}\n")
        f.write(f"Number of clusters: {n_clusters}\n")
        f.write(f"Confidence threshold: {confidence_threshold}\n\n")
        
        f.write("CLUSTER DISTRIBUTION:\n")
        for cluster_id, class_name in cluster_to_class_mapping.items():
            count = np.sum(cluster_labels == cluster_id)
            percentage = (count / len(cluster_labels)) * 100
            f.write(f"  Cluster {cluster_id} ({class_name}): {count} images ({percentage:.1f}%)\n")
        
        f.write(f"\nCONFIDENCE DISTRIBUTION:\n")
        for category in ['High', 'Medium', 'Low']:
            if category in conf_dist:
                count = conf_dist[category]
                percentage = (count / len(results_df)) * 100
                f.write(f"  {category} confidence: {count} images ({percentage:.1f}%)\n")
        
        f.write(f"\nHIGH CONFIDENCE PREDICTIONS:\n")
        f.write(f"  Count: {high_conf_count} ({high_conf_count/len(results_df)*100:.1f}%)\n")
        f.write(f"  These predictions are most reliable for decision making.\n")
        
        f.write(f"\nCONFIDENCE STATISTICS:\n")
        f.write(f"  Mean: {results_df['Combined_Confidence'].mean():.3f}\n")
        f.write(f"  Median: {results_df['Combined_Confidence'].median():.3f}\n")
        f.write(f"  Standard deviation: {results_df['Combined_Confidence'].std():.3f}\n")
        
        # Top 10 most confident predictions
        f.write(f"\nTOP 10 MOST CONFIDENT PREDICTIONS:\n")
        top_10 = results_df.head(10)[['Image_Name', 'Predicted_Class', 'Combined_Confidence']]
        for _, row in top_10.iterrows():
            f.write(f"  {row['Image_Name']}: {row['Predicted_Class']} (confidence: {row['Combined_Confidence']:.3f})\n")
        
        # Bottom 10 least confident predictions
        f.write(f"\nBOTTOM 10 LEAST CONFIDENT PREDICTIONS:\n")
        bottom_10 = results_df.tail(10)[['Image_Name', 'Predicted_Class', 'Combined_Confidence']]
        for _, row in bottom_10.iterrows():
            f.write(f"  {row['Image_Name']}: {row['Predicted_Class']} (confidence: {row['Combined_Confidence']:.3f})\n")
    
    print(f"📊 Summary report saved to: {summary_path}")
    
    print(f"\n🎉 Label generation completed successfully!")
    print(f"📁 Results saved to: {output_file}")
    
    return results_df, kmeans

def batch_process_directories(
    model_path,
    directories_config,
    base_output_dir,
    base_model='mobilenet_v2',
    device='auto',
    batch_size=32,
    confidence_threshold=0.5
):
    """
    Process multiple directories with different cluster mappings
    
    Args:
        model_path: Path to trained SSL model
        directories_config: List of dicts with config for each directory
                           e.g., [{'dir': 'path/to/images', 
                                  'mapping': {0: 'Normal', 1: 'Anomaly'},
                                  'name': 'appliance'}]
        base_output_dir: Base directory for outputs
        Other args: Same as generate_labels_with_confidence
    """
    
    print("🚀 Starting batch processing...")
    
    results = {}
    
    for config in directories_config:
        print(f"\n{'='*60}")
        print(f"Processing: {config['name']}")
        print(f"{'='*60}")
        
        # Create output paths
        output_dir = os.path.join(base_output_dir, config['name'])
        output_file = os.path.join(output_dir, f"{config['name']}_labels_with_confidence.csv")
        
        # Process directory
        try:
            df, kmeans_model = generate_labels_with_confidence(
                model_path=model_path,
                images_directory=config['dir'],
                cluster_to_class_mapping=config['mapping'],
                output_file=output_file,
                base_model=base_model,
                device=device,
                batch_size=batch_size,
                confidence_threshold=confidence_threshold
            )
            
            results[config['name']] = {
                'dataframe': df,
                'kmeans_model': kmeans_model,
                'output_file': output_file
            }
            
        except Exception as e:
            print(f"❌ Error processing {config['name']}: {e}")
            results[config['name']] = {'error': str(e)}
    
    print(f"\n🎉 Batch processing completed!")
    return results

# Example usage functions
def example_single_directory():
    """Example: Process a single directory"""
    
    config = {
        'model_path': 'models/door/door_ssl_model.pth',
        'images_directory': 'data/full_dataset/door_images',
        'cluster_to_class_mapping': {0: 'Normal', 1: 'Anomaly'},
        'output_file': 'output/door_labels_with_confidence.csv',
        'base_model': 'mobilenet_v2',
        'device': 'auto',
        'batch_size': 32,
        'confidence_threshold': 0.7
    }
    
    df, kmeans_model = generate_labels_with_confidence(**config)
    
    # Optional: Filter high-confidence predictions only
    high_conf_df = df[df['High_Confidence'] == True]
    high_conf_path = config['output_file'].replace('.csv', '_high_confidence_only.csv')
    high_conf_df.to_csv(high_conf_path, index=False)
    print(f"High confidence predictions saved to: {high_conf_path}")

def example_batch_processing():
    """Example: Process multiple directories"""
    
    directories_config = [
        {
            'dir': 'data/full_dataset/door_images',
            'mapping': {0: 'Normal', 1: 'Anomaly'},
            'name': 'door'
        },
        {
            'dir': 'data/full_dataset/appliance_images', 
            'mapping': {0: 'Normal', 1: 'Anomaly'},
            'name': 'appliance'
        },
        {
            'dir': 'data/full_dataset/window_images',
            'mapping': {0: 'Normal', 1: 'Anomaly'}, 
            'name': 'window'
        }
    ]
    
    results = batch_process_directories(
        model_path='models/ssl_model.pth',
        directories_config=directories_config,
        base_output_dir='output/batch_labeling',
        confidence_threshold=0.6
    )
    
    return results

# Main execution
if __name__ == "__main__":
    # Run single directory example
    print("Running single directory example...")
    example_single_directory()
    
    # Uncomment to run batch processing example
    # print("\nRunning batch processing example...")
    # example_batch_processing()