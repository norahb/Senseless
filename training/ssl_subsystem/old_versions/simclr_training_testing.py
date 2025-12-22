import ssl
import csv
import shutil
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Dataset
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import time
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.cluster import KMeans

# Define the SimCLR model
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

# Custom dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Data augmentation for training
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

# NT-Xent loss
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

# Training function
def train(model, train_loader, optimizer, epochs):
    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for images in train_loader:
            images = torch.cat(images, dim=0).cuda()
            batch_size = images.shape[0] // 2
            _, z = model(images)
            loss = nt_xent_loss(z[:batch_size], z[batch_size:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
    end_time = time.time()
    training_duration = (end_time - start_time) / 60
    print(f"Training Duration: {training_duration:.2f} minutes")
    return training_duration

# Clustering functions
def kmeans_cluster(features, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels

def gmm_cluster(features, n_clusters=2):
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    cluster_labels = gmm.fit_predict(features)
    return cluster_labels, gmm.bic(features)

def dbscan_cluster(features, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(features)
    return cluster_labels

def agglomerative_cluster(features, n_clusters=2):
    agglo = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = agglo.fit_predict(features)
    return cluster_labels

# Load ground truth labels
def load_ground_truth_labels(csv_file):
    true_labels_df = pd.read_csv(csv_file)
    return true_labels_df

# Merge clustering results with ground truth labels
def merge_with_ground_truth(cluster_labels, eval_dataset, true_labels_df):
    cluster_results_df = pd.DataFrame({
        'Image Name': eval_dataset.images,
        'Cluster': cluster_labels
    })
    merged_df = pd.merge(cluster_results_df, true_labels_df, on='Image Name')
    return merged_df

# Map clusters to classes based on confusion matrix
def map_clusters_to_classes(merged_df):
    cluster_to_class_mapping = merged_df.groupby('Cluster')['true-label'].agg(lambda x: x.value_counts().idxmax())
    print("Cluster to Class Mapping:")
    print(cluster_to_class_mapping)
    return cluster_to_class_mapping.to_dict()

# Save cluster results
def save_cluster_results(merged_df, cluster_to_class, filename='clustering_results_with_classes.csv'):
    merged_df['Assigned Class'] = merged_df['Cluster'].map(cluster_to_class)
    merged_df.to_csv(filename, index=False)
    print(f"Mapped clusters to classes and saved results to '{filename}'")

# Evaluate clusters
def evaluate_clusters(merged_df, cluster_to_class, method_name=None):
    if method_name:
        print(f"Evaluating clusters for {method_name}...")
    y_true = merged_df['true-label']
    y_pred = merged_df['Cluster'].map(cluster_to_class)
    
    # Filter out noise points (where cluster is -1 for DBSCAN)
    valid_indices = merged_df['Cluster'] != -1
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]

    conf_mat = confusion_matrix(y_true_filtered, y_pred_filtered, labels=list(cluster_to_class.values()))
    print("Confusion Matrix:")
    print(conf_mat)

    print("Classification Report:")
    print(classification_report(y_true_filtered, y_pred_filtered))
    
    return conf_mat

# Calculate model size
def calculate_model_size(model):
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024**2)
    print(f"Model Size: {model_size:.2f} MB")
    return model_size

# Main execution
if __name__ == "__main__":
    # Hyperparameters and paths
    batch_size = 256
    epochs = 100
    learning_rate = 3e-4
    train_dataset_path = "../Door_dataset_split/train"  # Training data
    test_dataset_path = "../Door_dataset_split/test"    # Testing data
    true_labels_csv = '../Door-true-labels2.csv'
    model_path = "../Door_simclr_model.pth"

    # Step 1: Training data loading
    train_transform = SimCLRDataTransform(size=256)
    train_dataset = ImageDataset(root_dir=train_dataset_path, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
    
    print(f"Training images: {len(train_dataset)}")
    
    # Step 2: Model setup
    model = SimCLR().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Step 3: Train or load the model
    if os.path.exists(model_path):
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        training_duration = 0
    else:
        print("Training new model...")
        training_duration = train(model, train_loader, optimizer, epochs)
        torch.save(model.state_dict(), model_path)
        print("Model saved!")

    # Step 4: Calculate model size
    calculate_model_size(model)

    # Step 5: Feature extraction on TEST data
    model.eval()
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageDataset(root_dir=test_dataset_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
    
    print(f"Testing images: {len(test_dataset)}")

    features = []
    with torch.no_grad():
        for images in test_loader:
            images = images.cuda()
            h, _ = model(images)
            features.append(h.cpu().numpy())

    features = np.concatenate(features, axis=0)

    # Step 6: Perform clustering on test features
    cluster_labels_kmeans = kmeans_cluster(features, n_clusters=2)
    print("KMeans clustering done.")
    
    cluster_labels_gmm, bic_score = gmm_cluster(features)
    print(f"GMM clustering done. BIC Score: {bic_score:.2f}")
    
    cluster_labels_dbscan = dbscan_cluster(features, eps=0.3, min_samples=10)
    print("DBSCAN clustering done.")
    
    cluster_labels_agglomerative = agglomerative_cluster(features, n_clusters=2)
    print("Agglomerative clustering done.")

    # Step 7: Calculate silhouette scores
    silhouette_avg_kmeans = silhouette_score(features, cluster_labels_kmeans)
    silhouette_avg_gmm = silhouette_score(features, cluster_labels_gmm)
    print(f"Silhouette Score (KMeans): {silhouette_avg_kmeans:.4f}")
    print(f"Silhouette Score (GMM): {silhouette_avg_gmm:.4f}")

    # Step 8: Load ground truth labels
    true_labels_df = load_ground_truth_labels(true_labels_csv)

    # Step 9: Merge clustering results with ground truth
    merged_df_kmeans = merge_with_ground_truth(cluster_labels_kmeans, test_dataset, true_labels_df)
    merged_df_gmm = merge_with_ground_truth(cluster_labels_gmm, test_dataset, true_labels_df)
    merged_df_dbscan = merge_with_ground_truth(cluster_labels_dbscan, test_dataset, true_labels_df)
    merged_df_agglo = merge_with_ground_truth(cluster_labels_agglomerative, test_dataset, true_labels_df)

    # Step 10: Map clusters to classes
    cluster_to_class_kmeans = map_clusters_to_classes(merged_df_kmeans)
    cluster_to_class_gmm = map_clusters_to_classes(merged_df_gmm)
    cluster_to_class_dbscan = map_clusters_to_classes(merged_df_dbscan)
    cluster_to_class_agglo = map_clusters_to_classes(merged_df_agglo)

    # Step 11: Save results
    save_cluster_results(merged_df_kmeans, cluster_to_class_kmeans, 'test_kmeans_results.csv')
    save_cluster_results(merged_df_gmm, cluster_to_class_gmm, 'test_gmm_results.csv')
    save_cluster_results(merged_df_dbscan, cluster_to_class_dbscan, 'test_dbscan_results.csv')
    save_cluster_results(merged_df_agglo, cluster_to_class_agglo, 'test_agglo_results.csv')

    # Step 12: Evaluate clusters
    evaluate_clusters(merged_df_kmeans, cluster_to_class_kmeans, "KMeans")
    evaluate_clusters(merged_df_gmm, cluster_to_class_gmm, "GMM")
    evaluate_clusters(merged_df_dbscan, cluster_to_class_dbscan, "DBSCAN")
    evaluate_clusters(merged_df_agglo, cluster_to_class_agglo, "Agglomerative")