import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import UMAP
from sklearn.preprocessing import StandardScaler
from typing import Dict
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def _to_numpy(cluster_vectors: Dict[str, list[torch.Tensor]]):
    data = []
    labels = []
    for label, vecs in cluster_vectors.items():
        for v in vecs:
            data.append(v.cpu().numpy())
            labels.append(label)
    return np.array(data), labels


def _project(data, method="pca", dim=2):
    data = StandardScaler().fit_transform(data)
    if method == "pca":
        return PCA(n_components=dim).fit_transform(data)
    elif method == "umap":
        return UMAP(n_components=dim, random_state=42).fit_transform(data)
    else:
        raise ValueError("Unsupported projection method.")


def _draw_ellipse(ax, points, color):
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:, 1][::-1]))
    width, height = 2 * np.sqrt(eigvals)
    mean = points.mean(axis=0)
    ellipse = Ellipse(mean, width, height, angle=angle, color=color, alpha=0.2)
    ax.add_patch(ellipse)


def plot_2d_latent_clusters(cluster_vectors: Dict[str, list[torch.Tensor]],
                             save_path: str = "outputs/latent_2d.png",
                             method: str = "pca",
                             overlay_stats: bool = True):
    """
    Projects pooled latent vectors to 2D and plots clusters, centroids, and covariance.
    """
    X, labels = _to_numpy(cluster_vectors)
    X_proj = _project(X, method=method, dim=2)

    unique_labels = sorted(set(labels))
    palette = {label: sns.color_palette("tab10")[i] for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == label]
        points = X_proj[idxs]
        ax.scatter(points[:, 0], points[:, 1], label=label,
                   alpha=0.6, s=40, color=palette[label])

        # Centroid
        centroid = points.mean(axis=0)
        ax.scatter(*centroid, marker='X', s=180, color='black', edgecolors='white')

        # Covariance Ellipse
        _draw_ellipse(ax, points, palette[label])

        if overlay_stats:
            compactness = np.linalg.norm(points - centroid, axis=1).mean()
            ax.text(centroid[0], centroid[1], f"{label}\nσ={compactness:.2f}",
                    fontsize=9, ha='center', va='center', color='black',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    ax.set_title(f"Latent Cluster Projection (2D - {method.upper()})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_3d_latent_clusters(cluster_vectors: Dict[str, list[torch.Tensor]],
                             save_path: str = "outputs/latent_3d.png",
                             method: str = "pca"):
    """
    Projects pooled latent vectors to 3D and plots clusters + centroids.
    """
    X, labels = _to_numpy(cluster_vectors)
    X_proj = _project(X, method=method, dim=3)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = sorted(set(labels))
    palette = {label: sns.color_palette("tab10")[i] for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == label]
        points = X_proj[idxs]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   label=label, s=35, alpha=0.6, color=palette[label])
        centroid = points.mean(axis=0)
        ax.scatter(*centroid, marker='X', s=200, color='black', edgecolors='white')

    ax.set_title(f"Latent Cluster Projection (3D - {method.upper()})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
