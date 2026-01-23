# ==================================================
# NetLSD Clustering: TUDatasets (MUTAG, ENZYMES, IMDB-MULTI)
# ==================================================

import time
import psutil
import numpy as np
import tracemalloc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
import umap

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from karateclub import NetLSD

sns.set_style("whitegrid")

# -----------------------------
# Load dataset and convert to NetworkX
# -----------------------------
def load_dataset(dataset_name, root="data"):
    dataset = TUDataset(root=root, name=dataset_name)
    graphs = []
    labels = []

    for data in dataset:
        g = to_networkx(data, to_undirected=True)
        graphs.append(g)
        labels.append(int(data.y))
    
    return graphs, np.array(labels)

# -----------------------------
# Generate NetLSD embeddings
# -----------------------------
def generate_netlsd_embeddings(graphs, dim=128):
    embeddings = []

    tracemalloc.start()
    start_time = time.time()

    for g in graphs:
        num_nodes = g.number_of_nodes()
        if num_nodes < 6:
            embeddings.append(np.zeros(dim))
            continue
        approximations = max(1, min(200, (num_nodes - 2) // 2))
        model = NetLSD(scale_steps=dim, approximations=approximations)
        model.fit([g])
        embeddings.append(model.get_embedding()[0])

    X = np.array(embeddings)
    total_time = time.time() - start_time
    embed_mem = X.nbytes / 1024**2
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    avg_time_per_graph = total_time / max(1, len(graphs))

    return X, total_time, embed_mem, peak_mem / 1024**2, avg_time_per_graph

# -----------------------------
# Clustering and ARI evaluation
# -----------------------------
def cluster_and_evaluate(X, y, method="kmeans"):
    n_clusters = len(np.unique(y))

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_pred = model.fit_predict(X)
    elif method == "spectral":
        model = SpectralClustering(n_clusters=n_clusters, random_state=42,
                                   affinity='nearest_neighbors', n_init=10)
        y_pred = model.fit_predict(X)
    else:
        raise ValueError("Unknown clustering method")

    ari = adjusted_rand_score(y, y_pred)
    return y_pred, ari

# -----------------------------
# Visualization with t-SNE / UMAP
# -----------------------------
def visualize_embeddings(X, y, title="t-SNE", method="tsne"):
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Unknown visualization method")

    X_2d = reducer.fit_transform(X)
    plt.figure(figsize=(6,6))
    palette = sns.color_palette("tab10", np.max(y)+1)
    sns.scatterplot(x=X_2d[:,0], y=X_2d[:,1], hue=y, palette=palette, legend="full", s=50)
    plt.title(f"{title} Visualization")
    plt.show()

# -----------------------------
# Main clustering experiment
# -----------------------------
def run_clustering_experiment(dataset_name, dataset_path, dim=128):
    print(f"\n===== Dataset: {dataset_name} =====")
    graphs, labels = load_dataset(dataset_name, root=dataset_path)
    print(f"Number of graphs: {len(graphs)}")

    # Generate NetLSD embeddings
    print("Generating NetLSD embeddings...")
    X, total_time, embed_mem, peak_mem, avg_time = generate_netlsd_embeddings(graphs, dim=dim)
    print(f"Embedding time (s): {total_time:.2f}, memory used: {embed_mem:.2f} MB, peak memory: {peak_mem:.2f} MB")

    # -----------------------------
    # K-Means clustering
    # -----------------------------
    print("Performing K-Means clustering...")
    y_pred_km, ari_km = cluster_and_evaluate(X, labels, method="kmeans")
    print(f"K-Means ARI: {ari_km:.4f}")
    visualize_embeddings(X, y_pred_km, title=f"{dataset_name} - KMeans t-SNE", method="tsne")
    visualize_embeddings(X, y_pred_km, title=f"{dataset_name} - KMeans UMAP", method="umap")

    # -----------------------------
    # Spectral clustering
    # -----------------------------
    print("Performing Spectral clustering...")
    y_pred_sc, ari_sc = cluster_and_evaluate(X, labels, method="spectral")
    print(f"Spectral ARI: {ari_sc:.4f}")
    visualize_embeddings(X, y_pred_sc, title=f"{dataset_name} - Spectral t-SNE", method="tsne")
    visualize_embeddings(X, y_pred_sc, title=f"{dataset_name} - Spectral UMAP", method="umap")

    results = {
        "embedding_dim": dim,
        "embedding_time_sec": total_time,
        "embedding_memory_mb": embed_mem,
        "kmeans_ari": ari_km,
        "spectral_ari": ari_sc
    }

    return results

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    run_clustering_experiment()
