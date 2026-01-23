# ==================================================
# NetLSD Clustering: TUDatasets (MUTAG, ENZYMES, IMDB-MULTI)
# ==================================================

import time
import os
import numpy as np
import tracemalloc
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
import umap

from karateclub import NetLSD
import networkx as nx

sns.set_style("whitegrid")

# -----------------------------
# Load dataset (manual TUDataset format)
# -----------------------------

def load_dataset(dataset_name, root="data"):
    """
    Loads a graph dataset manually from the provided TUDataset files
    and converts it to NetworkX objects, which KarateClub’s NetLSD expects.
    Supports MUTAG, ENZYMES, and IMDB-MULTI formats.
    """
    dataset_dir = os.path.join(root)
    edge_file = os.path.join(dataset_dir, f"{dataset_name}_A.txt")
    indicator_file = os.path.join(dataset_dir, f"{dataset_name}_graph_indicator.txt")
    label_file = os.path.join(dataset_dir, f"{dataset_name}_graph_labels.txt")

    # Optional files
    node_label_file = os.path.join(dataset_dir, f"{dataset_name}_node_labels.txt")
    node_attr_file = os.path.join(dataset_dir, f"{dataset_name}_node_attributes.txt")

    # Read edges
    edges = []
    with open(edge_file, "r") as f:
        for line in f:
            u, v = map(int, line.strip().split(","))
            edges.append((u, v))

    # Read graph indicators
    with open(indicator_file, "r") as f:
        indicators = [int(line.strip()) for line in f]
    num_graphs = max(indicators)

    # Build graphs
    graphs = [nx.Graph() for _ in range(num_graphs)]
    node_id_to_graph = {}
    for node_id, graph_id in enumerate(indicators, start=1):
        graphs[graph_id - 1].add_node(node_id)
        node_id_to_graph[node_id] = graphs[graph_id - 1]

    # Add edges
    for u, v in edges:
        node_id_to_graph[u].add_edge(u, v)

    # Relabel nodes to consecutive integers starting from 0
    graphs = [nx.convert_node_labels_to_integers(g) for g in graphs]

    # Read graph labels
    with open(label_file, "r") as f:
        labels = [int(line.strip()) for line in f]

    # Optional: attach node labels or attributes if available
    if os.path.exists(node_label_file):
        with open(node_label_file, "r") as f:
            node_labels = [line.strip() for line in f]
        # assign node labels to graphs
        node_counter = 0
        for g in graphs:
            for n in g.nodes():
                g.nodes[n]["label"] = node_labels[node_counter]
                node_counter += 1

    if os.path.exists(node_attr_file):
        node_attrs = np.loadtxt(node_attr_file, delimiter=",")
        node_counter = 0
        for g in graphs:
            for n in g.nodes():
                g.nodes[n]["attr"] = node_attrs[node_counter]
                node_counter += 1

    return graphs, np.array(labels)

# -----------------------------
# Generate NetLSD embeddings
# -----------------------------
def generate_netlsd_embeddings(graphs, dim=250):
    embeddings = []
    tracemalloc.start()
    start_time = time.time()

    for g in graphs:
        n = g.number_of_nodes()
        if n < 6:
            embeddings.append(np.zeros(dim))
            continue

        #approximations = max(1, min(200, (n - 2) // 2))
        model = NetLSD(scale_steps=dim) #, approximations=approximations)
        model.fit([g])
        embeddings.append(model.get_embedding()[0])

    X = np.array(embeddings)
    total_time = time.time() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return X, total_time, peak_mem / 1024**2

''' Overall note over classification and clustering:
Classification: Confirms embeddings are useful for prediction.
    Are embeddings linearly separable for the target task?
    Can a classifier map the embedding space to known classes?
Clustering: Evaluates intrinsic structure of embeddings.
    Are embeddings naturally clustered by graph similarity?
    Do similar graphs lie close together in embedding space?
    Reveals emergent structure, independent of supervised task.
'''
# -----------------------------
# Clustering + ARI
# -----------------------------
def cluster_and_evaluate(X, y, method="kmeans"):
    n_clusters = len(np.unique(y)) # so many clusters as the #of the labels

    if method == "kmeans": # centroid-based clustering
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        y_pred = model.fit_predict(X) # X : input embeddings
    elif method == "spectral": # clustering using graph theory concepts
        # Leverages global structure, unlike K-Means, which only sees local distances.
        # so may give better results for NetLSD
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="rbf",
            random_state=42
        )
        y_pred = model.fit_predict(X)
    else:
        raise ValueError("Unknown clustering method")

    ''' How ARI works?
    Taking all pairs of points in the dataset:
    For each pair: 
    Agreement -> if same or different cluster in both true and predicted labels, 
    Disagreement -> if same in one, different in the other
    Count all agr/disagr and normalize + Adjust for chance.
    1 -> Perfect match (predicted = true clusters)
    0 -> Random Labeling
    -1 -> systematic disagreement 
    '''
    ari = adjusted_rand_score(y, y_pred)
    return y_pred, ari

# -----------------------------
# Visualization
# -----------------------------
def visualize(X, clusters, title, method="tsne"):
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)

    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=clusters,
        palette="tab10",
        legend="full",
        s=40
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Visualization 3D
# -----------------------------
def visualize_3d(X, clusters, title, method="tsne"):
    if method == "tsne":
        reducer = TSNE(n_components=3, random_state=42)
    else:
        reducer = umap.UMAP(n_components=3, random_state=42)

    X_3d = reducer.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        X_3d[:, 0],
        X_3d[:, 1],
        X_3d[:, 2],
        c=clusters,
        cmap="tab10",
        s=50,
        alpha=0.8
    )

    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main experiment
# -----------------------------
def run_clustering_experiment(dataset_name, dataset_path, dim=250, do_visualize=False):
    print(f"\n===== Dataset: {dataset_name} =====")
    graphs, labels = load_dataset(dataset_name, dataset_path)

    print("Generating NetLSD embeddings...")
    X, time_sec, peak_mem = generate_netlsd_embeddings(graphs, dim)
    print(f"Embedding time: {time_sec:.2f}s | Peak memory: {peak_mem:.2f} MB")
    
    if do_visualize:
            visualize_3d(X, labels, f"{dataset_name} – Original t-SNE", "tsne")
            visualize_3d(X, labels, f"{dataset_name} – Original UMAP", "umap")

    print("\nK-Means clustering...")
    km_labels, km_ari = cluster_and_evaluate(X, labels, "kmeans")
    print(f"K-Means ARI: {km_ari:.4f}")
    if do_visualize:
        visualize_3d(X, km_labels, f"{dataset_name} – KMeans t-SNE", "tsne")
        visualize_3d(X, km_labels, f"{dataset_name} – KMeans UMAP", "umap")

    print("\nSpectral clustering...")
    sc_labels, sc_ari = cluster_and_evaluate(X, labels, "spectral")
    print(f"Spectral ARI: {sc_ari:.4f}")
    if do_visualize:
        visualize_3d(X, sc_labels, f"{dataset_name} – Spectral t-SNE", "tsne")
        visualize_3d(X, sc_labels, f"{dataset_name} – Spectral UMAP", "umap")

    return {
        "dim": dim,
        "kmeans_ari": km_ari,
        "spectral_ari": sc_ari
    }
