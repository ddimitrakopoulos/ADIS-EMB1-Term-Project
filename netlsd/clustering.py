import time
import os
import numpy as np
import tracemalloc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import TSNE
import umap

from karateclub import NetLSD
import networkx as nx

import netlsd 

sns.set_style("whitegrid")

# -----------------------------
# Load dataset and (manual TUDataset format) convert to NetworkX
# -----------------------------

def load_dataset(dataset_name, root="data"):
    dataset_dir = os.path.join(root)
    edge_file = os.path.join(dataset_dir, f"{dataset_name}_A.txt")
    indicator_file = os.path.join(dataset_dir, f"{dataset_name}_graph_indicator.txt")
    label_file = os.path.join(dataset_dir, f"{dataset_name}_graph_labels.txt")

    # Optional files
    node_label_file = os.path.join(dataset_dir, f"{dataset_name}_node_labels.txt")
    node_attr_file = os.path.join(dataset_dir, f"{dataset_name}_node_attributes.txt")
    edge_label_file = os.path.join(dataset_dir, f"{dataset_name}_edge_labels.txt")

    # Read edges
    edges = []

    if os.path.exists(edge_label_file):
        with open(edge_file, "r") as ef, open(edge_label_file, "r") as elf:
            for e_line, l_line in zip(ef, elf):
                u, v = map(int, e_line.strip().split(","))
                label = int(l_line.strip())
                edges.append((u, v, label))
    else:
        with open(edge_file, "r") as f:
            for line in f:
                u, v = map(int, line.strip().split(","))
                edges.append((u, v))

    # Read graph indicators
    with open(indicator_file, "r") as f:
        indicators = [int(line.strip()) for line in f] 
    num_graphs = max(indicators) 

    # Build graphs
    # nx.Graph() is for undirected graphs, so add_edge doesnt include both entries of the same graph
    graphs = [nx.Graph() for _ in range(num_graphs)] 
    node_id_to_graph = {}
    for node_id, graph_id in enumerate(indicators, start=1):
        graphs[graph_id - 1].add_node(node_id) 
        node_id_to_graph[node_id] = graphs[graph_id - 1]

    # Add edges (with optional labels)
    if len(edges) > 0 and len(edges[0]) == 3:
        for u, v, label in edges:
            node_id_to_graph[u].add_edge(u, v, label=label)
    else:
        for u, v in edges:
            node_id_to_graph[u].add_edge(u, v) 

    # Relabel nodes to consecutive integers starting from 0
    graphs = [nx.convert_node_labels_to_integers(g) for g in graphs]

    # Read graph labels
    with open(label_file, "r") as f:
        labels = [int(line.strip()) for line in f]

    # Attach node labels or attributes if available
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
    import numpy as np
    import tracemalloc
    import time
    from karateclub import NetLSD

    num_graphs = len(graphs)
    embeddings = np.zeros((num_graphs, dim), dtype=np.float32)  

    # Start memory tracking
    tracemalloc.start()
    start_time = time.time()

    # Instantiate a single NetLSD model (scale_steps=dim is constant across graphs)
    model = NetLSD(scale_steps=dim) #, approximations=...)

    for i, g in enumerate(graphs):
        num_nodes = g.number_of_nodes()

        if num_nodes < 6:
            continue

        # model.infer to avoid creating a new model and calling fit every time
        embeddings[i] = model.infer([g])[0]

    total_time = time.time() - start_time

    # Memory usage
    embed_mem = embeddings.nbytes / 1024**2  # MB
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return (
        embeddings,
        total_time,
        peak_mem / 1024**2
    )

def generate_netlsd_embeddings_wave(graphs, dim=250, kernel="heat"):

    embeddings = []
    tracemalloc.start()
    start_time = time.time()

    for g in graphs:
        n = g.number_of_nodes()
        if n < 6:
            embeddings.append(np.zeros(dim))
            continue

        # compute NetLSD descriptor using selected kernel
        if kernel == "heat":
            emb = netlsd.heat(g, timescales=np.logspace(-2, 2, dim)) # default approximations = 200
        elif kernel == "wave":
            emb = netlsd.wave(g, timescales=np.logspace(-2, 2, dim)) # default approximations = 200
        else:
            raise ValueError("Unknown kernel. Use 'heat' or 'wave'.")
        
        embeddings.append(emb)

    total_time = time.time() - start_time
    X = np.array(embeddings)
    embed_mem = X.nbytes / 1024**2
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return X, total_time, peak_mem / 1024**2


# -----------------------------
# Clustering + ARI
# -----------------------------

def cluster_and_evaluate(X, y, method="kmeans", n_runs=50):
    n_clusters = len(np.unique(y))
    ari_scores = []
    y_preds = []
    
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1_000_000, size=n_runs)
    
    for seed in seeds:
        if method == "kmeans":
            model = KMeans(
                n_clusters=n_clusters,
                random_state=seed,
                n_init=20
            )
            y_pred = model.fit_predict(X)

        elif method == "spectral":
            model = SpectralClustering(
                n_clusters=n_clusters,
                affinity="rbf",
                random_state=seed,
                assign_labels="cluster_qr"
            )
            y_pred = model.fit_predict(X)

        else:
            raise ValueError("Unknown clustering method")

        ari = adjusted_rand_score(y, y_pred)
        ari_scores.append(ari)
        y_preds.append(y_pred)

    ari_scores = np.array(ari_scores)

    mean_ari = ari_scores.mean()
    std_ari = ari_scores.std()

    # pick the run closest to the mean for visualization
    idx = np.argmin(np.abs(ari_scores - mean_ari))
    y_pred_vis = y_preds[idx]

    return {
        "mean_ari": mean_ari,
        "std_ari": std_ari,
        "ari_all": ari_scores,
        "y_pred_vis": y_pred_vis
    }



# -----------------------------
# Visualization
# -----------------------------

def visualize_2d(
    X,
    clusters,
    title,
    dataset_name,
    emb_dim,
    method="tsne",
    cluster_method="original",
    out_dir="agglo_plots"
):
    os.makedirs(out_dir, exist_ok=True)

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

    filename = f"{dataset_name}_dim{emb_dim}_{cluster_method}_{method}_2d.png"
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main experiment
# -----------------------------

def run_clustering_experiment(dataset_name, dataset_path, dim=250, do_visualize=False):
    print(f"\n===== Dataset: {dataset_name} =====")
    graphs, labels = load_dataset(dataset_name, dataset_path)

    print("Generating NetLSD embeddings...")
    X, time_sec, peak_mem = generate_netlsd_embeddings(graphs, dim)
    #X, time_sec, peak_mem = generate_netlsd_embeddings_wave(graphs, dim) --> experimenting with the wave kernel
    print(f"Embedding time: {time_sec:.2f}s | Peak memory: {peak_mem:.2f} MB")
    
    '''For visualizing the original clusters of the dataset:
    if do_visualize:
        visualize_2d(
            X, labels,
            f"{dataset_name} – Original t-SNE",
            dataset_name, dim,
            method="tsne",
            cluster_method="original"
        )
        visualize_2d(
            X, labels,
            f"{dataset_name} – Original UMAP",
            dataset_name, dim,
            method="umap",
            cluster_method="original"
        )
    '''

    # ---------- KMEANS ----------
    print("\nK-Means clustering (50 runs)...")
    km_results = cluster_and_evaluate(X, labels, method="kmeans", n_runs=50)

    print(
        f"K-Means ARI: {km_results['mean_ari']:.4f} "
        f"± {km_results['std_ari']:.4f}"
    )

    if do_visualize:
        visualize_2d(
            X, km_results["y_pred_vis"],
            f"{dataset_name} – KMeans t-SNE",
            dataset_name, dim,
            method="tsne",
            cluster_method="kmeans"
        )
        visualize_2d(
            X, km_results["y_pred_vis"],
            f"{dataset_name} – KMeans UMAP",
            dataset_name, dim,
            method="umap",
            cluster_method="kmeans"
        )

    # ---------- SPECTRAL ----------
    print("\nSpectral clustering (50 runs)...")
    sc_results = cluster_and_evaluate(X, labels, method="spectral", n_runs=50)

    print(
        f"Spectral ARI: {sc_results['mean_ari']:.4f} "
        f"± {sc_results['std_ari']:.4f}"
    )

    if do_visualize:
        visualize_2d(
            X, sc_results["y_pred_vis"],
            f"{dataset_name} – Spectral t-SNE",
            dataset_name, dim,
            method="tsne",
            cluster_method="spectral"
        )
        visualize_2d(
            X, sc_results["y_pred_vis"],
            f"{dataset_name} – Spectral UMAP",
            dataset_name, dim,
            method="umap",
            cluster_method="spectral"
        )

    return {
        "dim": dim,
        "kmeans_mean_ari": km_results["mean_ari"],
        "kmeans_std_ari": km_results["std_ari"],
        "spectral_mean_ari": sc_results["mean_ari"],
        "spectral_std_ari": sc_results["std_ari"],
    }
