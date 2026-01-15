# ==================================================
# NetLSD Stability Analysis: MUTAG, ENZYMES, IMDB-MULTI
# ==================================================

import time
import random
import numpy as np
import networkx as nx
import tracemalloc
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from karateclub import NetLSD

# -----------------------------
# Load dataset and convert to NetworkX
# -----------------------------

def load_dataset(dataset_name, root="data"):
    # Load from local TUDataset files (not torch_geometric)
    import os
    import networkx as nx
    import numpy as np
    dataset_dir = os.path.join(root)
    edge_file = os.path.join(dataset_dir, f"{dataset_name}_A.txt")
    indicator_file = os.path.join(dataset_dir, f"{dataset_name}_graph_indicator.txt")
    label_file = os.path.join(dataset_dir, f"{dataset_name}_graph_labels.txt")

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
    for u, v in edges:
        node_id_to_graph[u].add_edge(u, v)

    # Relabel nodes to consecutive integers starting from 0
    graphs = [nx.convert_node_labels_to_integers(g) for g in graphs]

    # Read graph labels
    with open(label_file, "r") as f:
        labels = [int(line.strip()) for line in f]
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

    total_time = time.time() - start_time
    X = np.array(embeddings)
    embed_mem = X.nbytes / 1024**2
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return X, total_time, embed_mem, peak_mem / 1024**2


# -----------------------------
# Graph perturbation functions
# -----------------------------
def perturb_edges_alter(G, perturb_ratio=0.05):
    """Randomly remove/add p% of edges (safe)."""
    Gp = deepcopy(G)

    edges = list(Gp.edges())
    nodes = list(Gp.nodes())

    if len(edges) == 0 or len(nodes) < 2:
        return Gp

    num_changes = max(1, int(len(edges) * perturb_ratio))

    # ---- Remove edges safely ----
    for _ in range(min(num_changes, len(edges))):
        u, v = random.choice(edges)
        if Gp.has_edge(u, v):
            Gp.remove_edge(u, v)
        edges.remove((u, v))

    # ---- Add random edges ----
    for _ in range(num_changes):
        u, v = random.sample(nodes, 2)
        if not Gp.has_edge(u, v):
            Gp.add_edge(u, v)

    return Gp


def shuffle_node_labels(G):
    """Randomly shuffle node labels (if present)."""
    Gp = deepcopy(G)

    # Check if labels exist
    if "label" not in next(iter(Gp.nodes(data=True)))[1]:
        return Gp  # nothing to shuffle

    labels = [Gp.nodes[n]["label"] for n in Gp.nodes()]
    random.shuffle(labels)

    for n, new_label in zip(Gp.nodes(), labels):
        Gp.nodes[n]["label"] = new_label

    return Gp


# to observe how much graph changed when perturbated
def edge_jaccard(G, Gp):
    E1, E2 = set(G.edges()), set(Gp.edges())
    if len(E1 | E2) == 0:
        return 1.0  # both empty
    return len(E1 & E2) / len(E1 | E2)


# -----------------------------
# Stability analysis
# -----------------------------
def stability_analysis(graphs, orig_embeddings, dim=128, edge_perturb=0.05, shuffle_labels=False):
    perturbed_embeddings = []
    edge_jaccards = []

    for G in graphs:
        Gp = perturb_edges_alter(G, perturb_ratio=edge_perturb)
        if shuffle_labels:
            Gp = shuffle_node_labels(Gp)

        edge_jaccards.append(edge_jaccard(G, Gp))

        num_nodes = Gp.number_of_nodes()
        if num_nodes < 6:
            perturbed_embeddings.append(np.zeros(dim))
            continue

        approximations = max(1, min(200, (num_nodes - 2) // 2))
        model = NetLSD(scale_steps=dim, approximations=approximations)
        model.fit([Gp])
        perturbed_embeddings.append(model.get_embedding()[0])

    perturbed_embeddings = np.array(perturbed_embeddings)

    # ---- Metrics ----
    cosine_sims = [] # shows shape preservation
    l2_dists = [] # shows magnitude sensitivity
    rel_l2_dists = []

    for x, y in zip(orig_embeddings, perturbed_embeddings):
        cosine_sims.append(
            cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0, 0]
        )
        l2 = np.linalg.norm(x - y)
        l2_dists.append(l2)
        rel_l2_dists.append(l2 / (np.linalg.norm(x) + 1e-10))

    return {
        "edge_jaccard": np.array(edge_jaccards),
        "cosine": np.array(cosine_sims),
        "l2": np.array(l2_dists),
        "rel_l2": np.array(rel_l2_dists),
    }


# -----------------------------
# Run stability experiments
# -----------------------------
def run_stability(datasets, dim=128, dataset_path=None, edge_perturb=0.05, shuffle_labels=False):
    results = {}

    for dataset_name in datasets:
        print(f"\n===== Dataset: {dataset_name} =====")

        graphs, labels = load_dataset(dataset_name, root=dataset_path)

        # Original embeddings
        orig_embeddings, t, mem, peak = generate_netlsd_embeddings(graphs, dim=dim)
        print(f"Original embeddings generated in {t:.2f}s | Mem: {mem:.2f} MB | Peak: {peak:.2f} MB")

        # Perturbed embeddings & stability
        metrics = stability_analysis(
            graphs, orig_embeddings, dim=dim,
            edge_perturb=edge_perturb, shuffle_labels=shuffle_labels
        )

        print(f"Edge Jaccard (mean ± std): {metrics['edge_jaccard'].mean():.3f} ± {metrics['edge_jaccard'].std():.3f}")
        print(f"Cosine similarity (mean ± std): {metrics['cosine'].mean():.4f} ± {metrics['cosine'].std():.4f}")
        print(f"Relative L2 change (mean ± std): {metrics['rel_l2'].mean():.4f} ± {metrics['rel_l2'].std():.4f}")

        '''
        # Histogram of cosine similarities
        plt.figure(figsize=(6,4))
        plt.hist(metrics['cosine'], bins=20, color='skyblue', edgecolor='black')
        plt.title(f"Stability (Cosine similarity) - {dataset_name}")
        plt.xlabel("Cosine similarity with original embeddings")
        plt.ylabel("Number of graphs")
        plt.show()

        # Scatter plot Edge Jaccard vs Relative L2
        plt.figure(figsize=(5,4))
        plt.scatter(metrics['edge_jaccard'], metrics['rel_l2'], alpha=0.5)
        plt.xlabel("Edge Jaccard Similarity")
        plt.ylabel("Relative L2 Change (NetLSD)")
        plt.title(f"Perturbation vs Embedding Change ({dataset_name})")
        plt.grid(True)
        plt.show()
        '''
        results[dataset_name] = metrics

    return results


# -----------------------------
# Execute
# -----------------------------
if __name__ == "__main__":
    stability_results = run_stability(
        datasets=["MUTAG", "ENZYMES", "IMDB-MULTI"],
        dim=128,
        edge_perturb=0.8, 
        shuffle_labels=False  # node labels not shuffled (doesnt make a difference for netlsd)
    )

'''
NetLSD embeddings are invariant to node-label permutations, as the method is purely structural. 
Consequently, shuffling node labels does not affect the embeddings.
'''
