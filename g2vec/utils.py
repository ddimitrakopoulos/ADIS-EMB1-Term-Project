import os
import psutil
import numpy as np
import networkx as nx
import copy
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# Memory helper
# ==================================================
process = psutil.Process(os.getpid())
def mem_mb():
    return process.memory_info().rss / 1024 / 1024

# ==================================================
# Load TUDataset graphs
# ==================================================
def load_tudataset(dataset_path):
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    edges = np.loadtxt(os.path.join(dataset_path, f"{dataset_name}_A.txt"), delimiter=",", dtype=int) - 1
    graph_indicator = np.loadtxt(os.path.join(dataset_path, f"{dataset_name}_graph_indicator.txt"), dtype=int) - 1
    graph_labels = np.loadtxt(os.path.join(dataset_path, f"{dataset_name}_graph_labels.txt"), dtype=int)
    node_labels_path = os.path.join(dataset_path, f"{dataset_name}_node_labels.txt")
    node_labels = np.loadtxt(node_labels_path, dtype=int) if os.path.exists(node_labels_path) else None

    graphs = []
    for g_id in range(graph_labels.shape[0]):
        node_ids = np.where(graph_indicator == g_id)[0]
        id_map = {old: new for new, old in enumerate(node_ids)}
        G = nx.Graph()
        for old_id, new_id in id_map.items():
            G.add_node(new_id)
            if node_labels is not None:
                G.nodes[new_id]["label"] = int(node_labels[old_id])
        for u, v in edges:
            if u in id_map and v in id_map:
                G.add_edge(id_map[u], id_map[v])
        graphs.append(G)

    return graphs, graph_labels

# ==================================================
# Perturb a graph for stability analysis
# ==================================================
def perturb_graph(G, edge_perturb_ratio=0.1, shuffle_node_labels=True):
    G_perturbed = copy.deepcopy(G)
    num_edges = G_perturbed.number_of_edges()
    num_remove = int(edge_perturb_ratio * num_edges)
    edges = list(G_perturbed.edges())
    edges_to_remove = random.sample(edges, min(num_remove, len(edges)))
    G_perturbed.remove_edges_from(edges_to_remove)
    nodes = list(G_perturbed.nodes())
    added = 0
    while added < num_remove:
        u, v = random.sample(nodes, 2)
        if not G_perturbed.has_edge(u, v):
            G_perturbed.add_edge(u, v)
            added += 1
    if shuffle_node_labels and nx.get_node_attributes(G_perturbed, "label"):
        labels = list(nx.get_node_attributes(G_perturbed, "label").values())
        random.shuffle(labels)
        for i, node in enumerate(G_perturbed.nodes()):
            G_perturbed.nodes[node]["label"] = labels[i]
    return G_perturbed

# ==================================================
# Feature scaling
# ==================================================
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

# ==================================================
# Embedding stability score
# ==================================================
def embedding_stability(X_original, X_perturbed):
    sim = np.diag(cosine_similarity(X_original, X_perturbed))
    return np.mean(sim)
