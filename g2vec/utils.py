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
def perturb_graph(G, edge_perturb_ratio=0.1, mode='both', shuffle_node_labels=False):

    G_perturbed = copy.deepcopy(G)
    edges = list(G_perturbed.edges())
    nodes = list(G_perturbed.nodes())
    
    if len(nodes) < 2:
        return G_perturbed
    
    num_edges = len(edges)
    num_perturb = max(1, int(num_edges * edge_perturb_ratio)) if num_edges > 0 else 0
    
    # Remove edges
    if mode in ['remove', 'both'] and num_edges > 0:
        num_to_remove = min(num_perturb, len(edges))
        for _ in range(num_to_remove):
            if len(edges) == 0:
                break
            e = random.choice(edges)
            if G_perturbed.has_edge(*e):
                G_perturbed.remove_edge(*e)
            edges.remove(e)
    
    # Add edges
    if mode in ['add', 'both']:
        added = 0
        max_attempts = num_perturb * 100
        attempts = 0
        while added < num_perturb and attempts < max_attempts:
            u, v = random.sample(nodes, 2)
            if not G_perturbed.has_edge(u, v):
                G_perturbed.add_edge(u, v)
                added += 1
            attempts += 1
    
    # mode='none' skips edge perturbation (used for label-only shuffling)
    
    # Shuffle node labels (for GIN-style perturbation)
    # Shuffle proportionally to edge_perturb_ratio
    if shuffle_node_labels and nx.get_node_attributes(G_perturbed, "label"):
        labels_dict = nx.get_node_attributes(G_perturbed, "label")
        node_list = list(G_perturbed.nodes())
        n_to_shuffle = max(2, int(len(node_list) * edge_perturb_ratio))
        n_to_shuffle = min(n_to_shuffle, len(node_list))
        
        # Pick random subset of nodes to shuffle labels among
        shuffle_nodes = random.sample(node_list, n_to_shuffle)
        shuffle_labels_list = [labels_dict[n] for n in shuffle_nodes]
        random.shuffle(shuffle_labels_list)
        for i, n in enumerate(shuffle_nodes):
            G_perturbed.nodes[n]["label"] = shuffle_labels_list[i]
    
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

def edge_jaccard(G, Gp):
    E1, E2 = set(G.edges()), set(Gp.edges())
    if len(E1 | E2) == 0:
        return 1.0
    return len(E1 & E2) / len(E1 | E2)
