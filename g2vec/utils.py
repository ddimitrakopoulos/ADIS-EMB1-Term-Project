import os
import psutil
import numpy as np
import networkx as nx
import copy
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.datasets import TUDataset

process = psutil.Process(os.getpid())

def mem_mb():
    return process.memory_info().rss / 1024 / 1024

def _to_numpy(tensor):
    if hasattr(tensor, 'cpu'):
        return tensor.cpu().numpy()
    elif hasattr(tensor, 'numpy'):
        return tensor.numpy()
    return tensor

def download_tudataset(dataset_name, download_dir="./datasets"):
    """
    Download a TUDataset dataset using PyG (PyTorch Geometric).
    Automatically handles node/edge attributes with use_node_attr and use_edge_attr.
    
    Supported datasets: MUTAG, IMDB-MULTI, ENZYMES
    """
    supported = ["MUTAG", "IMDB-MULTI", "ENZYMES"]
    if dataset_name not in supported:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from: {', '.join(supported)}")
    
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Loading '{dataset_name}' dataset using PyG (use_node_attr=True, use_edge_attr=True)...")
    try:
        dataset = TUDataset(
            root=download_dir,
            name=dataset_name,
            use_node_attr=True,
            use_edge_attr=True
        )
        
        graphs, labels = [], []
        
        for pyg_graph in dataset:
            graph_label = int(pyg_graph.y.item()) if hasattr(pyg_graph, 'y') else 0
            labels.append(graph_label)
            
            G = nx.Graph()
            edge_index = pyg_graph.edge_index
            
            for node_idx in range(pyg_graph.num_nodes):
                G.add_node(node_idx)
                if hasattr(pyg_graph, 'x') and pyg_graph.x is not None:
                    try:
                        node_features = _to_numpy(pyg_graph.x[node_idx])
                        G.nodes[node_idx]["feature"] = node_features
                    except Exception:
                        G.nodes[node_idx]["feature"] = node_idx
                else:
                    G.nodes[node_idx]["feature"] = node_idx
            
            for i in range(edge_index.shape[1]):
                u, v = int(edge_index[0, i]), int(edge_index[1, i])
                G.add_edge(u, v)
                if hasattr(pyg_graph, 'edge_attr') and pyg_graph.edge_attr is not None:
                    G[u][v]["attr"] = pyg_graph.edge_attr[i]
            
            graphs.append(G)
        
        print(f"Successfully loaded '{dataset_name}': {len(graphs)} graphs")
        return graphs, np.array(labels)
        
    except Exception as e:
        raise RuntimeError(f"Failed to download/load dataset '{dataset_name}': {e}")

def load_tudataset(dataset_path):
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    edges = np.loadtxt(os.path.join(dataset_path, f"{dataset_name}_A.txt"), delimiter=",", dtype=int) - 1
    graph_indicator = np.loadtxt(os.path.join(dataset_path, f"{dataset_name}_graph_indicator.txt"), dtype=int) - 1
    graph_labels = np.loadtxt(os.path.join(dataset_path, f"{dataset_name}_graph_labels.txt"), dtype=int)
    node_labels_path = os.path.join(dataset_path, f"{dataset_name}_node_labels.txt")
    node_labels = np.loadtxt(node_labels_path, dtype=int) if os.path.exists(node_labels_path) else None

    graphs = []
    for g_id in range(len(graph_labels)):
        node_ids = np.where(graph_indicator == g_id)[0]
        id_map = {old: new for new, old in enumerate(node_ids)}
        G = nx.Graph()
        for old_id, new_id in id_map.items():
            G.add_node(new_id)
            feature = int(node_labels[old_id]) if node_labels is not None else new_id
            G.nodes[new_id]["feature"] = feature
        for u, v in edges:
            if u in id_map and v in id_map:
                G.add_edge(id_map[u], id_map[v])
        graphs.append(G)

    return graphs, graph_labels

def perturb_graph(G, edge_perturb_ratio=0.1, mode='both'):
    G_perturbed = copy.deepcopy(G)
    edges = list(G_perturbed.edges())
    nodes = list(G_perturbed.nodes())
    
    if len(nodes) < 2:
        return G_perturbed
    
    num_edges = len(edges)
    num_perturb = max(1, int(num_edges * edge_perturb_ratio)) if num_edges > 0 else 0
    
    if mode in ['remove', 'both'] and num_edges > 0:
        for _ in range(min(num_perturb, len(edges))):
            if not edges:
                break
            e = random.choice(edges)
            if G_perturbed.has_edge(*e):
                G_perturbed.remove_edge(*e)
            edges.remove(e)
    
    if mode in ['add', 'both']:
        added = 0
        for _ in range(num_perturb * 100):
            if added >= num_perturb:
                break
            u, v = random.sample(nodes, 2)
            if not G_perturbed.has_edge(u, v):
                G_perturbed.add_edge(u, v)
                added += 1
    
    if mode == 'shuffle':
        features = nx.get_node_attributes(G_perturbed, "feature")
        if features:
            node_list = list(G_perturbed.nodes())
            n_shuffle = max(2, min(int(len(node_list) * edge_perturb_ratio), len(node_list)))
            shuffle_nodes = random.sample(node_list, n_shuffle)
            shuffle_features = [features[n] for n in shuffle_nodes]
            random.shuffle(shuffle_features)
            for node, feature in zip(shuffle_nodes, shuffle_features):
                G_perturbed.nodes[node]["feature"] = feature
    
    return G_perturbed

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def embedding_stability(X_original, X_perturbed):
    return np.mean(np.diag(cosine_similarity(X_original, X_perturbed)))

def edge_jaccard(G, Gp):
    E1, E2 = set(G.edges()), set(Gp.edges())
    union = E1 | E2
    return len(E1 & E2) / len(union) if union else 1.0
