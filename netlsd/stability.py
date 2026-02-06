import os
import time
import random
import numpy as np
import networkx as nx
import tracemalloc
from copy import deepcopy

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from karateclub import NetLSD

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import netlsd 

# --------------------------------------------------
# Load TUDataset from raw files
# --------------------------------------------------
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
    edge_labels = None

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
        embed_mem,
        peak_mem / 1024**2
    )

def generate_netlsd_embeddings_wave(graphs, dim=250, kernel="wave"):

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

    return X, total_time, embed_mem, peak_mem / 1024**2

# --------------------------------------------------
# Graph perturbations
# --------------------------------------------------

def perturb_edges(G, ratio=0.05):
    Gp = deepcopy(G)
    edges = list(Gp.edges())
    nodes = list(Gp.nodes())

    if len(edges) == 0 or len(nodes) < 2:
        return Gp

    k = max(1, int(len(edges) * ratio))

    # remove 
    for _ in range(min(k, len(edges))):
        e = random.choice(edges)
        if Gp.has_edge(*e):
            Gp.remove_edge(*e)
        edges.remove(e)

    # add
    for _ in range(k):
        u, v = random.sample(nodes, 2)
        if not Gp.has_edge(u, v):
            Gp.add_edge(u, v)
    
    return Gp

def perturb_add(G, ratio=0.05):
    """Only add edges"""
    Gp = deepcopy(G)
    nodes = list(Gp.nodes())
    edges = list(Gp.edges())
    if len(nodes) < 2:
        return Gp

    k = max(1, int(len(edges) * ratio))
    for _ in range(k):
        u, v = random.sample(nodes, 2)
        if not Gp.has_edge(u, v):
            Gp.add_edge(u, v)
    return Gp

def perturb_remove(G, ratio=0.05):
    """Only remove edges"""
    Gp = deepcopy(G)
    edges = list(Gp.edges())
    if len(edges) == 0:
        return Gp

    k = max(1, int(len(edges) * ratio))
    for _ in range(min(k, len(edges))):
        e = random.choice(edges)
        if Gp.has_edge(*e):
            Gp.remove_edge(*e)
        edges.remove(e)
    return Gp


''' edge_jaccard: 
Measures how much the graph structure itself changed after perturbation.
(for sanity check)
1 → graphs have exactly the same edges
0 → graphs share no edges
'''
def edge_jaccard(G, Gp):
    E1, E2 = set(G.edges()), set(Gp.edges())
    if len(E1 | E2) == 0:
        return 1.0
    return len(E1 & E2) / len(E1 | E2)

# --------------------------------------------------
# Stability analysis
# --------------------------------------------------

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

def stability_analysis(graphs, X_orig, y, dim, edge_perturb=0.05):
    perturbations = {
        "both": perturb_edges,
        "add": perturb_add,
        "remove": perturb_remove
    }

    # Task A classifiers
    SVM_C = 75
    LOGREG_C = 3
    MLP_HIDDEN_LAYER_SIZES = (512, 256)
    MLP_MAX_ITER = 2000
    MLP_EARLY_STOPPING = True
    KNN_N_NEIGHBORS = 5

    classifiers = {
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=SVM_C, probability=True, random_state=42))
        ]),
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=LOGREG_C, max_iter=500, random_state=42))
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                                  max_iter=MLP_MAX_ITER,
                                  early_stopping=MLP_EARLY_STOPPING,
                                  random_state=42))
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS))
        ])
    }

    results = {}

    for name, perturb_func in perturbations.items():
        X_pert = []
        jaccards = []

        # -----------------------------
        # Perturb graphs & compute embeddings
        # -----------------------------
        for G in graphs:
            Gp = perturb_func(G, edge_perturb)

            # Edge Jaccard similarity
            jacc = edge_jaccard(G, Gp)
            jaccards.append(jacc)

            if Gp.number_of_nodes() < 2:
                X_pert.append(np.zeros(dim))
                continue

            #try:
            model = NetLSD(scale_steps=dim)
            model.fit([Gp])
            X_pert.append(model.get_embedding()[0])
            #except ValueError:
            #   X_pert.append(np.zeros(dim))

        X_pert = np.vstack(X_pert)

        # -----------------------------
        # Embedding stability
        # -----------------------------
        cosine = []
        rel_l2 = []
        for x, y_ in zip(X_orig, X_pert):
            cosine.append(cosine_similarity(x.reshape(1, -1), y_.reshape(1, -1))[0,0])
            rel_l2.append(np.linalg.norm(x - y_) / (np.linalg.norm(x) + 1e-9))

        # -----------------------------
        # Classification stability
        # -----------------------------
        acc_orig_list = []
        acc_pert_list = []

        for clf_name, clf in classifiers.items():
            clf.fit(X_orig, y)
            acc_orig_list.append(accuracy_score(y, clf.predict(X_orig)))
            acc_pert_list.append(accuracy_score(y, clf.predict(X_pert)))

        # Store results for this perturbation type
        results[name] = {
            "edge_jaccard": np.mean(jaccards),
            "cosine": np.mean(cosine),
            "rel_l2": np.mean(rel_l2),
            "acc_orig_avg": np.mean(acc_orig_list),
            "acc_pert_avg": np.mean(acc_pert_list),
            "acc_drop_avg": np.mean(np.array(acc_orig_list) - np.array(acc_pert_list))
        }

    return results

# --------------------------------------------------
# Run experiment
# --------------------------------------------------

def run_stability(datasets, dataset_path, dim=250, edge_perturb=0.05):
    if isinstance(edge_perturb, str) and edge_perturb.lower() == "all":
        perturb_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    else:
        perturb_values = [float(edge_perturb)]

    for name in datasets:
        print(f"\n===== Dataset: {name} =====")

        graphs, labels = load_dataset(name, dataset_path)
        X, t, mem, peak = generate_netlsd_embeddings(graphs, dim)
        #X, t, mem, peak = generate_netlsd_embeddings_wave(graphs, dim, kernel="wave") --> for wave kernel experiments

        print(f"Embeddings: {t:.2f}s | peak_mem {peak:.2f} MB")

        for perturb in perturb_values:
            print(f"\n--- Edge perturbation ratio: {perturb:.2f} ---")

            metrics = stability_analysis(
                graphs, X, labels, dim, edge_perturb=perturb
            )

            for perturb_type, metrics in metrics.items():
                print(f"\nPerturbation: {perturb_type}")
                print(f"Edge Jaccard        : {metrics['edge_jaccard']:.3f}")
                print(f"Cosine similarity   : {metrics['cosine']:.4f}")
                print(f"Relative L2 change  : {metrics['rel_l2']:.4f}")
                print(f"Accuracy (orig)     : {metrics['acc_orig_avg']:.4f}")
                print(f"Accuracy (perturbed): {metrics['acc_pert_avg']:.4f}")
                print(f"Accuracy drop       : {metrics['acc_drop_avg']:.4f}")

