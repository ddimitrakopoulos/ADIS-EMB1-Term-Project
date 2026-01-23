# ==================================================
# NetLSD Stability Analysis: MUTAG, ENZYMES, IMDB-MULTI
# ==================================================

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

# --------------------------------------------------
# Load TUDataset from raw files
# --------------------------------------------------

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


# --------------------------------------------------
# NetLSD embeddings
# --------------------------------------------------
def generate_netlsd_embeddings(graphs, dim):
    tracemalloc.start()
    start = time.time()

    embeddings = []

    for g in graphs:
        if g.number_of_nodes() < 2:
            embeddings.append(np.zeros(dim))
            continue

        #approximations = min(200, max(1, g.number_of_nodes() // 2))
        model = NetLSD(scale_steps=dim) #, approximations=approximations)
        model.fit([g])
        embeddings.append(model.get_embedding()[0])

    X = np.vstack(embeddings)

    total_time = time.time() - start
    mem = X.nbytes / 1024**2
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return X, total_time, mem, peak / 1024**2


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

# ==================================================
# Alternative -> Perturb a graph for stability analysis
# ==================================================

def perturb_graph(G, edge_perturb_ratio=0.1, shuffle_node_labels=False):
    G_perturbed = deepcopy(G)
    num_edges = G_perturbed.number_of_edges()
    num_remove = int(edge_perturb_ratio * num_edges)
    edges = list(G_perturbed.edges())
    edges_to_remove = random.sample(edges, min(num_remove, len(edges)))
    G_perturbed.remove_edges_from(edges_to_remove)
    nodes = list(G_perturbed.nodes())
    added = 0 
    '''because of variable added, if k edges are removed, exactly k are added
    at perturb_edges this doesnt get checked, so when trying to add a random edge
    if this edge already exists at GP, it will continue to the next iteration
    so the peturbed graph's size is roughly similar to the original's
    and much more noise is introduced
    '''
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

def stability_analysis(graphs, X_orig, y, dim, edge_perturb):
    X_pert = []
    jaccards = []

    for G in graphs:
        Gp = perturb_edges(G, edge_perturb)
        #Gp = perturb_graph(G, edge_perturb)
        jaccards.append(edge_jaccard(G, Gp))

        if Gp.number_of_nodes() < 2:
            X_pert.append(np.zeros(dim))
            continue

        #approximations = min(200, max(1, Gp.number_of_nodes() // 2))
        model = NetLSD(scale_steps=dim) #, approximations=approximations)
        model.fit([Gp])
        X_pert.append(model.get_embedding()[0])

    X_pert = np.vstack(X_pert)

    # ---- embedding stability ----
    cosine = []
    rel_l2 = []

    for x, y_ in zip(X_orig, X_pert):
        '''cosine similarity:
        x → original NetLSD embedding
        y → perturbed embedding
        1 → embeddings point in the same direction → very stable (preffered)
        0 → embeddings are orthogonal → completely different
        -1 → embeddings are opposite
        It's important cause NetLSD embeddings are directional vectors.
        Cosine similarity shows how well the “shape” of the graph is preserved.
        '''
        cosine.append(cosine_similarity(x.reshape(1,-1), y_.reshape(1,-1))[0,0])
        '''Relative L2 Distance:
        Computes Euclidean distance between the original and perturbed embeddings.
        Normalized by the magnitude of the original embedding to get relative change.
        Captures magnitude differences, unlike cosine which only captures direction.
        Helps detect if embeddings shrink/expand under perturbation.
        0 → no change, Higher → bigger perturbation effect
        '''
        rel_l2.append(np.linalg.norm(x - y_) / (np.linalg.norm(x) + 1e-9))

    # ---- classification stability ----
    ''' note to self:
    introduce more classifiers and take the average accuracy 
    though classification results at task A didnt show much 
    difference between different classifiers
    '''
    clf = SVC(kernel="linear") # no train/test split, purely evaluating stability 
    # --> introducing split would mess with generalization noise and hide true stability effects
    clf.fit(X_orig, y)
    acc_orig = accuracy_score(y, clf.predict(X_orig))
    acc_pert = accuracy_score(y, clf.predict(X_pert))

    return {
        "edge_jaccard": np.mean(jaccards),
        "cosine": np.mean(cosine),
        "rel_l2": np.mean(rel_l2),
        "acc_orig": acc_orig,
        "acc_pert": acc_pert,
        "acc_drop": acc_orig - acc_pert
    }


# --------------------------------------------------
# Run experiment
# --------------------------------------------------

def run_stability(datasets, dim, dataset_path, edge_perturb=0.05):
    for name in datasets:
        print(f"\n===== Dataset: {name} =====")

        graphs, labels = load_dataset(name, dataset_path)
        X, t, mem, peak = generate_netlsd_embeddings(graphs, dim)

        print(f"Embeddings: {t:.2f}s | mem {mem:.2f} MB")

        metrics = stability_analysis(
            graphs, X, labels, dim, edge_perturb
        )

        print(f"Edge Jaccard        : {metrics['edge_jaccard']:.3f}")
        print(f"Cosine similarity   : {metrics['cosine']:.4f}")
        print(f"Relative L2 change  : {metrics['rel_l2']:.4f}")
        print(f"Accuracy (orig)     : {metrics['acc_orig']:.4f}")
        print(f"Accuracy (perturbed): {metrics['acc_pert']:.4f}")
        print(f"Accuracy drop       : {metrics['acc_drop']:.4f}")
