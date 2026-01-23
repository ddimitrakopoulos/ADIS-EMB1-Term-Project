import time
import psutil
import os
import numpy as np
import torch
import networkx as nx
import tracemalloc

from tqdm import tqdm
from karateclub import NetLSD

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------
# Load dataset and convert to NetworkX
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

    # track all memory allocations in Python 
    # during the block of code where we keep tracemalloc active
    tracemalloc.start()               # start tracemalloc first

    start_time = time.time()

    for g in graphs:
        num_nodes = g.number_of_nodes()
        # avoid errors for small graphs based on how eigenvalues are calculated at the karateclub impl of netlsd
        
        if num_nodes < 6: 
            # there are only 4 graphs in enzymes dataset with less than 6 nodes 
            # so this doesnt affect the accuracy output much
            embeddings.append(np.zeros(dim))
            continue
        '''# choose a reasonable number of eigenvalues to be calculated based on the graph size
        approximations = max(1, min(200, (num_nodes - 2) // 2)) '''
        model = NetLSD(scale_steps=dim) #, approximations=approximations)
        model.fit([g])
        embeddings.append(model.get_embedding()[0])

    total_time = time.time() - start_time

    X = np.array(embeddings) # holding all the NetLSD embeddings for all graphs
    # X.nbytes gives the number of bytes used in memory by X array
    embed_mem = X.nbytes / 1024**2  # memory in MB for embedding array (instead of mem_mb function)
    _, peak_mem = tracemalloc.get_traced_memory() # maximum memory Python allocated at any point while generating embeddings
    tracemalloc.stop()

    return (
        X,
        total_time,                # total time it took to generate embeddings for all graphs
        embed_mem,                 
        peak_mem / 1024**2,        
        total_time / max(1, len(graphs)) # average time spent generating the NetLSD embedding per graph
    )

# -----------------------------
# Classification using SVM, MLP, Logistic Regression, KNN
# -----------------------------

def evaluate_classifiers(X, y, classifier=None):
    """
    X  -> feature matrix: embeddings from NetLSD, shape (num_graphs, dim)
    y  -> graph labels: actual labels from the dataset, shape (num_graphs,)
    classifier -> only train and evaluate this classifier if specified
    """

    # 80% of graphs go into training, 20% into testing
    # stratified split for ensuring same class distribution in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    classifiers = {
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, random_state=42))
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64),
                                  max_iter=500,
                                  random_state=42))
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    results = {}
    clf_names = [classifier] if classifier else classifiers.keys()
    for clf_name in clf_names:
        clf = classifiers[clf_name]
        print(f"\nTraining {clf_name.upper()}...")
        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = clf.predict(X_test) # predicted class labels
        y_prob = None # predicted class probabilities, if the classifier supports it (for AUC)
        try:
            y_prob = clf.predict_proba(X_test)
        except:
            pass

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        auc = None
        if y_prob is not None:
            try:
                if len(np.unique(y)) == 2: # if there are only 2 classes (binary classification)
                    auc = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
            except:
                auc = None

        results[clf_name.upper()] = {
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "train_time": train_time
        }

    return results

# -----------------------------
# Main experiment loop
# -----------------------------

def run_classification_experiment(dataset_name, dataset_path, dim=250, classifier="svm"):
    print(f"\n===== Dataset: {dataset_name} =====")
    graphs, labels = load_dataset(dataset_name, root=dataset_path)

    print("Generating NetLSD embeddings...")
    embeddings, total_time, embed_mem, peak_mem, avg_time_per_graph = generate_netlsd_embeddings(
        graphs, dim=dim
    )        

    ''' debugging info
    print("NetLSD embeddings (first few graphs):")
    for i, emb in enumerate(embeddings[:10]):
        print(f"Graph {i}: {emb}")
    '''

    print(f"Total embedding time (s)       : {total_time:.2f}")
    print(f"Embedding memory usage (MB)    : {embed_mem:.2f}")
    print(f"Peak embedding memory (MB)     : {peak_mem:.2f}")
    print(f"Average embedding time/graph (s): {avg_time_per_graph:.4f}")

    print("Training classifier...")
    metrics = evaluate_classifiers(embeddings, labels, classifier=classifier)

    print(f"\nResults for {dataset_name}:")
    clf_metrics = metrics.get(classifier.upper())
    print(f"  {classifier.upper()}:")
    for k, v in clf_metrics.items():
        print(f"    {k}: {v}")

    return metrics
