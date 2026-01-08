import time
import psutil
import os
import numpy as np
import torch
import networkx as nx
import tracemalloc

from tqdm import tqdm
from karateclub import NetLSD
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

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
    Loads a graph dataset from PyTorch Geometric 
    and converts it to NetworkX objects, 
    which KarateClub’s NetLSD expects.

    """
    dataset = TUDataset(root=root, name=dataset_name) # PyG dataset
    graphs = [] # NetworkX graphs
    labels = [] # graph labels

    for data in dataset: # loops over each graph in the dataset
        g = to_networkx(data, to_undirected=True) 
        # to_undirected=True ensures edges are undirected, which NetLSD expects.
        graphs.append(g)
        labels.append(int(data.y)) # graph label in data.y

    # returns also NumPy array of graph labels, ready for classifiers
    return graphs, np.array(labels)


# -----------------------------
# Generate NetLSD embeddings
# -----------------------------

def generate_netlsd_embeddings(graphs, dim=128):
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
        # choose a reasonable number of eigenvalues to be calculated based on the graph size
        approximations = max(1, min(200, (num_nodes - 2) // 2)) 
        model = NetLSD(scale_steps=dim, approximations=approximations)
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

def evaluate_classifiers(X, y):
    """
    X  -> feature matrix: embeddings from NetLSD, shape (num_graphs, dim)
    y  -> graph labels: actual labels from the dataset, shape (num_graphs,)
    """

    # 80% of graphs go into training, 20% into testing
    # stratified split for ensuring same class distribution in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    classifiers = {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, random_state=42))
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64),
                                  max_iter=500,
                                  random_state=42))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    results = {}

    for clf_name, clf in classifiers.items():
        print(f"\nTraining {clf_name}...")
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

        results[clf_name] = {
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "train_time": train_time
        }

    return results

# -----------------------------
# Main experiment loop
# -----------------------------

def run_experiment():
    datasets = ["MUTAG", "ENZYMES", "IMDB-MULTI"]
    embedding_dim = 128
    results = {}

    for dataset_name in datasets:
        print(f"\n===== Dataset: {dataset_name} =====")
        graphs, labels = load_dataset(dataset_name)

        print("Generating NetLSD embeddings...")
        embeddings, total_time, embed_mem, peak_mem, avg_time_per_graph = generate_netlsd_embeddings(
            graphs, dim=embedding_dim
        )        
        print(f"Total embedding time (s)       : {total_time:.2f}")
        print(f"Embedding memory usage (MB) : {embed_mem:.2f}")
        print(f"Peak embedding memory (MB)     : {peak_mem:.2f}")
        print(f"Average embedding time/graph (s): {avg_time_per_graph:.4f}")

        print("Training classifiers...")
        metrics = evaluate_classifiers(embeddings, labels)

        results[dataset_name] = {
            "embedding_dim": embedding_dim,
            "embedding_time_sec": total_time,
            "embedding_memory_mb": embed_mem,  # X.nbytes in MB
            "embedding_peak_memory_mb": peak_mem,
            "avg_embedding_time_per_graph_sec": avg_time_per_graph,
            "classifiers": metrics
        }

        print(f"\nResults for {dataset_name}:")
        for clf_name, clf_metrics in metrics.items():
            print(f"  {clf_name}:")
            for k, v in clf_metrics.items():
                print(f"    {k}: {v}")

    return results

# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    run_experiment()
