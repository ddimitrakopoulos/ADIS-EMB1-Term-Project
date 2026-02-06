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

import netlsd # for experiments with the wave kernel

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
        peak_mem / 1024**2,
        total_time / max(1, num_graphs)
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

    return X, total_time, embed_mem, peak_mem / 1024**2, total_time / max(1, len(graphs))


def evaluate_classifiers(X, y, classifier=None, n_trials=100):
    # ==================================================
    # Classifier hyperparameters
    # ==================================================
    SVM_C = 75
    LOGREG_C = 3
    MLP_HIDDEN_LAYER_SIZES = (512, 256)
    MLP_MAX_ITER = 2000
    MLP_EARLY_STOPPING = True
    KNN_N_NEIGHBORS = 5

    # ==================================================
    # Classifiers
    # ==================================================
    classifiers = {
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=SVM_C,
                probability=True,
                random_state=42
            ))
        ]),

        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=LOGREG_C,
                max_iter=500,
                random_state=42
            ))
        ]),

        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
                max_iter=MLP_MAX_ITER,
                early_stopping=MLP_EARLY_STOPPING,
                random_state=42
            ))
        ]),

        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=KNN_N_NEIGHBORS
            ))
        ])
    }
    
    if classifier == "all" or classifier is None:
        clf_names = list(classifiers.keys())
    else:
        clf_names = [classifier]

    # -------------------------------------------------
    # Storage for metrics across trials
    # -------------------------------------------------

    metrics_storage = {
        name.upper(): {
            "accuracy": [],
            "f1": [],
            "auc": [],
            "train_time": [],
            "inference_time": []
        }
        for name in clf_names
    }

    # -------------------------------------------------
    # Repeated classification experiments
    # -------------------------------------------------
    
    test_size = 0.2 # stays the same at each trial 
    rng = np.random.default_rng(42) # master seed controls randomness

    for trial in range(n_trials):

        random_state = rng.integers(0, 1_000_000)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        for clf_name in clf_names:
            clf = classifiers[clf_name]

            # -------- Training time --------
            start_time = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # -------- Inference time --------
            start_time = time.time()
            y_pred = clf.predict(X_test)
            inference_time = time.time() - start_time

            # -------- Probabilities --------
            try:
                y_prob = clf.predict_proba(X_test)
            except AttributeError:
                y_prob = None

            # -------- Metrics --------
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")

            auc = None
            if y_prob is not None:
                try:
                    if len(np.unique(y)) == 2:
                        auc = roc_auc_score(y_test, y_prob[:, 1])
                    else:
                        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                except:
                    auc = None

            store = metrics_storage[clf_name.upper()]
            store["accuracy"].append(acc)
            store["f1"].append(f1)
            store["train_time"].append(train_time)
            store["inference_time"].append(inference_time)
            if auc is not None:
                store["auc"].append(auc)

    # -------------------------------------------------
    #  Averages
    # -------------------------------------------------
    results = {}
    avg_all = {"accuracy": [], "f1": [], "auc": [], "train_time": [], "inference_time": [] }

    for clf_name, vals in metrics_storage.items():
        results[clf_name] = {
            "accuracy": float(np.mean(vals["accuracy"])),
            "f1": float(np.mean(vals["f1"])),
            "auc": float(np.mean(vals["auc"])) if vals["auc"] else None,
            "train_time": float(np.mean(vals["train_time"])),
            "inference_time": float(np.mean(vals["inference_time"]))
        }

        avg_all["accuracy"].append(results[clf_name]["accuracy"])
        avg_all["f1"].append(results[clf_name]["f1"])
        avg_all["train_time"].append(results[clf_name]["train_time"])
        avg_all["inference_time"].append(results[clf_name]["inference_time"])
        if results[clf_name]["auc"] is not None:
            avg_all["auc"].append(results[clf_name]["auc"])

    # -------------------------------------------------
    # Final average across classifiers
    # -------------------------------------------------
    results["AVERAGE"] = {
        "accuracy": float(np.mean(avg_all["accuracy"])),
        "f1": float(np.mean(avg_all["f1"])),
        "auc": float(np.mean(avg_all["auc"])) if avg_all["auc"] else None,
        "train_time": float(np.mean(avg_all["train_time"])),
        "inference_time": float(np.mean(avg_all["inference_time"]))
    }

    return results

# -----------------------------
# Main experiment loop
# -----------------------------

def run_classification_experiment(dataset_name, dataset_path, dim, classifier="svm", kernel="heat"):
    print(f"\n===== Dataset: {dataset_name} =====")
    graphs, labels = load_dataset(dataset_name, root=dataset_path)

    print("Generating NetLSD embeddings...")
    
    embeddings, total_time, embed_mem, peak_mem, avg_time_per_graph = generate_netlsd_embeddings(
        graphs, dim=dim
    )     
    
    ''' For the wave kernel utilization:
    embeddings, total_time, embed_mem, peak_mem, avg_time_per_graph = generate_netlsd_embeddings_wave(
            graphs, dim=dim, kernel="wave"
        )
    '''
    
    print(f"Total embedding time (s)       : {total_time:.2f}")
    print(f"Embedding memory usage (MB)    : {embed_mem:.2f}")
    print(f"Peak embedding memory (MB)     : {peak_mem:.2f}")
    print(f"Average embedding time/graph (s): {avg_time_per_graph:.4f}")

    print("Training classifier...")
    metrics = evaluate_classifiers(embeddings, labels, classifier=classifier)

    print(f"\nResults for {dataset_name}:")

    for clf_name, clf_metrics in metrics.items():
        print(f"  {clf_name}:")
        for k, v in clf_metrics.items():
            print(f"    {k}: {v}")

    return metrics
