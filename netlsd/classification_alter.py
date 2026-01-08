# ==================================================
# Offline TUDataset Classification
# NetLSD + Simple Classifiers + Memory + Dimension
# Works for ENZYMES, IMDB-MULTI, MUTAG, etc.
# ==================================================

import os
import time
import psutil
import numpy as np
import networkx as nx
import tracemalloc

from karateclub import NetLSD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# ==================================================
# Memory helper
# ==================================================
process = psutil.Process(os.getpid())

def mem_mb():
    return process.memory_info().rss / 1024 / 1024


# ==================================================
# User input: dataset folder
# ==================================================
dataset_path = os.path.expanduser(input("Enter path to dataset folder (e.g., ~/study/datasets/IMDB-MULTI): ").strip())

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

dataset_name = os.path.basename(os.path.normpath(dataset_path))
print(f"\nLoading dataset: {dataset_name}")


# ==================================================
# User input: classifier
# ==================================================
print("\nChoose classifier to use:")
print("1 = SVM (RBF)")
print("2 = Logistic Regression")
print("3 = MLP")
print("4 = k-NN")

while True:
    try:
        classifier_choice = int(input("Enter a number (1-4): "))
        if classifier_choice in [1, 2, 3, 4]:
            break
        print("Invalid choice.")
    except ValueError:
        print("Enter a number.")


# ==================================================
# User input: embedding dimension
# ==================================================
print("\nChoose NetLSD embedding dimension (e.g. 64, 128, 256):")

while True:
    try:
        dim = int(input("Enter embedding dimension: "))
        if dim > 0:
            break
        print("Dimension must be positive.")
    except ValueError:
        print("Enter an integer.")


# ==================================================
# Load TUDataset files dynamically
# ==================================================
edges = np.loadtxt(
    os.path.join(dataset_path, f"{dataset_name}_A.txt"),
    delimiter=",",
    dtype=int
) - 1

graph_indicator = np.loadtxt(
    os.path.join(dataset_path, f"{dataset_name}_graph_indicator.txt"),
    dtype=int
) - 1

graph_labels = np.loadtxt(
    os.path.join(dataset_path, f"{dataset_name}_graph_labels.txt"),
    dtype=int
)

node_labels_path = os.path.join(dataset_path, f"{dataset_name}_node_labels.txt")
node_labels = (
    np.loadtxt(node_labels_path, dtype=int)
    if os.path.exists(node_labels_path)
    else None
)

num_graphs = graph_labels.shape[0]

print("\nBuilding NetworkX graphs (NetLSD-safe)...")
graphs = []

for g_id in range(num_graphs):
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

y = graph_labels
print(f"Loaded {len(graphs)} graphs.")


# ==================================================
# NetLSD embeddings
# ==================================================
print(f"\nGenerating NetLSD embeddings (dim={dim})...")
mem_before_embed = mem_mb()
embed_start = time.time()

tracemalloc.start()
embeddings = []

for g in graphs:
    num_nodes = g.number_of_nodes()

    # Handle tiny graphs
    if num_nodes < 6:
        embeddings.append(np.zeros(dim))
        continue

    # Safe approximation
    approximations = min(200, max(1, (num_nodes - 2) // 2))

    model = NetLSD(scale_steps=dim, approximations=approximations)
    model.fit([g])
    embeddings.append(model.get_embedding()[0])

X = np.array(embeddings)
embed_time = time.time() - embed_start
mem_after_embed = mem_mb()
embed_mem = mem_after_embed - mem_before_embed
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Embedding generation time: {embed_time:.2f} s")
print(f"Embedding memory usage   : {embed_mem:.2f} MB")
print(f"Peak memory usage (tracemalloc) : {peak_mem / 1024 ** 2:.2f} MB")


# ==================================================
# Train / Test split
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# ==================================================
# Feature scaling
# ==================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==================================================
# Classifier selection
# ==================================================
if classifier_choice == 1:
    clf_name = "SVM (RBF)"
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, max_iter=3000)

elif classifier_choice == 2:
    clf_name = "Logistic Regression"
    clf = LogisticRegression(max_iter=3000, solver="lbfgs", C=1.0, n_jobs=-1)

elif classifier_choice == 3:
    clf_name = "MLP"
    clf = MLPClassifier(hidden_layer_sizes=(128, 128), activation="relu",
                        max_iter=500, early_stopping=True, random_state=42)

elif classifier_choice == 4:
    clf_name = "k-NN"
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance")

print(f"\nTraining classifier: {clf_name}")


# ==================================================
# Train classifier (memory-aware)
# ==================================================
mem_before_train = mem_mb()
train_start = time.time()
clf.fit(X_train, y_train)
train_time = time.time() - train_start
mem_after_train = mem_mb()
train_mem = mem_after_train - mem_before_train


# ==================================================
# Evaluation
# ==================================================
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n===== Classification Results ({}) =====".format(dataset_name))
print(f"Embedding dim        : {dim}")
print(f"Classifier           : {clf_name}")
print(f"Accuracy             : {accuracy:.4f}")
print(f"Macro-F1             : {macro_f1:.4f}")
print(f"Train time (s)       : {train_time:.4f}")
print(f"Train memory (MB)    : {train_mem:.2f}")

auc = "N/A"
if classifier_choice in [1, 2, 3]:  # classifiers that support probability
    try:
        y_prob = clf.predict_proba(X_test)
        if len(np.unique(y)) == 2:  # binary classification
            # Only take probability of positive class
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:  # multi-class classification
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    except Exception as e:
        auc = f"N/A (error: {e})"

print(f"AUC                  : {auc}")
print(f"Embedding time (s)   : {embed_time:.4f}")
print(f"Embedding memory(MB) : {embed_mem:.2f}")
print("==========================================")
