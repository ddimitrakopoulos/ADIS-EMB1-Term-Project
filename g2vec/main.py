# ==================================================
# Offline TUDataset: Classification + Clustering + Stability
# Graph2Vec + Simple Classifiers + Memory + Dimension
# ==================================================

import os
import time
import psutil
import numpy as np
import networkx as nx
import random
import copy
import warnings
from karateclub import Graph2Vec
# ===========================
# Suppress UMAP warnings
# ===========================
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import umap



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
print("\nChoose Graph2Vec embedding dimension (e.g. 64, 128, 256):")
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
edges = np.loadtxt(os.path.join(dataset_path, f"{dataset_name}_A.txt"), delimiter=",", dtype=int) - 1
graph_indicator = np.loadtxt(os.path.join(dataset_path, f"{dataset_name}_graph_indicator.txt"), dtype=int) - 1
graph_labels = np.loadtxt(os.path.join(dataset_path, f"{dataset_name}_graph_labels.txt"), dtype=int)
node_labels_path = os.path.join(dataset_path, f"{dataset_name}_node_labels.txt")
node_labels = np.loadtxt(node_labels_path, dtype=int) if os.path.exists(node_labels_path) else None

num_graphs = graph_labels.shape[0]
print("\nBuilding NetworkX graphs (Graph2Vec-safe)...")
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
# Graph2Vec embeddings
# ==================================================
print(f"\nGenerating Graph2Vec embeddings (dim={dim})...")
mem_before_embed = mem_mb()
embed_start = time.time()

g2v = Graph2Vec(dimensions=dim, wl_iterations=3, epochs=40, learning_rate=0.025, seed=42)
g2v.fit(graphs)
X = g2v.get_embedding()

embed_time = time.time() - embed_start
mem_after_embed = mem_mb()
embed_mem = mem_after_embed - mem_before_embed
print(f"Embedding generation time: {embed_time:.2f} s")
print(f"Embedding memory usage   : {embed_mem:.2f} MB")

# ==================================================
# Train / Test split with indices
# ==================================================
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

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
    clf = MLPClassifier(hidden_layer_sizes=(128,128), activation="relu", max_iter=500, early_stopping=True, random_state=42)
elif classifier_choice == 4:
    clf_name = "k-NN"
    clf = KNeighborsClassifier(n_neighbors=5, weights="distance")

print(f"\nTraining classifier: {clf_name}")

# ==================================================
# Train classifier
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

# Binary-safe AUC computation
if classifier_choice in [1,2,3]:
    y_prob = clf.predict_proba(X_test)
    if y_prob.shape[1] == 2:
        auc = roc_auc_score(y_test, y_prob[:,1])
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    print(f"AUC (OvR)            : {auc:.4f}")
else:
    print("AUC                  : N/A")

print(f"Embedding time (s)   : {embed_time:.4f}")
print(f"Embedding memory(MB) : {embed_mem:.2f}")
print("==========================================")

# ==================================================
# Clustering
# ==================================================
print("\nPerforming clustering...")
kmeans = KMeans(n_clusters=len(np.unique(y)), n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X)
ari_score = adjusted_rand_score(y, cluster_labels)
print(f"KMeans ARI: {ari_score:.4f}")

# t-SNE and UMAP visualization
tsne_emb = TSNE(n_components=2, random_state=42).fit_transform(X)
umap_emb = umap.UMAP(n_components=2, random_state=42).fit_transform(X)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(tsne_emb[:,0], tsne_emb[:,1], c=y, cmap='tab10')
plt.title("t-SNE")
plt.subplot(1,2,2)
plt.scatter(umap_emb[:,0], umap_emb[:,1], c=y, cmap='tab10')
plt.title("UMAP")
plt.show()

# ==================================================
# Optional Stability Analysis (interactive)
# ==================================================
do_stability = input("\nDo you want to perform stability analysis? (y/n): ").strip().lower()

while do_stability == 'y':
    print("\nPerforming stability analysis...")

    def perturb_graph(G, edge_perturb_ratio=0.1, shuffle_node_labels=True):
        """Randomly perturb a graph"""
        G_perturbed = copy.deepcopy(G)
        num_edges = G_perturbed.number_of_edges()
        num_remove = int(edge_perturb_ratio * num_edges)
        edges = list(G_perturbed.edges())
        edges_to_remove = random.sample(edges, min(num_remove, len(edges)))
        G_perturbed.remove_edges_from(edges_to_remove)
        # Add random edges
        nodes = list(G_perturbed.nodes())
        added = 0
        while added < num_remove:
            u, v = random.sample(nodes, 2)
            if not G_perturbed.has_edge(u, v):
                G_perturbed.add_edge(u, v)
                added += 1
        # Shuffle node labels
        if shuffle_node_labels and nx.get_node_attributes(G_perturbed, "label"):
            labels = list(nx.get_node_attributes(G_perturbed, "label").values())
            random.shuffle(labels)
            for i, node in enumerate(G_perturbed.nodes()):
                G_perturbed.nodes[node]["label"] = labels[i]
        return G_perturbed

    perturbed_graphs = [perturb_graph(G) for G in graphs]

    # Recompute embeddings
    g2v_perturbed = Graph2Vec(dimensions=dim, wl_iterations=2, epochs=15, learning_rate=0.05, seed=42)
    g2v_perturbed.fit(perturbed_graphs)
    X_perturbed = g2v_perturbed.get_embedding()

    # Compute embedding stability
    sim = np.diag(cosine_similarity(X, X_perturbed))
    stability_score = np.mean(sim)
    print(f"Average embedding stability score: {stability_score:.4f}")

    # Classification on perturbed embeddings using original train/test split
    X_train_pert = X_perturbed[train_idx]
    X_test_pert = X_perturbed[test_idx]

    scaler_pert = StandardScaler()
    X_train_pert = scaler_pert.fit_transform(X_train_pert)
    X_test_pert = scaler_pert.transform(X_test_pert)

    clf_pert = copy.deepcopy(clf)
    clf_pert.fit(X_train_pert, y_train)
    y_pred_pert = clf_pert.predict(X_test_pert)
    acc_pert = accuracy_score(y_test, y_pred_pert)
    print(f"Classification accuracy on perturbed embeddings: {acc_pert:.4f}")
    print(f"Change in accuracy: {acc_pert - accuracy:.4f}")

    do_stability = input("\nDo you want to run another stability trial? (y/n): ").strip().lower()

print("\nStability analysis finished. Exiting script.")
