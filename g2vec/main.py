# ==================================================
# Offline TUDataset: Classification + Clustering + Stability
# Graph2Vec + Simple Classifiers + Memory + Dimension
# ==================================================

import os
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import copy
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from karateclub import Graph2Vec
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from dotenv import load_dotenv
from utils import load_tudataset, mem_mb, perturb_graph, scale_features, embedding_stability

# ==================================================
# Load .env defaults
# ==================================================
load_dotenv()

G2V_WL_ITER = int(os.getenv("G2V_WL_ITER", 3))
G2V_EPOCHS = int(os.getenv("G2V_EPOCHS", 40))
G2V_LR = float(os.getenv("G2V_LR", 0.025))
G2V_SEED = int(os.getenv("G2V_SEED", 42))
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))

# ==================================================
# User input: dataset folder
# ==================================================
dataset_path = os.path.expanduser(input("Enter path to dataset folder (e.g., ~/study/datasets/IMDB-MULTI): ").strip())
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
dataset_name = os.path.basename(os.path.normpath(dataset_path))
print(f"\nLoading dataset: {dataset_name}")

# Load graphs
graphs, y = load_tudataset(dataset_path)
print(f"Loaded {len(graphs)} graphs.")

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
# Graph2Vec embeddings
# ==================================================
print(f"\nGenerating Graph2Vec embeddings (dim={dim})...")
mem_before_embed = mem_mb()
embed_start = time.time()

g2v = Graph2Vec(dimensions=dim, wl_iterations=G2V_WL_ITER, epochs=G2V_EPOCHS, learning_rate=G2V_LR, seed=G2V_SEED)
g2v.fit(graphs)
X = g2v.get_embedding()

embed_time = time.time() - embed_start
mem_after_embed = mem_mb()
embed_mem = mem_after_embed - mem_before_embed
print(f"Embedding generation time: {embed_time:.2f} s")
print(f"Embedding memory usage   : {embed_mem:.2f} MB")

# ==================================================
# Train / Test split
# ==================================================
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Feature scaling
X_train, X_test = scale_features(X_train, X_test)

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
kmeans = KMeans(n_clusters=len(np.unique(y)), n_init=10, random_state=RANDOM_SEED)
cluster_labels = kmeans.fit_predict(X)
ari_score = adjusted_rand_score(y, cluster_labels)
print(f"KMeans ARI: {ari_score:.4f}")

# t-SNE and UMAP visualization
tsne_emb = TSNE(n_components=2, random_state=RANDOM_SEED).fit_transform(X)
umap_emb = umap.UMAP(n_components=2, random_state=RANDOM_SEED).fit_transform(X)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(tsne_emb[:,0], tsne_emb[:,1], c=y, cmap='tab10')
plt.title("t-SNE")
plt.subplot(1,2,2)
plt.scatter(umap_emb[:,0], umap_emb[:,1], c=y, cmap='tab10')
plt.title("UMAP")
plt.show()

# ==================================================
# Optional Stability Analysis
# ==================================================
do_stability = input("\nDo you want to perform stability analysis? (y/n): ").strip().lower()

while do_stability == 'y':
    print("\nPerforming stability analysis...")
    perturbed_graphs = [perturb_graph(G) for G in graphs]

    g2v_perturbed = Graph2Vec(dimensions=dim, wl_iterations=2, epochs=15, learning_rate=0.05, seed=G2V_SEED)
    g2v_perturbed.fit(perturbed_graphs)
    X_perturbed = g2v_perturbed.get_embedding()

    sim = embedding_stability(X, X_perturbed)
    print(f"Average embedding stability score: {sim:.4f}")

    X_train_pert = X_perturbed[train_idx]
    X_test_pert = X_perturbed[test_idx]
    X_train_pert, X_test_pert = scale_features(X_train_pert, X_test_pert)

    clf_pert = copy.deepcopy(clf)
    clf_pert.fit(X_train_pert, y_train)
    y_pred_pert = clf_pert.predict(X_test_pert)
    acc_pert = accuracy_score(y_test, y_pred_pert)
    print(f"Classification accuracy on perturbed embeddings: {acc_pert:.4f}")
    print(f"Change in accuracy: {acc_pert - accuracy:.4f}")

    do_stability = input("\nDo you want to run another stability trial? (y/n): ").strip().lower()

print("\nStability analysis finished. Exiting script.")
