import os
import time
import tracemalloc
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from utils import scale_features

def get_classifier(name):
    load_dotenv()
    name = name.lower()

    if name == "svm":
        svm_c = float(os.getenv("SVM_C", 1.0))
        svm_kernel = os.getenv("SVM_KERNEL", "rbf")
        svm_gamma = os.getenv("SVM_GAMMA", "scale")
        svm_max_iter = int(os.getenv("SVM_MAX_ITER", 3000))
        return f"SVM ({svm_kernel.upper()})", SVC(
            kernel=svm_kernel, C=svm_c, gamma=svm_gamma,
            probability=True, max_iter=svm_max_iter
        )
    elif name == "logreg":
        logreg_c = float(os.getenv("LOGREG_C", 1.0))
        logreg_max_iter = int(os.getenv("LOGREG_MAX_ITER", 3000))
        logreg_solver = os.getenv("LOGREG_SOLVER", "lbfgs")
        return "Logistic Regression", LogisticRegression(
            max_iter=logreg_max_iter, solver=logreg_solver, C=logreg_c, n_jobs=-1
        )
    elif name == "mlp":
        hidden_sizes_str = os.getenv("MLP_HIDDEN_LAYER_SIZES", "128,128")
        hidden_sizes = tuple(int(x) for x in hidden_sizes_str.split(","))
        mlp_max_iter = int(os.getenv("MLP_MAX_ITER", 500))
        mlp_early_stopping = os.getenv("MLP_EARLY_STOPPING", "True").lower() == "true"
        mlp_activation = os.getenv("MLP_ACTIVATION", "relu")
        mlp_lr_init = float(os.getenv("MLP_LEARNING_RATE_INIT", 0.001))
        return "MLP", MLPClassifier(
            hidden_layer_sizes=hidden_sizes, activation=mlp_activation,
            max_iter=mlp_max_iter, early_stopping=mlp_early_stopping,
            learning_rate_init=mlp_lr_init, random_state=42
        )
    elif name == "knn":
        knn_n_neighbors = int(os.getenv("KNN_N_NEIGHBORS", 5))
        knn_weights = os.getenv("KNN_WEIGHTS", "distance")
        knn_metric = os.getenv("KNN_METRIC", "minkowski")
        return "k-NN", KNeighborsClassifier(
            n_neighbors=knn_n_neighbors, weights=knn_weights, metric=knn_metric
        )
    else:
        raise ValueError("Invalid classifier. Choose from: svm, logreg, mlp, knn")

def run_classification(X, y, classifier_name, test_size, seed,
                       dataset_name, dim, embed_time, embed_mem):

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=seed
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train, X_test = scale_features(X_train, X_test)

    clf_name, clf = get_classifier(classifier_name)

    # Measure training time and memory with tracemalloc
    tracemalloc.start()
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    _, train_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    train_mem = train_peak / 1024**2  # Convert to MB

    # Measure inference time
    start = time.time()
    y_pred = clf.predict(X_test)
    inference_time = time.time() - start

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # AUC calculation
    auc = None
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_test, y_prob[:,1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

    print(f"\n===== Classification Results ({dataset_name}) =====")
    print(f"Embedding dim        : {dim}")
    print(f"Classifier           : {clf_name}")
    print(f"Accuracy             : {accuracy:.4f}")
    print(f"Macro-F1             : {macro_f1:.4f}")
    print(f"Train time (s)       : {train_time:.4f}")
    print(f"Inference time (s)   : {inference_time:.6f}")
    print(f"Train memory (MB)    : {train_mem:.2f}")
    if auc is not None:
        print(f"AUC (OvR)            : {auc:.4f}")
    else:
        print("AUC                  : N/A")
    print(f"Embedding time (s)   : {embed_time:.4f}")
    print(f"Embedding memory(MB) : {embed_mem:.2f}")
    print("==========================================")

    # Return metrics dict for averaging
    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "auc": auc if auc is not None else 0.0,
        "train_time": train_time,
        "inference_time": inference_time,
        "train_mem": train_mem,
        "embed_time": embed_time,
        "embed_mem": embed_mem,
    }

    return clf, train_idx, test_idx, metrics
