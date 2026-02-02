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

def _get_env_int(key, default):
    return int(os.getenv(key, default))

def _get_env_float(key, default):
    return float(os.getenv(key, default))

def _str_to_bool(s):
    return s.lower() == "true"

def get_classifier(name):
    load_dotenv()
    name = name.lower()
    
    classifiers = {
        "svm": lambda: (
            f"SVM ({os.getenv('SVM_KERNEL', 'rbf').upper()})",
            SVC(
                kernel=os.getenv("SVM_KERNEL", "rbf"),
                C=_get_env_float("SVM_C", 1.0),
                gamma=os.getenv("SVM_GAMMA", "scale"),
                probability=True,
                max_iter=_get_env_int("SVM_MAX_ITER", 3000)
            )
        ),
        "logreg": lambda: (
            "Logistic Regression",
            LogisticRegression(
                max_iter=_get_env_int("LOGREG_MAX_ITER", 3000),
                solver=os.getenv("LOGREG_SOLVER", "lbfgs"),
                C=_get_env_float("LOGREG_C", 1.0),
                n_jobs=-1
            )
        ),
        "mlp": lambda: (
            "MLP",
            MLPClassifier(
                hidden_layer_sizes=tuple(int(x) for x in os.getenv("MLP_HIDDEN_LAYER_SIZES", "128,128").split(",")),
                activation=os.getenv("MLP_ACTIVATION", "relu"),
                max_iter=_get_env_int("MLP_MAX_ITER", 500),
                early_stopping=_str_to_bool(os.getenv("MLP_EARLY_STOPPING", "True")),
                learning_rate_init=_get_env_float("MLP_LEARNING_RATE_INIT", 0.001),
                random_state=42
            )
        ),
        "knn": lambda: (
            "k-NN",
            KNeighborsClassifier(
                n_neighbors=_get_env_int("KNN_N_NEIGHBORS", 5),
                weights=os.getenv("KNN_WEIGHTS", "distance"),
                metric=os.getenv("KNN_METRIC", "minkowski")
            )
        ),
    }
    
    if name not in classifiers:
        raise ValueError("Invalid classifier. Choose from: svm, logreg, mlp, knn")
    
    return classifiers[name]()

def run_classification(X, y, classifier_name, test_size, seed,
                       dataset_name, dim, embed_time, embed_mem):

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=y, random_state=seed)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = scale_features(X_train, X_test)

    clf_name, clf = get_classifier(classifier_name)

    tracemalloc.start()
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    _, train_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    train_mem = train_peak / 1024**2

    start = time.time()
    y_pred = clf.predict(X_test)
    inference_time = time.time() - start

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    auc = None
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
        auc = (roc_auc_score(y_test, y_prob[:, 1]) if y_prob.shape[1] == 2 
               else roc_auc_score(y_test, y_prob, multi_class="ovr"))

    print(f"\n===== Classification Results ({dataset_name}) =====")
    print(f"Embedding dim        : {dim}")
    print(f"Classifier           : {clf_name}")
    print(f"Accuracy             : {accuracy:.4f}")
    print(f"Macro-F1             : {macro_f1:.4f}")
    print(f"Train time (s)       : {train_time:.4f}")
    print(f"Inference time (s)   : {inference_time:.6f}")
    print(f"Train memory (MB)    : {train_mem:.2f}")
    print(f"AUC (OvR)            : {auc:.4f}" if auc is not None else "AUC                  : N/A")
    print(f"Embedding time (s)   : {embed_time:.4f}")
    print(f"Embedding memory(MB) : {embed_mem:.2f}")
    print("==========================================")

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
