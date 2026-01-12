import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from utils import scale_features, mem_mb

def get_classifier(name):
    name = name.lower()

    if name == "svm":
        return "SVM (RBF)", SVC(kernel="rbf", C=1.0, gamma="scale",
                                probability=True, max_iter=3000)
    elif name == "logreg":
        return "Logistic Regression", LogisticRegression(
            max_iter=3000, solver="lbfgs", C=1.0, n_jobs=-1
        )
    elif name == "mlp":
        return "MLP", MLPClassifier(
            hidden_layer_sizes=(128,128), activation="relu",
            max_iter=500, early_stopping=True, random_state=42
        )
    elif name == "knn":
        return "k-NN", KNeighborsClassifier(
            n_neighbors=5, weights="distance"
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

    mem_before = mem_mb()
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    mem_after = mem_mb()
    train_mem = mem_after - mem_before

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n===== Classification Results ({dataset_name}) =====")
    print(f"Embedding dim        : {dim}")
    print(f"Classifier           : {clf_name}")
    print(f"Accuracy             : {accuracy:.4f}")
    print(f"Macro-F1             : {macro_f1:.4f}")
    print(f"Train time (s)       : {train_time:.4f}")
    print(f"Train memory (MB)    : {train_mem:.2f}")

    if hasattr(clf, "predict_proba"):
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

    return clf, train_idx, test_idx
