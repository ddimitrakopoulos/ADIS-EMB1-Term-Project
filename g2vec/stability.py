import copy
from karateclub import Graph2Vec
from sklearn.metrics import accuracy_score

from utils import perturb_graph, scale_features, embedding_stability

def run_stability(graphs, X, clf, y, train_idx, test_idx, dim, seed):
    perturbed = [perturb_graph(G) for G in graphs]

    g2v = Graph2Vec(dimensions=dim, wl_iterations=2, epochs=15, learning_rate=0.05, seed=seed)
    g2v.fit(perturbed)
    Xp = g2v.get_embedding()

    sim = embedding_stability(X, Xp)
    print(f"Embedding stability: {sim:.4f}")

    Xp_train = Xp[train_idx]
    Xp_test = Xp[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    Xp_train, Xp_test = scale_features(Xp_train, Xp_test)

    clf2 = copy.deepcopy(clf)
    clf2.fit(Xp_train, y_train)
    y_pred = clf2.predict(Xp_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on perturbed data: {acc:.4f}")
