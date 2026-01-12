import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

def run_clustering(X, y, seed):
    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)

    ari = adjusted_rand_score(y, labels)
    print(f"KMeans ARI: {ari:.4f}")

    tsne = TSNE(n_components=2, random_state=seed).fit_transform(X)
    umap_emb = umap.UMAP(n_components=2, random_state=seed).fit_transform(X)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.scatter(tsne[:,0], tsne[:,1], c=y, cmap="tab10")
    plt.title("t-SNE")

    plt.subplot(1,2,2)
    plt.scatter(umap_emb[:,0], umap_emb[:,1], c=y, cmap="tab10")
    plt.title("UMAP")

    plt.show()
