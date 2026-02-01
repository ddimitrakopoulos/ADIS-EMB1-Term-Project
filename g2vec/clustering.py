import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

def run_clustering(X, y, seed, plot_3d=False):
    num_classes = len(np.unique(y))
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=num_classes, random_state=seed, n_init=20)
    labels_kmeans = kmeans.fit_predict(X)
    ari_kmeans = adjusted_rand_score(y, labels_kmeans)
    print(f"KMeans ARI: {ari_kmeans:.4f}")

    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=num_classes,
                                  assign_labels='cluster_qr',
                                  random_state=seed,
                                  affinity='nearest_neighbors')
    labels_spectral = spectral.fit_predict(X)
    ari_spectral = adjusted_rand_score(y, labels_spectral)
    print(f"Spectral Clustering ARI: {ari_spectral:.4f}")

    # t-SNE and UMAP visualization
    n_components = 3 if plot_3d else 2
    tsne = TSNE(n_components=n_components, random_state=seed).fit_transform(X)
    umap_emb = umap.UMAP(n_components=n_components, random_state=seed).fit_transform(X)

    plt.figure(figsize=(12,5))

    if plot_3d:
        from mpl_toolkits.mplot3d import Axes3D
        ax1 = plt.subplot(1,2,1, projection='3d')
        ax1.scatter(tsne[:,0], tsne[:,1], tsne[:,2], c=y, cmap="tab10")
        ax1.set_title("t-SNE 3D")
        ax2 = plt.subplot(1,2,2, projection='3d')
        ax2.scatter(umap_emb[:,0], umap_emb[:,1], umap_emb[:,2], c=y, cmap="tab10")
        ax2.set_title("UMAP 3D")
    else:
        plt.subplot(1,2,1)
        plt.scatter(tsne[:,0], tsne[:,1], c=y, cmap="tab10")
        plt.title("t-SNE")
        plt.subplot(1,2,2)
        plt.scatter(umap_emb[:,0], umap_emb[:,1], c=y, cmap="tab10")
        plt.title("UMAP")

    plt.show()

    return ari_kmeans, ari_spectral
