import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

def run_clustering(X, y, seed, plot_3d=False):
    num_classes = len(np.unique(y))
    
    # KMeans Clustering
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

    # Agglomerative Clustering (Hierarchical)
    agglomerative = AgglomerativeClustering(n_clusters=num_classes, linkage='ward')
    labels_agglomerative = agglomerative.fit_predict(X)
    ari_agglomerative = adjusted_rand_score(y, labels_agglomerative)
    print(f"Agglomerative Clustering ARI: {ari_agglomerative:.4f}")

    # DBSCAN (density-based clustering)
    # Use heuristic for eps based on data scale
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, -1], axis=0)
    eps = np.percentile(distances, 90)  # Use 90th percentile as eps
    
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels_dbscan = dbscan.fit_predict(X)
    
    # DBSCAN may not find all clusters or may mark points as noise (-1)
    # For ARI calculation, we still compare with true labels
    ari_dbscan = adjusted_rand_score(y, labels_dbscan)
    n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
    n_noise = list(labels_dbscan).count(-1)
    print(f"DBSCAN ARI: {ari_dbscan:.4f} (Found {n_clusters_dbscan} clusters, {n_noise} noise points)")

    n_components = 3 if plot_3d else 2
    tsne = TSNE(n_components=n_components, random_state=seed).fit_transform(X)
    umap_emb = umap.UMAP(n_components=n_components, random_state=seed).fit_transform(X)

    plt.figure(figsize=(12, 5))

    if plot_3d:
        from mpl_toolkits.mplot3d import Axes3D
        for idx, (data, title) in enumerate([(tsne, "t-SNE 3D"), (umap_emb, "UMAP 3D")], 1):
            ax = plt.subplot(1, 2, idx, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=y, cmap="tab10")
            ax.set_title(title)
    else:
        for idx, (data, title) in enumerate([(tsne, "t-SNE"), (umap_emb, "UMAP")], 1):
            plt.subplot(1, 2, idx)
            plt.scatter(data[:, 0], data[:, 1], c=y, cmap="tab10")
            plt.title(title)

    plt.show()

    return {
        'kmeans': ari_kmeans,
        'spectral': ari_spectral,
        'agglomerative': ari_agglomerative,
        'dbscan': ari_dbscan
    }
