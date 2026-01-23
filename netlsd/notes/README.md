# NetLSD Stability Analysis Notes

---
## NetLSD Eigenvalue Approximations

NetLSD embeddings are based on the **heat kernel trace** of the graph’s normalized Laplacian, which depends on its **eigenvalues**. For large graphs, computing all eigenvalues is expensive, so NetLSD allows **approximating the spectrum**: only a subset of the smallest and largest eigenvalues is computed exactly, while the middle ones are **linearly interpolated**.

Varying the number of approximations changes the **fidelity of the spectral representation**:

- **Fewer approximations** → More interpolation → Coarser embedding, subtle structural differences may be smoothed out / expect small comp. cost too  
- **More approximations** → Less interpolation → More precise embedding, better capturing fine-grained structural features / expect higher comp. cost though

---

## 1. Classification vs Clustering

- **Classification**  
  - Confirms that embeddings are useful for prediction.  
  - Shows whether embeddings are **linearly separable** for the target task.  
  - A classifier maps the embedding space to the known labels.  

- **Clustering**  
  - Evaluates the **intrinsic structure** of embeddings without supervision.  
  - Reveals whether **similar graphs lie close together** in embedding space.  
  - Provides insight into the **emergent structure**, independent of the supervised task.  

---

## 2. Spectral vs K-Means Clustering

- **K-Means**  
  - Centroid-based clustering; assumes roughly spherical clusters.  
  - Relies only on local Euclidean distances in the embedding space.  

- **Spectral Clustering**  
  - Uses graph theory concepts to capture **global structure**.  
  - May better reflect relationships in NetLSD embeddings due to their structural nature.  

---

## 3. Adjusted Rand Index (ARI)

- Measures the agreement between predicted clusters and true labels.  
- Works by comparing all pairs of points:  
  - **Agreement**: same/different in both predicted and true clusters.  
  - **Disagreement**: same in one, different in the other.  
- Normalized and adjusted for chance.  
- Values:  
  - 1 → Perfect match  
  - 0 → Random labeling  
  - -1 → Systematic disagreement  

---

## 4. Visualization

- Visualizations provide intuition about the **relative distances and grouping** of embeddings.  
- **Important**: t-SNE/UMAP distort high-dimensional distances; apparent clusters in plots may not perfectly reflect true separability.  

---

## 5. Graph Perturbations

- **`perturb_edges`**  
  - Randomly removes `k` edges (fraction of total) and adds `k` new edges.  
  - New edges are sampled randomly; if an edge already exists, it is skipped.  
  - Graph size is **roughly maintained**, but more structural noise is introduced.

- **`perturb_graph`**  
  - Removes exactly `k` edges and **ensures exactly `k` edges are added**.  
  - Guarantees graph size is preserved.  
  - Optional node label shuffling included, but **NetLSD is structural-only**, so labels have no effect.

---

## 6. Metrics

- **Edge Jaccard**  
  - Measures how much the graph structure changed after perturbation.  
  - `1 → graphs identical`, `0 → no edges shared`.  
  - Serves as a sanity check for perturbation intensity.

- **Cosine Similarity**  
  - Measures directional similarity of embeddings (vector “shape”).  
  - `1 → same direction`, `0 → orthogonal`, `-1 → opposite`.  
  - Important because NetLSD embeddings encode graph structure in vector direction.

- **Relative L2 Distance**  
  - Normalized Euclidean distance between original and perturbed embeddings.  
  - Captures **magnitude differences**, unlike cosine which only captures direction.  
  - Helps detect shrink/expansion of embeddings under perturbation.

---

## 7. Classification Stability

- SVM is trained on original embeddings and used to predict perturbed embeddings.  
- No train/test split is used intentionally to **measure stability directly**, not generalization.  
- `acc_drop = acc_orig - acc_pert` provides a numeric stability score.

**Notes:**  
- Using multiple classifiers can provide more robust assessment.  
- Classification results are mostly consistent across different classifiers as task A shows.
