# ADIS-EMB1-Term-Project

##  Project Overview

This repository contains the implementation and evaluation framework for the project:

**Evaluation of Graph Embedding Methods on Real-World Datasets**

Graph embedding methods map entire graphs into fixed-length vector representations that capture structural and semantic information, enabling downstream tasks such as graph classification, clustering, and visualization. This project benchmarks representative unsupervised and supervised graph embedding techniques, analyzing their performance, scalability, and robustness on real-world datasets.

## Objectives

The main goals of this project are to:

- Compare graph embedding methods in terms of **classification performance**
- Analyze **computational efficiency and resource usage**
- Evaluate **clustering quality**
- Study **stability under structural perturbations**
- Investigate the effect of **embedding dimensionality** for highlighting accuracy–efficiency trade-offs.
- Examine the influence of **method-specific hyperparameters** on embedding quality and downstream task performance.


## Graph Embedding Methods

The following methods are evaluated:

### Unsupervised
- **Graph2Vec**
- **NetLSD**

### Supervised
- **Graph Isomorphism Network (GIN)**

Unsupervised methods are evaluated by training downstream classifiers  on the extracted embeddings, while supervised models are trained end-to-end.

## Datasets

The project uses the following benchmark datasets:

- **MUTAG**
- **ENZYMES**
- **IMDB-MULTI**

A dedicated `datasets/` folder is included in the repository.  
Currently, **MUTAG, ENZYMES, and IMDB-MULTI are hardcoded and provided directly**.

Each dataset subfolder contains:
- Preprocessed graph files
- A **README.md explaining the internal file structure and usage**

Please refer to the README inside each dataset directory for detailed information.

## Experimental Tasks

### 1. Graph Classification
- Use graph embeddings as input features for standard classifiers (SVM, LogReg, MLP, KNN).
- Report **Accuracy, F1-score, and AUC**.
- Record **training time**, **inference time**, **embedding generation time**, and **memory usage**.
- Analyze the effect of **embedding dimensionality** on performance and cost.

### 2. Clustering
- Apply **k-means** and **spectral clustering** on embeddings.
- Report **Adjusted Rand Index (ARI)**.
- Provide qualitative visualization using **t-SNE** or **UMAP**.

### 3. Stability Analysis
- Introduce random graph perturbations (edge addition/removal, attribute shuffling).
- Recompute embeddings and measure changes.
- Evaluate **embedding stability** and **classification performance degradation**.


## Running Experiments

Each graph embedding method is **self-contained** within its corresponding folder.

To run experiments:
1. Navigate to the desired method directory (e.g., `netlsd/`, `graph2vec/`, `gin/`)
2. Follow the instructions provided in the **README.md inside that folder**
3. Dataset-specific details and expected file formats are documented in the respective dataset README files


## Notes

- All embedding methods are evaluated using **comparable embedding dimensionalities** to ensure fair comparison.
- **Results for each experiment (classification, clustering, and stability analysis) are stored under the corresponding method folders** for clear organization and reproducibility.





