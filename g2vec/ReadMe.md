# Graph2Vec Experiment Pipeline

This repository contains a Python pipeline for **graph embedding, classification, clustering, and stability analysis** using **Graph2Vec** on TUDatasets.

---

## Setup

### Option 1: Using conda environment

1. Create the environment from `env.yml`:

   ```bash
   conda env create -f env.yml
   ```
2. Activate the environment:

   ```bash
   conda activate g2vec
   ```

### Option 2: Using pip requirements

1. Create a Python 3.10 environment manually.
2. Install dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Script

Run the main script:

```bash
python main.py
```

* You will be prompted to enter:

  * Dataset folder path (e.g., `../datasets/MUTAG`)
  * Classifier choice (SVM, Logistic Regression, MLP, k-NN)
  * Graph2Vec embedding dimension (e.g., 64, 128, 256)
* The script outputs:

  * Classification results (Accuracy, Macro-F1, AUC)
  * Clustering results (ARI, t-SNE & UMAP plots)
  * Optional stability analysis

---

## Stability Analysis

* After classification and clustering, you will be prompted:

  ```
  Do you want to perform stability analysis? (y/n):
  ```

* If `y`, the script will:

  * Randomly perturb the graphs (add/remove edges, shuffle node labels)
  * Recompute embeddings
  * Compute **embedding stability score**
  * Evaluate **classification on perturbed embeddings**

* You can run multiple stability trials interactively.

---
