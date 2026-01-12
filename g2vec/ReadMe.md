# Graph2Vec Experiment Pipeline

This repository contains a Python pipeline for **graph embedding, classification, clustering, and stability analysis** using **Graph2Vec** on TUDatasets.

---

## Setup

### Option 1: Using conda environment

1. Create the environment from `environment.yml`:

   ```bash
   conda env create -f environment.yml
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

The script is now run via command-line arguments. Example usage:

```bash
python main.py --dataset <DATASET_PATH> [--mode <MODE>] [--classifier <CLASSIFIER>] [--dim <EMBED_DIM>]
```

- `--dataset` (required): Path to TUDataset folder (e.g., `../datasets/MUTAG`)
- `--mode`: Which experiment to run. Options: `classify`, `cluster`, `stability`, `all` (default: `all`)
- `--classifier`: Classifier to use. Options: `svm`, `logreg`, `mlp`, `knn` (default: `svm`)
- `--dim`: Graph2Vec embedding dimension (default: `128`)

### Example

```bash
python main.py --dataset ../datasets/MUTAG --mode all --classifier svm --dim 128
```

---

## Output

- Classification results (Accuracy, Macro-F1, AUC)
- Clustering results (ARI, t-SNE & UMAP plots)
- Optional stability analysis

---

You can control the number of trials and other parameters by editing environment variables in a `.env` file or via the code.

---

