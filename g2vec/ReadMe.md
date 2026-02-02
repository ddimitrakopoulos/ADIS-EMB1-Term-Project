# Graph2Vec Experiment Pipeline

This repository contains a Python pipeline for **graph embedding, classification, clustering, and stability analysis** using **Graph2Vec** on TUDatasets.

---

## Setup

### Option 1: Using conda environment (Recommended)

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

## Configuration

### Environment Variables (.env file)

You can control Graph2Vec and experiment parameters via a `.env` file in the project root:

```bash
# Test/Train split
TEST_SIZE=0.2

# Random seeds
RANDOM_SEED=42
G2V_SEED=42

# Graph2Vec parameters
G2V_WL_ITER=3        # Weisfeiler-Lehman iterations
G2V_EPOCHS=40        # Training epochs
G2V_LR=0.025         # Learning rate
```

---

## Running the Script

The script is run via command-line arguments. Two dataset options:

### Using local dataset:
```bash
python main.py --dataset ../datasets/MUTAG [OPTIONS]
```

### Using auto-download TUDataset (requires PyTorch Geometric):
```bash
python main.py --tudataset MUTAG [OPTIONS]
```

### Available Options:

- `--mode`: Experiment mode. Options: `classify`, `cluster`, `stability`, `all` (default: `all`)
- `--classifier`: Classifier for classification/stability. Options: `svm`, `logreg`, `mlp`, `knn` (default: `svm`)
- `--dim`: Graph2Vec embedding dimension (default: `128`)
- `--perturb_pct`: Edge perturbation percentage for stability (0.0-1.0, default: `0.1`)
- `--perturb_mode`: Perturbation mode. Options: `add`, `remove`, `both`, `shuffle` (default: `both`)
- `--full_stability`: Run full stability analysis (0-30% perturbation for all modes)
- `--3d`: Plot t-SNE and UMAP in 3D (default: 2D)
- `--repeat`: Number of repetitions with different seeds for averaging (default: `1`)
- `--attributed`: Use node labels/attributes in Graph2Vec (default: False)

### Examples

**Classification with SVM:**
```bash
python main.py --dataset ../datasets/MUTAG --mode classify --classifier svm
```

**Classification with auto-download:**
```bash
python main.py --tudataset MUTAG --mode classify --classifier mlp --dim 256
```

**Clustering with 3D visualization:**
```bash
python main.py --tudataset IMDB-MULTI --mode cluster --3d
```

**Stability analysis (remove 10% edges):**
```bash
python main.py --tudataset ENZYMES --mode stability --perturb_pct 0.1 --perturb_mode remove
```

**Full stability analysis (0-30% for all modes):**
```bash
python main.py --tudataset MUTAG --mode stability --full_stability
```

**All experiments with 10 repetitions:**
```bash
python main.py --tudataset MUTAG --mode all --repeat 10
```

**Stability with node label shuffling:**
```bash
python main.py --tudataset MUTAG --mode stability --perturb_pct 0.1 --perturb_mode shuffle
```

---

## Output

- **Classification**: Accuracy, Macro-F1, AUC, train/inference times, memory usage
- **Clustering**: Adjusted Rand Index (ARI) for KMeans and Spectral Clustering, t-SNE & UMAP visualizations
- **Stability Analysis**: Embedding stability, edge Jaccard similarity, accuracy drop under perturbations
- **Full Stability**: Generates `stability_analysis.png` with curves for all perturbation modes

---

## Dependencies

Core dependencies (from `requirements.txt`):
- numpy, networkx, scikit-learn, matplotlib, umap-learn, psutil, python-dotenv, karateclub
- torch, torch-geometric (for TUDataset auto-download)

See `environment.yml` for conda-based setup.

---



