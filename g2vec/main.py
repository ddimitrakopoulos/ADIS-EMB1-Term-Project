import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import argparse
from dotenv import load_dotenv

from utils import load_tudataset
from embeddings import compute_graph2vec_embeddings
from classification import run_classification
from clustering import run_clustering
from stability import run_stability

# ==================================================
# Argument parser
# ==================================================
parser = argparse.ArgumentParser(description="Graph2Vec Experiments")

parser.add_argument("--dataset", type=str, required=True,
                    help="Path to TUDataset folder (e.g. ../datasets/MUTAG)")

parser.add_argument("--mode", type=str, default="all",
                    choices=["classify", "cluster", "stability", "all"],
                    help="Which experiment to run")

parser.add_argument("--classifier", type=str, default="svm",
                    choices=["svm", "logreg", "mlp", "knn"],
                    help="Classifier to use")

parser.add_argument("--dim", type=int, default=128,
                    help="Graph2Vec embedding dimension")

args = parser.parse_args()

# ==================================================
# Load environment variables
# ==================================================
load_dotenv()

TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
SEED = int(os.getenv("RANDOM_SEED", 42))
WL = int(os.getenv("G2V_WL_ITER", 3))
EPOCHS = int(os.getenv("G2V_EPOCHS", 40))
LR = float(os.getenv("G2V_LR", 0.025))
G2V_SEED = int(os.getenv("G2V_SEED", 42))

# ==================================================
# Load dataset
# ==================================================
dataset_path = args.dataset
dataset_name = os.path.basename(os.path.normpath(dataset_path))

print(f"\nLoading dataset: {dataset_name}")
graphs, y = load_tudataset(dataset_path)
print(f"Loaded {len(graphs)} graphs.")

# ==================================================
# Embeddings (always needed)
# ==================================================
X, embed_time, embed_mem = compute_graph2vec_embeddings(
    graphs, args.dim, WL, EPOCHS, LR, G2V_SEED
)

# ==================================================
# Run selected mode
# ==================================================
clf = None
train_idx = test_idx = None

if args.mode in ["classify", "all"]:
    clf, train_idx, test_idx = run_classification(
        X, y, args.classifier, TEST_SIZE, SEED,
        dataset_name, args.dim, embed_time, embed_mem
    )

if args.mode in ["cluster", "all"]:
    run_clustering(X, y, SEED)

if args.mode in ["stability", "all"]:
    if clf is None:
        print("\nStability requires a trained classifier. Running classification first...")
        clf, train_idx, test_idx = run_classification(
            X, y, args.classifier, TEST_SIZE, SEED,
            dataset_name, args.dim, embed_time, embed_mem
        )

    run_stability(graphs, X, clf, y, train_idx, test_idx, args.dim, SEED)

print("\nDone.")
