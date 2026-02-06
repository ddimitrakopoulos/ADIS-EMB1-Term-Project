import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import argparse

from classification import run_classification_experiment
from clustering import run_clustering_experiment
from stability import run_stability

# ==================================================
# Argument parser
# ==================================================
parser = argparse.ArgumentParser(description="NetLSD Experiments")

parser.add_argument("--dataset", type=str, required=True,
                    help="Path to TUDataset folder (e.g. ../datasets/MUTAG)")

parser.add_argument("--mode", type=str, default="all",
                    choices=["classify", "cluster", "stability", "all"],
                    help="Which experiment to run")

parser.add_argument("--classifier", type=str, default="svm",
                    choices=["svm", "logreg", "mlp", "knn", "all"],
                    help="Classifier to use (for classification mode)")

parser.add_argument("--dim", type=int, default=250,
                    help="NetLSD embedding dimension")

parser.add_argument("--perturb", type=str, default="0.05",
                    help="Edge perturbation ratio for stability analysis (e.g. 0.05 = 5%% edges). Use 'all' for multiple ratios.")

parser.add_argument("--visualize", action="store_true",
                    help="Visualize clustering results (scatter plots, etc.)")

args = parser.parse_args()


# ==================================================
# Load dataset name
# ==================================================
dataset_path = args.dataset
dataset_name = os.path.basename(os.path.normpath(dataset_path))

print(f"\nLoading dataset: {dataset_name}")

# ==================================================
# Run selected mode
# ==================================================
if args.mode in ["classify", "all"]:
    print("\nRunning classification experiment...")
    run_classification_experiment(dataset_name=dataset_name, dataset_path=dataset_path, dim=args.dim, classifier=args.classifier)
        
if args.mode in ["cluster", "all"]:
    print("\nRunning clustering experiment...")
    run_clustering_experiment(dataset_name=dataset_name, dataset_path=dataset_path, dim=args.dim, do_visualize=args.visualize)
    
if args.mode in ["stability", "all"]:
    print("\nRunning stability experiment...")
    run_stability(datasets=[dataset_name], dim=args.dim, dataset_path=dataset_path, edge_perturb=args.perturb)

print("\nDone.")
