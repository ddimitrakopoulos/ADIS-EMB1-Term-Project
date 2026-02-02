import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import argparse
from dotenv import load_dotenv

from utils import load_tudataset, download_tudataset
from embeddings import compute_graph2vec_embeddings
from classification import run_classification
from clustering import run_clustering
from stability import run_stability, run_full_stability_analysis

parser = argparse.ArgumentParser(
    description="Graph2Vec Experiments: Classification, Clustering, and Stability Analysis",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Run classification with SVM (using local dataset)
  python main.py --dataset ../datasets/MUTAG --mode classify --classifier svm

  # Run classification with auto-download TUDataset
  python main.py --tudataset MUTAG --mode classify --classifier svm

  # Run clustering with 3D visualization
  python main.py --tudataset IMDB-MULTI --mode cluster --3d

  # Run stability analysis (remove 10%% edges)
  python main.py --tudataset ENZYMES --mode stability --permute_pct 0.1 --perturb_mode remove

  # Run full stability analysis (tests add, remove, both, shuffle at 0-30%%)
  # Generates stability_analysis.png with embedding stability and accuracy curves
  python main.py --tudataset MUTAG --mode stability --full_stability

  # Run all experiments with 10 repetitions
  python main.py --tudataset MUTAG --mode all --repeat 10

  # Run stability with node label shuffling (shuffle mode)
  python main.py --tudataset MUTAG --mode stability --perturb_pct 0.1 --perturb_mode shuffle
"""
)

dataset_group = parser.add_mutually_exclusive_group(required=True)
dataset_group.add_argument("--dataset", type=str,
                    help="Path to TUDataset folder (e.g., ../datasets/MUTAG)")
dataset_group.add_argument("--tudataset", type=str, choices=["MUTAG", "IMDB-MULTI", "ENZYMES"],
                    help="TUDataset name to auto-download and use (mutually exclusive with --dataset). Choices: MUTAG, IMDB-MULTI, ENZYMES")

parser.add_argument("--mode", type=str, default="all",
                    choices=["classify", "cluster", "stability", "all"],
                    help="Experiment mode: 'classify' (classification), 'cluster' (clustering), 'stability' (stability analysis), 'all' (run all). Default: all")

parser.add_argument("--classifier", type=str, default="svm",
                    choices=["svm", "logreg", "mlp", "knn"],
                    help="Classifier for classification/stability: 'svm', 'logreg', 'mlp', 'knn'. Default: svm")

parser.add_argument("--dim", type=int, default=128,
                    help="Graph2Vec embedding dimension. Default: 128")

parser.add_argument("--perturb_pct", type=float, default=0.1,
                    help="Edge perturbation percentage (0.0 to 1.0) for stability analysis. Default: 0.1 (10%%)")

parser.add_argument("--perturb_mode", type=str, default="both",
                    choices=["add", "remove", "both", "shuffle"],
                    help="Perturbation mode for stability: 'add', 'remove', 'both', 'shuffle'. Default: both")

parser.add_argument("--full_stability", action='store_true',
                    help="Run full stability analysis: tests add, remove, both, and shuffle at 0-30%%. Saves stability_analysis.png")

parser.add_argument('--3d', action='store_true',
                    help='Plot t-SNE and UMAP visualizations in 3D. Default: 2D')

parser.add_argument('--repeat', type=int, default=1,
                    help='Number of repetitions with different random seeds for averaging. Default: 1')

parser.add_argument('--attributed', action='store_true',
                    help='Use node labels/attributes in Graph2Vec (Weisfeiler-Lehman). Default: False')

args = parser.parse_args()

if args.tudataset:
    graphs, y = download_tudataset(args.tudataset, download_dir="./datasets")
    dataset_name = args.tudataset
else:
    dataset_name = os.path.basename(os.path.normpath(args.dataset))
    graphs, y = load_tudataset(args.dataset)

print("\n" + "="*60)
print("GRAPH2VEC EXPERIMENT CONFIGURATION")
print("="*60)
dataset_source = "--tudataset (auto-download)" if args.tudataset else "--dataset (local)"
print(f"Dataset ({dataset_source}) : {dataset_name}")
print(f"Mode             : {args.mode}")
print(f"Classifier       : {args.classifier}")
print(f"Embedding dim    : {args.dim}")
print(f"Use node labels  : {args.attributed}")
print(f"Repeat           : {args.repeat}")
if args.mode in ["stability", "all"]:
    print(f"Perturb pct      : {args.perturb_pct*100:.0f}%")
    print(f"Perturb mode     : {args.perturb_mode}")
    print(f"Full stability   : {args.full_stability}")
if args.mode in ["cluster", "all"]:
    print(f"3D visualization : {args.__dict__.get('3d', False)}")
print("="*60)

load_dotenv()

TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
SEED = int(os.getenv("RANDOM_SEED", 42))
WL = int(os.getenv("G2V_WL_ITER", 3))
EPOCHS = int(os.getenv("G2V_EPOCHS", 40))
LR = float(os.getenv("G2V_LR", 0.025))
G2V_SEED = int(os.getenv("G2V_SEED", 42))

print(f"\nDataset loaded: {dataset_name}")
print(f"Number of graphs: {len(graphs)}")

X, embed_time, embed_mem = compute_graph2vec_embeddings(
    graphs, args.dim, WL, EPOCHS, LR, G2V_SEED, attributed=args.attributed
)

clf = None
train_idx = test_idx = None

import numpy as np

if args.mode in ["classify", "all"]:
    all_metrics = []
    for i in range(args.repeat):
        this_seed = SEED + i
        clf, train_idx, test_idx, metrics = run_classification(
            X, y, args.classifier, TEST_SIZE, this_seed,
            dataset_name, args.dim, embed_time, embed_mem
        )
        all_metrics.append(metrics)
    if args.repeat > 1:
        print(f"\n===== Averaged Classification Results ({dataset_name}) =====")
        print(f"Embedding dim        : {args.dim}")
        print(f"Classifier           : {args.classifier.upper()}")
        print(f"Number of runs       : {args.repeat}")
        print("-" * 50)
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            print(f"{key:<20}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        print("=" * 50)

if args.mode in ["cluster", "all"]:
    ari_kmeans_scores = []
    ari_spectral_scores = []
    for i in range(args.repeat):
        this_seed = SEED + i
        ari_kmeans, ari_spectral = run_clustering(X, y, this_seed, plot_3d=args.__dict__.get('3d', False))
        ari_kmeans_scores.append(ari_kmeans)
        ari_spectral_scores.append(ari_spectral)
    if args.repeat > 1:
        print(f"\n===== Averaged Clustering Results =====")
        print(f"KMeans ARI over {args.repeat} runs: {np.mean(ari_kmeans_scores):.4f} ± {np.std(ari_kmeans_scores):.4f}")
        print(f"Spectral ARI over {args.repeat} runs: {np.mean(ari_spectral_scores):.4f} ± {np.std(ari_spectral_scores):.4f}")
        print("=" * 40)

if args.mode in ["stability", "all"]:
    clf, train_idx, test_idx, metrics = run_classification(
        X, y, args.classifier, TEST_SIZE, SEED,
        dataset_name, args.dim, embed_time, embed_mem
    )
    orig_acc = metrics["accuracy"]
    
    if args.full_stability:
        run_full_stability_analysis(
            graphs, X, clf, y, train_idx, test_idx, args.dim, SEED, orig_acc,
            attributed=args.attributed
        )
    else:
        result = run_stability(
            graphs, X, clf, y, train_idx, test_idx, args.dim, SEED,
            permute_pct=args.perturb_pct,
            mode=args.perturb_mode,
            orig_acc=orig_acc,
            attributed=args.attributed
        )
        print(f"\n===== Stability Analysis Results =====")
        print(f"Perturbation         : {args.perturb_pct*100:.0f}%")
        print(f"Mode                 : {args.perturb_mode}")
        print(f"Embedding stability  : {result['embedding_stability']:.4f}")
        print(f"Edge Jaccard         : {result['edge_jaccard']:.4f}")
        print(f"Original accuracy    : {orig_acc:.4f}")
        print(f"Perturbed accuracy   : {result['accuracy']:.4f}")
        print(f"Accuracy drop        : {result['accuracy_drop']:.4f}")
        print("=" * 40)

print("\nDone.")
