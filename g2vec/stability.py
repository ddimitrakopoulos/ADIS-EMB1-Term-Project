import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from karateclub import Graph2Vec
from sklearn.metrics import accuracy_score

from utils import perturb_graph, scale_features, embedding_stability, edge_jaccard

def run_stability(graphs, X, clf, y, train_idx, test_idx, dim, seed, 
                  permute_pct=0.1, mode='both', shuffle_labels=False, orig_acc=None):

    load_dotenv()
    wl_iterations = int(os.getenv('G2V_WL_ITER', 2))
    epochs = int(os.getenv('G2V_EPOCHS', 15))
    learning_rate = float(os.getenv('G2V_LR', 0.05))
    env_seed = int(os.getenv('G2V_SEED', seed))

    # Perturb graphs
    perturbed = [perturb_graph(G, edge_perturb_ratio=permute_pct, mode=mode, shuffle_node_labels=shuffle_labels) for G in graphs]

    # Recompute embeddings
    g2v = Graph2Vec(dimensions=dim, wl_iterations=wl_iterations, epochs=epochs, learning_rate=learning_rate, seed=env_seed)
    g2v.fit(perturbed)
    Xp = g2v.get_embedding()

    # Embedding stability (cosine similarity)
    sim = embedding_stability(X, Xp)

    # Classification on perturbed data
    Xp_train = Xp[train_idx]
    Xp_test = Xp[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    Xp_train, Xp_test = scale_features(Xp_train, Xp_test)

    clf2 = copy.deepcopy(clf)
    clf2.fit(Xp_train, y_train)
    y_pred = clf2.predict(Xp_test)

    acc = accuracy_score(y_test, y_pred)
    acc_drop = (orig_acc - acc) if orig_acc is not None else 0.0

    # Compute edge jaccard
    jaccards = [edge_jaccard(G, Gp) for G, Gp in zip(graphs, perturbed)]
    avg_jaccard = np.mean(jaccards)

    return {
        "permute_pct": permute_pct,
        "mode": mode,
        "shuffle_labels": shuffle_labels,
        "embedding_stability": sim,
        "edge_jaccard": avg_jaccard,
        "accuracy": acc,
        "accuracy_drop": acc_drop,
    }


def run_full_stability_analysis(graphs, X, clf, y, train_idx, test_idx, dim, seed, orig_acc, shuffle_labels=False):

    load_dotenv()
    
    perturbation_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = []
    
    # Storage for plotting curves
    emb_curves = {'add': [], 'remove': [], 'both': [], 'shuffle': []}
    acc_curves = {'add': [], 'remove': [], 'both': [], 'shuffle': []}
    
    print(f"\n{'='*80}")
    print(f"STABILITY ANALYSIS")
    if shuffle_labels:
        print(f"(Label Shuffling ENABLED)")
    print(f"{'='*80}")
    
    # Test removing edges (Drop Edges)
    print(f"\n--- DROP EDGES ---")
    print(f"{'Pct':<8} {'Emb Stability':<15} {'Edge Jaccard':<15} {'Accuracy':<12} {'Acc Drop':<12}")
    print("-" * 62)
    for pct in perturbation_levels:
        if pct == 0.0:
            emb_curves['remove'].append(1.0)
            acc_curves['remove'].append(orig_acc)
            print(f"{pct*100:>5.0f}%   {1.0:<15.4f} {1.0:<15.4f} {orig_acc:<12.4f} {0.0:<12.4f}")
        else:
            result = run_stability(
                graphs, X, clf, y, train_idx, test_idx, dim, seed,
                permute_pct=pct, mode='remove', shuffle_labels=False, orig_acc=orig_acc
            )
            results.append(result)
            emb_curves['remove'].append(result['embedding_stability'])
            acc_curves['remove'].append(result['accuracy'])
            print(f"{pct*100:>5.0f}%   {result['embedding_stability']:<15.4f} {result['edge_jaccard']:<15.4f} {result['accuracy']:<12.4f} {result['accuracy_drop']:<12.4f}")
    
    # Test adding edges (Add Edges)
    print(f"\n--- ADD EDGES ---")
    print(f"{'Pct':<8} {'Emb Stability':<15} {'Edge Jaccard':<15} {'Accuracy':<12} {'Acc Drop':<12}")
    print("-" * 62)
    for pct in perturbation_levels:
        if pct == 0.0:
            emb_curves['add'].append(1.0)
            acc_curves['add'].append(orig_acc)
            print(f"{pct*100:>5.0f}%   {1.0:<15.4f} {1.0:<15.4f} {orig_acc:<12.4f} {0.0:<12.4f}")
        else:
            result = run_stability(
                graphs, X, clf, y, train_idx, test_idx, dim, seed,
                permute_pct=pct, mode='add', shuffle_labels=False, orig_acc=orig_acc
            )
            results.append(result)
            emb_curves['add'].append(result['embedding_stability'])
            acc_curves['add'].append(result['accuracy'])
            print(f"{pct*100:>5.0f}%   {result['embedding_stability']:<15.4f} {result['edge_jaccard']:<15.4f} {result['accuracy']:<12.4f} {result['accuracy_drop']:<12.4f}")
    
    # Test both (Mixed: remove and add edges)
    print(f"\n--- MIXED (ADD + DROP EDGES) ---")
    print(f"{'Pct':<8} {'Emb Stability':<15} {'Edge Jaccard':<15} {'Accuracy':<12} {'Acc Drop':<12}")
    print("-" * 62)
    for pct in perturbation_levels:
        if pct == 0.0:
            emb_curves['both'].append(1.0)
            acc_curves['both'].append(orig_acc)
            print(f"{pct*100:>5.0f}%   {1.0:<15.4f} {1.0:<15.4f} {orig_acc:<12.4f} {0.0:<12.4f}")
        else:
            result = run_stability(
                graphs, X, clf, y, train_idx, test_idx, dim, seed,
                permute_pct=pct, mode='both', shuffle_labels=False, orig_acc=orig_acc
            )
            results.append(result)
            emb_curves['both'].append(result['embedding_stability'])
            acc_curves['both'].append(result['accuracy'])
            print(f"{pct*100:>5.0f}%   {result['embedding_stability']:<15.4f} {result['edge_jaccard']:<15.4f} {result['accuracy']:<12.4f} {result['accuracy_drop']:<12.4f}")
    
    # Test shuffle features (label shuffling only, no edge perturbation)
    print(f"\n--- SHUFFLE FEATURES ---")
    print(f"{'Pct':<8} {'Emb Stability':<15} {'Edge Jaccard':<15} {'Accuracy':<12} {'Acc Drop':<12}")
    print("-" * 62)
    for pct in perturbation_levels:
        if pct == 0.0:
            emb_curves['shuffle'].append(1.0)
            acc_curves['shuffle'].append(orig_acc)
            print(f"{pct*100:>5.0f}%   {1.0:<15.4f} {1.0:<15.4f} {orig_acc:<12.4f} {0.0:<12.4f}")
        else:
            result = run_stability(
                graphs, X, clf, y, train_idx, test_idx, dim, seed,
                permute_pct=pct, mode='none', shuffle_labels=True, orig_acc=orig_acc
            )
            results.append(result)
            emb_curves['shuffle'].append(result['embedding_stability'])
            acc_curves['shuffle'].append(result['accuracy'])
            print(f"{pct*100:>5.0f}%   {result['embedding_stability']:<15.4f} {result['edge_jaccard']:<15.4f} {result['accuracy']:<12.4f} {result['accuracy_drop']:<12.4f}")
    
    print(f"\n{'='*80}")
    print(f"Original Accuracy: {orig_acc:.4f}")
    print(f"{'='*80}")
    
    # ================= Plot and save PNG =================
    pct_vals = [p * 100 for p in perturbation_levels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Embedding Stability (Cosine Sim)
    ax1.plot(pct_vals, emb_curves['add'], marker='o', color='blue', linewidth=2, label='Add Edges')
    ax1.plot(pct_vals, emb_curves['remove'], marker='s', color='red', linewidth=2, label='Drop Edges')
    ax1.plot(pct_vals, emb_curves['both'], marker='D', color='purple', linewidth=2, label='Mixed (Add+Drop)')
    ax1.plot(pct_vals, emb_curves['shuffle'], marker='^', color='green', linewidth=2, label='Shuffle Features')
    ax1.set_xlabel('Perturbation (%)')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Embedding Stability (Cosine Sim)')
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlim(-1, 32)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left')
    
    # Right: Accuracy Robustness
    ax2.plot(pct_vals, acc_curves['add'], marker='o', color='blue', linewidth=2, label='Add Edges')
    ax2.plot(pct_vals, acc_curves['remove'], marker='s', color='red', linewidth=2, label='Drop Edges')
    ax2.plot(pct_vals, acc_curves['both'], marker='D', color='purple', linewidth=2, label='Mixed (Add+Drop)')
    ax2.plot(pct_vals, acc_curves['shuffle'], marker='^', color='green', linewidth=2, label='Shuffle Features')
    ax2.set_xlabel('Perturbation (%)')
    ax2.set_ylabel('Classification Accuracy')
    ax2.set_title('Accuracy Robustness')
    ax2.set_ylim(0.0, 1.05)
    ax2.set_xlim(-1, 32)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left')
    
    plt.tight_layout()
    
    # Save to PNG
    output_path = 'stability_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nStability plot saved to: {output_path}")
    plt.show()
    
    return results
