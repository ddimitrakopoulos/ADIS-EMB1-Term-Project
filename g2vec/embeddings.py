import time
from karateclub import Graph2Vec
from utils import mem_mb

def compute_graph2vec_embeddings(graphs, dim, wl_iter, epochs, lr, seed, attributed=False):
    print(f"\nGenerating Graph2Vec embeddings (dim={dim}, attributed={attributed})...")

    mem_before = mem_mb()
    start = time.time()

    g2v = Graph2Vec(
        dimensions=dim,
        wl_iterations=wl_iter,
        epochs=epochs,
        learning_rate=lr,
        seed=seed,
        attributed=attributed
    )
    g2v.fit(graphs)
    X = g2v.get_embedding()

    elapsed = time.time() - start
    mem_after = mem_mb()
    mem_used = mem_after - mem_before

    print(f"Embedding generation time: {elapsed:.2f} s")
    print(f"Embedding memory usage   : {mem_used:.2f} MB")

    return X, elapsed, mem_used
