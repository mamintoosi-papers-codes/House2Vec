# embeddings_utils.py
"""
Embedding utilities: DeepWalk and Node2Vec implementations.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from csrgraph import csrgraph


# -----------------------------
# DeepWalk
# -----------------------------
def random_walk(G, start, length):
    """Perform one random walk starting from a given node."""
    walk = [str(start)]
    for _ in range(length):
        neighbors = [node for node in G.neighbors(start)]
        if len(neighbors) == 0:
            next_node = start
        else:
            next_node = np.random.choice(neighbors, 1)[0]
        walk.append(str(next_node))
        start = next_node
    return walk


def generate_random_walks_deepwalk(G, num_walks=80, walk_length=10):
    """Generate random walks for DeepWalk."""
    walks = []
    for node in tqdm(G.nodes, desc="DeepWalk Nodes"):
        for _ in range(num_walks):
            walks.append(random_walk(G, node, walk_length))
    return walks


def create_word2vec_model(walks, vector_size):
    """Train Word2Vec on walks."""
    model = Word2Vec(
        walks,
        hs=1,
        sg=1,
        vector_size=vector_size,
        window=5,
        workers=4,
        seed=1
    )
    return model


def get_embeddings(model, G, prefix="deepwalk"):
    """Extract embeddings into dataframe."""
    embeddings = np.array([model.wv[str(i)] for i in G.nodes()])
    embeddings_df = pd.DataFrame(
        embeddings,
        columns=[f"{prefix}_emb_{i}" for i in range(embeddings.shape[1])]
    )
    return embeddings_df


def train_deepwalk_embeddings(G, df, vector_size=16):
    """Full pipeline for DeepWalk embeddings."""
    walks = generate_random_walks_deepwalk(G)
    wv_model = create_word2vec_model(walks, vector_size=vector_size)
    emb_df = get_embeddings(wv_model, G, prefix="deepwalk")
    df_with_embeddings = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
    X = df_with_embeddings.drop(['price', 'id'], axis=1)
    y = df_with_embeddings['price']
    return X, y, emb_df


# -----------------------------
# Node2Vec
# -----------------------------
def generate_random_walks_node2vec(G, num_walks=80, walk_length=10,
                                   return_weight=3, neighbor_weight=1):
    """Generate biased random walks for Node2Vec using csrgraph."""
    cg = csrgraph(G)
    walks = []
    for _ in tqdm(range(num_walks), desc="Node2Vec Walks"):
        random_walks = cg.random_walks(
            walklen=walk_length,
            return_weight=return_weight,
            neighbor_weight=neighbor_weight
        )
        for walk in random_walks:
            walk_str = [str(node) for node in walk.tolist()]
            walks.append(walk_str)
    return walks


def train_node2vec_embeddings(G, df, vector_size=16,
                              return_weight=3, neighbor_weight=1):
    """Full pipeline for Node2Vec embeddings."""
    walks = generate_random_walks_node2vec(
        G, num_walks=80, walk_length=10,
        return_weight=return_weight, neighbor_weight=neighbor_weight
    )
    wv_model = create_word2vec_model(walks, vector_size=vector_size)
    emb_df = get_embeddings(wv_model, G, prefix="node2vec")
    df_with_embeddings = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
    X = df_with_embeddings.drop(['price', 'id'], axis=1)
    y = df_with_embeddings['price']
    return X, y, emb_df
