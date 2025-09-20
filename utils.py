"""
Utilities for dataset loading, graph creation, embeddings (DeepWalk & Node2Vec),
regression, evaluation, grid search, and experiment runner.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

from gensim.models import Word2Vec
from csrgraph import csrgraph


# -----------------------------
# Dataset loader
# -----------------------------
def load_dataset(dataset_name: str):
    """Load dataset by name (CA or MHD) and return df, numeric_features, threshold."""
    if dataset_name == "CA":
        df = pd.read_csv("data/California-housing.csv")
        df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
        df = df.dropna().reset_index(drop=True)
        df['id'] = df.index

        numeric_features = [
            'median_income', 'housing_median_age',
            'total_rooms', 'total_bedrooms',
            'population', 'households'
        ]
        threshold = 4000   # meters
        return df, numeric_features, threshold

    elif dataset_name == "MHD":

        data= pd.read_excel('data/MHD-housing.xlsx')
        # Filter the data for the specified region
        filtered_data = data[(data['longitude'] >= 59.4) & (data['longitude'] <= 59.7) &
                            (data['latitude'] >= 36.2) & (data['latitude'] <= 36.45)]
        np.random.seed(42)
        shuffle_indices = np.random.choice(np.arange(filtered_data.shape[0]), size=5000, replace=False,)
        df = filtered_data.iloc[shuffle_indices].reset_index(drop=True)
        df['id'] = df.index  # Add this line to create a unique identifier for each house

        # df = pd.read_csv("data/MHD-housing.csv")
        # df = df.dropna().reset_index(drop=True)
        # df['id'] = df.index

        numeric_features = [
            'area_sq_m', 'age_years',
            'floor_number', 'number_of_bedrooms'
        ]
        threshold = 2000   # meters
        return df, numeric_features, threshold

    else:
        raise ValueError("Unknown dataset name. Use 'CA' or 'MHD'.")


# -----------------------------
# Graph construction
# -----------------------------
def create_graph_from_dataframe(df, numeric_features, distance_threshold=4000):
    """Create graph of properties based on spatial distance."""
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    coordinates = df[['latitude', 'longitude']].to_numpy()
    tree = cKDTree(coordinates)
    G = nx.Graph()

    for i, row in df.iterrows():
        node_id = int(row['id'])
        G.add_node(node_id, **row.to_dict())

    pairs = tree.query_pairs(distance_threshold / 111000)

    # Define a function to compute edge weight  
    def calculate_weight(df, idx1, idx2):  
        if 'type' in df.columns:
            # no edge if the types are different in MHD dataset
            if df.loc[idx1, 'type'] != df.loc[idx2, 'type']:
                return 0
        geo_distance = np.linalg.norm(coordinates[idx1] - coordinates[idx2])  
        # num_distance = np.linalg.norm(df.loc[idx1, numeric_features] - df.loc[idx2, numeric_features])  
        # binary_similarity = np.sum(df.loc[idx1, binary_features] == df.loc[idx2, binary_features])   
        # weight = (1 / (1 + geo_distance)) * (1 + binary_similarity) / (1 + num_distance)  

        weight = (1 / (1 + geo_distance))  
        return weight  

    # Add edges with weights  
    for idx1, idx2 in tqdm(pairs,  desc="Building graph"):  
        weight = calculate_weight(df, idx1, idx2)  
        if weight != 0:  
            G.add_edge(df.at[idx1, 'id'], df.at[idx2, 'id'], weight=weight)  # Use ids for edges  

    return G


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


# -----------------------------
# Regression & evaluation
# -----------------------------
def fit_and_evaluate(model, X_train, y_train, X_test, y_test, filename=None, verbose=True):
    """Train regression model and evaluate with multiple metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    eps = 1e-8
    try:
        mse_log = mean_squared_error(
            np.log10(y_test + eps),
            np.log10(np.maximum(y_pred, eps))
        )
    except ValueError:
        mse_log = np.nan   # in case log fails

    acc = np.mean(np.abs(y_test - y_pred) <= 0.2 * y_test)

    if verbose:
        print(f"R2: {r2:.3f}, MAPE: {mape:.3f}, RMSE: {rmse:.3f}, Acc: {acc:.3f}")

    if filename:
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.savefig(filename)
        plt.close()

    return r2, mape, acc, rmse, mse_log


# -----------------------------
# Grid Search for embedding size
# -----------------------------
def grid_search_embedding_size(df, G, embedding_sizes, method="deepwalk",
                               score_name="r2", random_state=42, dataset_name="CA"):
    """Perform grid search for embedding size for DeepWalk or Node2Vec."""
    best_score = np.inf if score_name == 'rmse' else -np.inf
    best_params, best_X, best_y = None, None, None
    results = []

    for vector_size in embedding_sizes:
        print(f"[{method}] Evaluating embedding size: {vector_size}")

        if method == "deepwalk":
            X, y, _ = train_deepwalk_embeddings(G, df, vector_size)
        elif method == "node2vec":
            X, y, _ = train_node2vec_embeddings(G, df, vector_size)
        else:
            raise ValueError("Unknown method")

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=random_state)
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = []

        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            model = GradientBoostingRegressor(loss='huber', n_estimators=100,
                                              max_depth=10, random_state=random_state)
            # _, _, _, rmse, _ = fit_and_evaluate(model, X_tr, y_tr, X_val, y_val, verbose=False)
            if score_name == 'rmse':
                _, _, _, score, _ = fit_and_evaluate(model, X_tr, y_tr, X_val, y_val, verbose=False)
            else: # r2
                score, _, _, _, _ = fit_and_evaluate(model, X_tr, y_tr, X_val, y_val, verbose=False)
            
            scores.append(score)

        mean_score = np.mean(scores)
        results.append((vector_size, mean_score))

                # Update best score and parameters
        if score_name == 'rmse':
            condition = mean_score < best_score
        else:
            condition = mean_score > best_score
        if condition == True:
            best_score, best_params, best_X, best_y = mean_score, vector_size, X, y

    results_df = pd.DataFrame(results, columns=['Embedding Size', score_name])
    print(f"[{method}] Best embedding size: {best_params} with {score_name}: {best_score:.3f}")

    os.makedirs(f"results/{dataset_name}", exist_ok=True)
    results_df.to_excel(f"results/{dataset_name}/{method}_embedding_size_results.xlsx", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Embedding Size'], results_df[score_name], marker='o')
    plt.scatter(best_params, best_score, color='red')
    plt.title(f"{method.upper()} - Embedding Size vs {score_name.upper()}")
    plt.xlabel("Embedding Size")
    plt.ylabel(score_name.upper())
    plt.grid(True)
    plt.savefig(f"results/{dataset_name}/{method}_embedding_size_plot.png")
    plt.close()

    return best_params, best_X, best_y, results_df
