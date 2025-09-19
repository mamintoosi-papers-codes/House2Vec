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
from sklearn.neighbors import NearestNeighbors

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
        binary_features = ['ocean_proximity_INLAND', 'ocean_proximity_ISLAND',
       'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']
        threshold = 4000   # meters
        return df, numeric_features, binary_features

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
        binary_features = ['elevator', 'parking', 'storage', 'balcony', 'parquet',   
                    'ceramic_flooring', 'stone_façade', 'garden', 'renovated'] 
        threshold = 40   # meters
        return df, numeric_features, binary_features

    else:
        raise ValueError("Unknown dataset name. Use 'CA' or 'MHD'.")


# -----------------------------
# Graph construction
# -----------------------------
def create_graph_from_dataframe(
    df,
    numeric_features,
    binary_features,
    k=35,
    scale_numeric=True,
    metric="euclidean",
    use_edge_weights=True
):
    """
    KNN-based graph construction for housing datasets.

    - numeric_features: continuous attributes (scaled if scale_numeric=True)
    - binary_features: categorical/binary attributes (0/1 encoding)
    - k: number of neighbors per node
    - metric: distance metric for NearestNeighbors ('euclidean', 'cosine', etc.)
    - use_edge_weights: if True, edges get weights = 1 / (1 + distance)
    """
    df_copy = df.copy()

    # Scale only numeric features if requested
    if scale_numeric and numeric_features:
        scaler = StandardScaler()
        df_copy[numeric_features] = scaler.fit_transform(df_copy[numeric_features])

    # Build feature space: coords (NOT scaled) + numeric + binary
    feature_list = ["latitude", "longitude"] + numeric_features + (binary_features if binary_features else [])
    X = df_copy[feature_list].to_numpy()

    # Fit Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)  # k+1 because first neighbor is itself
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Build graph
    G = nx.Graph()
    for i, row in df.iterrows():
        node_id = int(row['id'])
        G.add_node(node_id, **row.to_dict())

    for i in tqdm(range(len(df)), desc="Building KNN graph"):
        for j_idx, j in enumerate(indices[i][1:]):  # skip itself
            # enforce type constraint for MHD dataset
            if "type" in df.columns and df.loc[i, "type"] != df.loc[j, "type"]:
                continue  

            if use_edge_weights:
                dist = distances[i][j_idx + 1]  # distance to neighbor j
                weight = 1.0 / (1.0 + dist)
                G.add_edge(df.at[i, "id"], df.at[j, "id"], weight=weight)
            else:
                G.add_edge(df.at[i, "id"], df.at[j, "id"])

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


def train_deepwalk_embeddings(G, df, vector_size=16, num_walks=40, walk_length=15):
    """Full pipeline for DeepWalk embeddings."""
    walks = generate_random_walks_deepwalk(G, num_walks=num_walks, walk_length=walk_length)
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


def train_node2vec_embeddings(G, df, vector_size=16, num_walks=40, walk_length=15,
                              return_weight=3, neighbor_weight=1):
    """Full pipeline for Node2Vec embeddings."""
    walks = generate_random_walks_node2vec(
        G,  num_walks=num_walks, walk_length=walk_length, 
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
                               score_name="r2", random_state=42, dataset_name="CA",
                               num_walks=60, walk_length=15, p=3, q=1):
    """Perform grid search for embedding size for DeepWalk or Node2Vec."""
    best_score = np.inf if score_name == 'rmse' else -np.inf
    best_params, best_X, best_y = None, None, None
    results = []

    for vector_size in embedding_sizes:
        print(f"[{method}] Evaluating embedding size: {vector_size}")

        if method == "deepwalk":
            X, y, _ = train_deepwalk_embeddings(G, df, vector_size, num_walks=num_walks, walk_length=walk_length)
        elif method == "node2vec":
            X, y, _ = train_node2vec_embeddings(G, df, vector_size, num_walks=num_walks, walk_length=walk_length,
                                                 return_weight=p, neighbor_weight=q)
        else:
            raise ValueError("Unknown method")

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=random_state)
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = []

        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            # model = GradientBoostingRegressor(loss='huber', n_estimators=100,
            #                                   max_depth=10, random_state=random_state)
            model = RandomForestRegressor(random_state=random_state)
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


def graph_report(G):
    # Assuming G is your NetworkX graph created from the df DataFrame  
    # and df contains the relevant house data with longitude and latitude  

    # Step 1: Calculate basic statistics  
    num_nodes = G.number_of_nodes()  
    num_edges = G.number_of_edges()  
    degrees = [len(list(G.neighbors(node))) for node in G.nodes]  # List of number of neighbors for each node  

    # Step 2: Compute minimum, maximum, and average number of neighbors  
    min_neighbors = min(degrees)  
    max_neighbors = max(degrees)  
    avg_neighbors = sum(degrees) / num_nodes if num_nodes > 0 else 0  

    # Step 3: Create a DataFrame for a better overview  
    degree_distribution = pd.DataFrame({  
        'Node ID': list(G.nodes),  
        'Num Neighbors': degrees  
    })  

    # Step 4: Summary statistics of the degree distribution  
    degree_stats = degree_distribution['Num Neighbors'].describe()  

    # Step 5: Output the results  
    print(f"Total number of nodes: {num_nodes}")  
    print(f"Total number of edges: {num_edges}")  
    print(f"Minimum number of neighbors: {min_neighbors}")  
    print(f"Maximum number of neighbors: {max_neighbors}")  
    print(f"Average number of neighbors: {avg_neighbors:.2f}")  
    # print("\nDegree Distribution Summary:")  
    # print(degree_stats)  