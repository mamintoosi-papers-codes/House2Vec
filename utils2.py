"""
Utilities for dataset loading, graph creation, embeddings (DeepWalk & Node2Vec),
regression, evaluation, grid search, and experiment runner.
Using PyTorch Geometric for graph embeddings.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors


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
        df = pd.read_excel('data/MHD-housing.xlsx')
        df = df.dropna().reset_index(drop=True)
        df['id'] = df.index

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
# Graph construction (برای PyG)
# -----------------------------
def create_pyg_graph_from_dataframe(
    df,
    numeric_features,
    binary_features,
    k=10,
    scale_numeric=True,
    metric="euclidean"
):
    """
    KNN-based graph construction for PyTorch Geometric.
    Returns PyG Data object.
    """
    # Use only Geo features for graph structure
    X_geo = df[["latitude", "longitude"]].to_numpy()
    
    # Fit Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nn.fit(X_geo)
    distances, indices = nn.kneighbors(X_geo)

    # Create edge_index for PyG
    edges = []
    for i in range(len(df)):
        for j_idx, j in enumerate(indices[i][1:]):  # skip itself
            # enforce type constraint for MHD dataset
            if "type" in df.columns and df.loc[i, "type"] != df.loc[j, "type"]:
                continue  
            edges.append([i, j])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Node features: numeric + binary features
    if scale_numeric and numeric_features:
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(df[numeric_features])
    else:
        numeric_scaled = df[numeric_features].values
    
    # Combine numeric and binary features
    if binary_features:
        binary_data = df[binary_features].values
        node_features = np.concatenate([numeric_scaled, binary_data], axis=1)
    else:
        node_features = numeric_scaled
    
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(df['price'].values, dtype=torch.float)
    
    # Create PyG Data object
    pyg_data = Data(x=x, edge_index=edge_index, y=y)
    pyg_data.num_nodes = len(df)
    
    return pyg_data


# -----------------------------
# تابع یکپارچه برای DeepWalk/Node2Vec با PyG
# -----------------------------
def train_graph_embeddings_pyg(pyg_data, method="deepwalk", vector_size=16, 
                              walk_length=10, context_size=5, walks_per_node=10, 
                              p=None, q=None, epochs=50):
    """
    Train graph embeddings using PyG's Node2Vec.
    
    Parameters:
    - method: "deepwalk" or "node2vec"
    - p, q: if None and method="deepwalk", automatically set to 1.0
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # تعیین پارامترهای p و q بر اساس روش
    if method == "deepwalk":
        p = 1.0 if p is None else p
        q = 1.0 if q is None else q
        method_name = "DeepWalk"
    elif method == "node2vec":
        p = p if p is not None else 1.0
        q = q if q is not None else 1.0
        method_name = "Node2Vec"
    else:
        raise ValueError("Method must be 'deepwalk' or 'node2vec'")
    
    print(f"Training {method_name} with p={p}, q={q}, vector_size={vector_size}")
    
    # ایجاد مدل
    model = Node2Vec(
        pyg_data.edge_index,
        embedding_dim=vector_size,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=p,
        q=q,
        num_negative_samples=1,
        sparse=True
    ).to(device)
    
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
    
    # Training loop با tqdm
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    # استفاده از tqdm برای نمایش پیشرفت
    losses = []
    with tqdm(total=epochs, desc=f"{method_name} Training") as pbar:
        for epoch in range(epochs):
            loss = train()
            losses.append(loss)
            
            # بروزرسانی progress bar
            pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Best': f'{min(losses):.4f}' if losses else 'N/A'
            })
            pbar.update(1)
    
    # Get embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.embedding.weight.cpu().numpy()
    
    return embeddings


# -----------------------------
# تابع یکپارچه برای pipeline کامل
# -----------------------------
def train_graph_embeddings_pipeline_pyg(pyg_data, df, method="deepwalk", 
                                       vector_size=16, num_walks=40, 
                                       walk_length=15, p=None, q=None, epochs=50):
    """
    Full pipeline for graph embeddings using PyG.
    
    Parameters:
    - method: "deepwalk" or "node2vec"
    - p, q: parameters for Node2Vec (ignored for DeepWalk if not specified)
    """
    
    # آموزش embeddings
    embeddings = train_graph_embeddings_pyg(
        pyg_data, 
        method=method,
        vector_size=vector_size,
        walk_length=walk_length,
        walks_per_node=num_walks,
        p=p,
        q=q,
        epochs=epochs
    )
    
    # تعیین prefix بر اساس روش
    prefix = method.lower()
    
    # Create embeddings DataFrame
    emb_df = pd.DataFrame(
        embeddings,
        columns=[f"{prefix}_emb_{i}" for i in range(embeddings.shape[1])]
    )
    
    # Combine with original features
    feature_columns = [col for col in df.columns if col not in ['price', 'id']]
    X_original = df[feature_columns]
    X_combined = pd.concat([X_original.reset_index(drop=True), emb_df], axis=1)
    y = df['price']
    
    return X_combined, y, emb_df


# -----------------------------
# توابع قدیمی برای compatibility (اختیاری)
# -----------------------------
def train_deepwalk_embeddings_pyg(pyg_data, df, vector_size=16, num_walks=40, 
                                 walk_length=15, epochs=50):
    """Compatibility function - uses the unified pipeline."""
    return train_graph_embeddings_pipeline_pyg(
        pyg_data, df, method="deepwalk", vector_size=vector_size,
        num_walks=num_walks, walk_length=walk_length, epochs=epochs
    )


def train_node2vec_embeddings_pyg(pyg_data, df, vector_size=16, num_walks=40, 
                                 walk_length=15, p=1.0, q=1.0, epochs=50):
    """Compatibility function - uses the unified pipeline."""
    return train_graph_embeddings_pipeline_pyg(
        pyg_data, df, method="node2vec", vector_size=vector_size,
        num_walks=num_walks, walk_length=walk_length, p=p, q=q, epochs=epochs
    )


# -----------------------------
# Regression & evaluation (بدون تغییر)
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
# Grid Search for embedding size (با PyG)
# -----------------------------
def grid_search_embedding_size_pyg(pyg_data, df, embedding_sizes, method="deepwalk",
                                 score_name="r2", random_state=42, dataset_name="CA",
                                 num_walks=60, walk_length=15, p=3, q=1, epochs=30):
    """Perform grid search for embedding size using PyG."""
    best_score = np.inf if score_name == 'rmse' else -np.inf
    best_params, best_X, best_y = None, None, None
    results = []

    # Progress bar برای grid search
    pbar_embedding = tqdm(embedding_sizes, desc=f"{method.upper()} Grid Search")
    
    for vector_size in pbar_embedding:
        pbar_embedding.set_postfix({'Testing': f'vec_size={vector_size}'})
        
        # استفاده از تابع یکپارچه
        if method == "deepwalk":
            X, y, _ = train_graph_embeddings_pipeline_pyg(
                pyg_data, df, method="deepwalk", vector_size=vector_size, 
                num_walks=num_walks, walk_length=walk_length, epochs=epochs
            )
        elif method == "node2vec":
            X, y, _ = train_graph_embeddings_pipeline_pyg(
                pyg_data, df, method="node2vec", vector_size=vector_size, 
                num_walks=num_walks, walk_length=walk_length, p=p, q=q, epochs=epochs
            )
        else:
            raise ValueError("Unknown method")

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=random_state)
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = []

        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            model = RandomForestRegressor(random_state=random_state)
            
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
            
        if condition:
            best_score, best_params, best_X, best_y = mean_score, vector_size, X, y

    results_df = pd.DataFrame(results, columns=['Embedding Size', score_name])
    print(f"\n[{method}] Best embedding size: {best_params} with {score_name}: {best_score:.3f}")

    os.makedirs(f"results/{dataset_name}", exist_ok=True)
    results_df.to_csv(f"results/{dataset_name}/{method}_embedding_size_results_pyg.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Embedding Size'], results_df[score_name], marker='o')
    plt.scatter(best_params, best_score, color='red')
    plt.title(f"{method.upper()} (PyG) - Embedding Size vs {score_name.upper()}")
    plt.xlabel("Embedding Size")
    plt.ylabel(score_name.upper())
    plt.grid(True)
    plt.savefig(f"results/{dataset_name}/{method}_embedding_size_plot_pyg.png")
    plt.close()

    return best_params, best_X, best_y, results_df


def graph_report(G):
    num_nodes = G.number_of_nodes()  
    num_edges = G.number_of_edges()  
    degrees = [len(list(G.neighbors(node))) for node in G.nodes]

    min_neighbors = min(degrees)  
    max_neighbors = max(degrees)  
    avg_neighbors = sum(degrees) / num_nodes if num_nodes > 0 else 0  

    print(f"Total number of nodes: {num_nodes}")  
    print(f"Total number of edges: {num_edges}")  
    print(f"Minimum number of neighbors: {min_neighbors}")  
    print(f"Maximum number of neighbors: {max_neighbors}")  
    print(f"Average number of neighbors: {avg_neighbors:.2f}")