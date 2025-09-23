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
import time
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
        # threshold = 4000   # meters
        return df, numeric_features, binary_features

    elif dataset_name == "MHD":
        data= pd.read_excel('data/MHD-housing.xlsx')

        filtered_data = data.copy()
        # Filter the data for the specified region
        # filtered_data = data[(data['longitude'] >= 59.4) & (data['longitude'] <= 59.7) &
        #                     (data['latitude'] >= 36.2) & (data['latitude'] <= 36.45)]
        np.random.seed(42)
        shuffle_indices = np.random.choice(np.arange(filtered_data.shape[0]), size=2000, replace=False,)
        df = filtered_data.iloc[shuffle_indices].reset_index(drop=True)

        df = df.dropna().reset_index(drop=True)
        df['id'] = df.index

        numeric_features = [
            'area_sq_m', 'age_years',
            'floor_number', 'number_of_bedrooms'
        ]
        binary_features = ['elevator', 'parking', 'storage', 'balcony', 'parquet',   
                    'ceramic_flooring', 'stone_façade', 'garden', 'renovated'] 
        # threshold = 40   # meters
        return df, numeric_features, binary_features

    else:
        raise ValueError("Unknown dataset name. Use 'CA' or 'MHD'.")


# -----------------------------
# Graph construction for PyG
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
# Unified graph embeddings training function
# -----------------------------
def train_graph_embeddings_pyg(pyg_data, vector_size=16, walk_length=10, 
                              context_size=5, walks_per_node=10, p=1.0, q=1.0, 
                              epochs=50):
    """
    Train graph embeddings using PyG's Node2Vec.
    
    Parameters:
    - p, q: if both are 1.0, equivalent to DeepWalk
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine method name based on parameters
    if p == 1.0 and q == 1.0:
        method_name = "DeepWalk"
    else:
        method_name = f"Node2Vec (p={p}, q={q})"
    
    print(f"Training {method_name} with vector_size={vector_size}")
    
    start_time = time.time()
    
    # Create model
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
    
    # Training loop with tqdm
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
    
    # Use tqdm for progress tracking
    losses = []
    with tqdm(total=epochs, desc=f"{method_name} Training") as pbar:
        for epoch in range(epochs):
            loss = train()
            losses.append(loss)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Best': f'{min(losses):.4f}' if losses else 'N/A'
            })
            pbar.update(1)
    
    # Get embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.embedding.weight.cpu().numpy()
    
    training_time = time.time() - start_time
    print(f"{method_name} training completed in {training_time:.2f} seconds")
    
    return embeddings, training_time, method_name


# -----------------------------
# Unified pipeline for graph embeddings
# -----------------------------
def train_graph_embeddings_pipeline_pyg(pyg_data, df, vector_size=16, num_walks=40, 
                                       walk_length=15, p=1.0, q=1.0, epochs=50):
    """
    Full pipeline for graph embeddings using PyG.
    
    Parameters:
    - p, q: parameters for random walks (p=q=1 for DeepWalk)
    """
    
    start_time = time.time()
    
    # Train embeddings
    embeddings, emb_training_time, method_name = train_graph_embeddings_pyg(
        pyg_data, 
        vector_size=vector_size,
        walk_length=walk_length,
        walks_per_node=num_walks,
        p=p,
        q=q,
        epochs=epochs
    )
    
    # Determine prefix based on method
    if p == 1.0 and q == 1.0:
        prefix = "deepwalk"
    else:
        prefix = "node2vec"
    
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
    
    total_time = time.time() - start_time
    print(f"Full pipeline completed in {total_time:.2f} seconds")
    
    return X_combined, y, emb_df, emb_training_time, total_time


# -----------------------------
# Regression & evaluation with timing
# -----------------------------
def fit_and_evaluate(model, X_train, y_train, X_test, y_test, filename=None, verbose=True):
    """Train regression model and evaluate with multiple metrics."""
    start_time = time.time()
    
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
    
    training_time = time.time() - start_time

    if verbose:
        print(f"R2: {r2:.3f}, MAPE: {mape:.3f}, RMSE: {rmse:.3f}, Acc: {acc:.3f}")
        print(f"Regression training time: {training_time:.2f} seconds")

    if filename:
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.savefig(filename)
        plt.close()

    return r2, mape, acc, rmse, mse_log, training_time

# -----------------------------
# Grid Search for embedding size with timing
# -----------------------------
def grid_search_embedding_size_pyg(pyg_data, df, embedding_sizes, method="node2vec",
                                 score_name="r2", random_state=42, dataset_name="CA",
                                 num_walks=60, walk_length=15, p=1.0, q=1.0, epochs=30):
    """Perform grid search for embedding size using PyG."""
    best_score = np.inf if score_name == 'rmse' else -np.inf
    best_params, best_X, best_y = None, None, None
    results = []
    timing_results = []

    # Determine method name for display
    if p == 1.0 and q == 1.0:
        method_name = "DeepWalk"
    else:
        method_name = f"Node2Vec (p={p}, q={q})"

    # Progress bar for grid search
    pbar_embedding = tqdm(embedding_sizes, desc=f"{method_name} Grid Search")
    
    for vector_size in pbar_embedding:
        pbar_embedding.set_postfix({'Testing': f'vec_size={vector_size}'})
        
        # Train embeddings with timing
        X, y, _, emb_time, total_time = train_graph_embeddings_pipeline_pyg(
            pyg_data, df, vector_size=vector_size, 
            num_walks=num_walks, walk_length=walk_length, 
            p=p, q=q, epochs=epochs
        )

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=random_state)
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = []
        reg_times = []

        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            model = RandomForestRegressor(random_state=random_state)
            
            if score_name == 'rmse':
                _, _, _, score, _, reg_time = fit_and_evaluate(model, X_tr, y_tr, X_val, y_val, verbose=False)
            else: # r2
                score, _, _, _, _, reg_time = fit_and_evaluate(model, X_tr, y_tr, X_val, y_val, verbose=False)
            
            scores.append(score)
            reg_times.append(reg_time)

        mean_score = np.mean(scores)
        mean_reg_time = np.mean(reg_times)
        results.append((vector_size, mean_score))
        timing_results.append((vector_size, emb_time, total_time, mean_reg_time))

        # Update best score and parameters
        if score_name == 'rmse':
            condition = mean_score < best_score
        else:
            condition = mean_score > best_score
            
        if condition:
            best_score, best_params, best_X, best_y = mean_score, vector_size, X, y

    # Save results to results-gpu folder
    os.makedirs(f"results-gpu/{dataset_name}", exist_ok=True)
    
    # Save main results
    results_df = pd.DataFrame(results, columns=['Embedding_Size', score_name])
    results_df.to_csv(f"results-gpu/{dataset_name}/{method_name.replace(' ', '_').replace('(', '').replace(')', '')}_embedding_size_results.csv", index=False)

    # Save timing results
    timing_df = pd.DataFrame(timing_results, 
                           columns=['Embedding_Size', 'Embedding_Time', 'Total_Pipeline_Time', 'Regression_Time'])
    timing_df.to_csv(f"results-gpu/{dataset_name}/{method_name.replace(' ', '_').replace('(', '').replace(')', '')}_timing_results.csv", index=False)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Embedding_Size'], results_df[score_name], marker='o')
    plt.scatter(best_params, best_score, color='red')
    plt.title(f"{method_name} - Embedding Size vs {score_name.upper()}")
    plt.xlabel("Embedding Size")
    plt.ylabel(score_name.upper())
    plt.grid(True)
    plt.savefig(f"results-gpu/{dataset_name}/{method_name.replace(' ', '_').replace('(', '').replace(')', '')}_embedding_size_plot.png")
    plt.close()

    print(f"\n[{method_name}] Best embedding size: {best_params} with {score_name}: {best_score:.3f}")

    return best_params, best_X, best_y, results_df, timing_df

def graph_report(G):
    """Generate basic graph statistics report."""
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