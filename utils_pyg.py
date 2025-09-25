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

from matplotlib.ticker import FuncFormatter
import re

# -----------------------------
# Baseline results cache function
# -----------------------------
def load_or_run_baseline(dataset_name, df, quiet=False):
    """Load baseline results from file or run and save them."""
    baseline_file = f"results-gpu/{dataset_name}/baseline_results.csv"
    os.makedirs(f"results-gpu/{dataset_name}", exist_ok=True)
    
    # Load from file if exists
    if os.path.exists(baseline_file):
        if not quiet:
            print("Loading baseline results from file...")
        baseline_df = pd.read_csv(baseline_file)
        return baseline_df.values.tolist()
    
    # Otherwise run and save
    if not quiet:
        print("Running baseline models...")
    
    X_base = df.drop(['price', 'id'], axis=1)
    y = df['price']
    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X_base, y, test_size=0.1, random_state=42
    )

    baseline_results = []
    for model_name, model in [
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
        ("RandomForest", RandomForestRegressor(random_state=42)),
    ]:
        baseline_start = time.time()
        metrics = fit_and_evaluate(model, X_train_base, y_train, X_test_base, y_test, verbose=False)
        baseline_time = time.time() - baseline_start
        
        baseline_results.append([
            model_name, "Raw", None, None, None, None, None,
            *metrics[:-1], 0, metrics[-1], baseline_time
        ])

    # Save results
    columns = [
        "BaseModel", "Method", "EmbeddingDim", "NumWalks", "WalkLength", "p", "q",
        "R2", "MAPE", "ACC", "RMSE", "MSE_log", 
        "Embedding_Time", "Regression_Time", "Total_Time"
    ]
    baseline_df = pd.DataFrame(baseline_results, columns=columns)
    baseline_df.to_csv(baseline_file, index=False)
    
    if not quiet:
        print(f"Baseline results saved to {baseline_file}")
    
    return baseline_results

# -----------------------------
# Enhanced dataset loader with spatial features
# -----------------------------
def load_dataset(dataset_name: str):
    """Load dataset by name (CA or MHD) and return df, numeric_features, binary_features."""
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
        
        # Add spatial features
        centroid_lat = df['latitude'].mean()
        centroid_lon = df['longitude'].mean()
        df['distance_from_center'] = np.sqrt(
            (df['latitude'] - centroid_lat)**2 + 
            (df['longitude'] - centroid_lon)**2
        )
        df['income_per_room'] = df['median_income'] / (df['total_rooms'] + 1)
        df['rooms_per_household'] = df['total_rooms'] / (df['households'] + 1)
        
        numeric_features.extend(['distance_from_center', 'income_per_room', 'rooms_per_household'])
        
        return df, numeric_features, binary_features

    elif dataset_name == "MHD":
        data = pd.read_excel('data/MHD-housing.xlsx')

        # # Whole dataset
        # df = data.dropna().reset_index(drop=True)

        # random subset
        filtered_data = data.copy()
        np.random.seed(42)
        shuffle_indices = np.random.choice(np.arange(filtered_data.shape[0]), size=20000, replace=False)
        df = filtered_data.iloc[shuffle_indices].reset_index(drop=True)
        df = df.dropna().reset_index(drop=True)

        df['id'] = df.index

        # Convert price from Rials to Million Rials
        df['price'] = df['price'] / 1000000  # Convert to million rials
        
        numeric_features = [
            'area_sq_m', 'age_years',
            'floor_number', 'number_of_bedrooms'
        ]
        binary_features = ['elevator', 'parking', 'storage', 'balcony', 'parquet',   
                    'ceramic_flooring', 'stone_façade', 'garden', 'renovated']
        
        # Add spatial features
        centroid_lat = df['latitude'].mean()
        centroid_lon = df['longitude'].mean()
        df['distance_from_center'] = np.sqrt(
            (df['latitude'] - centroid_lat)**2 + 
            (df['longitude'] - centroid_lon)**2
        )
        df['price_per_sqm'] = df['price'] / (df['area_sq_m'] + 1)  # Now in million rials per sqm
        df['age_sq'] = df['age_years'] ** 2
        
        numeric_features.extend(['distance_from_center', 'price_per_sqm', 'age_sq'])
        
        # print(f"MHD Dataset: Price converted to million rials")
        # print(f"Price range: {df['price'].min():.1f} to {df['price'].max():.1f} million rials")
        # print(f"Mean price: {df['price'].mean():.1f} million rials")
        
        return df, numeric_features, binary_features

    else:
        raise ValueError("Unknown dataset name. Use 'CA' or 'MHD'.")

# -----------------------------
# Optimized graph construction for PyG
# -----------------------------
def create_pyg_graph_from_dataframe(
    df,
    numeric_features,
    binary_features,
    k=15,  # Increased k for better connectivity
    scale_numeric=True,
    metric="euclidean",
    weight_edges=False,
    threshold_filter=None
):
    """
    Optimized KNN-based graph construction for PyTorch Geometric.
    Returns PyG Data object with optional edge weights and threshold filtering.
    """
    # Use only Geo features for graph structure
    X_geo = df[["latitude", "longitude"]].to_numpy()
    
    # Fit Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric)
    nn.fit(X_geo)
    distances, indices = nn.kneighbors(X_geo)

    # Create edge_index for PyG
    edges = []
    edge_weights = []
    
    for i in range(len(df)):
        for j_idx, j in enumerate(indices[i][1:]):  # skip itself
            dist = distances[i][j_idx+1]
            
            # Apply threshold filter if specified
            if threshold_filter and dist > threshold_filter:
                continue
                
            # enforce type constraint for MHD dataset
            if "type" in df.columns and df.loc[i, "type"] != df.loc[j, "type"]:
                continue  
            
            edges.append([i, j])
            if weight_edges:
                # Inverse distance weighting - closer points have higher weight
                edge_weights.append(1.0 / (dist + 1e-8))
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Add edge weights if enabled
    if weight_edges and edge_weights:
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    else:
        edge_attr = None
    
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
    pyg_data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
    pyg_data.num_nodes = len(df)
    
    return pyg_data

# -----------------------------
# Hybrid graph construction (KNN + Threshold)
# -----------------------------
def create_hybrid_graph(
    df,
    numeric_features,
    binary_features,
    k=10,
    threshold=1000,
    scale_numeric=True,
    dataset_name="CA"
):
    """
    Hybrid graph construction: KNN + Threshold-based connections.
    """
    X_geo = df[["latitude", "longitude"]].to_numpy()
    
    # 1. KNN for local connections
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nn.fit(X_geo)
    knn_distances, knn_indices = nn.kneighbors(X_geo)
    
    # 2. Threshold-based for regional connections
    threshold_nn = NearestNeighbors(radius=threshold, metric="euclidean")
    threshold_nn.fit(X_geo)
    threshold_distances, threshold_indices = threshold_nn.radius_neighbors(X_geo)
    
    edges = set()  # Use set to avoid duplicates
    
    # Add KNN connections
    for i in range(len(df)):
        for j_idx, j in enumerate(knn_indices[i][1:]):  # skip itself
            if "type" in df.columns and df.loc[i, "type"] != df.loc[j, "type"]:
                continue
            edges.add((min(i, j), max(i, j)))
    
    # Add threshold-based connections
    for i in range(len(df)):
        for j in threshold_indices[i]:
            if i != j:  # skip itself
                if "type" in df.columns and df.loc[i, "type"] != df.loc[j, "type"]:
                    continue
                edges.add((min(i, j), max(i, j)))
    
    edges = list(edges)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Node features
    if scale_numeric and numeric_features:
        scaler = StandardScaler()
        numeric_scaled = scaler.fit_transform(df[numeric_features])
    else:
        numeric_scaled = df[numeric_features].values
    
    if binary_features:
        binary_data = df[binary_features].values
        node_features = np.concatenate([numeric_scaled, binary_data], axis=1)
    else:
        node_features = numeric_scaled
    
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(df['price'].values, dtype=torch.float)
    
    pyg_data = Data(x=x, edge_index=edge_index, y=y)
    pyg_data.num_nodes = len(df)
    
    return pyg_data

# -----------------------------
# Unified graph embeddings training function
# -----------------------------
def train_graph_embeddings_pyg(pyg_data, vector_size=16, walk_length=10, 
                              context_size=5, walks_per_node=10, p=1.0, q=1.0, 
                              epochs=50, quiet=False):
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
    
    if not quiet:
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
    
    loader = model.loader(batch_size=4096, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
    
    # Training loop with optional tqdm
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
    
    losses = []
    if not quiet:
        # With progress bar
        with tqdm(total=epochs, desc=f"{method_name} Training") as pbar:
            for epoch in range(epochs):
                loss = train()
                losses.append(loss)
                pbar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    'Best': f'{min(losses):.4f}' if losses else 'N/A'
                })
                pbar.update(1)
    else:
        # Without progress bar
        for epoch in range(epochs):
            loss = train()
            losses.append(loss)
    
    # Get embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.embedding.weight.cpu().numpy()
    
    training_time = time.time() - start_time
    if not quiet:
        print(f"{method_name} training completed in {training_time:.2f} seconds")
    
    return embeddings, training_time, method_name

# -----------------------------
# Unified pipeline for graph embeddings
# -----------------------------
def train_graph_embeddings_pipeline_pyg(pyg_data, df, vector_size=16, num_walks=40, 
                                       walk_length=15, p=1.0, q=1.0, epochs=50, quiet=False):
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
        epochs=epochs,
        quiet=quiet
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
    if not quiet:
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
# Grid Search for embedding size with timing and LinearRegression
# -----------------------------
def grid_search_embedding_size_pyg(pyg_data, df, embedding_sizes, method="node2vec",
                                 score_name="r2", random_state=42, dataset_name="CA",
                                 num_walks=60, walk_length=15, p=1.0, q=1.0, epochs=30,
                                 quiet=False):
    """Perform grid search for embedding size using PyG with LinearRegression."""
    best_score = np.inf if score_name == 'rmse' else -np.inf
    best_params, best_X, best_y = None, None, None
    results = []
    timing_results = []

    # Determine method name for display
    if p == 1.0 and q == 1.0:
        method_name = "DeepWalk"
    else:
        method_name = f"Node2Vec (p={p}, q={q})"

    # Progress bar for grid search (only if not quiet)
    if not quiet:
        pbar_embedding = tqdm(embedding_sizes, desc=f"{method_name} Grid Search")
    else:
        pbar_embedding = embedding_sizes
    
    for vector_size in pbar_embedding:
        if not quiet:
            pbar_embedding.set_postfix({'Testing': f'vec_size={vector_size}'})
        
        # Train embeddings with timing
        X, y, _, emb_time, total_time = train_graph_embeddings_pipeline_pyg(
            pyg_data, df, vector_size=vector_size, 
            num_walks=num_walks, walk_length=walk_length, 
            p=p, q=q, epochs=epochs, quiet=quiet
        )

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=random_state)
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = []
        reg_times = []

        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            # Use LinearRegression instead of RandomForest
            model = GradientBoostingRegressor(random_state=random_state)            
            
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

    # Unified saving of all results in single files
    os.makedirs(f"results-gpu/{dataset_name}", exist_ok=True)
    
    # Master file for all embedding size results
    master_file = f"results-gpu/{dataset_name}/embedding_size_results.csv"
    
    # Load existing file if exists and append new results
    if os.path.exists(master_file):
        master_df = pd.read_csv(master_file)
    else:
        master_df = pd.DataFrame(columns=['Method', 'Embedding_Size', 'Score_Type', 'Score'])
    
    # Add new results
    new_results = pd.DataFrame({
        'Method': [method_name] * len(results),
        'Embedding_Size': [r[0] for r in results],
        'Score_Type': [score_name] * len(results),
        'Score': [r[1] for r in results]
    })
    
    master_df = pd.concat([master_df, new_results], ignore_index=True)
    master_df.to_csv(master_file, index=False)

    # Save timing results
    timing_file = f"results-gpu/{dataset_name}/timing_results.csv"
    if os.path.exists(timing_file):
        timing_master_df = pd.read_csv(timing_file)
    else:
        timing_master_df = pd.DataFrame(columns=['Method', 'Embedding_Size', 'Embedding_Time', 
                                               'Total_Pipeline_Time', 'Regression_Time'])
    
    new_timing = pd.DataFrame({
        'Method': [method_name] * len(timing_results),
        'Embedding_Size': [r[0] for r in timing_results],
        'Embedding_Time': [r[1] for r in timing_results],
        'Total_Pipeline_Time': [r[2] for r in timing_results],
        'Regression_Time': [r[3] for r in timing_results]
    })
    
    timing_master_df = pd.concat([timing_master_df, new_timing], ignore_index=True)
    timing_master_df.to_csv(timing_file, index=False)

    if not quiet:
        print(f"\n[{method_name}] Best embedding size: {best_params} with {score_name}: {best_score:.3f}")

    return best_params, best_X, best_y, new_results, new_timing

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

def compare_models(dataset_name, metric='RMSE', res_file_name="final_results.csv", quiet=False):
    """
    Single plot for selected metric comparison.
    - For each BaseModel and Method category, selects the best configuration based on the specified metric
    - For RMSE/MSE_log: lower is better (minimize)
    - For R2: higher is better (maximize)
    - Highlights the single best model across all methods with a red star.
    - Only saves the plot, does not display it.
    """
    # Load results
    file_path = f"results-gpu/{dataset_name}/{res_file_name}"
    df = pd.read_csv(file_path)

    # Categorize methods
    def categorize(method):
        if "DeepWalk" in str(method):
            return "DeepWalk"
        elif "Node2Vec" in str(method):
            return "Node2Vec"
        else:
            return "Raw"

    df["Method_Category"] = df["Method"].apply(categorize)

    # Determine optimization direction
    if metric in ['RMSE', 'MSE_log']:
        # Lower is better
        optimization_func = lambda x: x.idxmin()
        best_func = np.nanargmin
    else:  # R2
        # Higher is better  
        optimization_func = lambda x: x.idxmax()
        best_func = np.nanargmax

    # For each BaseModel and Method category, select the best configuration based on the specified metric
    best_configs = []
    
    for base_model in df["BaseModel"].unique():
        for method_cat in ["Raw", "DeepWalk", "Node2Vec"]:
            subset = df[(df["BaseModel"] == base_model) & (df["Method_Category"] == method_cat)]
            
            if not subset.empty:
                if method_cat == "Raw":
                    # For Raw, there's only one configuration
                    best_row = subset.iloc[0]
                else:
                    # For DeepWalk and Node2Vec, select the best based on the specified metric
                    best_idx = optimization_func(subset[metric])
                    best_row = subset.loc[best_idx]
                
                best_configs.append(best_row)

    # Create DataFrame with best configurations
    best_df = pd.DataFrame(best_configs)

    # Pivot table using best configurations
    order_types = ["Raw", "DeepWalk", "Node2Vec"]
    metric_pivot = best_df.pivot_table(index="BaseModel", columns="Method_Category", values=metric, aggfunc="mean").reindex(columns=order_types)

    # Base models for x-axis
    models = metric_pivot.index.tolist()
    x = np.arange(len(models))
    bar_width = 0.25

    # Create single plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

    # Single plot: Selected metric comparison
    all_metric_bars = []
    bar_positions = []
    
    for i, method_type in enumerate(order_types):
        if method_type in metric_pivot.columns:
            vals = metric_pivot[method_type].values
            positions = x + i * bar_width - bar_width
            ax.bar(positions, vals, width=bar_width, label=method_type, color=colors[i], alpha=0.8)
            all_metric_bars.extend(vals)
            bar_positions.extend(positions)

    # Highlight single best based on metric
    if all_metric_bars:
        best_idx = best_func(all_metric_bars)
        ax.scatter(bar_positions[best_idx], all_metric_bars[best_idx],
                   color="red", zorder=5, s=150, marker="*", label="Best Overall")

    # Set title and labels based on metric
    if metric == 'R2':
        title = f"R² Score Comparison - {dataset_name}"
        ylabel = "R²"
    else:
        title = f"{metric} Comparison - {dataset_name}"
        ylabel = metric

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Set y-axis limits
    if all_metric_bars:
        metric_min, metric_max = np.nanmin(all_metric_bars), np.nanmax(all_metric_bars)
        if metric == 'R2':
            ax.set_ylim(max(0, metric_min - 0.05), min(1, metric_max + 0.05))
        else:
            ax.set_ylim(metric_min * 0.9, metric_max * 1.1)

    # Add value labels on bars with scientific notation
    for i, method_type in enumerate(order_types):
        if method_type in metric_pivot.columns:
            vals = metric_pivot[method_type].values
            positions = x + i * bar_width - bar_width
            for pos, val in zip(positions, vals):
                if not np.isnan(val):
                    ax.text(pos, val * 1.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                    # # Use scientific notation for large/small numbers
                    # if abs(val) >= 1000 or (abs(val) <= 0.01 and val != 0):
                    #     ax.text(pos, val * 1.01, f'{val:.2e}', ha='center', va='bottom', fontsize=9)
                    # else:
                    #     ax.text(pos, val * 1.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save figure (without displaying)
    out_path = f"results-gpu/{dataset_name}/best_{metric}_comparison.png"
    os.makedirs(f"results-gpu/{dataset_name}", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to prevent display
    
    if not quiet:
        print(f"Best configurations comparison plot saved to {out_path}")
        print(f"Selected best configurations for each method (based on {metric}):")
        
        # Print best configurations summary
        for base_model in best_df["BaseModel"].unique():
            print(f"\n{base_model}:")
            for method_cat in ["Raw", "DeepWalk", "Node2Vec"]:
                subset = best_df[(best_df["BaseModel"] == base_model) & (best_df["Method_Category"] == method_cat)]
                if not subset.empty:
                    row = subset.iloc[0]
                    if method_cat == "Node2Vec":
                        p_val = row["p"] if "p" in row and not pd.isna(row["p"]) else "N/A"
                        q_val = row["q"] if "q" in row and not pd.isna(row["q"]) else "N/A"
                        print(f"  {method_cat}: p={p_val}, q={q_val}, {metric}={row[metric]:.4f}, R²={row['R2']:.4f}")
                    else:
                        print(f"  {method_cat}: {metric}={row[metric]:.4f}, R²={row['R2']:.4f}")

    return best_df