"""
Main experiment runner for housing price prediction (CA or MHD).
Using PyTorch Geometric for graph embeddings.
"""

import argparse
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import time

from utils_pyg import (
    load_dataset,
    create_pyg_graph_from_dataframe,
    create_hybrid_graph,
    fit_and_evaluate,
    grid_search_embedding_size_pyg,
    compare_models,
    load_or_run_baseline
)

def get_unified_walk_strategies():
    """
    Return unified walk strategies for both CA and MHD datasets.
    These parameters are chosen based on theoretical foundations and preliminary results.
    """
    strategies = {
        "dfs_balanced": {
            "p": 0.5, 
            "q": 0.5,
            "justification": "Balanced DFS-style exploration (p<1, q<1) for discovering structurally similar communities beyond immediate neighborhoods"
        },
        "dfs_local": {
            "p": 0.5, 
            "q": 1.0,
            "justification": "DFS-style with local bias (p<1, q=1) for exploring distant nodes while maintaining some local connectivity"
        },
        "deepwalk": {
            "p": 1.0, 
            "q": 1.0,
            "justification": "Standard DeepWalk (p=1, q=1) as baseline for uniform random walk exploration"
        },
        "bfs_local": {
            "p": 1.0, 
            "q": 0.5,
            "justification": "BFS-style with exploration bias (p=1, q<1) for focusing on local neighborhoods with some diversity"
        }
    }
    return strategies

def optimize_walk_parameters(dataset_name):
    """
    Return unified parameters for both datasets.
    """
    unified_params = {
        'num_walks': 80,
        'walk_length': 15,
        'epochs': 80,
        'k': 10,
        'strategies': get_unified_walk_strategies()
    }
    
    # فقط threshold بر اساس dataset متفاوت باشد
    if dataset_name == "CA":
        unified_params['threshold'] = 4000
    else:  # MHD
        unified_params['threshold'] = 100
        
    return unified_params


def run_experiments_pyg(dataset_name, embedding_sizes=[2, 4, 8], 
                       graph_method="threshold", quiet=False):
    """Run experiments with unified parameters for both datasets."""
    
    experiment_start_time = time.time()
    
    # Get unified parameters
    optimized_params = optimize_walk_parameters(dataset_name)
    strategies = optimized_params['strategies']
    
    # Load dataset
    if not quiet:
        print(f"Loading dataset: {dataset_name}")
    df, numeric_features, binary_features = load_dataset(dataset_name)
    
    # Graph construction
    graph_start_time = time.time()
    
    if graph_method == "threshold":
        pyg_data = create_pyg_graph_from_dataframe(
            df, numeric_features, binary_features,
            k=optimized_params['k'],
            threshold_filter=optimized_params['threshold']
        )
        graph_type = "KNN with Threshold"
    elif graph_method == "hybrid":
        pyg_data = create_hybrid_graph(
            df, numeric_features, binary_features,
            k=optimized_params['k'], 
            threshold=optimized_params['threshold'],
            dataset_name=dataset_name
        )
        graph_type = "Hybrid"
    else:  # "knn"
        pyg_data = create_pyg_graph_from_dataframe(
            df, numeric_features, binary_features,
            k=optimized_params['k']
        )
        graph_type = "KNN"
    
    graph_time = time.time() - graph_start_time
    if not quiet:
        print(f"{graph_type} graph construction completed in {graph_time:.2f} seconds")
        print(f"PyG Graph: {pyg_data.num_nodes} nodes, {pyg_data.edge_index.shape[1]} edges")
        print(f"Testing {len(strategies)} unified walk strategies")

    # Results columns
    columns = [
        "BaseModel", "Method", "EmbeddingDim", "NumWalks", "WalkLength", "p", "q",
        "R2", "MAPE", "ACC", "RMSE", "MSE_log", 
        "Embedding_Time", "Regression_Time", "Total_Time", "Graph_Type", "Strategy"
    ]
    results = []
    overall_timing = {"graph_construction": graph_time}

    # Baseline models
    if not quiet:
        print("\n=== Running Baseline Models ===")
    
    baseline_results = load_or_run_baseline(dataset_name, df, quiet=quiet)
    for result in baseline_results:
        result.extend([graph_type, "Baseline"])
    results.extend(baseline_results)

    # Test unified walk strategies
    if not quiet:
        print("\n=== Testing Unified Walk Strategies ===")
    
    for strategy_name, strategy_config in strategies.items():
        ip = strategy_config["p"]
        iq = strategy_config["q"]
        justification = strategy_config["justification"]
        
        if ip == 1.0 and iq == 1.0:
            method_display_name = "DeepWalk"
        else:
            method_display_name = f"Node2Vec (p={ip}, q={iq})"
        
        if not quiet:
            print(f"\n--- Strategy: {strategy_name} ---")
            print(f"Parameters: p={ip}, q={iq}")
            print(f"Testing {method_display_name} ---")
        
        n2v_start_time = time.time()
        best_size, X_emb, y_emb, emb_results, timing_info = grid_search_embedding_size_pyg(
            pyg_data, df, embedding_sizes, method="node2vec", dataset_name=dataset_name,
            num_walks=optimized_params['num_walks'], 
            walk_length=optimized_params['walk_length'], 
            p=ip, q=iq, epochs=optimized_params['epochs'],
            quiet=quiet
        )
        overall_timing[f"{strategy_name}"] = time.time() - n2v_start_time
        
        # Test best configuration
        X_train, X_test, y_train, y_test = train_test_split(
            X_emb, y_emb, test_size=0.1, random_state=42
        )
        
        best_timing = timing_info[timing_info['Embedding_Size'] == best_size].iloc[0]
        
        for model_name, model in [
            ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
            ("RandomForest", RandomForestRegressor(random_state=42)),
        ]:
            reg_start = time.time()
            metrics = fit_and_evaluate(model, X_train, y_train, X_test, y_test, verbose=False)
            reg_time = time.time() - reg_start
            
            results.append([
                model_name, method_display_name, best_size, 
                optimized_params['num_walks'], optimized_params['walk_length'], 
                ip, iq,
                *metrics[:-1], best_timing['Embedding_Time'], reg_time, 
                best_timing['Total_Pipeline_Time'], graph_type, strategy_name
            ])

    # Save results
    df_results = pd.DataFrame(results, columns=columns)
    
    # Format numeric columns
    numeric_cols = df_results.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_results[col] = df_results[col].apply(
            lambda x: int(x) if pd.notna(x) and x == int(x) else round(x, 4) if pd.notna(x) else x
        )
    
    out_file = f"results-gpu/{dataset_name}/final_results.csv"
    os.makedirs(f"results-gpu/{dataset_name}", exist_ok=True)
    df_results.to_csv(out_file, index=False)
    
    # Save strategy information
    strategy_df = pd.DataFrame([
        {**{"strategy": k}, **v} for k, v in strategies.items()
    ])
    strategy_df.to_csv(f"results-gpu/{dataset_name}/walk_strategies.csv", index=False)
    
    # Timing information
    overall_timing["total_experiment"] = time.time() - experiment_start_time
    timing_df = pd.DataFrame(list(overall_timing.items()), columns=['Component', 'Time_Seconds'])
    timing_df.to_csv(f"results-gpu/{dataset_name}/experiment_timing.csv", index=False)
    
    # if not quiet:
    print(f"\n=== Experiment Summary ===")
    print(f"Dataset: {dataset_name}")
    print(f"Graph type: {graph_type}") 
    print(f"Unified parameters applied to both datasets")
    print(f"Saved results to {out_file}")

    compare_models(dataset_name, 'R2', quiet=quiet)
    compare_models(dataset_name, 'RMSE', quiet=quiet)
    compare_models(dataset_name, 'MSE_log', quiet=quiet)

    return df_results, timing_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run housing price prediction experiments with PyG")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: CA or MHD")
    parser.add_argument("--graph-method", type=str, default="hybrid", 
                       choices=["knn", "hybrid", "threshold"],
                       help="Graph construction method: knn, hybrid, or threshold")
    parser.add_argument("--embedding-sizes", nargs="+", type=int, default=[8, 16, 32, 64],
                       help="List of embedding sizes to try for grid search")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()

    run_experiments_pyg(
        args.dataset, 
        embedding_sizes=args.embedding_sizes,
        graph_method=args.graph_method,
        quiet=args.quiet
    )