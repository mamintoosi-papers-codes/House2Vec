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


def optimize_walk_parameters(dataset_name):
    """
    Return optimized walk parameters based on dataset characteristics.
    """
    if dataset_name == "CA":
        return {
            'num_walks': 50,  # Increased for larger dataset
            'walk_length': 10,  # Increased for better exploration
            'p': 0.5,  # Decreased for more exploration
            'q': 2.0,  # Increased to focus on local neighbors
            'epochs': 30,
            'k': 10,  # Increased k for better connectivity
            'threshold': 4000  # 4km threshold for CA
        }
    elif dataset_name == "MHD":
        return {
            'num_walks': 50,
            'walk_length': 10,
            'p': 0.5,  # Decreased for more exploration
            'q': 2.0,  # Increased to focus on local neighbors
            'epochs': 40,
            'k': 10,  # Moderate k for MHD
            'threshold': 40  # 40m threshold for MHD
        }


def run_experiments_pyg(dataset_name, embedding_sizes=[2, 8, 16], 
                       graph_method="hybrid",  # Options: "knn", "hybrid", "threshold"
                       quiet=False):
    """Run complete experiments with timing tracking and optimized parameters."""
    
    experiment_start_time = time.time()
    
    # Get optimized parameters based on dataset
    optimized_params = optimize_walk_parameters(dataset_name)
    
    # -----------------------------
    # Load dataset & build PyG graph
    # -----------------------------
    if not quiet:
        print(f"Loading dataset: {dataset_name}")
    df, numeric_features, binary_features = load_dataset(dataset_name)
    
    graph_start_time = time.time()
    
    # Choose graph construction method
    if graph_method == "hybrid":
        pyg_data = create_hybrid_graph(
            df, numeric_features, binary_features,
            k=optimized_params['k'], 
            threshold=optimized_params['threshold'],
            dataset_name=dataset_name
        )
        graph_type = "Hybrid (KNN + Threshold)"
    elif graph_method == "threshold":
        # Use KNN with threshold filtering
        threshold = optimized_params['threshold'] if dataset_name == "CA" else 40
        pyg_data = create_pyg_graph_from_dataframe(
            df, numeric_features, binary_features,
            k=optimized_params['k'],
            threshold_filter=threshold
        )
        graph_type = "KNN with Threshold"
    else:  # "knn" - default
        pyg_data = create_pyg_graph_from_dataframe(
            df, numeric_features, binary_features,
            k=optimized_params['k']
        )
        graph_type = "KNN"
    
    graph_time = time.time() - graph_start_time
    if not quiet:
        print(f"{graph_type} graph construction completed in {graph_time:.2f} seconds")
        print(f"PyG Graph: {pyg_data.num_nodes} nodes, {pyg_data.edge_index.shape[1]} edges")

    # Results columns with timing information
    columns = [
        "BaseModel", "Method", "EmbeddingDim", "NumWalks", "WalkLength", "p", "q",
        "R2", "MAPE", "ACC", "RMSE", "MSE_log", 
        "Embedding_Time", "Regression_Time", "Total_Time", "Graph_Type"
    ]
    results = []
    overall_timing = {"graph_construction": graph_time}

    # -----------------------------
    # Baseline (raw features only) - with caching
    # -----------------------------
    if not quiet:
        print("\n=== Running Baseline Models ===")
    
    # Use cached baseline results
    baseline_results = load_or_run_baseline(dataset_name, df, quiet=quiet)
    # Add graph type to baseline results
    for result in baseline_results:
        result.append(graph_type)
    results.extend(baseline_results)

    # -----------------------------
    # Node2Vec grid search (includes DeepWalk when p=q=1)
    # -----------------------------
    if not quiet:
        print("\n=== Running Node2Vec Grid Search (includes DeepWalk) ===")
    
    # Test optimized parameter combinations
    p_values = [optimized_params['p']]
    q_values = [optimized_params['q']]
    
    # Also include DeepWalk (p=q=1) for comparison
    if optimized_params['p'] != 1.0 or optimized_params['q'] != 1.0:
        p_values.append(1.0)
        q_values.append(1.0)
    
    for ip in p_values:
        for iq in q_values:
            # Determine method name based on parameters
            if ip == 1.0 and iq == 1.0:
                method_display_name = "DeepWalk"
            else:
                method_display_name = f"Node2Vec (p={ip}, q={iq})"
            
            if not quiet:
                print(f"\n--- Testing {method_display_name} ---")
            
            n2v_start_time = time.time()
            best_size, X_emb, y_emb, emb_results, timing_info = grid_search_embedding_size_pyg(
                pyg_data, df, embedding_sizes, method="node2vec", dataset_name=dataset_name,
                num_walks=optimized_params['num_walks'], 
                walk_length=optimized_params['walk_length'], 
                p=ip, q=iq, epochs=optimized_params['epochs'],
                quiet=quiet
            )
            overall_timing[f"{method_display_name.replace(' ', '_')}"] = time.time() - n2v_start_time
            
            # Test best configuration
            X_train, X_test, y_train, y_test = train_test_split(
                X_emb, y_emb, test_size=0.1, random_state=42
            )
            
            # Get timing for best configuration
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
                    best_timing['Total_Pipeline_Time'], graph_type
                ])

    # -----------------------------
    # Save results to CSV
    # -----------------------------
    df_results = pd.DataFrame(results, columns=columns)
    
    # Format numeric columns
    numeric_cols = df_results.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_results[col] = df_results[col].apply(
            lambda x: int(x) if pd.notna(x) and x == int(x) else round(x, 4) if pd.notna(x) else x
        )
    
    # Save to results-gpu folder
    out_file = f"results-gpu/{dataset_name}/final_results.csv"
    os.makedirs(f"results-gpu/{dataset_name}", exist_ok=True)
    df_results.to_csv(out_file, index=False)
    
    # Save timing information
    overall_timing["total_experiment"] = time.time() - experiment_start_time
    timing_df = pd.DataFrame(list(overall_timing.items()), columns=['Component', 'Time_Seconds'])
    timing_df.to_csv(f"results-gpu/{dataset_name}/experiment_timing.csv", index=False)
    
    # if not quiet:
    print(f"\n=== Experiment Summary ===")
    print(f"Graph type: {graph_type}")
    print(f"Optimized parameters: {optimized_params}")
    print(f"Saved results to {out_file}")
    print(f"Total experiment time: {overall_timing['total_experiment']:.2f} seconds")
    print(f"Timing breakdown saved to results-gpu/{dataset_name}/experiment_timing.csv")

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