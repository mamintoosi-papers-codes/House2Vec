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
    fit_and_evaluate,
    grid_search_embedding_size_pyg,
    compare_models,
    load_or_run_baseline  # تابع جدید
)


def run_experiments_pyg(dataset_name, embedding_sizes=[2, 8, 16, 32, 64], 
                       num_walks=80, walk_length=10, p=3, q=1, epochs=30,
                       quiet=False):
    """Run complete experiments with timing tracking."""
    
    experiment_start_time = time.time()
    
    # -----------------------------
    # Load dataset & build PyG graph
    # -----------------------------
    if not quiet:
        print(f"Loading dataset: {dataset_name}")
    df, numeric_features, binary_features = load_dataset(dataset_name)
    
    graph_start_time = time.time()
    pyg_data = create_pyg_graph_from_dataframe(
        df, numeric_features, binary_features
    )
    graph_time = time.time() - graph_start_time
    if not quiet:
        print(f"Graph construction completed in {graph_time:.2f} seconds")
        print(f"PyG Graph: {pyg_data.num_nodes} nodes, {pyg_data.edge_index.shape[1]} edges")

    # Results columns with timing information
    columns = [
        "BaseModel", "Method", "EmbeddingDim", "NumWalks", "WalkLength", "p", "q",
        "R2", "MAPE", "ACC", "RMSE", "MSE_log", 
        "Embedding_Time", "Regression_Time", "Total_Time"
    ]
    results = []
    overall_timing = {"graph_construction": graph_time}

    # -----------------------------
    # Baseline (raw features only) - با قابلیت ذخیره/بارگذاری
    # -----------------------------
    if not quiet:
        print("\n=== Running Baseline Models ===")
    
    # استفاده از تابع جدید برای baseline
    baseline_results = load_or_run_baseline(dataset_name, df, quiet=quiet)
    results.extend(baseline_results)

    # -----------------------------
    # Node2Vec grid search (includes DeepWalk when p=q=1)
    # -----------------------------
    if not quiet:
        print("\n=== Running Node2Vec Grid Search (includes DeepWalk) ===")
    
    # Test all combinations of p and q (p=q=1 represents DeepWalk)
    for ip in range(1, p+1):
        for iq in range(1, q+1):
            # Determine method name based on parameters
            if ip == 1 and iq == 1:
                method_display_name = "DeepWalk"
            else:
                method_display_name = f"Node2Vec (p={ip}, q={iq})"
            
            if not quiet:
                print(f"\n--- Testing {method_display_name} ---")
            
            n2v_start_time = time.time()
            best_size, X_emb, y_emb, emb_results, timing_info = grid_search_embedding_size_pyg(
                pyg_data, df, embedding_sizes, method="node2vec", dataset_name=dataset_name,
                num_walks=num_walks, walk_length=walk_length, p=ip, q=iq, epochs=epochs,
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
                    model_name, method_display_name, best_size, num_walks, walk_length, ip, iq,
                    *metrics[:-1], best_timing['Embedding_Time'], reg_time, best_timing['Total_Pipeline_Time']
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
    parser.add_argument("--num-walks", type=int, default=80)
    parser.add_argument("--walk-length", type=int, default=10)
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs for embeddings")
    parser.add_argument("--embedding_sizes", nargs="+", type=int, default=[8, 16, 32, 64],
                        help="List of embedding sizes to try for grid search")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()

    run_experiments_pyg(
        args.dataset, 
        embedding_sizes=args.embedding_sizes,
        num_walks=args.num_walks, 
        walk_length=args.walk_length, 
        p=args.p, 
        q=args.q,
        epochs=args.epochs,
        quiet=args.quiet
    )