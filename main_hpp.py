"""
Main experiment runner for housing price prediction (CA or MHD).
Performs baseline models, DeepWalk, and Node2Vec with grid search for embedding size.
"""

import argparse
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from utils import (
    load_dataset,
    create_graph_from_dataframe,
    fit_and_evaluate,
    grid_search_embedding_size,
    graph_report
)


def run_experiments(dataset_name, embedding_sizes=[2, 8, 16, 32, 64], num_walks=80, walk_length=10, p=3, q=1):
    # -----------------------------
    # Load dataset & build graph
    # -----------------------------
    df, numeric_features, binary_features = load_dataset(dataset_name)
    G = create_graph_from_dataframe(df, numeric_features, binary_features)

    graph_report(G)

    columns = [
        "BaseModel", "Method", "EmbeddingDim",
        "NumWalks", "WalkLength", "p", "q",
        "R2", "MAPE", "ACC", "RMSE", "MSE_log"
    ]
    results = []

    # -----------------------------
    # Baseline (raw features only)
    # -----------------------------
    X_base = df.drop(['price', 'id'], axis=1)
    y = df['price']
    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X_base, y, test_size=0.1, random_state=42
    )

    for model_name, model in [
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
        ("RandomForest", RandomForestRegressor(random_state=42)),
    ]:
        metrics = fit_and_evaluate(model, X_train_base, y_train, X_test_base, y_test, verbose=False)
        results.append([
            model_name, "Raw",
            None, None, None, None, None,   # EmbeddingDim, NumWalks, WalkLength, p, q
            *metrics
        ])

    # -----------------------------
    # DeepWalk with grid search
    # -----------------------------
    best_dw_size, X_dw, y_dw, _ = grid_search_embedding_size(
        df, G, embedding_sizes, method="deepwalk", dataset_name=dataset_name,
        num_walks=num_walks, walk_length=walk_length
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_dw, y_dw, test_size=0.1, random_state=42
    )
    for model_name, model in [
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
        ("RandomForest", RandomForestRegressor(random_state=42)),
    ]:
        metrics = fit_and_evaluate(model, X_train, y_train, X_test, y_test, verbose=False)
        results.append([
            model_name, "DeepWalk",
            best_dw_size, num_walks, walk_length,
            None, None,   # p, q
            *metrics
        ])

    # -----------------------------
    # Node2Vec with grid search
    # -----------------------------
    for ip in range(1, p+1):
        for iq in range(1, q+1):
            best_n2v_size, X_n2v, y_n2v, _ = grid_search_embedding_size(
                df, G, embedding_sizes, method="node2vec", dataset_name=dataset_name,
                num_walks=num_walks, walk_length=walk_length, p=ip, q=iq
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X_n2v, y_n2v, test_size=0.1, random_state=42
            )
            for model_name, model in [
                ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
                ("RandomForest", RandomForestRegressor(random_state=42)),
            ]:
                metrics = fit_and_evaluate(model, X_train, y_train, X_test, y_test, verbose=False)
                results.append([
                    model_name, "Node2Vec",
                    best_n2v_size, num_walks, walk_length,
                    ip, iq,
                    *metrics
                ])

    # -----------------------------
    # Save results to CSV
    # -----------------------------
    df_results = pd.DataFrame(results, columns=columns)
    out_file = f"results/{dataset_name}/final_results.csv"
    os.makedirs(f"results/{dataset_name}", exist_ok=True)
    df_results.to_csv(out_file, index=False)
    print(f"Saved results to {out_file}")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run housing price prediction experiments")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: CA or MHD")
    parser.add_argument("--num-walks", type=int, default=80)
    parser.add_argument("--walk-length", type=int, default=10)
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--embedding_sizes", nargs="+", type=int, default=[8, 16, 32, 64],
                        help="List of embedding sizes to try for grid search")
    args = parser.parse_args()

    run_experiments(args.dataset, embedding_sizes=args.embedding_sizes,
                    num_walks=args.num_walks, walk_length=args.walk_length, p=args.p, q=args.q)
