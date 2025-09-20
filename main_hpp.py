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
)


def run_experiments(dataset_name, embedding_sizes=[2, 8, 16, 32, 64]):
    # -----------------------------
    # Load dataset & build graph
    # -----------------------------
    df, numeric_features, threshold = load_dataset(dataset_name)
    G = create_graph_from_dataframe(df, numeric_features, threshold)

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
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(random_state=42)),
    ]:
        metrics = fit_and_evaluate(model, X_train_base, y_train, X_test_base, y_test, verbose=False)
        results.append([f"{model_name} (Raw)", *metrics])

    # -----------------------------
    # DeepWalk with grid search
    # -----------------------------
    best_dw_size, X_dw, y_dw, _ = grid_search_embedding_size(
        df, G, embedding_sizes, method="deepwalk", dataset_name=dataset_name
    )
    X_train, X_test, y_train, y_test = train_test_split(X_dw, y_dw, test_size=0.1, random_state=42)
    for model_name, model in [
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(random_state=42)),
    ]:
        metrics = fit_and_evaluate(model, X_train, y_train, X_test, y_test, verbose=False)
        results.append([f"{model_name} (DeepWalk-{best_dw_size})", *metrics])

    # -----------------------------
    # Node2Vec with grid search
    # -----------------------------
    best_n2v_size, X_n2v, y_n2v, _ = grid_search_embedding_size(
        df, G, embedding_sizes, method="node2vec", dataset_name=dataset_name
    )
    X_train, X_test, y_train, y_test = train_test_split(X_n2v, y_n2v, test_size=0.1, random_state=42)
    for model_name, model in [
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(random_state=42)),
    ]:
        metrics = fit_and_evaluate(model, X_train, y_train, X_test, y_test, verbose=False)
        results.append([f"{model_name} (Node2Vec-{best_n2v_size})", *metrics])

    # -----------------------------
    # Save results
    # -----------------------------
    os.makedirs(f"results/{dataset_name}", exist_ok=True)
    results_df = pd.DataFrame(
        results, columns=["Model", "R2", "MAPE", "Accuracy", "RMSE", "MSE_log"]
    )
    results_df.to_excel(f"results/{dataset_name}/final_results.xlsx", index=False)
    print(f"✅ Results saved to results/{dataset_name}/final_results.xlsx")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run housing price prediction experiments")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: CA or MHD")
    parser.add_argument("--embedding_sizes", nargs="+", type=int, default=[8, 16, 32, 64],
                        help="List of embedding sizes to try for grid search")
    args = parser.parse_args()

    run_experiments(args.dataset, embedding_sizes=args.embedding_sizes)
