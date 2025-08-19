# utils_embeddings.py
"""
Utilities for dataset loading, graph creation, regression, and experiment runner.
"""

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree

# import embedding functions
from embeddings_utils import train_deepwalk_embeddings, train_node2vec_embeddings


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
        df = pd.read_csv("data/MHD-housing.csv")
        df = df.dropna().reset_index(drop=True)
        df['id'] = df.index

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

    for _, row in df.iterrows():
        G.add_node(row['id'], **row.to_dict())

    pairs = tree.query_pairs(distance_threshold / 111000)

    for idx1, idx2 in tqdm(pairs, desc="Building graph"):
        geo_distance = np.linalg.norm(coordinates[idx1] - coordinates[idx2])
        weight = 1 / (1 + geo_distance)
        if weight > 0:
            G.add_edge(df.at[idx1, 'id'], df.at[idx2, 'id'], weight=weight)
    return G


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
# Main experiment runner
# -----------------------------
def run_experiments(df, G, dataset_name):
    """Run experiments for baseline, DeepWalk, and Node2Vec embeddings."""

    results = []

    # -------------------
    # Baseline (raw features)
    # -------------------
    X_base = df.drop(['price', 'id'], axis=1)
    y = df['price']
    X_train_base, X_test_base, y_train, y_test = train_test_split(X_base, y, test_size=0.1, random_state=42)

    for model_name, model in [
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(random_state=42))
    ]:
        metrics = fit_and_evaluate(model, X_train_base, y_train, X_test_base, y_test, verbose=False)
        results.append([f"{model_name} (Raw)", *metrics])

    # -------------------
    # DeepWalk embeddings
    # -------------------
    X_dw, y_dw, _ = train_deepwalk_embeddings(G, df, vector_size=16)
    X_train, X_test, y_train, y_test = train_test_split(X_dw, y_dw, test_size=0.1, random_state=42)

    for model_name, model in [
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(random_state=42))
    ]:
        metrics = fit_and_evaluate(model, X_train, y_train, X_test, y_test, verbose=False)
        results.append([f"{model_name} (DeepWalk)", *metrics])

    # -------------------
    # Node2Vec embeddings
    # -------------------
    X_n2v, y_n2v, _ = train_node2vec_embeddings(G, df, vector_size=16)
    X_train, X_test, y_train, y_test = train_test_split(X_n2v, y_n2v, test_size=0.1, random_state=42)

    for model_name, model in [
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(random_state=42))
    ]:
        metrics = fit_and_evaluate(model, X_train, y_train, X_test, y_test, verbose=False)
        results.append([f"{model_name} (Node2Vec)", *metrics])

    # Save results
    os.makedirs(f"results/{dataset_name}", exist_ok=True)
    results_df = pd.DataFrame(results, columns=["Model", "R2", "MAPE", "Accuracy", "RMSE", "MSE_log"])
    results_df.to_excel(f"results/{dataset_name}/model_results.xlsx", index=False)
    print(f"Results saved to results/{dataset_name}/model_results.xlsx")
    return results_df
