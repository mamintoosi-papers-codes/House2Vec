# main_hpp.py
"""
Main script to run HPP experiments with DeepWalk and Node2Vec embeddings
on different housing datasets (California, MHD).
"""

import os
import argparse
from utils_embeddings import load_dataset, create_graph_from_dataframe, run_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HPP experiments on housing datasets")
    parser.add_argument("--dataset", type=str, default="CA", choices=["CA", "MHD"],
                        help="Dataset name: 'CA' for California or 'MHD' for MHD dataset")
    args = parser.parse_args()
    dataset_name = args.dataset

    os.makedirs(f"results/{dataset_name}", exist_ok=True)

    # Load dataset and parameters
    df, numeric_features, threshold = load_dataset(dataset_name)

    # Create graph
    G = create_graph_from_dataframe(df, numeric_features, distance_threshold=threshold)

    # Run experiments
    results = run_experiments(df, G, dataset_name)
    print(results)
