from __future__ import annotations

import os
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import run_clustering

OUTPUT_DIR = "demo_output"
DATA_PATH = "data/simulated_data.csv"


def main(args: List[str]) -> None:
    # Allow optional path argument
    input_path = args[0] if args else DATA_PATH
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(input_path)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) < 2:
        print("Error: Dataset must have at least two numeric columns for clustering.")
        sys.exit(1)
    feature_cols = numeric_cols[:2]  # Use first two numeric columns

    base = os.path.splitext(os.path.basename(input_path))[0]

    # Try k = 2, 3, 4, 5 and collect metrics
    metrics_summary = []

    for k in (2, 3, 4, 5):
        print(f"\n=== Running k-means with k = {k} ===")
        result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k=k,
            standardise=True,
            output_path=os.path.join(OUTPUT_DIR, f"{base}_clustered_k{k}.csv"),
            random_state=42,
            compute_elbow=False
        )

        # Save cluster plot
        plot_path = os.path.join(OUTPUT_DIR, f"{base}_k{k}.png")
        result["fig_cluster"].savefig(plot_path, dpi=150)
        plt.close(result["fig_cluster"])

        # Save metrics
        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics_summary.append(metrics)

        print("Metrics:")
        for key, value in result.get("metrics", {}).items():
            print(f"  {key}: {value}")

    # Summarise metrics across k
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # Plot silhouette scores if available
    if "silhouette_score" in metrics_df.columns:
        plt.figure()
        plt.bar(metrics_df["k"], metrics_df["silhouette_score"])
        plt.xlabel("k")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette score for different k")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{base}_silhouette.png"), dpi=150)
        plt.close()

    # Additional plot: scatter of data colored by best k
    # Determine best k by maximum silhouette score
    if "silhouette_score" in metrics_df.columns:
        best_k = metrics_df.loc[metrics_df["silhouette_score"].idxmax(), "k"]
        print(f"\nBest k (max silhouette score): {best_k}")
        result_best = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k=int(best_k),
            standardise=True,
            random_state=42
        )
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            result_best["data"][feature_cols[0]],
            result_best["data"][feature_cols[1]],
            c=result_best["labels"],
            cmap="tab10",
            s=50,
            edgecolor="k"
        )
        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel(feature_cols[1])
        ax.set_title(f"Data colored by cluster (k={best_k})")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{base}_best_clusters.png"), dpi=150)
        plt.close(fig)

    print("\nAnalysis completed. Outputs saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main(sys.argv[1:])