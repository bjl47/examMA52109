import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker import agglomerative
from cluster_maker.plotting_clustered import plot_clusters_2d

OUTPUT_DIR = "demo_output"
DATA_PATH = "data/difficult_dataset.csv"

def main(args):
    input_path = args[0] if args else DATA_PATH
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(input_path)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        print("Error: at least two numeric columns required")
        sys.exit(1)

    X = df[numeric_cols[:2]].to_numpy()
    base = os.path.splitext(os.path.basename(input_path))[0]

    for k in range(2, 6):
        print(f"\n=== Agglomerative clustering with k={k} ===")
        labels, _ = agglomerative.agglomerative(X, k=k)
        print("Cluster labels:", labels)

        fig, ax = plot_clusters_2d(X, labels, title=f"k={k}")
        plot_path = os.path.join(OUTPUT_DIR, f"{base}_agglomerative_k{k}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main(sys.argv[1:])
