"""CLI entry point for generating all paper figures."""

import argparse
from pathlib import Path

from lilyvae.config import PAPER_DIR


ALL_FIGURES = [
    "expedition_map",
    "dataset_summary",
    "lithology_counts",
    "variables_dist",
    "reconstruction_scatter",
    "roc_comparison",
    "generated_variables_grid",
    "umap_lithology",
    "umap_expeditions",
    "r2_vs_observations",
    "zeroshot_scatter",
]


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--figure",
        choices=ALL_FIGURES + ["all"],
        default="all",
        help="Which figure to generate",
    )
    parser.add_argument("--output-dir", type=Path, default=PAPER_DIR)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    figures = ALL_FIGURES if args.figure == "all" else [args.figure]

    for fig_name in figures:
        print(f"\nGenerating {fig_name}...")
        try:
            if fig_name == "expedition_map":
                from lilyvae.plotting.data_figures import plot_expedition_map
                plot_expedition_map(args.output_dir)
            elif fig_name == "dataset_summary":
                from lilyvae.plotting.data_figures import plot_dataset_summary
                plot_dataset_summary(args.output_dir)
            elif fig_name == "lithology_counts":
                from lilyvae.plotting.data_figures import plot_lithology_counts
                plot_lithology_counts(args.output_dir)
            elif fig_name == "variables_dist":
                from lilyvae.plotting.data_figures import plot_variables_distribution
                plot_variables_distribution(args.output_dir)
            elif fig_name == "reconstruction_scatter":
                from lilyvae.plotting.model_figures import plot_reconstruction_scatter
                plot_reconstruction_scatter(args.output_dir, args.device)
            elif fig_name == "roc_comparison":
                from lilyvae.plotting.model_figures import plot_roc_comparison
                plot_roc_comparison(args.output_dir)
            elif fig_name == "generated_variables_grid":
                from lilyvae.plotting.model_figures import plot_generated_variables_grid
                plot_generated_variables_grid(args.output_dir, args.device)
            elif fig_name == "umap_lithology":
                from lilyvae.plotting.latent_figures import plot_umap_lithology
                plot_umap_lithology(args.output_dir)
            elif fig_name == "umap_expeditions":
                from lilyvae.plotting.latent_figures import plot_umap_expeditions
                plot_umap_expeditions(args.output_dir)
            elif fig_name == "r2_vs_observations":
                from lilyvae.plotting.zeroshot_figures import plot_r2_vs_observations
                plot_r2_vs_observations(args.output_dir)
            elif fig_name == "zeroshot_scatter":
                from lilyvae.plotting.zeroshot_figures import plot_zeroshot_scatter
                plot_zeroshot_scatter(args.output_dir)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
