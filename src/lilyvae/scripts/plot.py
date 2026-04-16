"""CLI entry point for generating all paper figures."""

import argparse
from pathlib import Path

from lilyvae.config import PAPER_DIR
from lilyvae.plotting.style import apply_style


ALL_FIGURES = [
    "dataset_summary",
    "variables_dist",
    "reconstruction_scatter",
    "roc_comparison",
    "umap_comparison",
    "alpha_sweep",
    "r2_unsup_vs_semi",
    "variable_correlations",
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

    apply_style()
    figures = ALL_FIGURES if args.figure == "all" else [args.figure]

    for fig_name in figures:
        print(f"\nGenerating {fig_name}...")
        try:
            if fig_name == "dataset_summary":
                from lilyvae.plotting.data_figures import plot_dataset_summary
                plot_dataset_summary(args.output_dir)
            elif fig_name == "variables_dist":
                from lilyvae.plotting.data_figures import plot_variables_distribution
                plot_variables_distribution(args.output_dir)
            elif fig_name == "reconstruction_scatter":
                from lilyvae.plotting.model_figures import plot_reconstruction_scatter
                plot_reconstruction_scatter(args.output_dir, args.device)
            elif fig_name == "roc_comparison":
                from lilyvae.plotting.model_figures import plot_roc_comparison
                plot_roc_comparison(args.output_dir)
            elif fig_name == "umap_comparison":
                from lilyvae.plotting.latent_figures import plot_umap_comparison
                plot_umap_comparison(args.output_dir)
            elif fig_name == "alpha_sweep":
                from lilyvae.plotting.model_figures import plot_alpha_sweep
                plot_alpha_sweep(args.output_dir)
            elif fig_name == "r2_unsup_vs_semi":
                from lilyvae.plotting.zeroshot_figures import plot_r2_unsup_vs_semi
                plot_r2_unsup_vs_semi(args.output_dir)
            elif fig_name == "variable_correlations":
                from lilyvae.plotting.data_figures import plot_variable_correlations
                plot_variable_correlations(args.output_dir)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
