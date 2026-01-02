#!/usr/bin/env python
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path("simulations/snn_long_run/report/.mplconfig").resolve()),
)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuro_mod.core.spiking_net.analysis import SNNAnalyzer
from neuro_mod.execution.helpers.logger import Logger
from neuro_mod.visualization import journal_style
from neuro_mod.core.spiking_net.analysis.logic import helpers


def main() -> int:
    logger = Logger(name="SNNLongRunReport")
    repo_dir = Path(__file__).resolve().parents[1]
    base_dir = repo_dir / "simulations" / "snn_long_run"
    analysis_dir = base_dir / "analysis"
    report_dir = base_dir / "report"
    figures_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading analysis from {analysis_dir}")
    analyzer = SNNAnalyzer(analysis_dir)
    logger.info("Loaded SNNAnalyzer data")
    attractors = analyzer.get_attractors_data()
    transition = analyzer.get_transition_matrix()

    logger.info("Computing attractor summary metrics")
    identities = helpers.get_attractor_identities_in_order(attractors)
    occurrences_arr = analyzer.get_occurrences()
    mean_lifespans_arr, _ = analyzer.get_life_spans()
    num_clusters_arr = analyzer.get_num_clusters()
    entries = []
    all_occurrence_durations = []
    for i, identity in enumerate(identities):
        data = attractors[identity]
        occurrences = int(occurrences_arr[i]) if i < len(occurrences_arr) else int(data.get("#", 0))
        total_duration = float(data.get("total_duration", 0.0))
        durations = data.get("occurrence_durations", [])
        if durations:
            all_occurrence_durations.extend(durations)
        entries.append(
            {
                "key": identity,
                "idx": int(data.get("idx", -1)),
                "occurrences": occurrences,
                "total_duration_ms": total_duration,
                "mean_duration_ms": float(mean_lifespans_arr[i]) if i < len(mean_lifespans_arr) else 0.0,
                "cluster_size": int(num_clusters_arr[i]) if i < len(num_clusters_arr) else len(identity),
            }
        )

    entries.sort(key=lambda x: x["total_duration_ms"], reverse=True)
    durations = np.array(all_occurrence_durations, dtype=float)
    total_attractors = analyzer.get_num_states()
    total_occurrences = int(occurrences_arr.sum()) if occurrences_arr.size else 0
    total_duration_ms = sum(e["total_duration_ms"] for e in entries)
    mean_duration_ms = (total_duration_ms / total_occurrences) if total_occurrences else 0.0

    top = entries[0] if entries else None
    top_share = (top["total_duration_ms"] / total_duration_ms) if top and total_duration_ms else 0.0

    row_sums = transition.sum(axis=1)
    nonzero_rows = row_sums > 0
    probs = np.divide(transition[nonzero_rows], row_sums[nonzero_rows][:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(probs * np.log2(probs + 1e-12), axis=1)
    mean_entropy = float(np.nanmean(entropy)) if entropy.size else 0.0
    mean_row_sum = float(np.mean(row_sums)) if row_sums.size else 0.0
    self_transition_mean = float(np.mean(np.diag(transition))) if transition.size else 0.0
    density = float(np.count_nonzero(transition > 1e-6) / transition.size) if transition.size else 0.0

    total_seconds = float(analyzer.total_duration_ms) / 1000.0
    bin_size_s = 5.0
    time_edges = np.arange(0.0, total_seconds + bin_size_s, bin_size_s)
    time_centers = (time_edges[:-1] + time_edges[1:]) / 2.0

    # Plot: new attractors per 5s bin (derivative of unique count)
    if time_edges.size > 1:
        logger.info("Plotting new attractor discovery rate")
        journal_style.set_figure_size("double", aspect=0.35)
        unique_counts = np.array(
            [
                analyzer.get_unique_attractors_count_until_time(edge * 1e3)
                for edge in time_edges
            ],
            dtype=float,
        )
        new_rate = np.diff(unique_counts) / bin_size_s
        plt.figure(figsize=(10, 3.6))
        plt.plot(time_centers, new_rate, color="#1b9e77", linewidth=1.5)
        plt.xlabel("Time (s)")
        plt.ylabel("New attractors per second")
        plt.title("Discovery rate of new attractors (5s bins)")
        plt.tight_layout()
        plt.savefig(figures_dir / "new_attractor_rate.png", dpi=150)
        plt.close()

    # Plot: L2 norms of transition matrices across time (Analyzer)
    if time_edges.size > 1:
        logger.info("Plotting transition matrix L2 norms over time")
        journal_style.set_figure_size("double", aspect=0.35)
        l2_times, l2_norms = analyzer.get_transition_matrix_l2_norms_until_time(
            t=total_seconds,
            dt=bin_size_s,
        )
        plt.figure(figsize=(10, 3.6))
        plt.plot(l2_times, l2_norms, color="#e76f51", linewidth=1.5)
        plt.xlabel("Time (s)")
        plt.ylabel("L2 norm")
        plt.title("Transition matrix L2 norms over time (5s bins)")
        plt.tight_layout()
        plt.savefig(figures_dir / "tm_l2_density.png", dpi=150)
        plt.close()

    # Plot: mean lifespan vs occurrences
    if entries:
        logger.info("Plotting mean lifespan vs occurrences")
        occ = np.array([e["occurrences"] for e in entries], dtype=float)
        mean_life = np.array([e["mean_duration_ms"] for e in entries], dtype=float)
        cluster_sizes = np.array([e["cluster_size"] for e in entries], dtype=int)
        journal_style.set_figure_size("double", aspect=0.55)
        plt.figure()
        unique_sizes = np.unique(cluster_sizes)
        cmap = plt.get_cmap("tab20", len(unique_sizes))
        for i, size in enumerate(unique_sizes):
            mask = cluster_sizes == size
            plt.scatter(
                occ[mask],
                mean_life[mask],
                s=18,
                color=cmap(i),
                alpha=0.7,
                label=str(size),
            )
        plt.xscale("log")
        plt.xlabel("Occurrences (log scale)")
        plt.ylabel("Mean lifespan (ms)")
        plt.title("Mean attractor lifespan vs occurrences")
        plt.legend(
            title="Cluster count",
            ncol=1,
            frameon=False,
            fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )
        plt.gcf().subplots_adjust(right=0.78)
        plt.savefig(figures_dir / "lifespan_vs_occurrences.png", dpi=150)
        plt.close()

    # Plot: attractor size vs frequency and lifespan correlations
    if entries:
        logger.info("Plotting attractor size correlations")
        sizes = np.array([e["cluster_size"] for e in entries], dtype=float)
        occurrences = np.array([e["occurrences"] for e in entries], dtype=float)
        mean_lifespan = np.array([e["mean_duration_ms"] for e in entries], dtype=float)
        unique_sizes = np.unique(sizes)
        mean_occ_by_size = [
            np.mean(occurrences[sizes == size]) for size in unique_sizes
        ]
        mean_life_by_size = [
            np.mean(mean_lifespan[sizes == size]) for size in unique_sizes
        ]
        journal_style.set_figure_size("double", aspect=0.5)
        fig, axes = plt.subplots(1, 2, sharex=True)
        axes[0].scatter(sizes, occurrences, s=16, alpha=0.4, color="#2a9d8f")
        axes[0].plot(unique_sizes, mean_occ_by_size, color="#1f6f5b", linewidth=2)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Attractor size (cluster count)")
        axes[0].set_ylabel("Occurrences (log scale)")
        axes[0].set_title("Size vs occurrences")
        axes[1].scatter(sizes, mean_lifespan, s=16, alpha=0.4, color="#f4a261")
        axes[1].plot(unique_sizes, mean_life_by_size, color="#c26d3b", linewidth=2)
        axes[1].set_xlabel("Attractor size (cluster count)")
        axes[1].set_ylabel("Mean lifespan (ms)")
        axes[1].set_title("Size vs mean lifespan")
        fig.suptitle("Attractor size correlations")
        fig.tight_layout()
        fig.savefig(figures_dir / "size_correlations.png", dpi=150)
        plt.close(fig)

    # Plot: histogram of mean lifespan per attractor
    if entries:
        logger.info("Plotting mean lifespan histogram")
        mean_lifespans = np.array([e["mean_duration_ms"] for e in entries], dtype=float)
        journal_style.set_figure_size("single", aspect=0.75)
        plt.figure()
        plt.hist(mean_lifespans, bins=40, color="#8d99ae", edgecolor="white")
        plt.xlabel("Mean lifespan (ms)")
        plt.ylabel("Count")
        plt.title("Mean lifespan distribution across attractors")
        plt.tight_layout()
        plt.savefig(figures_dir / "mean_lifespan_hist.png", dpi=150)
        plt.close()

    # Plot: mean lifespan by attractor size
    if entries:
        logger.info("Plotting mean lifespan by attractor size")
        sizes = np.array([e["cluster_size"] for e in entries], dtype=float)
        mean_lifespan = np.array([e["mean_duration_ms"] for e in entries], dtype=float)
        unique_sizes = np.unique(sizes)
        mean_life_by_size = [
            np.mean(mean_lifespan[sizes == size]) for size in unique_sizes
        ]
        journal_style.set_figure_size("single", aspect=0.75)
        plt.figure()
        plt.bar(unique_sizes, mean_life_by_size, color="#3d5a80")
        plt.xlabel("Attractor size (cluster count)")
        plt.ylabel("Mean lifespan (ms)")
        plt.title("Mean lifespan by attractor size")
        plt.tight_layout()
        plt.savefig(figures_dir / "mean_lifespan_by_size.png", dpi=150)
        plt.close()

    # Plot: top attractors by total duration
    top_n = min(20, len(entries))
    if top_n:
        logger.info("Plotting top attractors by occupancy")
        journal_style.set_figure_size("double", aspect=0.55)
        labels = [f"A{e['idx']}" for e in entries[:top_n]]
        values = [e["total_duration_ms"] for e in entries[:top_n]]
        plt.figure()
        plt.bar(labels, values, color="#2a6f97")
        plt.ylabel("Total duration (ms)")
        plt.title("Top attractors by total occupancy")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / "top_attractors.png", dpi=150)
        plt.close()

    # Plot: occurrence duration distribution
    if durations.size:
        logger.info("Plotting occurrence duration distribution")
        journal_style.set_figure_size("single", aspect=0.75)
        plt.figure()
        plt.hist(durations, bins=40, color="#6c757d", edgecolor="white")
        plt.xlabel("Occurrence duration (ms)")
        plt.ylabel("Count")
        plt.title("Occurrence duration distribution")
        plt.tight_layout()
        plt.savefig(figures_dir / "duration_hist.png", dpi=150)
        plt.close()

    # Plot: transition matrix for top attractors
    top_m = min(30, len(entries))
    if top_m:
        logger.info("Plotting transition matrix heatmap for top attractors")
        journal_style.set_figure_size("single", aspect=0.9)
        top_indices = [e["idx"] for e in entries[:top_m]]
        sub = transition[np.ix_(top_indices, top_indices)]
        plt.figure()
        plt.imshow(np.log10(sub + 1e-6), cmap="magma", aspect="auto")
        plt.colorbar(label="log10(prob + 1e-6)")
        plt.title("Transition matrix (top attractors)")
        plt.xlabel("To")
        plt.ylabel("From")
        plt.tight_layout()
        plt.savefig(figures_dir / "transition_heatmap_top.png", dpi=150)
        plt.close()

    report_path = report_dir / "snn_long_run_report.md"
    logger.info(f"Writing report to {report_path}")
    report_lines = [
        "# SNN long run analysis summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source: `{analysis_dir}`",
        "",
        "## KPIs",
        "",
        "| KPI | Value | Note |",
        "| --- | ---: | --- |",
        f"| Total attractors | {total_attractors} | Unique attractor states |",
        f"| Total occurrences | {total_occurrences} | Sum over all attractor visits |",
        f"| Total occupancy | {total_duration_ms:.1f} ms | Sum of occurrence durations |",
        f"| Mean occurrence | {mean_duration_ms:.2f} ms | Total occupancy / occurrences |",
        f"| Top attractor share | {top_share:.2%} | Share of total occupancy |",
        f"| Mean row sum | {mean_row_sum:.3f} | Transition row normalization check |",
        f"| Mean self-transition | {self_transition_mean:.4f} | Avg diagonal of transition matrix |",
        f"| Transition density | {density:.2%} | Entries > 1e-6 |",
        f"| Mean transition entropy | {mean_entropy:.3f} bits | Row-wise entropy |",
        "",
        "## Visuals",
        "",
        "![New attractor rate](figures/new_attractor_rate.png)",
        "",
        "Discovery rate of new attractors per second (5s bins).",
        "",
        "![TM L2 density](figures/tm_l2_density.png)",
        "",
        "L2 norm between consecutive transition matrices (Analyzer, 5s bins).",
        "",
        "![Lifespan vs occurrences](figures/lifespan_vs_occurrences.png)",
        "",
        "Mean attractor lifespan vs occurrences (log x-axis).",
        "",
        "![Size correlations](figures/size_correlations.png)",
        "",
        "Correlation between attractor size and frequency metrics. Lines show per-size means.",
        "",
        "![Mean lifespan histogram](figures/mean_lifespan_hist.png)",
        "",
        "Histogram of mean lifespan per attractor.",
        "",
        "![Mean lifespan by size](figures/mean_lifespan_by_size.png)",
        "",
        "Mean lifespan grouped by attractor size.",
        "",
        "![Top attractors](figures/top_attractors.png)",
        "",
        "Top 20 attractors by total occupancy. Highlights the dominant states over the run.",
        "",
        "![Duration histogram](figures/duration_hist.png)",
        "",
        "Distribution of individual occurrence durations. Shorter events dominate the tail.",
        "",
        "![Transition heatmap](figures/transition_heatmap_top.png)",
        "",
        "Transition probabilities among the top 30 attractors (log-scaled). Brighter means more likely transitions.",
        "",
        "## Notes",
        "",
        f"- dt: {analyzer.dt} s; total sim duration: {analyzer.total_duration_ms} ms.",
        "- starts/ends unit: seconds; occurrence durations are in ms.",
        f"- time bin size for time-series plots: {bin_size_s:.0f} s.",
    ]
    report_path.write_text("\n".join(report_lines))
    logger.info("Report generation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
