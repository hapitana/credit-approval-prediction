from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt


# 1. paths and configuration

project_root = Path(__file__).resolve().parent.parent

input_path = (
    project_root
    / "reports"
    / "a9_threshold_sensitivity_results.csv"
)

reports_path = project_root / "reports"
figures_path = reports_path / "figures"

reports_path.mkdir(parents=True, exist_ok=True)
figures_path.mkdir(parents=True, exist_ok=True)

reference_threshold = 0.50

scenario_order = [
    "all_features",
    "without_a9",
    "without_a9_top_proxy",
]

scenario_labels = {
    "all_features": "All features",
    "without_a9": "Without A9",
    "without_a9_top_proxy": "Without A9 + top proxies",
}


# 2. load results

if not input_path.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_path}"
    )

results_df = pl.read_csv(input_path)

required_columns = [
    "scenario",
    "threshold",
    "roc_auc_oof",
    "balanced_accuracy",
    "approval_rate_gap",
    "tpr_gap",
    "fpr_gap",
]

missing_columns = [
    column
    for column in required_columns
    if column not in results_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("A9 FAIRNESS-PERFORMANCE TRADE-OFF ANALYSIS")
print(f"Input file: {input_path}")
print(f"Total experiment rows: {results_df.height}")


# 3. compare scenarios at the same threshold

reference_df = results_df.filter(
    pl.col("threshold") == reference_threshold
)

if reference_df.height == 0:
    raise ValueError(
        f"No results found for threshold {reference_threshold}"
    )

reference_df = reference_df.with_columns(
    pl.col("scenario").alias("reference_scenario")
)

comparison_df = reference_df.select(
    [
        "scenario",
        "threshold",
        "roc_auc_oof",
        "balanced_accuracy",
        "approval_rate_gap",
        "tpr_gap",
        "fpr_gap",
    ]
).sort("scenario")

baseline_rows = comparison_df.filter(
    pl.col("scenario") == "all_features"
)

if baseline_rows.height != 1:
    raise ValueError(
        "Expected exactly one all_features baseline "
        f"at threshold {reference_threshold}"
    )

baseline = baseline_rows.row(0, named=True)

comparison_df = comparison_df.with_columns(
    [
        (
            pl.col("roc_auc_oof")
            - baseline["roc_auc_oof"]
        ).alias("roc_auc_change"),

        (
            pl.col("balanced_accuracy")
            - baseline["balanced_accuracy"]
        ).alias("balanced_accuracy_change"),

        (
            pl.col("approval_rate_gap")
            - baseline["approval_rate_gap"]
        ).alias("approval_rate_gap_change"),

        (
            pl.col("tpr_gap")
            - baseline["tpr_gap"]
        ).alias("tpr_gap_change"),

        (
            pl.col("fpr_gap")
            - baseline["fpr_gap"]
        ).alias("fpr_gap_change"),
    ]
)

comparison_df = comparison_df.with_columns(
    [
        pl.col("scenario")
        .replace(scenario_labels)
        .alias("scenario_label"),

        (
            -pl.col("approval_rate_gap_change")
        ).alias("approval_gap_reduction"),

        (
            -pl.col("tpr_gap_change")
        ).alias("tpr_gap_reduction"),

        (
            -pl.col("fpr_gap_change")
        ).alias("fpr_gap_reduction"),
    ]
)

comparison_path = (
    reports_path
    / "a9_fairness_performance_comparison.csv"
)

comparison_df.write_csv(comparison_path)

print("\nCOMPARISON AT THRESHOLD 0.50")
print(
    comparison_df.select(
        [
            "scenario_label",
            "roc_auc_oof",
            "balanced_accuracy",
            "approval_rate_gap",
            "tpr_gap",
            "fpr_gap",
        ]
    )
)

print(f"\nComparison saved to: {comparison_path}")


# 4. calculate Pareto frontier

objective_columns = [
    "roc_auc_oof",
    "balanced_accuracy",
    "approval_rate_gap",
    "tpr_gap",
    "fpr_gap",
]

maximize_columns = [
    "roc_auc_oof",
    "balanced_accuracy",
]

minimize_columns = [
    "approval_rate_gap",
    "tpr_gap",
    "fpr_gap",
]

candidate_rows = results_df.to_dicts()

pareto_flags = []

for i, candidate in enumerate(candidate_rows):
    dominated = False

    for j, challenger in enumerate(candidate_rows):
        if i == j:
            continue

        at_least_as_good = True
        strictly_better = False

        for column in maximize_columns:
            if challenger[column] < candidate[column]:
                at_least_as_good = False
                break

            if challenger[column] > candidate[column]:
                strictly_better = True

        if not at_least_as_good:
            continue

        for column in minimize_columns:
            if challenger[column] > candidate[column]:
                at_least_as_good = False
                break

            if challenger[column] < candidate[column]:
                strictly_better = True

        if at_least_as_good and strictly_better:
            dominated = True
            break

    pareto_flags.append(not dominated)

pareto_df = results_df.with_columns(
    pl.Series("is_pareto_optimal", pareto_flags)
)

pareto_df = pareto_df.with_columns(
    pl.col("scenario")
    .replace(scenario_labels)
    .alias("scenario_label")
)

pareto_path = (
    reports_path
    / "a9_fairness_performance_pareto.csv"
)

pareto_df.write_csv(pareto_path)

pareto_frontier = pareto_df.filter(
    pl.col("is_pareto_optimal")
).sort(
    ["roc_auc_oof", "balanced_accuracy"],
    descending=[True, True],
)

print("\nPARETO FRONTIER")
print(
    pareto_frontier.select(
        [
            "scenario_label",
            "threshold",
            "roc_auc_oof",
            "balanced_accuracy",
            "approval_rate_gap",
            "tpr_gap",
            "fpr_gap",
        ]
    )
)

print(f"\nPareto results saved to: {pareto_path}")


# 5. plot ROC-AUC versus approval rate gap

plt.figure(figsize=(10, 7))

for scenario_name in scenario_order:
    scenario_df = results_df.filter(
        pl.col("scenario") == scenario_name
    )

    x_values = scenario_df["approval_rate_gap"].to_list()
    y_values = scenario_df["roc_auc_oof"].to_list()

    plt.scatter(
        x_values,
        y_values,
        label=scenario_labels[scenario_name],
        alpha=0.75,
    )

pareto_plot_df = pareto_df.filter(
    pl.col("is_pareto_optimal")
)

plt.scatter(
    pareto_plot_df["approval_rate_gap"].to_list(),
    pareto_plot_df["roc_auc_oof"].to_list(),
    marker="x",
    s=90,
    label="Pareto-optimal",
)

plt.xlabel("Approval Rate Gap (absolute difference)")
plt.ylabel("Out-of-Fold ROC-AUC")
plt.title("Predictive Performance vs Approval Rate Gap")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

scatter_path = (
    figures_path
    / "a9_roc_auc_vs_approval_gap.png"
)

plt.savefig(scatter_path, dpi=300)
plt.close()

print(f"\nTrade-off plot saved to: {scatter_path}")


# 6. plot balanced accuracy at threshold 0.50

plot_df = comparison_df.sort("scenario")

scenario_names = plot_df["scenario_label"].to_list()

balanced_accuracy_values = (
    plot_df["balanced_accuracy"].to_list()
)

approval_gap_values = (
    plot_df["approval_rate_gap"].to_list()
)

x_positions = list(range(len(scenario_names)))
bar_width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(
    [
        position - bar_width / 2
        for position in x_positions
    ],
    balanced_accuracy_values,
    bar_width,
    label="Balanced accuracy",
)

ax1.set_xlabel("Model scenario")
ax1.set_ylabel("Balanced accuracy")
ax1.set_xticks(x_positions)
ax1.set_xticklabels(
    scenario_names,
    rotation=15,
    ha="right",
)

ax2 = ax1.twinx()

ax2.bar(
    [
        position + bar_width / 2
        for position in x_positions
    ],
    approval_gap_values,
    bar_width,
    label="Approval rate gap",
)

ax2.set_ylabel("Approval rate gap")

handles_1, labels_1 = ax1.get_legend_handles_labels()
handles_2, labels_2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles_1 + handles_2,
    labels_1 + labels_2,
    loc="best",
)

plt.title(
    "Balanced Accuracy and Approval Rate Gap "
    "at Threshold 0.50"
)

fig.tight_layout()

bar_path = (
    figures_path
    / "a9_balanced_accuracy_approval_gap.png"
)

plt.savefig(bar_path, dpi=300)
plt.close(fig)

print(f"Threshold comparison plot saved to: {bar_path}")


# 7. summarize the analysis

summary_rows = []

for row in comparison_df.to_dicts():
    summary_rows.append(
        {
            "scenario": row["scenario"],
            "threshold": row["threshold"],
            "roc_auc_oof": row["roc_auc_oof"],
            "roc_auc_change": row["roc_auc_change"],
            "balanced_accuracy": row["balanced_accuracy"],
            "balanced_accuracy_change": row[
                "balanced_accuracy_change"
            ],
            "approval_rate_gap": row["approval_rate_gap"],
            "approval_gap_reduction": row[
                "approval_gap_reduction"
            ],
            "tpr_gap": row["tpr_gap"],
            "tpr_gap_reduction": row["tpr_gap_reduction"],
            "fpr_gap": row["fpr_gap"],
            "fpr_gap_reduction": row["fpr_gap_reduction"],
        }
    )

summary_df = pl.DataFrame(summary_rows)

summary_path = (
    reports_path
    / "a9_fairness_performance_summary.csv"
)

summary_df.write_csv(summary_path)

print("\nFINAL SUMMARY")
print(summary_df)

print(f"\nSummary saved to: {summary_path}")
print("\nAnalysis completed.")