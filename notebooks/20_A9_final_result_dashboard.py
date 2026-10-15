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


# 2. load data

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

print("A9 FINAL RESULTS DASHBOARD")
print(f"Input rows: {results_df.height}")


# 3. calculate two-objective Pareto frontier

pareto_input = results_df.select(
    [
        "scenario",
        "threshold",
        "roc_auc_oof",
        "balanced_accuracy",
        "approval_rate_gap",
        "tpr_gap",
        "fpr_gap",
    ]
).to_dicts()

pareto_flags = []

for i, candidate in enumerate(pareto_input):
    dominated = False

    for j, challenger in enumerate(pareto_input):
        if i == j:
            continue

        roc_auc_not_worse = (
            challenger["roc_auc_oof"]
            >= candidate["roc_auc_oof"]
        )

        gap_not_worse = (
            challenger["approval_rate_gap"]
            <= candidate["approval_rate_gap"]
        )

        strictly_better = (
            challenger["roc_auc_oof"]
            > candidate["roc_auc_oof"]
            or challenger["approval_rate_gap"]
            < candidate["approval_rate_gap"]
        )

        if (
            roc_auc_not_worse
            and gap_not_worse
            and strictly_better
        ):
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
    / "a9_final_pareto_frontier.csv"
)

pareto_df.write_csv(pareto_path)

pareto_frontier = pareto_df.filter(
    pl.col("is_pareto_optimal")
).sort(
    ["approval_rate_gap", "roc_auc_oof"],
    descending=[False, True],
)

print("\nTWO-OBJECTIVE PARETO FRONTIER")
print(
    pareto_frontier.select(
        [
            "scenario_label",
            "threshold",
            "roc_auc_oof",
            "approval_rate_gap",
            "balanced_accuracy",
            "tpr_gap",
            "fpr_gap",
        ]
    )
)

print(
    f"\nPareto-optimal experiments: "
    f"{pareto_frontier.height} / {results_df.height}"
)

print(f"Pareto file saved to: {pareto_path}")


# 4. compare scenarios at threshold 0.50

comparison_df = results_df.filter(
    pl.col("threshold") == reference_threshold
)

comparison_df = comparison_df.with_columns(
    pl.col("scenario")
    .replace(scenario_labels)
    .alias("scenario_label")
)

baseline_rows = comparison_df.filter(
    pl.col("scenario") == "all_features"
)

if baseline_rows.height != 1:
    raise ValueError(
        "Expected one all_features row at threshold 0.50"
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

comparison_df = comparison_df.sort("scenario")

comparison_path = (
    reports_path
    / "a9_final_scenario_comparison.csv"
)

comparison_df.write_csv(comparison_path)

print("\nSCENARIO COMPARISON AT THRESHOLD 0.50")
print(
    comparison_df.select(
        [
            "scenario_label",
            "roc_auc_oof",
            "roc_auc_change",
            "balanced_accuracy",
            "balanced_accuracy_change",
            "approval_rate_gap",
            "approval_gap_reduction",
            "tpr_gap",
            "tpr_gap_reduction",
            "fpr_gap",
            "fpr_gap_reduction",
        ]
    )
)

print(f"\nComparison saved to: {comparison_path}")


# 5. plot ROC-AUC versus approval rate gap

plt.figure(figsize=(10, 7))

for scenario_name in scenario_order:
    scenario_df = results_df.filter(
        pl.col("scenario") == scenario_name
    ).sort("threshold")

    x_values = scenario_df[
        "approval_rate_gap"
    ].to_list()

    y_values = scenario_df[
        "roc_auc_oof"
    ].to_list()

    plt.scatter(
        x_values,
        y_values,
        label=scenario_labels[scenario_name],
        alpha=0.75,
    )

    plt.plot(
        x_values,
        y_values,
        alpha=0.4,
    )

frontier_df = pareto_frontier.sort(
    "approval_rate_gap"
)

plt.plot(
    frontier_df["approval_rate_gap"].to_list(),
    frontier_df["roc_auc_oof"].to_list(),
    linestyle="--",
    marker="x",
    label="Pareto frontier",
)

plt.xlabel("Approval Rate Gap")
plt.ylabel("Out-of-Fold ROC-AUC")

plt.title(
    "Credit Approval: Predictive Performance "
    "vs Approval Rate Gap"
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

pareto_plot_path = (
    figures_path
    / "a9_final_pareto_frontier.png"
)

plt.savefig(pareto_plot_path, dpi=300)
plt.close()

print(f"\nPareto plot saved to: {pareto_plot_path}")


# 6. plot performance and disparity metrics

metric_names = [
    "ROC-AUC",
    "Balanced accuracy",
    "Approval rate gap",
    "TPR gap",
    "FPR gap",
]

metric_columns = [
    "roc_auc_oof",
    "balanced_accuracy",
    "approval_rate_gap",
    "tpr_gap",
    "fpr_gap",
]

plot_df = comparison_df.sort("scenario")

scenario_names = plot_df[
    "scenario_label"
].to_list()

figure, axes = plt.subplots(
    nrows=1,
    ncols=5,
    figsize=(19, 5),
)

for index, (metric_name, metric_column) in enumerate(
    zip(metric_names, metric_columns)
):
    values = plot_df[metric_column].to_list()

    axes[index].bar(
        scenario_names,
        values,
    )

    axes[index].set_title(metric_name)
    axes[index].tick_params(
        axis="x",
        labelrotation=70,
    )

    axes[index].grid(
        axis="y",
        alpha=0.3,
    )

figure.suptitle(
    "Model Performance and Group Disparities "
    "at Threshold 0.50"
)

figure.tight_layout()

metrics_plot_path = (
    figures_path
    / "a9_final_metrics_comparison.png"
)

figure.savefig(
    metrics_plot_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(figure)

print(
    f"Metrics comparison plot saved to: "
    f"{metrics_plot_path}"
)


# 7. create concise final report

final_rows = []

for row in comparison_df.to_dicts():
    final_rows.append(
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

final_df = pl.DataFrame(final_rows)

final_path = (
    reports_path
    / "a9_final_results_dashboard.csv"
)

final_df.write_csv(final_path)

print("\nFINAL RESULTS")
print(final_df)

print(f"\nFinal dashboard data saved to: {final_path}")
print("\nAnalysis completed.")