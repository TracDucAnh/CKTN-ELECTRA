import json
from pathlib import Path
from textwrap import fill

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
OUT_PNG = OUT_DIR / "cktn_ablation_grouped_bars.png"
OUT_PDF = OUT_DIR / "cktn_ablation_grouped_bars.pdf"
OUT_SVG = OUT_DIR / "cktn_ablation_grouped_bars.svg"

# (Nhãn hiển thị, khóa nội bộ)
METHODS = [
    ("CKTN-ELECTRA", "main"),
    ("No script-aware", "no_script_aware"),
    ("No linear lambda", "no_linear_lambda"),
]

MAIN_VALUES = {
    "category_accuracy": 0.9214,
    "category_macro_f1": 0.7103,
    "retrieval_mrr_at_10": 0.7718,
    "retrieval_recall_at_10": 0.9039,
}

REPORTS = {
    "no_script_aware": {
        "category": ROOT
        / "quarter_no_script_aware_variant"
        / "downstream"
        / "after_cpt"
        / "category_classification"
        / "cls"
        / "cktn"
        / "report.json",
        "retrieval": ROOT
        / "quarter_no_script_aware_variant"
        / "downstream"
        / "after_cpt"
        / "information_retrieval"
        / "cktn"
        / "report.json",
    },
    "no_linear_lambda": {
        "category": ROOT
        / "quarter_and_no_linear_lambda_variant"
        / "downstream"
        / "after_cpt"
        / "category_classification"
        / "cls"
        / "cktn"
        / "report.json",
        "retrieval": ROOT
        / "quarter_and_no_linear_lambda_variant"
        / "downstream"
        / "after_cpt"
        / "information_retrieval"
        / "cktn"
        / "report.json",
    },
}


def best_category(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return max(rows, key=lambda row: row.get("macro_f1", -1.0))


def best_retrieval(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return max(rows, key=lambda row: row.get("mrr_at_10", -1.0))


def collect_values():
    """Gom giá trị của 3 method: main (hard-coded) + 2 ablation (đọc từ report.json)."""
    values = {"main": dict(MAIN_VALUES)}
    for method_key, reports in REPORTS.items():
        cat = best_category(reports["category"])
        ret = best_retrieval(reports["retrieval"])
        values[method_key] = {
            "category_accuracy": cat["accuracy"],
            "category_macro_f1": cat["macro_f1"],
            "retrieval_mrr_at_10": ret["mrr_at_10"],
            "retrieval_recall_at_10": ret["recall_at_10"],
        }
    return values


def plot_grouped(ax, values, metric_keys, metric_labels, title):
    """Vẽ bar chart nhóm: mỗi method có 2 cột kế nhau (2 metric) trên cùng 1 ax."""
    n_methods = len(METHODS)
    n_metrics = len(metric_keys)
    x = np.arange(n_methods)
    width = 0.32
    colors = ["#1f6f78", "#c65d32"]

    for i, (mkey, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        offset = (i - (n_metrics - 1) / 2) * width
        heights = [values[key][mkey] for _, key in METHODS]
        bars = ax.bar(
            x + offset,
            heights,
            width,
            label=mlabel,
            color=colors[i],
            edgecolor="black",
            linewidth=0.6,
        )
        labels = [f"{bar.get_height():.4f}" for bar in bars]
        ax.bar_label(bars, labels=labels, padding=4, fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels([fill(label, width=14) for label, _ in METHODS], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("Score", fontsize=15)
    ax.set_ylim(0, 1.3)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#dddddd",
        fontsize=13,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    values = collect_values()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 13,
            "axes.linewidth": 0.8,
        }
    )

    fig, (ax_cls, ax_ir) = plt.subplots(2, 1, figsize=(9.4, 9.2))

    plot_grouped(
        ax_cls,
        values,
        ["category_accuracy", "category_macro_f1"],
        ["Accuracy", "Macro-F1"],
        "Category Classification",
    )

    plot_grouped(
        ax_ir,
        values,
        ["retrieval_mrr_at_10", "retrieval_recall_at_10"],
        ["MRR@10", "Recall@10"],
        "Information Retrieval",
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.95,
        bottom=0.08,
        hspace=0.42,
    )

    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")

    print(OUT_PNG)
    print(OUT_PDF)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
