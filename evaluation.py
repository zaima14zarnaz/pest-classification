import torch
import torch.nn as nn
from tqdm import tqdm
import sys

from sklearn.metrics import precision_score, recall_score, f1_score

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in tqdm(loader, total=len(loader), desc="Evaluating", disable=not sys.stdout.isatty()):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)
            loss = criterion(out, y)

            pred = out.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.numel()
            loss_sum += loss.item() * y.size(0)

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    precision = precision_score(all_targets, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_targets, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return (
        loss_sum / max(total, 1),
        correct / max(total, 1),
        precision,
        recall,
        f1
    )

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

@torch.no_grad()
def evaluate_multiclass(model, loader, device, num_classes=None):
    model.eval()

    all_preds = []
    all_targets = []

    for x, y in tqdm(loader, total=len(loader), desc="Evaluating multiclass", disable=not sys.stdout.isatty()):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)

        if hasattr(out, "logits"):
            logits = out.logits
        else:
            logits = out

        pred = logits.argmax(dim=1)

        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average="macro")
    f1_weighted = f1_score(all_targets, all_preds, average="weighted")

    cm = confusion_matrix(all_targets, all_preds)

    if num_classes is None:
        num_classes = cm.shape[0]

    # per-class accuracy (recall)
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    # per-class F1
    per_class_f1 = f1_score(all_targets, all_preds, average=None)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_accuracy": per_class_acc,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
    }



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_per_class_df(metrics: dict, class_names=None) -> pd.DataFrame:
    per_acc = np.asarray(metrics["per_class_accuracy"])
    per_f1  = np.asarray(metrics["per_class_f1"])
    cm      = np.asarray(metrics["confusion_matrix"])

    n = len(per_acc)
    if class_names is None:
        names = [str(i) for i in range(n)]
    elif isinstance(class_names, dict):
        names = [class_names.get(i, str(i)) for i in range(n)]
    else:
        names = [class_names[i] if i < len(class_names) else str(i) for i in range(n)]

    df = pd.DataFrame({
        "class_id": np.arange(n),
        "class_name": names,
        "accuracy": per_acc,
        "f1": per_f1,
        "support": cm.sum(axis=1),
    })
    return df


def print_multiclass_summary(metrics: dict, df: pd.DataFrame, top_k: int = 15) -> None:
    n = len(df)
    df_sorted_f1 = df.sort_values(["f1", "support"], ascending=[True, True]).reset_index(drop=True)

    print("\n=== Overall Metrics ===")
    if "loss" in metrics:
        print(f"Loss          : {metrics['loss']:.4f}")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"F1 (macro)    : {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted) : {metrics['f1_weighted']:.4f}")

    print(f"\n=== Worst {min(top_k, n)} classes by F1 ===")
    print(df_sorted_f1.head(min(top_k, n)).to_string(index=False, justify="left"))

    print(f"\n=== Best {min(top_k, n)} classes by F1 ===")
    print(df_sorted_f1.tail(min(top_k, n)).sort_values("f1", ascending=False).to_string(index=False, justify="left"))


def save_per_class_csv(df: pd.DataFrame, save_dir: str, filename: str = "per_class_metrics.csv") -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    df.to_csv(path, index=False)
    return path


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_confusion_matrix_png(
    cm,
    save_dir,
    filename="confusion_matrix.png",
    normalize=True,
    class_names=None,
    title="Normalized Confusion Matrix"
):
    cm = np.asarray(cm).astype(float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar=True,
        linewidths=0.5
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    return save_path


def save_per_class_bar_png(
    df: pd.DataFrame,
    metric_col: str,                   # "accuracy" or "f1"
    save_dir: str,
    filename: str,
    top_k_by_support: int = None,       # optional for readability
    dpi: int = 200,
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    plot_df = df.copy()
    if top_k_by_support is not None and top_k_by_support < len(plot_df):
        plot_df = plot_df.sort_values("support", ascending=False).head(top_k_by_support)

    x_labels = plot_df["class_name"].astype(str).tolist()
    x = np.arange(len(plot_df))

    fig = plt.figure(figsize=(max(10, len(plot_df) * 0.35), 5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.bar(x, plot_df[metric_col].values)
    ax.set_title(f"Per-class {metric_col.capitalize()}")
    ax.set_xlabel("Class")
    ax.set_ylabel(metric_col.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()

    path = os.path.join(save_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_imbalance_vs_accuracy_scatter_png(
    df: pd.DataFrame,
    save_dir: str,
    filename: str = "imbalance_vs_accuracy.png",
    use_imbalance_ratio: bool = False,  # False: x=support, True: x=max_support/support
    log_x: bool = True,
    fit_trendline: bool = False,
    dpi: int = 200,
) -> str:
    os.makedirs(save_dir, exist_ok=True)

    support = df["support"].values.astype(float)
    acc = df["accuracy"].values.astype(float)

    if use_imbalance_ratio:
        max_support = float(np.max(support)) if len(support) else 1.0
        x_vals = max_support / np.clip(support, 1.0, None)
        x_label = "Imbalance Ratio (max_support / class_support)"
    else:
        x_vals = support
        x_label = "Class Support"

    fig = plt.figure(figsize=(6, 5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.scatter(x_vals, acc, s=20, alpha=0.7)

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(x_label + (" (log scale)" if log_x else ""))
    ax.set_ylabel("Per-class Accuracy")
    ax.set_title("Class Imbalance vs Accuracy")

    if fit_trendline and len(x_vals) >= 2:
        eps = 1e-12
        x_for_fit = np.log10(np.clip(x_vals, eps, None)) if log_x else x_vals
        coef = np.polyfit(x_for_fit, acc, 1)
        xs = np.linspace(np.min(x_for_fit), np.max(x_for_fit), 200)
        ys = coef[0] * xs + coef[1]
        if log_x:
            ax.plot(10 ** xs, ys)
        else:
            ax.plot(xs, ys)

    fig.tight_layout()
    path = os.path.join(save_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_model_metrics_comparison(
    csv_a_path: str,
    csv_b_path: str,
    csv_c_path: str,
    label_a: str = "Model A",
    label_b: str = "Model B",
    label_c: str = "Model C",
    metric: str = "accuracy",
    out_png: str = "metric_comparison_per_class.png",
    sort_by: str = "class_id",
    rotate_xticks: int = 0,
    class_order: list | None = None,
    show_support: bool = True,   # <-- new flag
):
    a = pd.read_csv(csv_a_path)
    b = pd.read_csv(csv_b_path)
    c = pd.read_csv(csv_c_path)

    # Validate metric
    for df_, name in [(a, "A"), (b, "B"), (c, "C")]:
        if metric not in df_.columns:
            raise ValueError(f"Metric '{metric}' not found in CSV {name}")

    # Merge + include support from B
    df = (
        a[["class_id", "class_name", metric]]
        .rename(columns={metric: f"{metric}_{label_a}"})
        .merge(
            b[["class_id", metric, "support"]]
            .rename(columns={metric: f"{metric}_{label_b}", "support": "support_B"}),
            on="class_id",
            how="inner",
        )
        .merge(
            c[["class_id", metric]].rename(columns={metric: f"{metric}_{label_c}"}),
            on="class_id",
            how="inner",
        )
    )

    # Sorting
    if class_order is not None:
        df["class_name"] = pd.Categorical(
            df["class_name"], categories=class_order, ordered=True
        )
        df = df.sort_values("class_name")
    elif sort_by == "class_name":
        df = df.sort_values("class_name")
    else:
        df = df.sort_values("class_id")

    # Convert to %
    for lab in [label_a, label_b, label_c]:
        df[f"{metric}_{lab}"] *= 100.0

    # Build class labels with support
    if show_support:
        classes = [
            f"{name}\n(n={int(sup)})"
            for name, sup in zip(df["class_name"], df["support_B"])
        ]
    else:
        classes = df["class_name"].tolist()

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, df[f"{metric}_{label_a}"], width, label=label_a)
    ax.bar(x,         df[f"{metric}_{label_b}"], width, label=label_b)
    ax.bar(x + width, df[f"{metric}_{label_c}"], width, label=label_c)

    ax.set_title(f"{metric.capitalize()} Comparison per Class")
    ax.set_ylabel(f"{metric.capitalize()} (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=rotate_xticks, ha="right")
    ax.legend(loc="upper left")

    ymax = max(
        df[f"{metric}_{label_a}"].max(),
        df[f"{metric}_{label_b}"].max(),
        df[f"{metric}_{label_c}"].max(),
    )
    ax.set_ylim(0, ymax * 1.10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return out_png, df

def save_report(
    metrics: dict,
    class_names=None,
    save_dir: str = "./results",
    top_k_print: int = 15,
    normalize_cm: bool = True,
    bars_top_k_by_support: int = None,
    scatter_use_imbalance_ratio: bool = False,
    scatter_log_x: bool = True,
    scatter_fit_trendline: bool = False,
    cf_title: str = None
):
    os.makedirs(save_dir, exist_ok=True)

    df = build_per_class_df(metrics, class_names=class_names)
    print_multiclass_summary(metrics, df, top_k=top_k_print)

    csv_path = save_per_class_csv(df, save_dir, filename="per_class_metrics.csv")
    cm_path = save_confusion_matrix_png(
        np.asarray(metrics["confusion_matrix"]),
        save_dir,
        filename="confusion_matrix.png",
        normalize=True,
        class_names=class_names,
        title=cf_title
    )

    acc_bar_path = save_per_class_bar_png(
        df, "accuracy", save_dir, filename="per_class_accuracy.png",
        top_k_by_support=bars_top_k_by_support
    )

    f1_bar_path = save_per_class_bar_png(
        df, "f1", save_dir, filename="per_class_f1.png",
        top_k_by_support=bars_top_k_by_support
    )

    scatter_path = save_imbalance_vs_accuracy_scatter_png(
        df,
        save_dir,
        filename="imbalance_vs_accuracy.png",
        use_imbalance_ratio=scatter_use_imbalance_ratio,
        log_x=scatter_log_x,
        fit_trendline=scatter_fit_trendline,
    )

    paths = {
        "per_class_csv": csv_path,
        "confusion_matrix_png": cm_path,
        "accuracy_bar_png": acc_bar_path,
        "f1_bar_png": f1_bar_path,
        "imbalance_vs_accuracy_png": scatter_path,
    }
    print("\nSaved artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    return df, paths


