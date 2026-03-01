from evaluation import plot_model_metrics_comparison
baseline1 = "/home/zaimaz/Desktop/research1/pest-classification/results_googlenet8/per_class_metrics.csv"
baseline2 = "/home/zaimaz/Desktop/research1/pest-classification/results_vgg8/per_class_metrics.csv"
baseline_main = "/home/zaimaz/Desktop/research1/pest-classification/baseline_per_class_metrics.csv"
title = "Accuracy Comparison Per Class"
metric = "accuracy"
out_fname = "acc_comparison_bar.png"
out_png, merged_df = plot_model_metrics_comparison(
    baseline_main,
    baseline1, 
    baseline2,
    label_a="Baseline",
    label_b="Googlenet",
    label_c="VGG16",
    out_png=out_fname,
    sort_by="class_id",
    metric=metric
)