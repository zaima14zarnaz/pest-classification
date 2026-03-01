import argparse
import torch
from dataloader import make_ip102_loader
from train import train_ip102
from evaluation import evaluate, evaluate_multiclass, save_report

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="googlenet", choices=["googlenet", "vgg16"])
parser.add_argument("--classes", type=int, default=102, choices=[8, 102])
parser.add_argument("--train_model", type=int, default=0, choices=[0,1])
args = parser.parse_args()

model_name = args.model_name
classes = args.classes
train_model = bool(args.train_model)

data_subdir = "superclasses" if classes == 8 else "classification"
cf_title = f"{model_name} - Normalized Confusion Matrix"

dataset_root = "/data/research/zaima/dataset/Dataset/IP102"

train_ds, train_dl = make_ip102_loader(
    dataset_root, "train",
    batch_size=64,
    img_size=224,
    num_workers=8,
    images_subdir=data_subdir
)

x, y = next(iter(train_dl))
print("batch:", x.shape, y.shape, "num_classes:", train_ds.num_classes)

if train_ds.idx_to_classname:
    print("label 0 name:", train_ds.idx_to_classname[0])

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- model initialization ----
ckpt_path = f"ckpt/{model_name}_ip{classes}.pt"

if model_name == "googlenet":
    from model_googlenet import build_googlenet_ip102
    model = build_googlenet_ip102(
        num_classes=train_ds.num_classes,
        aux_logits=True
    ).to(device)

elif model_name == "vgg16":
    from model_vggnet import build_vggnet_ip102
    model = build_vggnet_ip102(
        num_classes=train_ds.num_classes,
        variant="vgg16_bn",
        pretrained=False
    ).to(device)

if train_model:
    model = train_ip102(
        dataset_root,
        model,
        epochs=60,
        batch_size=64,
        img_size=224,
        num_workers=8,
        lr=0.01,
        use_weighted_sampler=True,
        sampler_alpha=0.25,
        save_path=ckpt_path,
        ckpt_path=ckpt_path,
        images_subdir=data_subdir
    )

print(ckpt_path)
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
print("Loaded model from epoch:", checkpoint.get("epoch"))

test_ds, test_dl = make_ip102_loader(
    dataset_root, "test",
    batch_size=64,
    img_size=224,
    num_workers=8,
    images_subdir=data_subdir
)

test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_dl, device=device)

print(
    f"\nResults for {model_name} on the {classes} class label configuration: \n"
    f"==================================================================== \n"
    f"Accuracy : {test_acc:.2%}\n"
    f"Precision: {test_prec:.2%}\n"
    f"Recall   : {test_rec:.2%}\n"
    f"F1-Score : {test_f1:.2%}"
)
metrics = evaluate_multiclass(model, test_dl, device)

df, paths = save_report(
    metrics,
    class_names=getattr(test_ds, "idx_to_classname", None),
    save_dir=f"./results_{model_name}_ip{classes}",
    top_k_print=15,
    normalize_cm=True,
    bars_top_k_by_support=50,
    scatter_use_imbalance_ratio=False,
    scatter_log_x=True,
    scatter_fit_trendline=False,
    cf_title=cf_title
)