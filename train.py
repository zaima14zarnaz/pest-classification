from tqdm import tqdm
import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
import sys

from dataloader import make_ip102_loader
from evaluation import evaluate


def train_ip102(
    dataset_root: str,
    model,
    epochs: int = 50,
    batch_size: int = 64,
    img_size: int = 224,
    num_workers: int = 8,
    lr: float = 0.1,
    weight_decay: float = 1e-4,
    aux_weight: float = 0.3,
    use_weighted_sampler: bool = False,
    sampler_alpha: float = 0.5,
    save_path: str = "googlenet_ip102.pt",
    ckpt_path: str = "/kaggle/input/pest-classification-ip102-checkpoint/googlenet_ip102.pt",
    loss_curve_path: str = "loss_curve.png",
    accuracy_curve_path: str = "acc_curve.png",
    images_subdir: str = "classification",
):
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, train_dl = make_ip102_loader(
        dataset_root,
        "train",
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        use_weighted_sampler=use_weighted_sampler,
        sampler_alpha=sampler_alpha,
        images_subdir=images_subdir,
    )

    val_ds, val_dl = make_ip102_loader(
        dataset_root,
        "val",
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        use_weighted_sampler=False,
        sampler_alpha=sampler_alpha,
        images_subdir=images_subdir,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_acc = 0.0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    def _ensure_parent_dir(path: str):
        d = os.path.dirname(os.path.abspath(path))
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)

    def _save_curves():
        if loss_curve_path:
            _ensure_parent_dir(loss_curve_path)
            plt.figure()
            plt.plot(range(1, len(train_losses) + 1), train_losses)
            plt.plot(range(1, len(val_losses) + 1), val_losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(["train", "val"])
            plt.tight_layout()
            plt.savefig(loss_curve_path, dpi=200)
            plt.close()

        if accuracy_curve_path:
            _ensure_parent_dir(accuracy_curve_path)
            plt.figure()
            plt.plot(range(1, len(train_accs) + 1), train_accs)
            plt.plot(range(1, len(val_accs) + 1), val_accs)
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend(["train", "val"])
            plt.tight_layout()
            plt.savefig(accuracy_curve_path, dpi=200)
            plt.close()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in tqdm(
            train_dl,
            total=len(train_dl),
            desc=f"Training {epoch}/{epochs}",
            disable=not sys.stdout.isatty(),
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(x)

                main_logits = None
                aux_logits = []

                if torch.is_tensor(out):
                    main_logits = out
                elif hasattr(out, "logits"):
                    main_logits = out.logits
                    if hasattr(out, "aux_logits1") and out.aux_logits1 is not None:
                        aux_logits.append(out.aux_logits1)
                    if hasattr(out, "aux_logits2") and out.aux_logits2 is not None:
                        aux_logits.append(out.aux_logits2)
                elif isinstance(out, (tuple, list)):
                    main_logits = out[0]
                    for t in out[1:]:
                        if torch.is_tensor(t):
                            aux_logits.append(t)
                else:
                    raise TypeError(f"Unsupported model output type: {type(out)}")

                loss_main = criterion(main_logits, y)
                loss = loss_main

                if len(aux_logits) > 0 and aux_weight > 0:
                    loss_aux = 0.0
                    for a in aux_logits:
                        loss_aux = loss_aux + criterion(a, y)
                    loss = loss + aux_weight * loss_aux

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred = main_logits.argmax(dim=1)
                running_correct += (pred == y).sum().item()
                running_total += y.numel()
                running_loss += loss.item() * y.size(0)

        scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_dl, device)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        train_accs.append(float(train_acc))
        val_accs.append(float(val_acc))

        _save_curves()

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.6f} | {dt:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "num_classes": train_ds.num_classes,
                    "img_size": img_size,
                    "best_val_acc": best_val_acc,
                    "last_lr": scheduler.get_last_lr()[0],
                    "use_weighted_sampler": use_weighted_sampler,
                    "sampler_alpha": sampler_alpha,
                    "images_subdir": images_subdir,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accs": train_accs,
                    "val_accs": val_accs,
                },
                save_path,
            )
            print(f"  saved best -> {save_path} (val_acc={best_val_acc:.4f})")

    print("Best val acc:", best_val_acc)
    return model