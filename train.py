"""Training script for MobileNetV2-UNet MSI segmentation with ablation support."""

import argparse
import json
import os
import sys
import time
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data.dataset import MSIDataset, get_dataset_kwargs
from data.augment import get_train_transforms, get_val_transforms
from data.split import get_data_splits
from model.model import build_model
from model.loss import SegmentationLoss
from utils.metrics import SegmentationMetrics


RESUME_CHECKPOINT_INTERVAL = 10


def set_seed(seed, deterministic=False):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed.
        deterministic: If True, force fully deterministic cuDNN (slower).
            Default False enables cudnn.benchmark for ~10-30% speedup.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _save_resume_checkpoint(path, epoch, model, optimizer, scheduler,
                            best_iou_c1, best_epoch, no_improve_count,
                            epoch_logs, cfg):
    """Save a single rolling checkpoint with full state for resume."""
    cuda_rng = (torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_iou_c1": best_iou_c1,
        "best_epoch": best_epoch,
        "no_improve_count": no_improve_count,
        "epoch_logs": epoch_logs,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": cuda_rng,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "config": cfg,
    }, path)


def _load_resume_checkpoint(path, model, optimizer, scheduler, device):
    """Load resume checkpoint and return (start_epoch, best_iou_c1, best_epoch,
    no_improve_count, epoch_logs).
    """
    # Resume checkpoints embed numpy/python RNG state and the config dict, which
    # PyTorch >=2.6 refuses to unpickle under the new weights_only=True default.
    # These checkpoints are produced by this script (trusted source), so load fully.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        except Exception:
            pass
    if "numpy_rng_state" in ckpt:
        np.random.set_state(ckpt["numpy_rng_state"])
    if "python_rng_state" in ckpt:
        random.setstate(ckpt["python_rng_state"])

    # backward compat: old checkpoints used "best_miou"
    best_val = ckpt.get("best_iou_c1", ckpt.get("best_miou", 0.0))

    return (
        ckpt["epoch"] + 1,
        best_val,
        ckpt.get("best_epoch", 0),
        ckpt.get("no_improve_count", 0),
        ckpt.get("epoch_logs", []),
    )


def count_parameters(model):
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, masks, _raw, _apple_masks, _stems in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, metrics, device):
    """Validate and compute metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    metrics.reset()

    for images, masks, _raw, _apple_masks, _stems in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    results = metrics.compute()
    return avg_loss, results


def train(cfg, seed, output_dir, splits=None):
    """Main training loop.

    Args:
        cfg: Config dict.
        seed: Random seed for reproducibility.
        output_dir: Output directory for checkpoints/logs.
        splits: Optional pre-computed splits dict {'train': [...], 'val': [...], 'test': [...]}.
                If None, splits are computed via get_data_splits().
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data splits
    data_dir = cfg["data"]["data_dir"]
    if splits is None:
        splits = get_data_splits(
            data_dir=data_dir,
            image_dir=cfg["data"]["image_dir"],
            seed=seed,
        )
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, "
          f"Test: {len(splits.get('test', []))}")

    # Transforms
    train_transform = get_train_transforms(cfg)
    val_transform = get_val_transforms(cfg)

    # Datasets
    ds_kwargs = get_dataset_kwargs(cfg)
    train_dataset = MSIDataset(
        splits["train"], data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=train_transform,
        num_classes=cfg["data"]["num_classes"],
        **ds_kwargs,
    )
    val_dataset = MSIDataset(
        splits["val"], data_dir=data_dir,
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        transform=val_transform,
        num_classes=cfg["data"]["num_classes"],
        **ds_kwargs,
    )

    # DataLoaders
    train_cfg = cfg["train"]
    num_workers = train_cfg.get("num_workers", 4)
    use_persistent = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=train_cfg.get("pin_memory", True),
        drop_last=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=train_cfg.get("pin_memory", True),
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Model
    model = build_model(cfg).to(device)
    param_count = count_parameters(model)
    print(f"Model: {cfg['experiment_name']} | Parameters: {param_count:.2f}M")

    # Loss
    criterion = SegmentationLoss(
        loss_type=train_cfg.get("loss", "ce_dice"),
        ce_weight=train_cfg.get("ce_weight", 0.5),
        dice_weight=train_cfg.get("dice_weight", 0.5),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        focal_alpha=train_cfg.get("focal_alpha", 0.25),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # Scheduler: optional linear warmup -> CosineAnnealing
    num_epochs = train_cfg["num_epochs"]
    eta_min = train_cfg.get("eta_min", 1e-6)
    use_warmup = train_cfg.get("use_warmup", False)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    if use_warmup and warmup_epochs > 0:
        warmup_start_factor = train_cfg.get("warmup_start_factor", 0.01)
        warmup = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_epochs - warmup_epochs),
            eta_min=eta_min,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        print(f"Scheduler: linear warmup({warmup_epochs}ep, start_factor={warmup_start_factor}) "
              f"-> CosineAnnealing({num_epochs - warmup_epochs}ep, eta_min={eta_min})")
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=eta_min,
        )

    # Metrics
    num_classes = cfg["data"]["num_classes"]
    metrics = SegmentationMetrics(num_classes=num_classes)

    # TensorBoard
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # Checkpoint dir
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt_path = os.path.join(ckpt_dir, "last_checkpoint.pth")
    training_done_flag = os.path.join(ckpt_dir, "training_done.flag")

    best_iou_c1 = 0.0
    best_epoch = 0
    patience = train_cfg.get("early_stopping_patience", 0)
    no_improve_count = 0
    epoch_logs = []
    start_epoch = 1

    # If a previous run already completed, skip training entirely.
    if os.path.exists(training_done_flag):
        print(f"\n[skip] training_done.flag found at {training_done_flag}, "
              f"skipping training loop.")
        log_path = os.path.join(output_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path, "r") as _f:
                cached_logs = json.load(_f)
            if cached_logs:
                best_entry = max(cached_logs, key=lambda x: x.get("IoU_class1", 0.0))
                best_iou_c1 = float(best_entry["IoU_class1"])
                best_epoch = int(best_entry["epoch"])
        return {
            "experiment": cfg["experiment_name"],
            "seed": seed,
            "params_M": param_count,
            "best_iou_c1": best_iou_c1,
            "best_epoch": best_epoch,
            "output_dir": output_dir,
            "resumed_from_done": True,
        }

    # Resume from last_checkpoint.pth if present
    if os.path.exists(last_ckpt_path):
        try:
            (start_epoch, best_iou_c1, best_epoch,
             no_improve_count, epoch_logs) = _load_resume_checkpoint(
                last_ckpt_path, model, optimizer, scheduler, device)
            print(f"\n[resume] Loaded {last_ckpt_path}: starting at epoch "
                  f"{start_epoch} (best_IoU_c1={best_iou_c1:.4f} @ epoch {best_epoch}, "
                  f"no_improve={no_improve_count})")
            if start_epoch > train_cfg["num_epochs"]:
                print(f"[resume] checkpoint epoch {start_epoch - 1} >= "
                      f"num_epochs {train_cfg['num_epochs']}, nothing to do.")
        except Exception as e:
            print(f"\n[resume] Failed to load {last_ckpt_path}: {e}. "
                  f"Starting from scratch.")
            start_epoch = 1
            best_iou_c1 = 0.0
            best_epoch = 0
            no_improve_count = 0
            epoch_logs = []

    print(f"\nStarting training for {train_cfg['num_epochs']} epochs "
          f"(from epoch {start_epoch})...")
    if patience > 0:
        print(f"Early stopping enabled: patience={patience} epochs")
    for epoch in range(start_epoch, train_cfg["num_epochs"] + 1):
        t0 = time.time()

        # Update band-gate schedule (tau anneal); no-op if model has no gate.
        if hasattr(model, "set_gate_progress"):
            model.set_gate_progress((epoch - 1) / max(1, train_cfg["num_epochs"] - 1))

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_results = validate(model, val_loader, criterion, metrics, device)

        # Step scheduler
        scheduler.step()

        miou = val_results["mIoU"]
        defect_iou = float(val_results["IoU_per_class"][1])
        elapsed = time.time() - t0

        # Log to TensorBoard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/mIoU", miou, epoch)
        writer.add_scalar("val/IoU_class1", defect_iou, epoch)
        writer.add_scalar("val/F1_macro", val_results["F1_macro"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        epoch_logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mIoU": miou,
            "IoU_class1": defect_iou,
            "F1_macro": float(val_results["F1_macro"]),
            "Precision_macro": float(val_results["Precision_macro"]),
            "Recall_macro": float(val_results["Recall_macro"]),
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            gate_msg = ""
            if hasattr(model, "get_selected_bands"):
                sel = model.get_selected_bands()
                if sel is not None:
                    gate_msg = f" | Gate bands: {sel}"
            print(f"Epoch {epoch:03d}/{train_cfg['num_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"IoU(defect): {defect_iou:.4f} | F1: {val_results['F1_macro']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s{gate_msg}")

        # Save best model + early stopping check
        if defect_iou > best_iou_c1:
            best_iou_c1 = defect_iou
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou_c1": best_iou_c1,
                "config": cfg,
                "selected_bands": (model.get_selected_bands()
                                   if hasattr(model, "get_selected_bands") else None),
            }, os.path.join(ckpt_dir, "best_model.pth"))
        else:
            no_improve_count += 1

        # Rolling resume checkpoint: written every N epochs to reduce I/O.
        if epoch % RESUME_CHECKPOINT_INTERVAL == 0 or epoch == train_cfg["num_epochs"]:
            _save_resume_checkpoint(
                last_ckpt_path, epoch, model, optimizer, scheduler,
                best_iou_c1, best_epoch, no_improve_count, epoch_logs, cfg,
            )

        # Flush epoch_logs to disk every epoch for external monitoring.
        with open(os.path.join(output_dir, "training_log.json"), "w") as _f:
            json.dump(epoch_logs, _f, indent=2)

        if patience > 0 and no_improve_count >= patience:
            print(f"\nEarly stopping at epoch {epoch}: "
                  f"IoU(defect) no improvement for {patience} epochs. "
                  f"Best IoU(defect): {best_iou_c1:.4f} at epoch {best_epoch}")
            # Save resume checkpoint on early stop so state is recoverable.
            _save_resume_checkpoint(
                last_ckpt_path, epoch, model, optimizer, scheduler,
                best_iou_c1, best_epoch, no_improve_count, epoch_logs, cfg,
            )
            break

        # Periodic checkpoints
        save_interval = train_cfg.get("save_interval", 50)
        if epoch % save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou_c1": best_iou_c1,
                "config": cfg,
            }, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

    # Save final model
    torch.save({
        "epoch": train_cfg["num_epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_iou_c1": best_iou_c1,
        "config": cfg,
    }, os.path.join(ckpt_dir, "final_model.pth"))

    writer.close()

    # Save training log
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(epoch_logs, f, indent=2)

    # Mark training as fully done so re-runs can skip the loop.
    with open(training_done_flag, "w") as _f:
        _f.write(f"completed_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        _f.write(f"best_iou_c1={best_iou_c1:.6f}\n")
        _f.write(f"best_epoch={best_epoch}\n")
        _f.write(f"final_epoch={epoch_logs[-1]['epoch'] if epoch_logs else 0}\n")

    # If a band gate was used, persist the deployment selection for Phase 2.
    # Reload the BEST checkpoint first so the recorded selection matches the
    # deployed model, not the (possibly drifted) last-epoch state.
    best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
    if os.path.exists(best_ckpt_path):
        _bc = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(_bc["model_state_dict"])
        model.eval()
    selected_bands = None
    if hasattr(model, "get_selected_bands"):
        selected_bands = model.get_selected_bands()
    if selected_bands is not None:
        # The gate operates on the (possibly pre-selected) input channels, so
        # selected_bands are indices into cfg.data.band_indices. Map them back
        # to global band indices when a candidate subset was used.
        candidate = cfg["data"].get("band_indices")
        if candidate is not None:
            global_bands = [candidate[i] for i in selected_bands]
        else:
            global_bands = selected_bands
        with open(os.path.join(output_dir, "selected_bands.json"), "w") as _f:
            json.dump({
                "selected_bands_local": selected_bands,
                "selected_bands_global": global_bands,
                "candidate_band_indices": candidate,
                "seed": seed,
            }, _f, indent=2)
        print(f"Band gate selected bands (local idx): {selected_bands} "
              f"-> global: {global_bands}")

    print(f"\nTraining complete. Best IoU(defect): {best_iou_c1:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {ckpt_dir}")

    return {
        "experiment": cfg["experiment_name"],
        "seed": seed,
        "params_M": param_count,
        "best_iou_c1": best_iou_c1,
        "best_epoch": best_epoch,
        "output_dir": output_dir,
        "selected_bands": selected_bands,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2-UNet for MSI segmentation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: outputs/<experiment>_seed<seed>)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g., 'cuda:0' or 'cpu')")
    args = parser.parse_args()

    # Load config (with _base inheritance)
    from utils.config import load_config
    cfg = load_config(args.config)

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "outputs", f"{cfg['experiment_name']}_seed{args.seed}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config copy
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    result = train(cfg, args.seed, args.output_dir)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
