# coding: utf-8
"""训练与评估工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import json
import os
import re

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm 可选
    tqdm = None


@dataclass
class TrainConfig:
    task_mode: str = "regression"  # regression | deterioration
    target_type: str = "total"  # total | organ
    deterioration_threshold: float = 2.0
    n_epochs: int = 30
    lr: float = 1e-4
    seed: int = 42
    device: str = "cpu"
    freeze_backbone: bool = False
    unfreeze_epoch: Optional[int] = None
    show_progress: bool = True
    compute_train_metrics: bool = True
    plot_history: bool = True
    early_stop_patience: Optional[int] = None
    early_stop_min_delta: float = 0.0
    weight_decay: float = 0.0
    aux_gram_loss_weight: float = 0.0
    gram_label_map: Optional[Dict[str, int]] = None
    grad_clip_norm: Optional[float] = None
    loss_type: str = "bce"
    pos_weight: Optional[float] = None
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    threshold_strategy: str = "fixed_0.5"
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_step: float = 0.05


def seed_everything(seed: int) -> None:
    """设置随机种子。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_valid(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """展开序列并按 mask 过滤。"""
    if preds.ndim == 3 and preds.shape[2] > 1:
        mask = np.repeat(mask[:, :, None], preds.shape[2], axis=2)
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    mask = mask.reshape(-1)
    valid = mask > 0
    return preds[valid], targets[valid]


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """回归指标。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    acc1 = float(np.mean(np.abs(y_pred - y_true) <= 1.0))
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "ACC@1": acc1,
    }


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """分类指标。"""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan
    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "ACC": acc,
        "F1": f1,
        "AUC": auc,
        "AUPRC": auprc,
        "CM": cm,
    }


def _build_threshold_grid(cfg: TrainConfig) -> List[float]:
    if cfg.threshold_step <= 0:
        return [0.5]
    values = np.arange(cfg.threshold_min, cfg.threshold_max + 1e-8, cfg.threshold_step)
    thresholds = [float(np.round(v, 6)) for v in values]
    if 0.5 not in thresholds:
        thresholds.append(0.5)
    return sorted(set(thresholds))


def _select_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Sequence[float]) -> Tuple[float, Dict[str, float]]:
    best_t = 0.5
    best_metrics: Dict[str, float] = {}
    best_f1 = -1.0
    for t in thresholds:
        metrics = compute_classification_metrics(y_true, y_prob, threshold=float(t))
        f1 = metrics.get("F1", np.nan)
        if np.isnan(f1):
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            best_metrics = metrics
    if not best_metrics:
        best_metrics = compute_classification_metrics(y_true, y_prob, threshold=best_t)
    return best_t, best_metrics


def _classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str,
    pos_weight: Optional[float],
    focal_alpha: float,
    focal_gamma: float,
) -> torch.Tensor:
    if loss_type == "focal":
        bce = nn.BCEWithLogitsLoss(reduction="none")(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = focal_alpha * targets + (1.0 - focal_alpha) * (1.0 - targets)
        return alpha_t * (1.0 - p_t).pow(focal_gamma) * bce

    if loss_type == "bce_pos_weight" and pos_weight is not None:
        weight = torch.tensor(pos_weight, device=logits.device)
        return nn.BCEWithLogitsLoss(reduction="none", pos_weight=weight)(logits, targets)

    return nn.BCEWithLogitsLoss(reduction="none")(logits, targets)


def compute_deterioration_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_total: np.ndarray,
    threshold: float,
    mask: np.ndarray,
) -> float:
    """用回归结果派生恶化准确率。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    prev_total = np.asarray(prev_total)
    mask = np.asarray(mask)

    delta_true = y_true - prev_total
    delta_pred = y_pred - prev_total

    label_true = delta_true >= threshold
    label_pred = delta_pred >= threshold

    valid = mask > 0
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(label_true[valid] == label_pred[valid]))


def _maybe_unfreeze(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def _freeze_backbone(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if "head" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False


def run_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_dim: int,
    task_mode: str,
    loss_type: str = "bce",
    pos_weight: Optional[float] = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    grad_clip_norm: Optional[float] = None,
    show_progress: bool = False,
    desc: str = "Train",
    aux_gram_loss_weight: float = 0.0,
    gram_label_map: Optional[Dict[str, int]] = None,
) -> float:
    """训练一个 epoch。"""
    model.train()
    if task_mode != "deterioration":
        criterion = nn.SmoothL1Loss(reduction="none")

    total_loss = 0.0
    total_count = 0.0

    iterable = loader
    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(loader, desc=desc, leave=False)
        iterable = pbar

    for x_time, x_static, y_seq, seq_mask, prev_mask, prev_total, _pid in iterable:
        x_time = x_time.to(device)
        x_static = x_static.to(device)
        y_seq = y_seq.to(device)
        seq_mask = seq_mask.to(device)

        preds, gram_logits = model(x_time, x_static)
        if task_mode == "deterioration":
            loss = _classification_loss(preds, y_seq, loss_type, pos_weight, focal_alpha, focal_gamma)
        else:
            loss = criterion(preds, y_seq)
        if target_dim > 1:
            loss = loss.mean(dim=2)
        else:
            loss = loss.squeeze(-1)

        # Gram auxiliary loss (training only)
        if aux_gram_loss_weight > 0 and gram_label_map:
            batch_pids = [str(pid) for pid in _pid]
            gram_labels = [gram_label_map.get(pid, -1) for pid in batch_pids]
            gram_targets = torch.tensor(gram_labels, dtype=torch.float, device=device)
            valid = gram_targets >= 0
            if valid.any():
                gram_loss = nn.functional.binary_cross_entropy_with_logits(
                    gram_logits[valid].squeeze(-1), gram_targets[valid])
                loss = loss + aux_gram_loss_weight * gram_loss

        loss = loss * seq_mask
        denom = seq_mask.sum().clamp(min=1.0)
        batch_loss = loss.sum() / denom

        optimizer.zero_grad()
        batch_loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += batch_loss.item() * denom.item()
        total_count += denom.item()

        if pbar is not None:
            pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

    return total_loss / max(total_count, 1.0)


def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    target_dim: int,
    task_mode: str,
    deterioration_threshold: float,
    organ_feature_indices: Optional[Sequence[int]] = None,
    loss_type: str = "bce",
    pos_weight: Optional[float] = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    threshold: float = 0.5,
    show_progress: bool = False,
    desc: str = "Eval",
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """评估并返回预测与指标。"""
    model.eval()
    if task_mode != "deterioration":
        criterion = nn.SmoothL1Loss(reduction="none")

    total_loss = 0.0
    total_count = 0.0

    all_preds = []
    all_targets = []
    all_masks = []
    all_prev_masks = []
    all_prev_totals = []
    all_patient_ids = []

    iterable = loader
    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(loader, desc=desc, leave=False)
        iterable = pbar

    with torch.no_grad():
        for x_time, x_static, y_seq, seq_mask, prev_mask, prev_total, _pid in iterable:
            x_time = x_time.to(device)
            x_static = x_static.to(device)
            y_seq = y_seq.to(device)
            seq_mask = seq_mask.to(device)
            prev_mask = prev_mask.to(device)

            preds, _gram = model(x_time, x_static)
            if task_mode == "deterioration":
                loss = _classification_loss(preds, y_seq, loss_type, pos_weight, focal_alpha, focal_gamma)
            else:
                loss = criterion(preds, y_seq)
            if target_dim > 1:
                loss = loss.mean(dim=2)
            else:
                loss = loss.squeeze(-1)

            loss = loss * seq_mask
            denom = seq_mask.sum().clamp(min=1.0)
            batch_loss = loss.sum() / denom

            total_loss += batch_loss.item() * denom.item()
            total_count += denom.item()

            # Always collect prev_total as it comes from dataset and is source of truth
            prev_total_np = prev_total.cpu().numpy()
            if prev_total_np.ndim == 3 and prev_total_np.shape[-1] == 1:
                prev_total_np = prev_total_np[..., 0]
            all_prev_totals.append(prev_total_np)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_seq.cpu().numpy())
            all_masks.append(seq_mask.cpu().numpy())
            all_prev_masks.append(prev_mask.cpu().numpy())
            all_patient_ids.extend(list(_pid))

            if pbar is not None:
                pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    prev_masks = np.concatenate(all_prev_masks, axis=0)
    prev_totals = np.concatenate(all_prev_totals, axis=0) if all_prev_totals else None

    # Sanity check: ensure prev_totals shape matches preds/targets
    if prev_totals is not None:
        pred_shape_check = preds[..., 0] if preds.ndim == 3 else preds.squeeze(-1)
        if prev_totals.shape[:2] != pred_shape_check.shape[:2]:
            raise ValueError(f"prev_totals shape {prev_totals.shape} does not match preds shape {preds.shape}")

    if task_mode == "deterioration":
        probs = 1 / (1 + np.exp(-preds))
        flat_probs, flat_targets = flatten_valid(probs, targets, masks)
        metrics = compute_classification_metrics(flat_targets, flat_probs, threshold=threshold)
    else:
        flat_preds, flat_targets = flatten_valid(preds, targets, masks)
        metrics = compute_regression_metrics(flat_targets, flat_preds)

        if target_dim == 1 and prev_totals is not None:
            det_mask = masks * prev_masks
            acc_det = compute_deterioration_accuracy(
                targets.squeeze(-1),
                preds.squeeze(-1),
                prev_totals,
                deterioration_threshold,
                det_mask,
            )
            metrics["ACC_DET@K"] = acc_det
        else:
            metrics["ACC_DET@K"] = np.nan

    return (
        total_loss / max(total_count, 1.0),
        metrics,
        preds,
        targets,
        masks,
        prev_masks,
        prev_totals,
        all_patient_ids,
    )


def _prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in metrics.items() if k != "CM"}


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def _save_history(history: List[Dict[str, object]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "history.json")
    with open(path, "w") as f:
        json.dump([{k: _to_jsonable(v) for k, v in row.items()} for row in history], f, indent=2)

    # 同步保存 CSV，方便分析
    try:
        import csv

        csv_path = os.path.join(output_dir, "history.csv")
        keys = sorted({k for row in history for k in row.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in history:
                writer.writerow({k: _to_jsonable(row.get(k)) for k in keys})
    except Exception:
        pass


def _log_line(text: str) -> None:
    if tqdm is not None:
        tqdm.write(text)
    else:
        print(text)


def _progress_bar(epoch: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return ""
    ratio = min(max(epoch / total, 0.0), 1.0)
    filled = int(round(ratio * width))
    return "#" * filled + "-" * (width - filled)


def _plot_history(history: List[Dict[str, object]], output_dir: str) -> None:
    """绘制损失与指标曲线。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        _log_line("[Warn] matplotlib 不可用，跳过绘图。")
        return

    if not history:
        return

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    # 1) loss 曲线
    train_loss = [row.get("train_loss") for row in history]
    val_loss = [row.get("val_loss") for row in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"))
    plt.close()

    # 2) 指标曲线（按 metric 名分别画）
    keys = sorted({k for row in history for k in row.keys()})
    metric_names = set()
    for key in keys:
        if key.startswith("train_"):
            metric_names.add(key.replace("train_", ""))
        if key.startswith("val_"):
            metric_names.add(key.replace("val_", ""))
    metric_names.discard("loss")

    for metric in sorted(metric_names):
        train_vals = [row.get(f"train_{metric}") for row in history]
        val_vals = [row.get(f"val_{metric}") for row in history]
        if all(v is None for v in train_vals) and all(v is None for v in val_vals):
            continue

        safe_metric = re.sub(r"[^0-9a-zA-Z]+", "_", metric.lower()).strip("_")
        plt.figure(figsize=(6, 4))
        if any(v is not None for v in train_vals):
            plt.plot(epochs, train_vals, label=f"train_{metric}")
        if any(v is not None for v in val_vals):
            plt.plot(epochs, val_vals, label=f"val_{metric}")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{safe_metric}_curve.png"))
        plt.close()


def _save_confusion_matrix(cm: np.ndarray, output_dir: str, tag: str) -> None:
    """保存混淆矩阵图片。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        _log_line("[Warn] matplotlib 不可用，跳过混淆矩阵绘图。")
        return

    cm = np.asarray(cm)
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{tag}_confusion_matrix.png"))
    plt.close()


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: TrainConfig,
    target_dim: int,
    organ_feature_indices: Optional[Sequence[int]] = None,
    output_dir: Optional[str] = None,
    tag: Optional[str] = None,
    log_prefix: Optional[str] = None,
) -> Dict[str, object]:
    """训练主入口，返回最佳权重与指标。"""
    seed_everything(cfg.seed)

    device = torch.device(cfg.device)
    model = model.to(device)

    if cfg.freeze_backbone:
        _freeze_backbone(model)

    if cfg.weight_decay and cfg.weight_decay > 0:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)

    best_val = float("inf")
    best_state = None
    history = []
    best_metrics = None
    best_cm = None
    no_improve = 0

    for epoch in range(1, cfg.n_epochs + 1):
        if cfg.unfreeze_epoch is not None and epoch == cfg.unfreeze_epoch:
            _maybe_unfreeze(model)
            if cfg.weight_decay and cfg.weight_decay > 0:
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            target_dim,
            cfg.task_mode,
            loss_type=cfg.loss_type,
            pos_weight=cfg.pos_weight,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            grad_clip_norm=cfg.grad_clip_norm,
            show_progress=cfg.show_progress,
            desc=f"Train {epoch}/{cfg.n_epochs}",
            aux_gram_loss_weight=cfg.aux_gram_loss_weight,
            gram_label_map=cfg.gram_label_map,
        )

        train_metrics = {}
        if cfg.compute_train_metrics:
            _train_eval_loss, train_metrics, *_ = evaluate(
                model,
                train_loader,
                device,
                target_dim,
                cfg.task_mode,
                cfg.deterioration_threshold,
                organ_feature_indices=organ_feature_indices,
                loss_type=cfg.loss_type,
                pos_weight=cfg.pos_weight,
                focal_alpha=cfg.focal_alpha,
                focal_gamma=cfg.focal_gamma,
                threshold=0.5,
                show_progress=cfg.show_progress,
                desc=f"TrainEval {epoch}/{cfg.n_epochs}",
            )

        val_loss, metrics, preds, targets, masks, prev_masks, prev_totals, pids = evaluate(
            model,
            val_loader,
            device,
            target_dim,
            cfg.task_mode,
            cfg.deterioration_threshold,
            organ_feature_indices=organ_feature_indices,
            loss_type=cfg.loss_type,
            pos_weight=cfg.pos_weight,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            threshold=0.5,
            show_progress=cfg.show_progress,
            desc=f"Val {epoch}/{cfg.n_epochs}",
        )

        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history_entry.update(_prefix_metrics(train_metrics, "train_"))
        history_entry.update(_prefix_metrics(metrics, "val_"))
        history.append(history_entry)

        # 输出本轮摘要
        metric_summary = {k: v for k, v in history_entry.items() if k.startswith("val_")}
        summary_str = ", ".join(
            [
                f"{k}={metric_summary[k]:.4f}"
                for k in sorted(metric_summary.keys())
                if isinstance(metric_summary[k], (int, float))
            ]
        )
        bar = _progress_bar(epoch, cfg.n_epochs, width=20)
        prefix = f"[{log_prefix}] " if log_prefix else ""
        if summary_str:
            _log_line(
                f"{prefix}[Epoch {epoch}/{cfg.n_epochs} |{bar}|] train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, {summary_str}"
            )
        else:
            _log_line(
                f"{prefix}[Epoch {epoch}/{cfg.n_epochs} |{bar}|] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

        if output_dir:
            _save_history(history, output_dir)

        improved = val_loss < (best_val - cfg.early_stop_min_delta)
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = metrics
            best_cm = metrics.get("CM")
            no_improve = 0
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                tag_str = tag or "model"
                torch.save(model.state_dict(), os.path.join(output_dir, f"{tag_str}.pt"))
                with open(os.path.join(output_dir, f"{tag_str}_metrics.json"), "w") as f:
                    json.dump({k: _to_jsonable(v) for k, v in history[-1].items()}, f, indent=2)

                if best_metrics:
                    with open(os.path.join(output_dir, "best_metrics.json"), "w") as f:
                        json.dump({k: _to_jsonable(v) for k, v in best_metrics.items()}, f, indent=2)
                if best_cm is not None:
                    with open(os.path.join(output_dir, "best_cm.json"), "w") as f:
                        json.dump(np.asarray(best_cm).tolist(), f, indent=2)
                    _save_confusion_matrix(best_cm, output_dir, tag_str)
        else:
            no_improve += 1

        if cfg.early_stop_patience is not None and no_improve >= cfg.early_stop_patience:
            _log_line(
                f"{prefix}[EarlyStop] epoch={epoch}, best_val={best_val:.4f}, patience={cfg.early_stop_patience}"
            )
            break

    best_threshold = 0.5
    if cfg.task_mode == "deterioration" and cfg.threshold_strategy == "best_f1":
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)
        _val_loss, _metrics, preds, targets, masks, prev_masks, _prev_totals, _pids = evaluate(
            model,
            val_loader,
            device,
            target_dim,
            cfg.task_mode,
            cfg.deterioration_threshold,
            organ_feature_indices=organ_feature_indices,
            loss_type=cfg.loss_type,
            pos_weight=cfg.pos_weight,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            threshold=0.5,
            show_progress=False,
            desc="ValBestThreshold",
        )
        probs = 1 / (1 + np.exp(-preds))
        flat_probs, flat_targets = flatten_valid(probs, targets, masks)
        if flat_targets.size > 0:
            thresholds = _build_threshold_grid(cfg)
            best_threshold, best_metrics = _select_best_threshold(flat_targets, flat_probs, thresholds)
            best_cm = best_metrics.get("CM")

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "best_threshold.json"), "w") as f:
                    json.dump({"threshold": float(best_threshold)}, f, indent=2)
                with open(os.path.join(output_dir, "best_metrics.json"), "w") as f:
                    json.dump({k: _to_jsonable(v) for k, v in best_metrics.items()}, f, indent=2)
                if best_cm is not None:
                    with open(os.path.join(output_dir, "best_cm.json"), "w") as f:
                        json.dump(np.asarray(best_cm).tolist(), f, indent=2)
                    _save_confusion_matrix(best_cm, output_dir, tag or "model")

    if output_dir and cfg.plot_history:
        _plot_history(history, output_dir)

    return {
        "best_state": best_state,
        "best_val_loss": best_val,
        "history": history,
        "best_metrics": best_metrics,
        "best_threshold": best_threshold,
    }
