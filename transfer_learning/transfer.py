# coding: utf-8
"""迁移学习主流程（两阶段 & 三阶段）。

两阶段:  Sepsis → BSI (5-fold CV) → external (SAB + KPB)
三阶段:  Sepsis → BSI (single) → target pathogen (5-fold CV) → external (SAB + KPB)
Scratch: BSI (5-fold CV) → external
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader

from . import schema
from .datasets import ORGAN_COLS, PatientSequenceDataset
from .models import build_model
from .train import TrainConfig, train_model, evaluate, compute_regression_metrics, compute_deterioration_accuracy, seed_everything

@dataclass
class DataBundle:
    static_df: pd.DataFrame
    dynamic_df: pd.DataFrame


def _load_feature_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_dynamic_medians(path: str) -> Dict[str, float]:
    with open(path, "r") as f:
        return json.load(f)


def _load_processed(processed_root: str, cohort: str) -> DataBundle:
    static_path = os.path.join(processed_root, f"{cohort}_static.csv")
    dynamic_path = os.path.join(processed_root, f"{cohort}_dynamic.csv")
    return DataBundle(
        static_df=pd.read_csv(static_path),
        dynamic_df=pd.read_csv(dynamic_path),
    )


def _load_labels(processed_root: str, cohort: str) -> pd.DataFrame:
    path = os.path.join(processed_root, f"{cohort}_labels.csv")
    return pd.read_csv(path)


def _concat_bundles(*bundles: DataBundle) -> DataBundle:
    static_frames = [b.static_df for b in bundles if b is not None]
    dynamic_frames = [b.dynamic_df for b in bundles if b is not None]
    return DataBundle(
        static_df=pd.concat(static_frames, ignore_index=True) if static_frames else pd.DataFrame(),
        dynamic_df=pd.concat(dynamic_frames, ignore_index=True) if dynamic_frames else pd.DataFrame(),
    )


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def _filter_numeric_metrics(metrics: Optional[Dict[str, object]]) -> Dict[str, float]:
    """过滤数值型指标（去除 CM 等非数值）。"""
    if not metrics:
        return {}
    out: Dict[str, float] = {}
    for key, val in metrics.items():
        if key == "CM":
            continue
        if _is_number(val):
            out[key] = float(val)
    return out


def _aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """计算指标均值/标准差。"""
    if not metrics_list:
        return {}
    keys = sorted({k for m in metrics_list for k in m.keys()})
    agg: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m and _is_number(m[key])]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        agg[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }
    return agg


def _aggregate_list(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def _compute_pos_weight(dataset: PatientSequenceDataset) -> Optional[float]:
    y = dataset.y_seq
    mask = dataset.seq_mask
    if y.ndim == 3:
        y = y[..., 0]
    valid = mask > 0
    if valid.sum() == 0:
        return None
    y_valid = y[valid]
    pos = float(np.sum(y_valid))
    neg = float(y_valid.size - pos)
    if pos <= 0 or neg <= 0:
        return None
    return float(neg / pos)


def _filter_by_ids(df: pd.DataFrame, id_col: str, keep_ids: Optional[set]) -> pd.DataFrame:
    if keep_ids is None:
        return df
    return df[df[id_col].astype(str).isin(keep_ids)].copy()


def _split_ids(ids: List[str], val_ratio: float, seed: int) -> Tuple[set, set]:
    rng = np.random.RandomState(seed)
    ids = np.array(sorted(ids))
    rng.shuffle(ids)
    split = int(len(ids) * (1 - val_ratio))
    train_ids = set(ids[:split])
    val_ids = set(ids[split:])
    return train_ids, val_ids


def _get_organ_indices(dynamic_features: List[str], include_optional: bool, dynamic_optional: List[str]) -> List[int]:
    feature_list = dynamic_features + (dynamic_optional if include_optional else [])
    indices = []
    for col in ORGAN_COLS:
        if col in feature_list:
            indices.append(feature_list.index(col))
    return indices


def strip_best_state_from_cv_result(stage2_res: Dict[str, object]) -> Dict[str, object]:
    out = dict(stage2_res)
    out["fold_results"] = [
        {
            k: v
            for k, v in fr.items()
            if k not in {"best_state", "train_ids", "val_ids"}
        }
        for fr in stage2_res["fold_results"]
    ]
    out.pop("oof_df", None)
    return out


def _build_dataset(
    bundle: DataBundle,
    id_col: str,
    static_features: List[str],
    dynamic_features: List[str],
    dynamic_optional: List[str],
    include_optional: bool,
    include_mask_features: bool,
    day_min: int,
    day_max: int,
    task_mode: str,
    target_type: str,
    deterioration_threshold: float,
    use_sofa_score_label: bool,
    dynamic_medians: Dict[str, float],
) -> PatientSequenceDataset:
    return PatientSequenceDataset(
        bundle.static_df,
        bundle.dynamic_df,
        id_col=id_col,
        static_features=static_features,
        dynamic_features=dynamic_features,
        dynamic_optional=dynamic_optional,
        include_optional=include_optional,
        include_mask_features=include_mask_features,
        day_min=day_min,
        day_max=day_max,
        task_mode=task_mode,
        target_type=target_type,
        deterioration_threshold=deterioration_threshold,
        use_sofa_score_label=use_sofa_score_label,
        dynamic_medians=dynamic_medians,
    )


def run_stage(
    stage_name: str,
    bundle: DataBundle,
    id_col: str,
    static_features: List[str],
    dynamic_features: List[str],
    dynamic_optional: List[str],
    include_optional: bool,
    include_mask_features: bool,
    day_min: int,
    day_max: int,
    task_mode: str,
    target_type: str,
    deterioration_threshold: float,
    use_sofa_score_label: bool,
    dynamic_medians: Dict[str, float],
    model_type: str,
    static_fusion: str,
    model_kwargs: Dict[str, object],
    cfg: TrainConfig,
    output_dir: str,
    batch_size: int = 32,
    init_state: Optional[Dict[str, torch.Tensor]] = None,
    reset_head: bool = False,
) -> Dict[str, object]:
    # 构建训练/验证划分
    ids = bundle.static_df[id_col].astype(str).unique().tolist()
    train_ids, val_ids = _split_ids(ids, val_ratio=0.1, seed=cfg.seed)

    train_bundle = DataBundle(
        static_df=_filter_by_ids(bundle.static_df, id_col, train_ids),
        dynamic_df=_filter_by_ids(bundle.dynamic_df, id_col, train_ids),
    )
    val_bundle = DataBundle(
        static_df=_filter_by_ids(bundle.static_df, id_col, val_ids),
        dynamic_df=_filter_by_ids(bundle.dynamic_df, id_col, val_ids),
    )

    train_ds = _build_dataset(
        train_bundle,
        id_col,
        static_features,
        dynamic_features,
        dynamic_optional,
        include_optional,
        include_mask_features,
        day_min,
        day_max,
        task_mode,
        target_type,
        deterioration_threshold,
        use_sofa_score_label,
        dynamic_medians,
    )
    val_ds = _build_dataset(
        val_bundle,
        id_col,
        static_features,
        dynamic_features,
        dynamic_optional,
        include_optional,
        include_mask_features,
        day_min,
        day_max,
        task_mode,
        target_type,
        deterioration_threshold,
        use_sofa_score_label,
        dynamic_medians,
    )

    stage_cfg = cfg
    if task_mode == "deterioration" and cfg.loss_type == "bce_pos_weight" and cfg.pos_weight is None:
        pos_weight = _compute_pos_weight(train_ds)
        if pos_weight is not None:
            stage_cfg = replace(cfg, pos_weight=pos_weight)
            print(f"[{stage_name}] pos_weight={pos_weight:.4f}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    dyn_features = dynamic_features + (dynamic_optional if include_optional else [])
    input_dim = len(dyn_features) + (len(dyn_features) if include_mask_features else 0)

    if task_mode == "deterioration":
        target_dim = 1
    else:
        target_dim = 1 if target_type == "total" else len(ORGAN_COLS)

    model = build_model(
        model_type,
        input_dim=input_dim,
        static_dim=len(static_features),
        target_dim=target_dim,
        static_fusion=static_fusion,
        **model_kwargs,
    )

    if init_state is not None:
        model.load_state_dict(init_state, strict=False)

    if reset_head:
        model.reset_head(target_dim)

    organ_indices = _get_organ_indices(dynamic_features, include_optional, dynamic_optional)

    stage_out_dir = os.path.join(output_dir, stage_name)

    result = train_model(
        model,
        train_loader,
        val_loader,
        stage_cfg,
        target_dim,
        organ_feature_indices=organ_indices,
        output_dir=stage_out_dir,
        tag=f"{stage_name}_best",
        log_prefix=stage_name,
    )

    result["train_ids"] = sorted(train_ids)
    result["val_ids"] = sorted(val_ids)

    return result


def run_stage_cv(
    stage_name: str,
    bundle: DataBundle,
    id_col: str,
    static_features: List[str],
    dynamic_features: List[str],
    dynamic_optional: List[str],
    include_optional: bool,
    include_mask_features: bool,
    day_min: int,
    day_max: int,
    task_mode: str,
    target_type: str,
    deterioration_threshold: float,
    use_sofa_score_label: bool,
    dynamic_medians: Dict[str, float],
    model_type: str,
    static_fusion: str,
    model_kwargs: Dict[str, object],
    cfg: TrainConfig,
    output_dir: str,
    batch_size: int = 32,
    init_state: Optional[Dict[str, torch.Tensor]] = None,
    reset_head: bool = False,
    num_folds: int = 5,
) -> Dict[str, object]:
    # ===== patient-level 5-fold split =====
    ids = np.array(sorted(bundle.static_df[id_col].astype(str).unique().tolist()))
    if len(ids) < num_folds:
        raise ValueError(
            f"[{stage_name}] number of unique IDs ({len(ids)}) is smaller than num_folds ({num_folds})"
        )

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=cfg.seed)

    fold_results: List[Dict[str, object]] = []
    val_losses: List[float] = []
    metrics_list: List[Dict[str, float]] = []
    oof_frames = []

    dyn_features = dynamic_features + (dynamic_optional if include_optional else [])
    input_dim = len(dyn_features) + (len(dyn_features) if include_mask_features else 0)

    if task_mode == "deterioration":
        target_dim = 1
    else:
        target_dim = 1 if target_type == "total" else len(ORGAN_COLS)

    organ_indices = _get_organ_indices(dynamic_features, include_optional, dynamic_optional)

    stage_root_dir = os.path.join(output_dir, stage_name)
    os.makedirs(stage_root_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(ids)):
        train_ids = set(ids[train_idx])
        val_ids = set(ids[val_idx])

        train_bundle = DataBundle(
            static_df=_filter_by_ids(bundle.static_df, id_col, train_ids),
            dynamic_df=_filter_by_ids(bundle.dynamic_df, id_col, train_ids),
        )
        val_bundle = DataBundle(
            static_df=_filter_by_ids(bundle.static_df, id_col, val_ids),
            dynamic_df=_filter_by_ids(bundle.dynamic_df, id_col, val_ids),
        )

        train_ds = _build_dataset(
            train_bundle,
            id_col,
            static_features,
            dynamic_features,
            dynamic_optional,
            include_optional,
            include_mask_features,
            day_min,
            day_max,
            task_mode,
            target_type,
            deterioration_threshold,
            use_sofa_score_label,
            dynamic_medians,
        )
        val_ds = _build_dataset(
            val_bundle,
            id_col,
            static_features,
            dynamic_features,
            dynamic_optional,
            include_optional,
            include_mask_features,
            day_min,
            day_max,
            task_mode,
            target_type,
            deterioration_threshold,
            use_sofa_score_label,
            dynamic_medians,
        )

        fold_cfg = cfg
        if task_mode == "deterioration" and cfg.loss_type == "bce_pos_weight" and cfg.pos_weight is None:
            pos_weight = _compute_pos_weight(train_ds)
            if pos_weight is not None:
                fold_cfg = replace(cfg, pos_weight=pos_weight)
                print(f"[{stage_name}_fold{fold}] pos_weight={pos_weight:.4f}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(cfg.seed))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = build_model(
            model_type,
            input_dim=input_dim,
            static_dim=len(static_features),
            target_dim=target_dim,
            static_fusion=static_fusion,
            **model_kwargs,
        )

        if init_state is not None:
            model.load_state_dict(init_state, strict=False)

        if reset_head:
            model.reset_head(target_dim)

        fold_out_dir = os.path.join(stage_root_dir, f"fold{fold}")

        result = train_model(
            model,
            train_loader,
            val_loader,
            fold_cfg,
            target_dim,
            organ_feature_indices=organ_indices,
            output_dir=fold_out_dir,
            tag=f"{stage_name}_fold{fold}_best",
            log_prefix=f"{stage_name}_fold{fold}",
        )

        fold_metrics = _filter_numeric_metrics(result.get("best_metrics"))
        fold_val_loss = float(result["best_val_loss"])

        fold_result = {
            "fold": fold,
            "best_val_loss": fold_val_loss,
            "best_metrics": fold_metrics,
            "best_threshold": float(result.get("best_threshold", 0.5)),
            "train_ids": sorted(train_ids),
            "val_ids": sorted(val_ids),
            "best_state": result.get("best_state"),
        }
        fold_results.append(fold_result)

        val_losses.append(fold_val_loss)
        metrics_list.append(fold_metrics)

        fold_oof = predict_on_bundle(
            val_bundle,
            id_col,
            static_features,
            dynamic_features,
            dynamic_optional,
            include_optional,
            include_mask_features,
            day_min,
            day_max,
            task_mode,
            target_type,
            deterioration_threshold,
            use_sofa_score_label,
            dynamic_medians,
            model_type,
            static_fusion,
            model_kwargs,
            fold_cfg,
            result["best_state"],
            batch_size=batch_size,
        )
        fold_oof["fold"] = fold
        oof_frames.append(fold_oof)

    oof_df = pd.concat(oof_frames, ignore_index=True)

    oof_eval = evaluate_oof_predictions(
        oof_df,
        deterioration_threshold=deterioration_threshold,
        name=f"{stage_name}_oof",
    )

    summary = {
        "fold_results": fold_results,
        "loss_stats": _aggregate_list(val_losses),
        "metrics_stats": _aggregate_metrics(metrics_list),
        "oof_eval": oof_eval,
        "oof_df": oof_df,  # 只在内存里留着，写 JSON 时别保存
    }

    # 写入磁盘时去掉 best_state（Tensor 无法 JSON 序列化）
    summary_to_save = {
        "fold_results": [
            {k: v for k, v in fr.items() if k != "best_state"}
            for fr in fold_results
        ],
        "loss_stats": summary["loss_stats"],
        "metrics_stats": summary["metrics_stats"],
        "oof_eval": summary["oof_eval"],
    }

    with open(os.path.join(stage_root_dir, "cv_summary.json"), "w") as f:
        json.dump(summary_to_save, f, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer learning V2 pipeline")

    # ===== 路径 =====
    parser.add_argument("--processed-root", default="dataset/processed")
    parser.add_argument("--output-root", default="outputs/transfer_learning")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-auto-run-dir", action="store_true")

    # ===== 迁移模式 =====
    # reg_eval_det：Stage1 Sepsis 回归预训练，Stage2 BSI 回归/恶化评估
    # reg_to_det：Stage1 Sepsis 回归预训练，Stage2 BSI 恶化分类
    # det_class：Stage1/Stage2 都按分类任务
    parser.add_argument("--transfer-mode", choices=["reg_eval_det", "reg_to_det", "det_class"], default="reg_eval_det")
    parser.add_argument("--no-transfer", action="store_true", help="跳过 stage1，直接在 MIMIC-BSI 上从头训练")
    parser.add_argument("--sepsis-only", action="store_true", help="仅 Stage1 Sepsis 预训练，跳过 Stage2，直接零样本评估 BSI + 外部队列")
    parser.add_argument("--stage2-cohort", default="MIMIC-BSI", help="Stage2 目标队列，默认 MIMIC-BSI；消融实验可改为 MIMIC-SAB/KPB")
    parser.add_argument("--stage3-target", default=None, help="三阶段目标队列名（MIMIC-SAB 或 MIMIC-KPB）")
    parser.add_argument("--stage3-lr", type=float, default=None)
    parser.add_argument("--stage3-epochs", type=int, default=None)
    parser.add_argument("--stage3-weight-decay", type=float, default=None)
    parser.add_argument("--stage3-grad-clip", type=float, default=None)
    parser.add_argument("--stage3-freeze-backbone", action="store_true")
    parser.add_argument("--stage3-unfreeze-epoch", type=int, default=None)
    parser.add_argument("--stage3-early-stop-patience", type=int, default=None)

    # ===== 时间窗口 / CV =====
    parser.add_argument("--day-min", type=int, default=0)
    parser.add_argument("--day-max", type=int, default=7)
    parser.add_argument("--num-folds", type=int, default=5)

    # ===== 模型 =====
    parser.add_argument("--model-type", choices=["transformer"], default="transformer")
    parser.add_argument("--static-fusion", choices=["concat"], default="concat")

    # ===== 特征 =====
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--include-mask-features", action="store_true")

    # ===== 任务 =====
    parser.add_argument("--task-mode", choices=["regression", "deterioration"], default="regression")
    parser.add_argument("--target-type", choices=["total", "organ"], default="total")
    parser.add_argument("--deterioration-threshold", type=float, default=2.0)

    # ===== loss =====
    parser.add_argument("--loss-type", choices=["bce", "bce_pos_weight", "focal"], default="bce")
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    # ===== threshold =====
    parser.add_argument("--threshold-strategy", choices=["fixed_0.5", "best_f1"], default="fixed_0.5")
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)

    # ===== 训练 =====
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-epoch", type=int, default=None)

    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=None)

    # ===== early stopping =====
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)

    # ===== SAHZU-SAB 模式 =====
    parser.add_argument("--sahzu-sab-mode", choices=["matched", "full"], default="matched", help="选择对齐还是全时段 SAHZU-SAB，用于外部验证")

    # ===== 评估 =====
    parser.add_argument("--skip-external-eval", action="store_true")
    parser.add_argument("--aux-gram-loss-weight", type=float, default=0.0, help="Gram+/Gram- 辅助分类损失权重（0=禁用，建议 0.1~0.5）")
    parser.add_argument("--local-adapt-target", default=None, help="本地适配目标队列，如 SAHZU-SAB 或 SAHZU-KPB")
    parser.add_argument("--local-adapt-train-ratio", type=float, default=0.1, help="本地适配训练集比例")
    parser.add_argument("--local-adapt-lr", type=float, default=1e-5)
    parser.add_argument("--local-adapt-epochs", type=int, default=10)
    parser.add_argument("--local-adapt-seeds", default="42", help="本地适配多seed，逗号分隔，如 42,123,456")

    # ===== 其他 =====
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-train-metrics", action="store_true")
    parser.add_argument("--no-plots", action="store_true")

    # ===== Transformer =====
    parser.add_argument("--tf-d-model", type=int, default=128)
    parser.add_argument("--tf-nhead", type=int, default=4)
    parser.add_argument("--tf-num-layers", type=int, default=2)
    parser.add_argument("--tf-ffn-dim", type=int, default=256)
    parser.add_argument("--tf-dropout", type=float, default=0.1)
    parser.add_argument("--tf-max-len", type=int, default=16)

    return parser.parse_args()


def evaluate_on_bundle(
    bundle: DataBundle,
    id_col: str,
    static_features: List[str],
    dynamic_features: List[str],
    dynamic_optional: List[str],
    include_optional: bool,
    include_mask_features: bool,
    day_min: int,
    day_max: int,
    task_mode: str,
    target_type: str,
    deterioration_threshold: float,
    use_sofa_score_label: bool,
    dynamic_medians: Dict[str, float],
    model_type: str,
    static_fusion: str,
    model_kwargs: Dict[str, object],
    cfg: TrainConfig,
    best_state: Dict[str, torch.Tensor],
    batch_size: int = 32,
    name: str = "eval",
) -> Dict[str, object]:
    dyn_features = dynamic_features + (dynamic_optional if include_optional else [])
    input_dim = len(dyn_features) + (len(dyn_features) if include_mask_features else 0)

    if task_mode == "deterioration":
        target_dim = 1
    else:
        target_dim = 1 if target_type == "total" else len(ORGAN_COLS)

    organ_indices = _get_organ_indices(dynamic_features, include_optional, dynamic_optional)

    ds = _build_dataset(
        bundle,
        id_col,
        static_features,
        dynamic_features,
        dynamic_optional,
        include_optional,
        include_mask_features,
        day_min,
        day_max,
        task_mode,
        target_type,
        deterioration_threshold,
        use_sofa_score_label,
        dynamic_medians,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = build_model(
        model_type,
        input_dim=input_dim,
        static_dim=len(static_features),
        target_dim=target_dim,
        static_fusion=static_fusion,
        **model_kwargs,
    )
    model.load_state_dict(best_state, strict=False)
    model = model.to(torch.device(cfg.device))

    threshold = float(getattr(cfg, "best_threshold", 0.5))

    loss, metrics, *_ = evaluate(
        model,
        loader,
        torch.device(cfg.device),
        target_dim,
        task_mode,
        deterioration_threshold,
        organ_feature_indices=organ_indices,
        loss_type=cfg.loss_type,
        pos_weight=cfg.pos_weight,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        threshold=threshold,
    )

    metrics_clean = _filter_numeric_metrics(metrics)
    metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics_clean.items()])
    print(f"[{name}] loss={loss:.4f}, {metrics_str}")

    return {
        "loss": loss,
        "metrics": metrics_clean,
        "threshold": threshold,
    }


def predict_on_bundle(
    bundle: DataBundle,
    id_col: str,
    static_features: List[str],
    dynamic_features: List[str],
    dynamic_optional: List[str],
    include_optional: bool,
    include_mask_features: bool,
    day_min: int,
    day_max: int,
    task_mode: str,
    target_type: str,
    deterioration_threshold: float,
    use_sofa_score_label: bool,
    dynamic_medians: Dict[str, float],
    model_type: str,
    static_fusion: str,
    model_kwargs: Dict[str, object],
    cfg: TrainConfig,
    best_state: Dict[str, torch.Tensor],
    batch_size: int = 32,
) -> pd.DataFrame:
    dyn_features = dynamic_features + (dynamic_optional if include_optional else [])
    input_dim = len(dyn_features) + (len(dyn_features) if include_mask_features else 0)

    if task_mode == "deterioration":
        target_dim = 1
    else:
        target_dim = 1 if target_type == "total" else len(ORGAN_COLS)

    organ_indices = _get_organ_indices(dynamic_features, include_optional, dynamic_optional)

    ds = _build_dataset(
        bundle,
        id_col,
        static_features,
        dynamic_features,
        dynamic_optional,
        include_optional,
        include_mask_features,
        day_min,
        day_max,
        task_mode,
        target_type,
        deterioration_threshold,
        use_sofa_score_label,
        dynamic_medians,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = build_model(
        model_type,
        input_dim=input_dim,
        static_dim=len(static_features),
        target_dim=target_dim,
        static_fusion=static_fusion,
        **model_kwargs,
    )
    model.load_state_dict(best_state, strict=False)
    model = model.to(torch.device(cfg.device))
    model.eval()

    y_true_all = []
    y_pred_all = []
    meta_rows = []

    with torch.no_grad():
        for batch in loader:
            x_seq, x_static, y, seq_mask, prev_mask_seq, prev_total_seq, patient_ids = batch

            x_seq = x_seq.to(cfg.device)
            x_static = x_static.to(cfg.device)
            y = y.to(cfg.device)
            seq_mask = seq_mask.to(cfg.device)

            preds, _gram = model(x_seq, x_static)

            # 统一成 [B, T, D]
            is_organ = (target_type == "organ" and task_mode != "deterioration")
            n_organs = preds.shape[-1] if preds.ndim == 3 else 1

            if not is_organ:
                if preds.ndim == 3:
                    preds = preds[..., 0]
                if y.ndim == 3:
                    y = y[..., 0]

            y_true_np = y.detach().cpu().numpy()
            y_pred_np = preds.detach().cpu().numpy()
            seq_mask_np = seq_mask.detach().cpu().numpy()
            prev_total_np = prev_total_seq.detach().cpu().numpy()
            if prev_total_np.ndim == 3:
                prev_total_np = prev_total_np[..., 0]

            patient_ids = list(patient_ids)

            if is_organ and y_true_np.ndim == 3:
                batch_size_now, seq_len = y_true_np.shape[0], y_true_np.shape[1]
            else:
                batch_size_now, seq_len = y_true_np.shape

            for b in range(batch_size_now):
                pid = str(patient_ids[b])

                for t in range(seq_len):
                    if is_organ:
                        mask_val = seq_mask_np[b, t]
                        if isinstance(mask_val, np.ndarray):
                            mask_val = mask_val[0]
                    else:
                        mask_val = seq_mask_np[b, t]
                    if float(mask_val) <= 0:
                        continue

                    if is_organ and y_true_np.ndim == 3:
                        row = {
                            id_col: pid,
                            "time_index": int(t),
                            "prev_total": float(prev_total_np[b, t]) if np.isfinite(prev_total_np[b, t]) else np.nan,
                        }
                        for o in range(n_organs):
                            row[f"y_true_{o}"] = float(y_true_np[b, t, o])
                            row[f"y_pred_{o}"] = float(y_pred_np[b, t, o])
                        # total as sum of organs
                        row["y_true"] = float(y_true_np[b, t].sum())
                        row["y_pred"] = float(y_pred_np[b, t].sum())
                    else:
                        row = {
                            id_col: pid,
                            "time_index": int(t),
                            "prev_total": float(prev_total_np[b, t]) if np.isfinite(prev_total_np[b, t]) else np.nan,
                            "y_true": float(y_true_np[b, t]),
                            "y_pred": float(y_pred_np[b, t]),
                        }
                    meta_rows.append(row)

    out = pd.DataFrame(meta_rows)
    return out


def evaluate_oof_predictions(
    oof_df: pd.DataFrame,
    deterioration_threshold: float,
    name: str = "oof",
) -> Dict[str, object]:
    y_true = oof_df["y_true"].to_numpy(dtype=float)
    y_pred = oof_df["y_pred"].to_numpy(dtype=float)

    metrics = compute_regression_metrics(y_true, y_pred)

    if "prev_total" in oof_df.columns:
        prev_total = oof_df["prev_total"].to_numpy(dtype=float)
        valid_prev = np.isfinite(prev_total)

        if valid_prev.sum() > 0:
            acc_det = compute_deterioration_accuracy(
                y_true=y_true[valid_prev],
                y_pred=y_pred[valid_prev],
                prev_total=prev_total[valid_prev],
                threshold=deterioration_threshold,
                mask=np.ones(valid_prev.sum(), dtype=float),
            )
        else:
            acc_det = np.nan
    else:
        acc_det = np.nan

    metrics["ACC_DET@K"] = acc_det
    mse = metrics.pop("MSE")

    out_metrics = {
        "MAE": float(metrics["MAE"]),
        "RMSE": float(metrics["RMSE"]),
        "R2": float(metrics["R2"]),
        "ACC@1": float(metrics["ACC@1"]),
        "ACC_DET@K": float(metrics["ACC_DET@K"]) if not np.isnan(metrics["ACC_DET@K"]) else float("nan"),
        "MSE": float(mse),
    }

    # Per-organ metrics if available
    organ_cols = sorted([c for c in oof_df.columns if c.startswith("y_true_")])
    if organ_cols:
        organ_metrics = {}
        for yt_col in organ_cols:
            o_idx = yt_col.split("_")[-1]
            yp_col = f"y_pred_{o_idx}"
            if yp_col not in oof_df.columns:
                continue
            yt = oof_df[yt_col].to_numpy(dtype=float)
            yp = oof_df[yp_col].to_numpy(dtype=float)
            om = compute_regression_metrics(yt, yp)
            organ_metrics[f"organ_{o_idx}"] = {
                "name": ORGAN_COLS[int(o_idx)] if int(o_idx) < len(ORGAN_COLS) else f"organ_{o_idx}",
                "R2": float(om["R2"]),
                "MAE": float(om["MAE"]),
                "RMSE": float(om["RMSE"]),
            }
        out_metrics["organ_breakdown"] = organ_metrics

    loss = float(out_metrics["MAE"])
    metrics_str = ", ".join([f"{k}={v:.4f}" if not np.isnan(v) else f"{k}=nan" for k, v in out_metrics.items() if k != "organ_breakdown"])
    print(f"[{name}] loss={loss:.4f}, {metrics_str}")

    return {
        "loss": loss,
        "metrics": out_metrics,
        "n_samples": int(len(oof_df)),
    }


def _make_stage_cfg(base_cfg: TrainConfig, args: argparse.Namespace, prefix: str = "stage3") -> TrainConfig:
    """从命令行参数构建 Stage3 专用 TrainConfig。"""
    return replace(
        base_cfg,
        lr=_first_non_none(getattr(args, f"{prefix}_lr"), base_cfg.lr),
        n_epochs=_first_non_none(getattr(args, f"{prefix}_epochs"), base_cfg.n_epochs),
        weight_decay=_first_non_none(getattr(args, f"{prefix}_weight_decay"), base_cfg.weight_decay),
        grad_clip_norm=_first_non_none(getattr(args, f"{prefix}_grad_clip"), base_cfg.grad_clip_norm),
        freeze_backbone=getattr(args, f"{prefix}_freeze_backbone", False),
        unfreeze_epoch=getattr(args, f"{prefix}_unfreeze_epoch", None),
        early_stop_patience=_first_non_none(getattr(args, f"{prefix}_early_stop_patience"), base_cfg.early_stop_patience),
    )


def _first_non_none(*values: object) -> object:
    for v in values:
        if v is not None:
            return v
    return None


def _bootstrap_external_r2(
    bundle: "DataBundle",
    id_col: str,
    static_features: List[str],
    dynamic_features: List[str],
    dynamic_optional: List[str],
    include_optional: bool,
    include_mask_features: bool,
    day_min: int,
    day_max: int,
    task_mode: str,
    target_type: str,
    deterioration_threshold: float,
    use_sofa_score_label: bool,
    dynamic_medians: Dict[str, float],
    model_type: str,
    static_fusion: str,
    model_kwargs: Dict[str, object],
    cfg: TrainConfig,
    best_state: Dict[str, torch.Tensor],
    batch_size: int,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, object]:
    """Bootstrap external predictions from one selected model state.

    The caller supplies one model state, usually the fold with the lowest
    validation loss in the corresponding training stage. This function performs
    patient-level bootstrap resampling for that selected model; it does not
    average predictions across all CV folds.
    """
    df = predict_on_bundle(
        bundle, id_col,
        static_features, dynamic_features, dynamic_optional,
        include_optional, include_mask_features,
        day_min, day_max,
        task_mode, target_type, deterioration_threshold,
        use_sofa_score_label, dynamic_medians,
        model_type, static_fusion, model_kwargs, cfg, best_state,
        batch_size=batch_size,
    )
    if df.empty:
        return {}

    patient_ids = df[id_col].unique()
    n_patients = len(patient_ids)

    # pre-compute per-patient y_true and y_pred arrays
    pid_to_y_true = {}
    pid_to_y_pred = {}
    for pid in patient_ids:
        mask = df[id_col] == pid
        pid_to_y_true[pid] = df.loc[mask, "y_true"].to_numpy(dtype=float)
        pid_to_y_pred[pid] = df.loc[mask, "y_pred"].to_numpy(dtype=float)

    rng = np.random.RandomState(seed)
    r2_samples, mae_samples, rmse_samples = [], [], []

    for _ in range(n_bootstrap):
        sampled_pids = rng.choice(patient_ids, size=n_patients, replace=True)
        yt_all, yp_all = [], []
        for pid in sampled_pids:
            yt_all.append(pid_to_y_true[pid])
            yp_all.append(pid_to_y_pred[pid])
        yt = np.concatenate(yt_all)
        yp = np.concatenate(yp_all)
        m = compute_regression_metrics(yt, yp)
        r2_samples.append(m["R2"])
        mae_samples.append(m["MAE"])
        mse = m.pop("MSE", np.nan)
        rmse_samples.append(float(np.sqrt(mse)) if np.isfinite(mse) else np.nan)

    def _pct(vals, p):
        arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        if len(arr) == 0:
            return float("nan")
        return float(np.percentile(arr, p))

    return {
        "n_patients": n_patients,
        "n_bootstrap": n_bootstrap,
        "R2_median": _pct(r2_samples, 50),
        "R2_95CI_low": _pct(r2_samples, 2.5),
        "R2_95CI_high": _pct(r2_samples, 97.5),
        "MAE_median": _pct(mae_samples, 50),
        "MAE_95CI_low": _pct(mae_samples, 2.5),
        "MAE_95CI_high": _pct(mae_samples, 97.5),
        "RMSE_median": _pct(rmse_samples, 50),
        "RMSE_95CI_low": _pct(rmse_samples, 2.5),
        "RMSE_95CI_high": _pct(rmse_samples, 97.5),
    }


def _run_three_stage(
    args: argparse.Namespace,
    cfg: TrainConfig,
    model_kwargs: Dict[str, object],
    static_features: List[str],
    dynamic_features: List[str],
    dynamic_optional: List[str],
    dynamic_medians: Dict[str, float],
    output_root: str,
    stage1_res: Optional[Dict[str, object]],
) -> None:
    """三阶段迁移：Stage1(已跑) → Stage2(BSI, 单次) → Stage3(目标病原体, 5-fold CV) → external。"""
    mimic_bsi = _load_processed(args.processed_root, "MIMIC-BSI")
    target_bundle = _load_processed(args.processed_root, args.stage3_target)
    target_id_col = schema.MIMIC_ID_COL

    # 从 BSI 中剔除 Stage3 目标患者（防泄露）
    target_ids = set(target_bundle.static_df[target_id_col].astype(str))
    mimic_bsi = DataBundle(
        static_df=_filter_by_ids(mimic_bsi.static_df, schema.MIMIC_ID_COL,
                                  set(mimic_bsi.static_df[schema.MIMIC_ID_COL].astype(str)) - target_ids),
        dynamic_df=_filter_by_ids(mimic_bsi.dynamic_df, schema.MIMIC_ID_COL,
                                   set(mimic_bsi.dynamic_df[schema.MIMIC_ID_COL].astype(str)) - target_ids),
    )
    print(f"[Info] Stage2 BSI after excluding {args.stage3_target}: static={len(mimic_bsi.static_df)}, dynamic={len(mimic_bsi.dynamic_df)}")

    stage_task = "regression" if args.transfer_mode in {"reg_eval_det", "reg_to_det"} else "deterioration"

    # ---- Stage2: BSI single run ----
    stage2_cfg = replace(cfg, task_mode=stage_task)
    stage2_init = stage1_res["best_state"] if stage1_res is not None else None

    stage2_res = run_stage(
        "stage2_bsi",
        mimic_bsi,
        schema.MIMIC_ID_COL,
        static_features, dynamic_features, dynamic_optional,
        args.include_optional, args.include_mask_features,
        args.day_min, args.day_max,
        stage_task, args.target_type,
        args.deterioration_threshold, True,
        dynamic_medians,
        args.model_type, args.static_fusion,
        model_kwargs, stage2_cfg, output_root,
        batch_size=args.batch_size, init_state=stage2_init,
    )

    # ---- Stage3: target pathogen 5-fold CV ----
    stage3_cfg = _make_stage_cfg(cfg, args, prefix="stage3")
    stage3_cfg = replace(stage3_cfg, task_mode=stage_task)

    stage3_res = run_stage_cv(
        f"stage3_{args.stage3_target}",
        target_bundle,
        target_id_col,
        static_features, dynamic_features, dynamic_optional,
        args.include_optional, args.include_mask_features,
        args.day_min, args.day_max,
        stage_task, args.target_type,
        args.deterioration_threshold, True,
        dynamic_medians,
        args.model_type, args.static_fusion,
        model_kwargs, stage3_cfg, output_root,
        batch_size=args.batch_size,
        init_state=stage2_res["best_state"],
        num_folds=args.num_folds,
    )

    # ---- External eval ----
    external_eval: Dict[str, Dict[str, object]] = {}
    if not args.skip_external_eval:
        sahzu_sab_cohort = "SAHZU-SAB_matched" if args.sahzu_sab_mode == "matched" else "SAHZU-SAB_full"
        sahzu_sab = _load_processed(args.processed_root, sahzu_sab_cohort)
        sahzu_kpb = _load_processed(args.processed_root, "SAHZU-KPB")

        sab_fold_losses, sab_fold_metrics = [], []
        kpb_fold_losses, kpb_fold_metrics = [], []

        for fold_item in stage3_res["fold_results"]:
            fold_state = fold_item["best_state"]
            fold_thr = float(fold_item.get("best_threshold", 0.5))
            eval_cfg = replace(stage3_cfg, threshold_strategy=stage3_cfg.threshold_strategy)
            setattr(eval_cfg, "best_threshold", fold_thr)

            for ext_bundle, ext_id_col, ext_name, losses, metrs in [
                (sahzu_sab, schema.SAHZU_ID_COL, "SAHZU-SAB", sab_fold_losses, sab_fold_metrics),
                (sahzu_kpb, schema.SAHZU_ID_COL, "SAHZU-KPB", kpb_fold_losses, kpb_fold_metrics),
            ]:
                r = evaluate_on_bundle(
                    ext_bundle, ext_id_col,
                    static_features, dynamic_features, dynamic_optional,
                    args.include_optional, args.include_mask_features,
                    args.day_min, args.day_max, stage_task,
                    args.target_type, args.deterioration_threshold, True,
                    dynamic_medians,
                    args.model_type, args.static_fusion,
                    model_kwargs, eval_cfg, fold_state,
                    batch_size=args.batch_size,
                    name=f"external_{ext_name}_fold{fold_item['fold']}",
                )
                losses.append(r["loss"])
                metrs.append(r["metrics"])

        for name, losses, metrs in [
            ("SAHZU-SAB", sab_fold_losses, sab_fold_metrics),
            ("SAHZU-KPB", kpb_fold_losses, kpb_fold_metrics),
        ]:
            external_eval[name] = {
                "loss_by_fold": losses,
                "loss_stats": _aggregate_list(losses),
                "metrics_by_fold": metrs,
                "metrics_stats": _aggregate_metrics(metrs),
            }

    # ---- Summary ----
    summary = {
        "transfer_mode": args.transfer_mode,
        "model_type": args.model_type,
        "no_transfer": args.no_transfer,
        "stage3_target": args.stage3_target,
        "day_min": args.day_min, "day_max": args.day_max, "num_folds": args.num_folds,
        "stage1": None if stage1_res is None else {
            "best_val_loss": stage1_res["best_val_loss"],
            "metrics": _filter_numeric_metrics(stage1_res.get("best_metrics")),
        },
        "stage2": {
            "best_val_loss": stage2_res["best_val_loss"],
            "metrics": _filter_numeric_metrics(stage2_res.get("best_metrics")),
        },
        f"stage3_{args.stage3_target}": strip_best_state_from_cv_result(stage3_res),
        "external_eval": external_eval,
    }

    # Bootstrap
    if not args.skip_external_eval:
        best_fold = min(stage3_res["fold_results"], key=lambda fr: fr["best_val_loss"])
        summary["external_bootstrap"] = {}
        for ext_name, ext_bundle, ext_id in [
            ("SAHZU-SAB", sahzu_sab, schema.SAHZU_ID_COL),
            ("SAHZU-KPB", sahzu_kpb, schema.SAHZU_ID_COL),
        ]:
            bs = _bootstrap_external_r2(
                ext_bundle, ext_id,
                static_features, dynamic_features, dynamic_optional,
                args.include_optional, args.include_mask_features,
                args.day_min, args.day_max,
                stage_task, args.target_type, args.deterioration_threshold, True,
                dynamic_medians,
                args.model_type, args.static_fusion, model_kwargs,
                stage3_cfg, best_fold["best_state"],
                batch_size=args.batch_size,
            )
            summary["external_bootstrap"][ext_name] = bs

    with open(os.path.join(output_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Done] Three-stage transfer completed. Output: {output_root}")


def _run_local_adapt(
    args: argparse.Namespace,
    cfg: TrainConfig,
    model_kwargs: Dict[str, object],
    static_features: List[str],
    dynamic_features: List[str],
    dynamic_optional: List[str],
    dynamic_medians: Dict[str, float],
    output_root: str,
    stage1_res: Optional[Dict[str, object]],
) -> None:
    """本地适配：Stage1 → Stage2(BSI全量) → SAHZU小样本微调 → SAHZU剩余评估。"""
    mimic_bsi = _load_processed(args.processed_root, args.stage2_cohort)

    # ---- Stage2: BSI full (single run, 不剔除任何患者) ----
    stage_task = "regression" if args.transfer_mode in {"reg_eval_det", "reg_to_det"} else "deterioration"
    stage2_cfg = replace(cfg, task_mode=stage_task)
    stage2_init = stage1_res["best_state"] if stage1_res is not None else None

    stage2_res = run_stage(
        "stage2_bsi",
        mimic_bsi, schema.MIMIC_ID_COL,
        static_features, dynamic_features, dynamic_optional,
        args.include_optional, args.include_mask_features,
        args.day_min, args.day_max,
        stage_task, args.target_type, args.deterioration_threshold, True,
        dynamic_medians,
        args.model_type, args.static_fusion, model_kwargs, stage2_cfg, output_root,
        batch_size=args.batch_size, init_state=stage2_init,
    )
    print(f"[LocalAdapt] Stage2 done: val_loss={stage2_res['best_val_loss']:.4f}")

    # ---- 解析 seeds ----
    seed_strs = args.local_adapt_seeds.split(",")
    local_seeds = [int(s.strip()) for s in seed_strs]

    # ---- 每个 seed 独立 split + adapt + evaluate ----
    sahzu_bundle = _load_processed(args.processed_root, args.local_adapt_target)
    sahzu_id_col = schema.SAHZU_ID_COL
    all_ids = sorted(sahzu_bundle.static_df[sahzu_id_col].astype(str).unique().tolist())

    seed_results = []
    for seed_idx, split_seed in enumerate(local_seeds):
        seed_dir = os.path.join(output_root, f"seed_{split_seed}")
        os.makedirs(seed_dir, exist_ok=True)

        rng = np.random.RandomState(split_seed)
        shuffled = list(all_ids)
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * args.local_adapt_train_ratio))
        train_ids = set(shuffled[:n_train])
        test_ids = set(shuffled[n_train:])
        print(f"[LocalAdapt] seed={split_seed}: train={len(train_ids)}, test={len(test_ids)}")

        train_bundle = DataBundle(
            static_df=_filter_by_ids(sahzu_bundle.static_df, sahzu_id_col, train_ids),
            dynamic_df=_filter_by_ids(sahzu_bundle.dynamic_df, sahzu_id_col, train_ids),
        )
        test_bundle = DataBundle(
            static_df=_filter_by_ids(sahzu_bundle.static_df, sahzu_id_col, test_ids),
            dynamic_df=_filter_by_ids(sahzu_bundle.dynamic_df, sahzu_id_col, test_ids),
        )

        adapt_cfg = replace(
            stage2_cfg,
            lr=args.local_adapt_lr,
            n_epochs=args.local_adapt_epochs,
            early_stop_patience=3,
            freeze_backbone=False,
            seed=split_seed,
        )

        adapt_res = run_stage(
            f"stage3_local_{args.local_adapt_target}_seed{split_seed}",
            train_bundle, sahzu_id_col,
            static_features, dynamic_features, dynamic_optional,
            args.include_optional, args.include_mask_features,
            args.day_min, args.day_max,
            stage_task, args.target_type, args.deterioration_threshold, True,
            dynamic_medians,
            args.model_type, args.static_fusion, model_kwargs, adapt_cfg, seed_dir,
            batch_size=args.batch_size,
            init_state=stage2_res["best_state"],
        )

        baseline_bs = _bootstrap_external_r2(
            test_bundle, sahzu_id_col,
            static_features, dynamic_features, dynamic_optional,
            args.include_optional, args.include_mask_features,
            args.day_min, args.day_max, stage_task, args.target_type,
            args.deterioration_threshold, True, dynamic_medians,
            args.model_type, args.static_fusion, model_kwargs,
            stage2_cfg, stage2_res["best_state"],
            batch_size=args.batch_size, seed=split_seed,
        )
        adapted_bs = _bootstrap_external_r2(
            test_bundle, sahzu_id_col,
            static_features, dynamic_features, dynamic_optional,
            args.include_optional, args.include_mask_features,
            args.day_min, args.day_max, stage_task, args.target_type,
            args.deterioration_threshold, True, dynamic_medians,
            args.model_type, args.static_fusion, model_kwargs,
            adapt_cfg, adapt_res["best_state"],
            batch_size=args.batch_size, seed=split_seed,
        )

        seed_results.append({
            "seed": split_seed,
            "train_n": n_train,
            "test_n": len(test_ids),
            "baseline": baseline_bs,
            "adapted": adapted_bs,
            "delta_R2": adapted_bs["R2_median"] - baseline_bs["R2_median"],
            "delta_MAE": adapted_bs["MAE_median"] - baseline_bs["MAE_median"],
            "delta_RMSE": adapted_bs["RMSE_median"] - baseline_bs["RMSE_median"],
        })
        print(f"  seed={split_seed}: baseline_R2={baseline_bs['R2_median']:.4f}, adapted_R2={adapted_bs['R2_median']:.4f}, delta={adapted_bs['R2_median']-baseline_bs['R2_median']:+.4f}")

    # ---- 跨 seed 汇总 ----
    def _agg(values: List[float]) -> Dict[str, float]:
        arr = np.array(values)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)),
                "min": float(np.min(arr)), "max": float(np.max(arr)),
                "values": [float(v) for v in values]}

    delta_r2s = [r["delta_R2"] for r in seed_results]
    delta_maes = [r["delta_MAE"] for r in seed_results]
    delta_rmses = [r["delta_RMSE"] for r in seed_results]

    summary = {
        "transfer_mode": args.transfer_mode,
        "model_type": args.model_type,
        "local_adapt_target": args.local_adapt_target,
        "local_adapt_train_ratio": args.local_adapt_train_ratio,
        "local_adapt_seeds": local_seeds,
        "n_seeds": len(local_seeds),
        "stage1": None if stage1_res is None else {
            "best_val_loss": stage1_res["best_val_loss"],
            "metrics": _filter_numeric_metrics(stage1_res.get("best_metrics")),
        },
        "stage2": {
            "best_val_loss": stage2_res["best_val_loss"],
            "metrics": _filter_numeric_metrics(stage2_res.get("best_metrics")),
        },
        "seed_details": seed_results,
        "aggregated_delta_R2": _agg(delta_r2s),
        "aggregated_delta_MAE": _agg(delta_maes),
        "aggregated_delta_RMSE": _agg(delta_rmses),
    }
    with open(os.path.join(output_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Done] Local adaptation across {len(local_seeds)} seeds. "
          f"delta_R2={_agg(delta_r2s)['mean']:+.4f} +/- {_agg(delta_r2s)['std']:.4f}. "
          f"Output: {output_root}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.model_type == "transformer" and args.static_fusion != "concat":
        raise ValueError("Transformer 仅支持 static_fusion=concat")

    def build_model_kwargs() -> Dict[str, object]:
        return {
            "d_model": args.tf_d_model,
            "nhead": args.tf_nhead,
            "num_layers": args.tf_num_layers,
            "dim_feedforward": args.tf_ffn_dim,
            "dropout": args.tf_dropout,
            "max_len": args.tf_max_len,
        }

    def build_run_name() -> str:
        prefix = "scratch_v2" if args.no_transfer else "transfer_v2"
        parts = [
            prefix,
            args.transfer_mode,
            args.model_type,
            f"target_{args.target_type}",
            f"d{args.day_min}_to_{args.day_max}",
            f"cv{args.num_folds}",
        ]
        if args.include_optional:
            parts.append("opt")
        if args.include_mask_features:
            parts.append("mask")
        return "_".join(parts)

    output_root = args.output_root
    if not args.no_auto_run_dir:
        run_name = args.run_name or build_run_name()
        output_root = os.path.join(output_root, run_name)

    feature_json = os.path.join(args.processed_root, "feature_intersection.json")
    features = _load_feature_json(feature_json)

    static_features = features["static_features"]
    dynamic_features = features["dynamic_features"]
    dynamic_optional = features["dynamic_optional"]

    dynamic_medians_path = os.path.join(args.processed_root, "artifacts", "dynamic_medians.json")
    dynamic_medians = _load_dynamic_medians(dynamic_medians_path)

    # ===== 加载数据 =====
    mimic_sepsis = _load_processed(args.processed_root, "MIMIC-Sepsis")
    mimic_bsi = _load_processed(args.processed_root, args.stage2_cohort)
    sahzu_sab_cohort = "SAHZU-SAB_matched" if args.sahzu_sab_mode == "matched" else "SAHZU-SAB_full"
    sahzu_sab = _load_processed(args.processed_root, sahzu_sab_cohort)
    sahzu_kpb = _load_processed(args.processed_root, "SAHZU-KPB")

    # 从 MIMIC-Sepsis 中剔除与 MIMIC-BSI 重叠患者，避免泄漏
    bsi_ids = set(mimic_bsi.static_df[schema.MIMIC_ID_COL].astype(str))
    sepsis_ids = set(mimic_sepsis.static_df[schema.MIMIC_ID_COL].astype(str))
    stage1_ids = sepsis_ids - bsi_ids

    mimic_sepsis = DataBundle(
        static_df=_filter_by_ids(mimic_sepsis.static_df, schema.MIMIC_ID_COL, stage1_ids),
        dynamic_df=_filter_by_ids(mimic_sepsis.dynamic_df, schema.MIMIC_ID_COL, stage1_ids),
    )

    # 加载 Gram 标签（仅 BSI 有，用于辅助分类损失）
    gram_label_map: Optional[Dict[str, int]] = None
    if args.aux_gram_loss_weight > 0:
        labels_path = os.path.join(args.processed_root, "MIMIC-BSI_labels.csv")
        if os.path.exists(labels_path):
            labels_df = pd.read_csv(labels_path)
            labels_df["stay_id"] = labels_df["stay_id"].astype(str)
            labels_df["gram_group"] = labels_df["gram_group"].astype(str).str.strip()
            gram_label_map = {}
            for _, row in labels_df.iterrows():
                g = row["gram_group"]
                if g == "Gram+":
                    gram_label_map[row["stay_id"]] = 1
                elif g == "Gram-":
                    gram_label_map[row["stay_id"]] = 0
            print(f"[Info] Gram label map: {len(gram_label_map)} patients (Gram+={sum(1 for v in gram_label_map.values() if v==1)}, Gram-={sum(1 for v in gram_label_map.values() if v==0)})")
        else:
            print(f"[Warn] Gram labels not found at {labels_path}")

    cfg = TrainConfig(
        task_mode=args.task_mode,
        target_type=args.target_type,
        deterioration_threshold=args.deterioration_threshold,
        n_epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        freeze_backbone=args.freeze_backbone,
        unfreeze_epoch=args.unfreeze_epoch,
        show_progress=not args.no_progress,
        compute_train_metrics=not args.no_train_metrics,
        plot_history=not args.no_plots,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        loss_type=args.loss_type,
        pos_weight=args.pos_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        threshold_strategy=args.threshold_strategy,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        aux_gram_loss_weight=args.aux_gram_loss_weight,
        gram_label_map=gram_label_map,
    )

    model_kwargs = build_model_kwargs()
    os.makedirs(output_root, exist_ok=True)

    stage1_bundle = mimic_sepsis
    stage2_bundle = mimic_bsi

    # ===== Stage1: Sepsis 预训练（只跑一次） =====
    stage1_res = None
    if not args.no_transfer:
        stage1_task = "regression" if args.transfer_mode in {"reg_eval_det", "reg_to_det"} else "deterioration"
        stage1_cfg = replace(cfg, task_mode=stage1_task)

        stage1_res = run_stage(
            "stage1_sepsis",
            stage1_bundle,
            schema.MIMIC_ID_COL,
            static_features,
            dynamic_features,
            dynamic_optional,
            args.include_optional,
            args.include_mask_features,
            args.day_min,
            args.day_max,
            stage1_task,
            args.target_type,
            args.deterioration_threshold,
            True,
            dynamic_medians,
            args.model_type,
            args.static_fusion,
            model_kwargs,
            stage1_cfg,
            output_root,
            batch_size=args.batch_size,
        )

    # ===== Sepsis-only 零样本评估 =====
    if args.sepsis_only:
        if args.no_transfer:
            raise ValueError("--sepsis-only 与 --no-transfer 互斥")
        if stage1_res is None:
            raise RuntimeError("Stage1 训练失败，无法进行 sepsis-only 评估")

        stage1_summary: Dict[str, object] = {
            "best_val_loss": stage1_res["best_val_loss"],
            "metrics": _filter_numeric_metrics(stage1_res.get("best_metrics")),
        }
        sepsis_only_summary: Dict[str, object] = {
            "transfer_mode": args.transfer_mode,
            "model_type": args.model_type,
            "sepsis_only": True,
            "day_min": args.day_min,
            "day_max": args.day_max,
            "stage1": stage1_summary,
        }

        # 评估 BSI（零样本）
        bsi_metrics = evaluate_on_bundle(
            stage2_bundle, schema.MIMIC_ID_COL,
            static_features, dynamic_features, dynamic_optional,
            args.include_optional, args.include_mask_features,
            args.day_min, args.day_max,
            "regression", args.target_type, args.deterioration_threshold,
            True, dynamic_medians,
            args.model_type, args.static_fusion, model_kwargs,
            cfg, stage1_res["best_state"], batch_size=args.batch_size, name="BSI_zero_shot",
        )
        sepsis_only_summary["bsi_zero_shot"] = _filter_numeric_metrics(bsi_metrics.get("metrics"))

        # Bootstrap 外部评估
        for ext_name, ext_bundle in [("SAHZU-SAB", sahzu_sab), ("SAHZU-KPB", sahzu_kpb)]:
            ext_boot = _bootstrap_external_r2(
                ext_bundle, schema.SAHZU_ID_COL,
                static_features, dynamic_features, dynamic_optional,
                args.include_optional, args.include_mask_features,
                args.day_min, args.day_max,
                "regression", args.target_type, args.deterioration_threshold,
                True, dynamic_medians,
                args.model_type, args.static_fusion, model_kwargs,
                cfg, stage1_res["best_state"],
                batch_size=args.batch_size, n_bootstrap=1000, seed=args.seed,
            )
            sepsis_only_summary[f"external_bootstrap_{ext_name}"] = ext_boot

        summary_path = os.path.join(output_root, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(sepsis_only_summary, f, indent=2, default=str)
        print(f"[sepsis-only] Summary saved to {summary_path}")
        return

    # ===== 三阶段 / 本地适配 分支 =====
    if args.stage3_target:
        _run_three_stage(args, cfg, model_kwargs, static_features, dynamic_features, dynamic_optional,
                         dynamic_medians, output_root, stage1_res=stage1_res)
        return
    # ===== 本地适配 分支 =====
    if args.local_adapt_target:
        _run_local_adapt(args, cfg, model_kwargs, static_features, dynamic_features, dynamic_optional,
                         dynamic_medians, output_root, stage1_res=stage1_res)
        return

    # ===== Stage2: BSI 5-fold CV =====
    if args.transfer_mode == "reg_to_det":
        stage2_task = "deterioration"
    elif args.transfer_mode == "det_class":
        stage2_task = "deterioration"
    else:
        stage2_task = "regression"

    stage2_cfg = replace(cfg, task_mode=stage2_task)
    stage2_init_state = None if args.no_transfer else stage1_res["best_state"]

    stage2_res = run_stage_cv(
        "stage2_bsi",
        stage2_bundle,
        schema.MIMIC_ID_COL,
        static_features,
        dynamic_features,
        dynamic_optional,
        args.include_optional,
        args.include_mask_features,
        args.day_min,
        args.day_max,
        stage2_task,
        args.target_type,
        args.deterioration_threshold,
        True,
        dynamic_medians,
        args.model_type,
        args.static_fusion,
        model_kwargs,
        stage2_cfg,
        output_root,
        batch_size=args.batch_size,
        init_state=stage2_init_state,
        reset_head=(args.transfer_mode == "reg_to_det"),
        num_folds=args.num_folds,
    )

    # ===== External eval: fold-level =====
    external_eval: Dict[str, Dict[str, object]] = {}

    if not args.skip_external_eval:
        print("[Info] Start external evaluation...")

        sab_fold_losses = []
        sab_fold_metrics = []
        kpb_fold_losses = []
        kpb_fold_metrics = []

        for fold_item in stage2_res["fold_results"]:
            fold_best_state = fold_item.get("best_state")
            fold_threshold = float(fold_item.get("best_threshold", 0.5))

            stage2_eval_cfg = replace(stage2_cfg, threshold_strategy=stage2_cfg.threshold_strategy)
            setattr(stage2_eval_cfg, "best_threshold", fold_threshold)

            sab_result = evaluate_on_bundle(
                sahzu_sab,
                schema.SAHZU_ID_COL,
                static_features,
                dynamic_features,
                dynamic_optional,
                args.include_optional,
                args.include_mask_features,
                args.day_min,
                args.day_max,
                stage2_task,
                args.target_type,
                args.deterioration_threshold,
                True,
                dynamic_medians,
                args.model_type,
                args.static_fusion,
                model_kwargs,
                stage2_eval_cfg,
                fold_best_state,
                batch_size=args.batch_size,
                name=f"external_SAHZU-SAB_fold{fold_item['fold']}",
            )
            sab_fold_losses.append(sab_result["loss"])
            sab_fold_metrics.append(sab_result["metrics"])

            kpb_result = evaluate_on_bundle(
                sahzu_kpb,
                schema.SAHZU_ID_COL,
                static_features,
                dynamic_features,
                dynamic_optional,
                args.include_optional,
                args.include_mask_features,
                args.day_min,
                args.day_max,
                stage2_task,
                args.target_type,
                args.deterioration_threshold,
                True,
                dynamic_medians,
                args.model_type,
                args.static_fusion,
                model_kwargs,
                stage2_eval_cfg,
                fold_best_state,
                batch_size=args.batch_size,
                name=f"external_SAHZU-KPB_fold{fold_item['fold']}",
            )
            kpb_fold_losses.append(kpb_result["loss"])
            kpb_fold_metrics.append(kpb_result["metrics"])

        external_eval["SAHZU-SAB"] = {
            "loss_by_fold": sab_fold_losses,
            "loss_stats": _aggregate_list(sab_fold_losses),
            "metrics_by_fold": sab_fold_metrics,
            "metrics_stats": _aggregate_metrics(sab_fold_metrics),
        }
        external_eval["SAHZU-KPB"] = {
            "loss_by_fold": kpb_fold_losses,
            "loss_stats": _aggregate_list(kpb_fold_losses),
            "metrics_by_fold": kpb_fold_metrics,
            "metrics_stats": _aggregate_metrics(kpb_fold_metrics),
        }

    # ===== Subgroup eval: 先用所有 fold val_ids 合并版 =====
    subgroup_eval: Dict[str, Dict[str, object]] = {}

    labels_path = os.path.join(args.processed_root, "MIMIC-BSI_labels.csv")
    if os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path)
        labels_df["stay_id"] = labels_df["stay_id"].astype(str)
        labels_df["gram_group"] = labels_df["gram_group"].astype(str).str.strip()
        labels_df["organism_norm"] = labels_df["organism_norm"].astype(str).str.strip()

        overall_ids = set(labels_df["stay_id"])
        gram_pos_df = labels_df[labels_df["gram_group"] == "Gram+"].copy()
        gram_neg_df = labels_df[labels_df["gram_group"] == "Gram-"].copy()

        gram_pos_top3 = gram_pos_df["organism_norm"].value_counts().head(3).index.tolist()
        gram_neg_top3 = gram_neg_df["organism_norm"].value_counts().head(3).index.tolist()

        print(f"[Info] Gram+ top3 organisms: {gram_pos_top3}")
        print(f"[Info] Gram- top3 organisms: {gram_neg_top3}")

        subgroup_id_sets = {
            "overall": overall_ids,
            "Gram+": set(gram_pos_df["stay_id"]),
            "Gram-": set(gram_neg_df["stay_id"]),
        }

        for idx, org in enumerate(gram_pos_top3, start=1):
            subgroup_id_sets[f"Gram+_top{idx}_{org}"] = set(
                gram_pos_df.loc[gram_pos_df["organism_norm"] == org, "stay_id"].astype(str)
            )

        for idx, org in enumerate(gram_neg_top3, start=1):
            subgroup_id_sets[f"Gram-_top{idx}_{org}"] = set(
                gram_neg_df.loc[gram_neg_df["organism_norm"] == org, "stay_id"].astype(str)
            )

        oof_df = stage2_res["oof_df"].copy()
        oof_df[schema.MIMIC_ID_COL] = oof_df[schema.MIMIC_ID_COL].astype(str)

        for subgroup_name, subgroup_ids in subgroup_id_sets.items():
            subgroup_ids = set(map(str, subgroup_ids))
            subgroup_oof = oof_df[oof_df[schema.MIMIC_ID_COL].isin(subgroup_ids)].copy()

            if subgroup_oof.empty:
                print(f"[subgroup_{subgroup_name}_oof] skipped: no OOF samples found")
                continue

            subgroup_eval[subgroup_name] = evaluate_oof_predictions(
                subgroup_oof,
                deterioration_threshold=args.deterioration_threshold,
                name=f"subgroup_{subgroup_name}_oof",
            )
    else:
        print(f"[Warn] subgroup labels file not found: {labels_path}")


    # ===== 写 summary.json 时去掉不可序列化的 model =====
    stage2_cv_to_save = {
        "fold_results": [
            {k: v for k, v in fr.items() if k not in {"best_state", "train_ids", "val_ids"}}
            for fr in stage2_res["fold_results"]
        ],
        "loss_stats": stage2_res["loss_stats"],
        "metrics_stats": stage2_res["metrics_stats"],
    }

    # ===== 保存汇总 =====
    summary = {
        "transfer_mode": args.transfer_mode,
        "model_type": args.model_type,
        "no_transfer": args.no_transfer,
        "day_min": args.day_min,
        "day_max": args.day_max,
        "num_folds": args.num_folds,
        "stage1": None if stage1_res is None else {
            "best_val_loss": stage1_res["best_val_loss"],
            "metrics": _filter_numeric_metrics(stage1_res.get("best_metrics")),
        },
        "stage2_cv": strip_best_state_from_cv_result(stage2_res),
        "external_eval": external_eval,
        "subgroup_eval": subgroup_eval,
    }

    # Bootstrap
    if not args.skip_external_eval:
        best_fold = min(stage2_res["fold_results"], key=lambda fr: fr["best_val_loss"])
        summary["external_bootstrap"] = {}
        for ext_name, ext_bundle, ext_id in [
            ("SAHZU-SAB", sahzu_sab, schema.SAHZU_ID_COL),
            ("SAHZU-KPB", sahzu_kpb, schema.SAHZU_ID_COL),
        ]:
            bs = _bootstrap_external_r2(
                ext_bundle, ext_id,
                static_features, dynamic_features, dynamic_optional,
                args.include_optional, args.include_mask_features,
                args.day_min, args.day_max,
                stage2_task, args.target_type, args.deterioration_threshold, True,
                dynamic_medians,
                args.model_type, args.static_fusion, model_kwargs,
                stage2_cfg, best_fold["best_state"],
                batch_size=args.batch_size,
            )
            summary["external_bootstrap"][ext_name] = bs

    with open(os.path.join(output_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] Transfer learning V2 CV completed. Output: {output_root}")


if __name__ == "__main__":
    main()



# python -m transfer_learning.transfer --processed-root dataset/processed /
#       --output-root outputs/transfer_learning /
#       --transfer-mode reg_eval_det /
#       --model-type transformer --target-type total --epochs 1 --batch-size 8 --device cpu