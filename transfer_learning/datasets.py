# coding: utf-8
"""数据集构建：长表动态 + 静态表组序列。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import schema


ORGAN_COLS = [
    "sofa_respiration",
    "sofa_coagulation",
    "sofa_liver",
    "sofa_cardiovascular",
    "sofa_cns",
    "sofa_renal",
]


def load_feature_config(path: str) -> Dict:
    """读取特征清单 JSON。"""
    with open(path, "r") as f:
        return json.load(f)


def load_dynamic_medians(path: str) -> Dict[str, float]:
    """读取动态特征中位数。"""
    with open(path, "r") as f:
        return json.load(f)


def _build_day_index(day_min: int, day_max: int) -> List[int]:
    return list(range(day_min, day_max + 1))


def _get_row_by_day(day_map: Dict[int, pd.Series], day: int) -> Optional[pd.Series]:
    return day_map.get(day)


@dataclass
class SequenceMeta:
    days: List[int]
    input_days: List[int]
    target_days: List[int]


class PatientSequenceDataset(Dataset):
    """按患者构建时序输入与标签。"""

    def __init__(
        self,
        static_df: pd.DataFrame,
        dynamic_df: pd.DataFrame,
        id_col: str,
        static_features: List[str],
        dynamic_features: List[str],
        dynamic_optional: Optional[List[str]] = None,
        include_optional: bool = False,
        include_mask_features: bool = False,
        day_min: int = -1,
        day_max: int = 7,
        task_mode: str = "regression",
        target_type: str = "total",
        deterioration_threshold: float = 2.0,
        use_sofa_score_label: bool = True,
        dynamic_medians: Optional[Dict[str, float]] = None,
        drop_empty: bool = False,
    ) -> None:
        self.id_col = id_col
        self.static_features = static_features
        self.dynamic_features = dynamic_features
        self.dynamic_optional = dynamic_optional or []
        self.include_optional = include_optional
        self.include_mask_features = include_mask_features
        self.task_mode = task_mode
        self.target_type = target_type
        self.deterioration_threshold = deterioration_threshold
        self.use_sofa_score_label = use_sofa_score_label
        self.dynamic_medians = dynamic_medians or {}
        self.drop_empty = drop_empty

        self.meta = self._build_meta(day_min, day_max)

        # 统一 ID 类型
        static_df = static_df.copy()
        dynamic_df = dynamic_df.copy()
        static_df[id_col] = static_df[id_col].astype(str)
        dynamic_df[id_col] = dynamic_df[id_col].astype(str)

        # 保留静态与动态共有的患者
        ids_static = set(static_df[id_col])
        ids_dynamic = set(dynamic_df[id_col])
        self.patient_ids = sorted(list(ids_static & ids_dynamic))

        static_df = static_df[static_df[id_col].isin(self.patient_ids)].copy()
        dynamic_df = dynamic_df[dynamic_df[id_col].isin(self.patient_ids)].copy()

        # 缺失 mask 列
        mask_cols = [f"{c}_mask" for c in self.dynamic_features + self.dynamic_optional]
        self.mask_cols = [c for c in mask_cols if c in dynamic_df.columns]

        # 构建动态索引：每个患者 day -> 行
        self.dynamic_map = self._build_dynamic_map(dynamic_df)

        # 构建静态矩阵
        self.static_map = static_df.set_index(id_col)[static_features]

        # 预构建序列，加速训练
        self.x_seq = []
        self.static_seq = []
        self.y_seq = []
        self.seq_mask = []
        self.prev_mask_seq = []
        self.prev_total_seq = []
        self._build_sequences()

    def _build_meta(self, day_min: int, day_max: int) -> SequenceMeta:
        days = _build_day_index(day_min, day_max)
        return SequenceMeta(
            days=days,
            input_days=days[:-1],
            target_days=days[1:],
        )

    def _build_dynamic_map(self, dynamic_df: pd.DataFrame) -> Dict[str, Dict[int, pd.Series]]:
        day_map = {}
        for pid, group in dynamic_df.groupby(self.id_col):
            group = group.sort_values("day")
            mapping = {int(row["day"]): row for _, row in group.iterrows()}
            day_map[str(pid)] = mapping
        return day_map

    def _fill_feature_value(self, feature: str) -> float:
        # 若无中位数，默认填 0
        return float(self.dynamic_medians.get(feature, 0.0))

    def _get_feature_values(self, row: Optional[pd.Series], features: List[str]) -> List[float]:
        values = []
        for col in features:
            if row is None or col not in row:
                values.append(self._fill_feature_value(col))
            else:
                val = row[col]
                if pd.isna(val):
                    values.append(self._fill_feature_value(col))
                else:
                    values.append(float(val))
        return values

    def _get_mask_values(self, row: Optional[pd.Series], features: List[str]) -> List[float]:
        masks = []
        for col in features:
            mask_col = f"{col}_mask"
            if row is None or mask_col not in row:
                masks.append(0.0)
            else:
                val = row[mask_col]
                masks.append(float(val) if not pd.isna(val) else 0.0)
        return masks

    def _get_total(self, row: Optional[pd.Series]) -> float:
        if row is None:
            return 0.0
        if self.use_sofa_score_label and "sofa_score" in row:
            val = row.get("sofa_score")
            return float(val) if not pd.isna(val) else 0.0
        # 若没有 sofa_score，则由 6 个器官评分求和
        vals = []
        for col in ORGAN_COLS:
            if col in row:
                vals.append(row[col])
        total = np.nansum(vals) if vals else 0.0
        return float(total)

    def _get_organs(self, row: Optional[pd.Series]) -> List[float]:
        vals = []
        for col in ORGAN_COLS:
            if row is None or col not in row:
                vals.append(0.0)
            else:
                val = row[col]
                vals.append(float(val) if not pd.isna(val) else 0.0)
        return vals

    def _get_sofa_score_mask(self, row: Optional[pd.Series]) -> int:
        if row is None or "sofa_score_mask" not in row:
            return 0
        return int(row["sofa_score_mask"])

    def _get_day_mask_sum(self, row: Optional[pd.Series]) -> float:
        if row is None:
            return 0.0
        masks = [row.get(f"{col}_mask", 0) for col in ORGAN_COLS]
        masks = [0.0 if pd.isna(v) else float(v) for v in masks]
        return float(np.sum(masks))

    def _build_sequences(self) -> None:
        dyn_features = self.dynamic_features + (self.dynamic_optional if self.include_optional else [])

        for pid in self.patient_ids:
            static_vals = self.static_map.loc[pid].values.astype(np.float32)

            x_seq = []
            y_seq = []
            seq_mask = []
            prev_mask_seq = []
            prev_total_seq = []

            day_map = self.dynamic_map.get(pid, {})

            for in_day, tgt_day in zip(self.meta.input_days, self.meta.target_days):
                in_row = _get_row_by_day(day_map, in_day)
                tgt_row = _get_row_by_day(day_map, tgt_day)

                x_vals = self._get_feature_values(in_row, dyn_features)
                if self.include_mask_features:
                    x_vals.extend(self._get_mask_values(in_row, dyn_features))
                x_seq.append(x_vals)

                prev_mask_sum = self._get_day_mask_sum(in_row)
                tgt_mask_sum = self._get_day_mask_sum(tgt_row)

                # prev_total must respect sofa_score_mask when use_sofa_score_label
                if self.use_sofa_score_label:
                    prev_sofa_mask = self._get_sofa_score_mask(in_row)
                    if prev_sofa_mask == 1:
                        prev_total = self._get_total(in_row)
                    else:
                        prev_total = np.nan
                else:
                    prev_total = self._get_total(in_row)
                prev_total_seq.append([prev_total])

                if self.task_mode == "deterioration":
                    tgt_total = self._get_total(tgt_row)
                    label = 1.0 if (tgt_total - prev_total) >= self.deterioration_threshold else 0.0
                    y_seq.append([label])
                    # 需要前一天和目标日都有原始有效的 sofa_score
                    prev_sofa_mask = self._get_sofa_score_mask(in_row)
                    tgt_sofa_mask = self._get_sofa_score_mask(tgt_row)
                    step_mask = 1.0 if (prev_sofa_mask == 1 and tgt_sofa_mask == 1) else 0.0
                else:
                    if self.target_type == "total":
                        y_seq.append([self._get_total(tgt_row)])
                    else:
                        y_seq.append(self._get_organs(tgt_row))
                    # 对于回归：step_mask 仅当 sofa_score_mask == 1 时为 1.0
                    if self.use_sofa_score_label:
                        tgt_sofa_mask = self._get_sofa_score_mask(tgt_row)
                        step_mask = 1.0 if tgt_sofa_mask == 1 else 0.0
                    else:
                        step_mask = 1.0 if tgt_mask_sum > 0 else 0.0

                seq_mask.append(step_mask)
                # prev_mask_seq should also follow sofa_score_mask when use_sofa_score_label
                if self.use_sofa_score_label:
                    prev_sofa_mask = self._get_sofa_score_mask(in_row)
                    prev_mask_seq.append(1.0 if prev_sofa_mask == 1 else 0.0)
                else:
                    prev_mask_seq.append(1.0 if prev_mask_sum > 0 else 0.0)

            if self.drop_empty and np.sum(seq_mask) == 0:
                continue

            self.x_seq.append(np.array(x_seq, dtype=np.float32))
            self.static_seq.append(static_vals.astype(np.float32))
            self.y_seq.append(np.array(y_seq, dtype=np.float32))
            self.seq_mask.append(np.array(seq_mask, dtype=np.float32))
            self.prev_mask_seq.append(np.array(prev_mask_seq, dtype=np.float32))
            self.prev_total_seq.append(np.array(prev_total_seq, dtype=np.float32))

        self.x_seq = np.array(self.x_seq, dtype=np.float32)
        self.static_seq = np.array(self.static_seq, dtype=np.float32)
        self.y_seq = np.array(self.y_seq, dtype=np.float32)
        self.seq_mask = np.array(self.seq_mask, dtype=np.float32)
        self.prev_mask_seq = np.array(self.prev_mask_seq, dtype=np.float32)
        self.prev_total_seq = np.array(self.prev_total_seq, dtype=np.float32)

        # 添加健全性检查
        valid_steps = self.seq_mask.sum()
        print(f"Valid target steps in dataset: {valid_steps}")
        for i, y in enumerate(self.y_seq):
            mask = self.seq_mask[i]
            valid_y = y[mask == 1.0]
            assert not np.any(np.isnan(valid_y)), f"NaN labels in valid targets for patient {i}"

        # 检查 prev_total 的有限值数量
        finite_prev = np.isfinite(self.prev_total_seq).sum()
        print(f"Finite prev_total entries (originally observed previous-day sofa_score): {finite_prev}")

    def __len__(self) -> int:
        return len(self.x_seq)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.x_seq[idx]),
            torch.tensor(self.static_seq[idx]),
            torch.tensor(self.y_seq[idx]),
            torch.tensor(self.seq_mask[idx]),
            torch.tensor(self.prev_mask_seq[idx]),
            torch.tensor(self.prev_total_seq[idx]),
            self.patient_ids[idx],
        )