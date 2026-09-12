# coding: utf-8
"""特征与字段配置。"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import numpy as np

# ====== 核心字段配置 ======

MIMIC_ID_COL = "stay_id"
SAHZU_ID_COL = "PATIENT_ID"

DYNAMIC_INDEX_COLS = ["day"]

# 动态输入的基础特征（与模型输入顺序一致）
DYNAMIC_BASE_COLS = [
    "temperature_max",
    "heart_rate_max",
    "resp_rate_max",
    "sofa_respiration",
    "sofa_coagulation",
    "sofa_liver",
    "sofa_cardiovascular",
    "sofa_cns",
    "sofa_renal",
]

# 动态可选特征（默认不入模，但可通过开关加入）
DYNAMIC_OPTIONAL_COLS = ["sofa_score"]

# BMI 相关配置
BMI_COL = "bmi"
HEIGHT_COL = "height"
WEIGHT_COL = "weight"
BMI_HEIGHT_RANGE = (0.5, 2.5)  # 合理身高范围（米）
DEFAULT_USE_BMI = True

# 静态中需要剔除的字段（索引/时间/结局/窗口）
STATIC_EXCLUDE_COLS = {
    "subject_id",
    "admittime",
    "dischtime",
    "icu_intime",
    "icu_outtime",
    "prediction_window_start",
    "prediction_window_end",
    "hosp_los_hours",
    "icu_los_hours",
    "died_in_hosp",
    "sofa",
}

# 静态里与动态重复的汇总生命体征，默认剔除
STATIC_VITALS_COLS = {
    "temperature_max",
    "heart_rate_max",
    "resp_rate_max",
}

# SAHZU 静态表中需要移除的占位列
SAHZU_STATIC_DROP_COLS = {"-", "Unnamed: 29"}


# ====== 工具函数 ======

def normalize_sahzu_id(series: pd.Series) -> pd.Series:
    """SAHZU 病例号去前导 0，统一为字符串。"""
    return series.astype(str).str.strip().str.lstrip("0")


def normalize_gender(series: pd.Series) -> pd.Series:
    """统一性别字段（M/F 或 男/女）为 1/0，其它保持为 NaN。"""
    if series is None:
        return series
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "m": 1,
        "male": 1,
        "男": 1,
        "f": 0,
        "female": 0,
        "女": 0,
    }
    mapped = s.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric)


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """将指定列尽量转换为数值型，无法转换的设为 NaN。"""
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_bmi(df: pd.DataFrame, height_unit: str = "cm") -> pd.Series:
    """根据身高/体重计算 BMI。若异常或缺失则返回 NaN。"""
    if HEIGHT_COL not in df.columns or WEIGHT_COL not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)

    height = pd.to_numeric(df[HEIGHT_COL], errors="coerce")
    weight = pd.to_numeric(df[WEIGHT_COL], errors="coerce")

    # 单位转换
    if height_unit == "cm":
        height_m = height / 100.0
    else:
        height_m = height

    # 异常值处理（避免不合理的身高/体重）
    height_m = height_m.where(
        (height_m > 0) & (height_m >= BMI_HEIGHT_RANGE[0]) & (height_m <= BMI_HEIGHT_RANGE[1])
    )
    weight = weight.where(weight > 0)

    bmi = weight / (height_m ** 2)
    return bmi
