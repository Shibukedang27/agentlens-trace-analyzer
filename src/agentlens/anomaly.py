"""Optional second signal: IsolationForest anomaly score on run-level features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES: tuple[str, ...] = (
    "total_tokens",
    "cost",
    "n_llm_calls",
    "n_tool_calls",
    "error_rate",
    "duration_ms",
    "max_tool_repeat",
    "input_growth_ratio",
    "output_ratio",
)

MIN_RUNS = 20


def feature_matrix(run_df: pd.DataFrame) -> pd.DataFrame:
    df = run_df.copy()
    df["output_ratio"] = df["output_tokens"] / df["total_tokens"].where(df["total_tokens"] > 0, 1)
    feats = df[list(FEATURES)].astype(float)
    # log-scale the heavy-tailed magnitudes so a few huge runs don't dominate
    for col in ("total_tokens", "cost", "duration_ms"):
        feats[col] = np.log1p(feats[col].clip(lower=0))
    return feats


def anomaly_scores(
    run_df: pd.DataFrame, contamination: float = 0.1, seed: int = 0
) -> tuple[pd.Series, pd.Series]:
    """Return ``(score, is_outlier)``; score is in [0, 1], higher = more anomalous.

    Features are scaled per agent (z-score inside each agent) because agents have
    very different "normal" shapes. With fewer than ``MIN_RUNS`` runs the model is
    not fitted and all scores are 0.
    """
    index = run_df.index
    if len(run_df) < MIN_RUNS:
        return pd.Series(0.0, index=index), pd.Series(False, index=index)
    feats = feature_matrix(run_df)
    grouped = feats.groupby(run_df["agent"])
    std = grouped.transform("std").replace(0, 1).fillna(1)
    scaled = ((feats - grouped.transform("mean")) / std).fillna(0.0)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    model.fit(scaled.to_numpy())
    raw = -model.score_samples(scaled.to_numpy())
    lo, hi = float(raw.min()), float(raw.max())
    score = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
    flags = model.predict(scaled.to_numpy()) == -1
    return pd.Series(score, index=index), pd.Series(flags, index=index)
