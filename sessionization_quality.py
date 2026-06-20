"""Sessionization quality checks for process telemetry.

These helpers are model-agnostic.  They exist to prevent a session detector from
silently reporting impressive metrics after connected components have merged a
large part of the dataset into one giant session.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SessionQualityThresholds:
    """Default guardrails for connected-component sessionization."""

    max_session_fraction: float = 0.05
    max_session_size: int | None = None
    min_positive_sessions: int | None = None


def session_quality_report(
    df: pd.DataFrame,
    session_ids: Sequence[object],
    label_cols: Sequence[str] = ("red_team", "bad_user"),
) -> dict[str, object]:
    """Summarize component sizes and optional label spread.

    Labels are used only for evaluation.  Missing label columns are skipped.
    """

    if len(df) != len(session_ids):
        raise ValueError("df and session_ids must have the same length")

    sessions = pd.Series(session_ids, index=df.index, name="session_id")
    sizes = sessions.value_counts(dropna=False)
    if sizes.empty:
        sizes = pd.Series([0])

    report: dict[str, object] = {
        "n_processes": int(len(df)),
        "n_sessions": int(sessions.nunique(dropna=False)),
        "singletons": int((sizes == 1).sum()),
        "median_session_size": float(sizes.median()),
        "p95_session_size": float(sizes.quantile(0.95)),
        "max_session_size": int(sizes.max()),
        "max_session_fraction": float(sizes.max() / max(len(df), 1)),
    }

    for label_col in label_cols:
        if label_col not in df.columns:
            continue
        labels = _binary_labels(df[label_col])
        session_labels = pd.DataFrame({"session_id": sessions, "label": labels}).groupby(
            "session_id", dropna=False
        )["label"].max()
        process_rows = int(labels.sum())
        positive_sessions = int(session_labels.sum())
        report[f"{label_col}_process_rows"] = process_rows
        report[f"{label_col}_sessions"] = positive_sessions
        report[f"{label_col}_processes_per_positive_session"] = (
            process_rows / positive_sessions if positive_sessions else np.nan
        )

    return report


def session_quality_issues(
    report: dict[str, object],
    thresholds: SessionQualityThresholds | None = None,
    label_cols: Sequence[str] = ("red_team", "bad_user"),
) -> list[str]:
    """Return human-readable quality issues for a sessionization report."""

    thresholds = thresholds or SessionQualityThresholds()
    issues: list[str] = []

    max_fraction = float(report.get("max_session_fraction", 0.0))
    if max_fraction > thresholds.max_session_fraction:
        issues.append(
            "max_session_fraction "
            f"{max_fraction:.3f} exceeds {thresholds.max_session_fraction:.3f}"
        )

    if thresholds.max_session_size is not None:
        max_size = int(report.get("max_session_size", 0))
        if max_size > thresholds.max_session_size:
            issues.append(
                f"max_session_size {max_size} exceeds {thresholds.max_session_size}"
            )

    if thresholds.min_positive_sessions is not None:
        for label_col in label_cols:
            key = f"{label_col}_sessions"
            if key in report and int(report[key]) < thresholds.min_positive_sessions:
                issues.append(
                    f"{key} {int(report[key])} is below {thresholds.min_positive_sessions}"
                )

    return issues


def _binary_labels(labels: Sequence[object]) -> np.ndarray:
    series = pd.Series(labels)
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0).to_numpy() > 0).astype(np.int64)

    clean = series.astype("string")
    present = clean.notna() & clean.str.strip().ne("") & clean.str.lower().ne("nan")
    return present.fillna(False).to_numpy(dtype=np.int64)

