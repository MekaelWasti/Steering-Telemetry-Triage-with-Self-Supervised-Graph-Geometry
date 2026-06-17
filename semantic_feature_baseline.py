"""Semantic process-feature baseline for ACME process telemetry.

This is a label-free baseline that enriches numeric process counters with
process identity, command-line text, file/path text, and a derived ancestor path.
It intentionally does not use graph labels or red-team labels during fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from hetero_graph_builder import DEFAULT_PROCESS_FEATURES


TOKEN_SPLIT_RE = re.compile(r"[^a-zA-Z0-9_.$:-]+")


@dataclass
class SemanticFeatureConfig:
    numeric_cols: tuple[str, ...] = DEFAULT_PROCESS_FEATURES
    max_text_features: int = 20_000
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: tuple[int, int] = (1, 2)
    ancestor_depth: int = 5
    rare_file_max_degree: int = 20
    include_numeric: bool = True
    include_process_name: bool = True
    include_args: bool = True
    include_process_path: bool = True
    include_file: bool = True
    include_ancestor_path: bool = True


class SemanticProcessFeatureBuilder:
    """Build numeric + TF-IDF process features aligned to process ids."""

    def __init__(self, config: SemanticFeatureConfig | None = None):
        self.config = config or SemanticFeatureConfig()
        self.numeric_scaler = StandardScaler(with_mean=False)
        self.text_vectorizer = TfidfVectorizer(
            lowercase=True,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            max_features=self.config.max_text_features,
            ngram_range=self.config.ngram_range,
            token_pattern=r"(?u)\b[^\s]+\b",
            sublinear_tf=True,
            norm="l2",
        )
        self.numeric_columns_: list[str] = []
        self.feature_names_: list[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame,
        process_ids: Iterable[str] | None = None,
    ) -> sparse.csr_matrix:
        ordered = self._align(df, process_ids)
        parts = []
        names = []

        if self.config.include_numeric:
            numeric = self._numeric_matrix(ordered)
            if numeric.shape[1]:
                parts.append(self.numeric_scaler.fit_transform(numeric))
                names.extend([f"num:{col}" for col in self.numeric_columns_])

        text_docs = self._text_documents(ordered)
        text_matrix = self.text_vectorizer.fit_transform(text_docs)
        parts.append(text_matrix)
        names.extend([f"text:{name}" for name in self.text_vectorizer.get_feature_names_out()])

        matrix = sparse.hstack(parts, format="csr") if len(parts) > 1 else parts[0].tocsr()
        self.feature_names_ = names
        return matrix

    def transform(
        self,
        df: pd.DataFrame,
        process_ids: Iterable[str] | None = None,
    ) -> sparse.csr_matrix:
        ordered = self._align(df, process_ids)
        parts = []

        if self.config.include_numeric and self.numeric_columns_:
            numeric = self._numeric_matrix(ordered, fit=False)
            parts.append(self.numeric_scaler.transform(numeric))

        text_docs = self._text_documents(ordered)
        parts.append(self.text_vectorizer.transform(text_docs))
        return sparse.hstack(parts, format="csr") if len(parts) > 1 else parts[0].tocsr()

    def _align(self, df: pd.DataFrame, process_ids: Iterable[str] | None) -> pd.DataFrame:
        if "pid_hash" not in df.columns:
            raise ValueError("df must contain pid_hash")
        work = df[df["pid_hash"].notna()].drop_duplicates("pid_hash", keep="first").copy()
        work["pid_hash"] = work["pid_hash"].astype(str)
        if process_ids is None:
            return work.reset_index(drop=True)

        ordered_ids = pd.Index([str(pid) for pid in process_ids], name="pid_hash")
        aligned = work.set_index("pid_hash").reindex(ordered_ids).reset_index()
        return aligned

    def _numeric_matrix(self, df: pd.DataFrame, fit: bool = True) -> sparse.csr_matrix:
        if fit:
            self.numeric_columns_ = [col for col in self.config.numeric_cols if col in df.columns]
        if not self.numeric_columns_:
            return sparse.csr_matrix((len(df), 0), dtype=np.float32)

        arr = df[self.numeric_columns_].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        arr[~np.isfinite(arr)] = 0.0
        arr = np.sign(arr) * np.log1p(np.abs(arr))
        return sparse.csr_matrix(arr)

    def _text_documents(self, df: pd.DataFrame) -> list[str]:
        ancestor_paths = self._ancestor_paths(df) if self.config.include_ancestor_path else {}
        rare_files = self._rare_files(df) if self.config.include_file and "filename" in df.columns else set()

        docs: list[str] = []
        for _, row in df.iterrows():
            tokens: list[str] = []

            if self.config.include_process_name:
                tokens.extend(self._field_tokens("proc", row.get("process_name")))

            if self.config.include_args:
                tokens.extend(self._field_tokens("arg", row.get("args")))

            if self.config.include_process_path:
                tokens.extend(self._field_tokens("path", row.get("process_path")))

            if self.config.include_file:
                filename = self._clean(row.get("filename"))
                if filename:
                    tokens.extend(self._field_tokens("file", filename))
                    if filename in rare_files:
                        tokens.append(f"rare_file={filename}")

            if self.config.include_ancestor_path:
                pid = self._clean(row.get("pid_hash"))
                ancestor_path = ancestor_paths.get(pid)
                if ancestor_path:
                    tokens.append(f"ancestor_path={ancestor_path}")
                    for ancestor in ancestor_path.split(">"):
                        tokens.append(f"ancestor={ancestor}")

            docs.append(" ".join(tokens))

        return docs

    def _ancestor_paths(self, df: pd.DataFrame) -> dict[str, str]:
        if "parent_pid_hash" not in df.columns or "process_name" not in df.columns:
            return {}

        pid_to_parent = {}
        pid_to_name = {}
        for _, row in df.iterrows():
            pid = self._clean(row.get("pid_hash"))
            if not pid:
                continue
            parent = self._clean(row.get("parent_pid_hash"))
            name = self._normalize_token(row.get("process_name")) or "unknown"
            pid_to_parent[pid] = parent
            pid_to_name[pid] = name

        paths: dict[str, str] = {}
        for pid in pid_to_name:
            lineage = []
            current = pid
            seen = set()
            for _ in range(self.config.ancestor_depth):
                if current in seen:
                    break
                seen.add(current)
                name = pid_to_name.get(current)
                if name:
                    lineage.append(name)
                parent = pid_to_parent.get(current)
                if not parent or parent not in pid_to_name:
                    break
                current = parent
            paths[pid] = ">".join(reversed(lineage))
        return paths

    def _rare_files(self, df: pd.DataFrame) -> set[str]:
        files = df["filename"].map(self._clean)
        counts = files.value_counts(dropna=True)
        return set(counts[(counts >= 2) & (counts <= self.config.rare_file_max_degree)].index)

    def _field_tokens(self, prefix: str, value: object) -> list[str]:
        cleaned = self._clean(value)
        if not cleaned:
            return []
        return [
            f"{prefix}={self._normalize_token(part)}"
            for part in TOKEN_SPLIT_RE.split(cleaned)
            if self._normalize_token(part)
        ]

    def _clean(self, value: object) -> str | None:
        if pd.isna(value):
            return None
        cleaned = str(value).strip().lower()
        return cleaned or None

    def _normalize_token(self, value: object) -> str | None:
        cleaned = self._clean(value)
        if not cleaned:
            return None
        return cleaned.replace("\\\\", "/")
