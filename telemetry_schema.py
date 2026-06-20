"""Canonical schema adapter for broad process telemetry dataframes.

The current prototype modules expect ACME-style canonical column names such as
``pid_hash``, ``parent_pid_hash``, ``process_name``, and ``args``.  Real telemetry
tables rarely agree on names, so this module provides a small, inspectable
adapter:

    raw dataframe -> inferred/explicit schema mapping -> canonical dataframe

The mapping object is also the future LLM-assisted integration point.  A model
can propose the same ``source_by_canonical`` dictionary from column names,
sample values, and analyst guidance; the rest of the prototype does not need to
know whether the mapping came from heuristics or from an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Mapping, Sequence

import pandas as pd


CANONICAL_COLUMNS = (
    "pid_hash",
    "parent_pid_hash",
    "process_name",
    "args",
    "process_path",
    "filename",
    "hostname",
    "user_name",
    "process_started",
    "first_seen",
    "last_seen",
    "red_team",
    "bad_user",
)


REQUIRED_CANONICAL_COLUMNS = ("pid_hash",)


COLUMN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "pid_hash": (
        "pid_hash",
        "process_guid",
        "processguid",
        "process_id",
        "processid",
        "event_id",
        "eventid",
        "id",
        "uuid",
        "process_uuid",
    ),
    "parent_pid_hash": (
        "parent_pid_hash",
        "parent_process_guid",
        "parentprocessguid",
        "parent_process_id",
        "parentprocessid",
        "parent_pid",
        "parentpid",
        "ppid",
    ),
    "process_name": (
        "process_name",
        "processname",
        "image_name",
        "imagename",
        "process",
        "executable",
        "exe",
        "binary",
        "program",
    ),
    "args": (
        "args",
        "command_line",
        "commandline",
        "cmdline",
        "cmd_line",
        "process_command_line",
        "processcommandline",
        "command",
        "arguments",
        "process_args",
    ),
    "process_path": (
        "process_path",
        "processpath",
        "image",
        "image_path",
        "imagepath",
        "executable_path",
        "filepath",
        "file_path",
        "process_file_path",
        "process_filepath",
    ),
    "filename": (
        "filename",
        "file_name",
        "target_filename",
        "targetfilename",
        "object_name",
        "objectname",
        "target_file",
        "file",
        "resource",
    ),
    "hostname": (
        "hostname",
        "host",
        "computer",
        "computer_name",
        "computername",
        "device",
        "device_name",
        "devicename",
        "endpoint",
        "agent_hostname",
    ),
    "user_name": (
        "user_name",
        "username",
        "user",
        "account",
        "account_name",
        "accountname",
        "subject_user_name",
        "target_user_name",
        "principal",
    ),
    "process_started": (
        "process_started",
        "process_start_time",
        "processstarttime",
        "timestamp",
        "event_time",
        "eventtime",
        "time",
        "@timestamp",
        "datetime",
        "created_at",
    ),
    "first_seen": (
        "first_seen",
        "firstseen",
        "first_time",
        "firsttime",
        "start_time",
        "starttime",
    ),
    "last_seen": (
        "last_seen",
        "lastseen",
        "last_time",
        "lasttime",
        "end_time",
        "endtime",
    ),
    "red_team": (
        "red_team",
        "redteam",
        "malicious",
        "is_malicious",
        "label",
        "attack",
    ),
    "bad_user": (
        "bad_user",
        "baduser",
        "bad_user_name",
        "malicious_user",
        "compromised_user",
    ),
}


@dataclass(frozen=True)
class TelemetrySchema:
    """Mapping from canonical column name to source dataframe column."""

    source_by_canonical: dict[str, str] = field(default_factory=dict)
    generated_columns: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def source_for(self, canonical: str) -> str | None:
        return self.source_by_canonical.get(canonical)


def infer_telemetry_schema(
    df: pd.DataFrame,
    overrides: Mapping[str, str] | None = None,
) -> TelemetrySchema:
    """Infer a canonical schema mapping from dataframe columns.

    ``overrides`` should map canonical names to source column names.  This is the
    same structure an LLM-assisted mapper should emit later.
    """

    overrides = dict(overrides or {})
    normalized_to_source = {_normalize_name(col): col for col in df.columns}
    source_by_canonical: dict[str, str] = {}
    notes: list[str] = []

    for canonical, source in overrides.items():
        if canonical not in CANONICAL_COLUMNS:
            notes.append(f"Ignored unknown canonical override: {canonical}")
            continue
        if source not in df.columns:
            notes.append(f"Override for {canonical} points to missing source column: {source}")
            continue
        source_by_canonical[canonical] = source

    for canonical in CANONICAL_COLUMNS:
        if canonical in source_by_canonical:
            continue
        for candidate in COLUMN_SYNONYMS.get(canonical, (canonical,)):
            source = normalized_to_source.get(_normalize_name(candidate))
            if source is not None:
                source_by_canonical[canonical] = source
                break

    generated: dict[str, str] = {}
    if "pid_hash" not in source_by_canonical:
        generated["pid_hash"] = "generated row id because no process/event id column was inferred"
        notes.append("Generated pid_hash from row position; parent-child linkage will be limited.")

    if "process_name" not in source_by_canonical and "process_path" in source_by_canonical:
        generated["process_name"] = "derived basename from process_path"
        notes.append("Derived process_name from process_path basename.")

    return TelemetrySchema(
        source_by_canonical=source_by_canonical,
        generated_columns=generated,
        notes=tuple(notes),
    )


def adapt_telemetry_dataframe(
    df: pd.DataFrame,
    schema: TelemetrySchema | None = None,
    overrides: Mapping[str, str] | None = None,
    keep_unmapped: bool = False,
) -> tuple[pd.DataFrame, TelemetrySchema, dict[str, object]]:
    """Return a dataframe using canonical process telemetry column names."""

    schema = schema or infer_telemetry_schema(df, overrides=overrides)
    adapted = pd.DataFrame(index=df.index)

    for canonical, source in schema.source_by_canonical.items():
        adapted[canonical] = df[source]

    if "pid_hash" in schema.generated_columns and "pid_hash" not in adapted.columns:
        adapted["pid_hash"] = [f"row-{idx}" for idx in range(len(df))]

    if "process_name" in schema.generated_columns and "process_name" not in adapted.columns:
        adapted["process_name"] = adapted["process_path"].map(_path_basename)

    if "first_seen" not in adapted.columns and "process_started" in adapted.columns:
        adapted["first_seen"] = adapted["process_started"]
    if "last_seen" not in adapted.columns and "process_started" in adapted.columns:
        adapted["last_seen"] = adapted["process_started"]

    if keep_unmapped:
        mapped_sources = set(schema.source_by_canonical.values())
        for col in df.columns:
            if col not in mapped_sources and col not in adapted.columns:
                adapted[col] = df[col]

    report = schema_report(df, adapted, schema)
    return adapted.reset_index(drop=True), schema, report


def schema_report(
    raw_df: pd.DataFrame,
    adapted_df: pd.DataFrame,
    schema: TelemetrySchema,
) -> dict[str, object]:
    """Summarize what was mapped, generated, and missing."""

    mapped = dict(schema.source_by_canonical)
    missing_required = [
        col
        for col in REQUIRED_CANONICAL_COLUMNS
        if col not in adapted_df.columns and col not in schema.generated_columns
    ]
    missing_recommended = [
        col
        for col in ("process_name", "args", "process_path", "hostname", "user_name", "process_started")
        if col not in adapted_df.columns
    ]

    return {
        "raw_shape": tuple(raw_df.shape),
        "adapted_shape": tuple(adapted_df.shape),
        "mapped_columns": mapped,
        "generated_columns": dict(schema.generated_columns),
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
        "notes": list(schema.notes),
    }


def llm_schema_prompt(df: pd.DataFrame, n_sample_values: int = 3) -> str:
    """Build a compact prompt for a future LLM-assisted schema mapper."""

    samples = []
    for col in df.columns:
        values = (
            df[col]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(n_sample_values)
            .tolist()
        )
        samples.append(f"- {col}: examples={values}")

    canonical = ", ".join(CANONICAL_COLUMNS)
    sample_text = "\n".join(samples)
    return (
        "Map the dataframe columns to these canonical telemetry fields:\n"
        f"{canonical}\n\n"
        "Return JSON as {\"source_by_canonical\": {canonical_name: source_column}}.\n"
        "Use only source columns that exist. Leave uncertain fields unmapped.\n\n"
        f"Columns and sample values:\n{sample_text}"
    )


def _normalize_name(name: object) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _path_basename(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace("\\", "/")
    if not text:
        return None
    return PurePath(text).name or text

