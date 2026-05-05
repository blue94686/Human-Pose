from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import TEMPLATE_LIBRARY_FILE


LIBRARY_SCHEMA_VERSION = 1


def load_template_library(path: Path | None = None) -> dict[str, Any]:
    library_path = path or TEMPLATE_LIBRARY_FILE
    if not library_path.exists():
        return _default_library()
    try:
        payload = json.loads(library_path.read_text(encoding="utf-8"))
    except Exception:
        return _default_library()
    if isinstance(payload, list):
        payload = {"version": LIBRARY_SCHEMA_VERSION, "templates": payload}
    if not isinstance(payload, dict):
        return _default_library()
    templates = payload.get("templates")
    if not isinstance(templates, list):
        payload["templates"] = []
    payload.setdefault("version", LIBRARY_SCHEMA_VERSION)
    payload.setdefault("updated_at", "")
    return payload


def list_template_entries(path: Path | None = None) -> list[dict[str, Any]]:
    library = load_template_library(path)
    entries = [item for item in library.get("templates", []) if isinstance(item, dict)]
    return sorted(entries, key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)


def save_template_result(result, template_payload: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    library_path = path or TEMPLATE_LIBRARY_FILE
    library = load_template_library(library_path)
    entries = [item for item in library.get("templates", []) if isinstance(item, dict)]

    template_name = str(template_payload.get("模板名称") or result.rules.template_name or "未命名模板")
    merge_index = _find_merge_index(entries, template_name)
    existing_entry = entries[merge_index] if merge_index is not None else None
    version = int((existing_entry or {}).get("version") or 1)
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry_id = (
        str(existing_entry.get("id"))
        if existing_entry and existing_entry.get("id")
        else f"{template_name}-{now_text}".replace("/", "_")
    )

    phase_snapshots = template_payload.get("阶段快照") or []
    phases = template_payload.get("模板阶段") or []
    summary = template_payload.get("模板摘要") or {}
    previous_artifacts = list((existing_entry or {}).get("previous_artifacts") or [])
    if existing_entry:
        previous_artifacts.append(
            {
                "updated_at": existing_entry.get("updated_at") or existing_entry.get("created_at"),
                "source_video": existing_entry.get("source_video"),
                "baseline_json_path": existing_entry.get("baseline_json_path"),
                "excel_report_path": existing_entry.get("excel_report_path"),
                "compact_data_json_path": existing_entry.get("compact_data_json_path"),
                "summary_txt_path": existing_entry.get("summary_txt_path"),
                "action_guidance_txt_path": existing_entry.get("action_guidance_txt_path"),
            }
        )
    entry = {
        "id": entry_id,
        "template_name": template_name,
        "template_key": _template_key(template_name),
        "version": version,
        "created_at": (existing_entry or {}).get("created_at") or now_text,
        "updated_at": now_text,
        "source_video": result.source,
        "description_text": result.rules.raw_text,
        "baseline_json_path": result.artifacts.get("template_baseline_json"),
        "excel_report_path": result.artifacts.get("excel_report"),
        "compact_data_json_path": result.artifacts.get("compact_data_json"),
        "summary_txt_path": result.artifacts.get("analysis_summary_txt"),
        "action_guidance_txt_path": result.artifacts.get("action_guidance_txt"),
        "comparison_json_path": result.artifacts.get("compact_data_json") or result.artifacts.get("comparison_json"),
        "total_score": summary.get("总分", result.summary.total_score),
        "pose_enabled": result.used_pose_estimator,
        "pose_backend_reason": result.summary.pose_backend_reason,
        "phases_count": len(phases),
        "phases": phases,
        "phase_snapshots": phase_snapshots,
        "merged_count": int((existing_entry or {}).get("merged_count") or 0) + (1 if existing_entry else 0),
        "previous_artifacts": previous_artifacts[-8:],
    }
    if merge_index is None:
        entries.append(entry)
    else:
        entries[merge_index] = entry

    library["version"] = LIBRARY_SCHEMA_VERSION
    library["updated_at"] = now_text
    library["templates"] = entries
    _write_library(library_path, library)
    return entry


def delete_template_entry(entry_id: str, path: Path | None = None) -> bool:
    library_path = path or TEMPLATE_LIBRARY_FILE
    library = load_template_library(library_path)
    entries = [item for item in library.get("templates", []) if isinstance(item, dict)]
    kept_entries = [item for item in entries if item.get("id") != entry_id]
    if len(kept_entries) == len(entries):
        return False
    library["templates"] = kept_entries
    library["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_library(library_path, library)
    return True


def clear_template_library(path: Path | None = None) -> None:
    library_path = path or TEMPLATE_LIBRARY_FILE
    payload = _default_library()
    payload["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_library(library_path, payload)


def get_template_entry(entry_id: str, path: Path | None = None) -> dict[str, Any] | None:
    for entry in list_template_entries(path):
        if entry.get("id") == entry_id:
            return entry
    return None


def _default_library() -> dict[str, Any]:
    return {"version": LIBRARY_SCHEMA_VERSION, "updated_at": "", "templates": []}


def _template_key(template_name: str) -> str:
    return "".join(str(template_name).split()).casefold()


def _find_merge_index(entries: list[dict[str, Any]], template_name: str) -> int | None:
    target_key = _template_key(template_name)
    for index, entry in enumerate(entries):
        entry_key = str(entry.get("template_key") or "")
        if not entry_key:
            entry_key = _template_key(str(entry.get("template_name") or ""))
        if entry_key == target_key:
            return index
    return None


def _write_library(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
