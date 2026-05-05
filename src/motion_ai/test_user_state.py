from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestUserEntry:
    name: str
    source: Path
    description: str
    terminology: str = ""
    project_type: str = "自动识别"
    test_type: str = "未标记"
    result: object | None = None
    status: str = "待分析"
    last_output_dir: Path | None = None
    last_analyzed_at: str | None = None
    baseline_name: str | None = None


def format_test_user_display(index: int, test_user: TestUserEntry) -> str:
    suffix = test_user.status
    if test_user.last_analyzed_at:
        suffix = f"{suffix} · {test_user.last_analyzed_at}"
    test_type = (test_user.test_type or "未标记").strip()
    return f"{index + 1}. {test_user.name} [{test_type} | {suffix}]"


def format_test_user_subject(test_user: TestUserEntry) -> str:
    subject_text = f"当前对象：{test_user.name} · 类型：{test_user.test_type or '未标记'} · 状态：{test_user.status}"
    if test_user.baseline_name:
        subject_text += f" · 模板：{test_user.baseline_name}"
    return subject_text


__all__ = ["TestUserEntry", "format_test_user_display", "format_test_user_subject"]
