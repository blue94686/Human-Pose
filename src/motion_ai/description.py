from __future__ import annotations

import json
from pathlib import Path

from .config import DEFAULT_TEMPLATE_FILE
from .models import RuleSet


class DescriptionParser:
    def __init__(self, template_file: Path | None = None) -> None:
        self.template_file = template_file or DEFAULT_TEMPLATE_FILE
        self.templates = self._load_templates(self.template_file)

    def parse(self, text: str) -> RuleSet:
        raw_text = (text or "").strip()
        matched_template_name, template = self._match_template(raw_text)
        keywords = self._extract_keywords(raw_text)

        rules = RuleSet(
            raw_text=raw_text,
            template_name=matched_template_name,
            expected_tempo=self._detect_tempo(raw_text, template),
            requires_symmetry=self._detect_symmetry(raw_text, template),
            requires_upright_torso=self._detect_upright_torso(raw_text, template),
            requires_hold=self._detect_hold(raw_text, template),
            focus_body_parts=self._detect_focus_parts(raw_text, template),
            thresholds=dict(template.get("thresholds", {})) if template else {},
            keywords=keywords,
            action_category=str(template.get("category") or self._detect_action_category(raw_text, template)),
        )
        return rules

    def _load_templates(self, path: Path) -> dict[str, dict]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _match_template(self, text: str) -> tuple[str | None, dict]:
        if not text:
            return None, {}
        for name, template in self.templates.items():
            keywords = template.get("keywords", [])
            if name in text or any(keyword in text for keyword in keywords):
                return name, template
        return None, {}

    def _detect_tempo(self, text: str, template: dict) -> str:
        if any(token in text for token in ["缓慢", "舒缓", "慢速", "柔和"]):
            return "slow"
        if any(token in text for token in ["快速", "迅速", "爆发", "有力"]):
            return "fast"
        return template.get("tempo", "medium")

    def _detect_symmetry(self, text: str, template: dict) -> bool:
        if any(token in text for token in ["对称", "左右一致", "平衡", "双侧协调"]):
            return True
        return bool(template.get("requires_symmetry", False))

    def _detect_upright_torso(self, text: str, template: dict) -> bool:
        if any(token in text for token in ["躯干直立", "上身直立", "挺胸", "腰背直", "中正"]):
            return True
        return bool(template.get("requires_upright_torso", False))

    def _detect_hold(self, text: str, template: dict) -> bool:
        if any(token in text for token in ["保持", "停顿", "定势", "停留"]):
            return True
        return bool(template.get("requires_hold", False))

    def _detect_focus_parts(self, text: str, template: dict) -> list[str]:
        mapping = {
            "手": "arms",
            "臂": "arms",
            "肩": "arms",
            "躯干": "torso",
            "上身": "torso",
            "腰": "torso",
            "背": "torso",
            "腿": "legs",
            "膝": "legs",
            "步": "legs",
            "重心": "balance",
            "平衡": "balance",
            "稳定": "balance",
            "节奏": "tempo",
            "呼吸": "tempo",
        }
        parts = {value for key, value in mapping.items() if key in text}
        parts.update(template.get("focus_body_parts", []))
        return sorted(parts)

    def _detect_action_category(self, text: str, template: dict) -> str:
        if template.get("category"):
            return str(template["category"])
        if any(token in text for token in ["八段锦", "健身气功", "五禽戏", "易筋经"]):
            return "健身气功"
        if any(token in text for token in ["太极", "云手", "揽雀尾", "单鞭"]):
            return "太极"
        if any(token in text for token in ["武术", "弓步", "马步", "冲拳", "踢腿", "弹腿"]):
            return "武术基本功"
        if any(token in text for token in ["民族舞", "舞蹈", "摆臂", "转身", "节拍"]):
            return "民族舞"
        if any(token in text for token in ["站姿", "站立", "定势"]):
            return "站姿稳定"
        if any(token in text for token in ["抬手", "举臂", "抬臂"]):
            return "上肢动作"
        return "通用动作"

    def _extract_keywords(self, text: str) -> list[str]:
        if not text:
            return []
        vocab = [
            "缓慢",
            "舒缓",
            "快速",
            "对称",
            "左右一致",
            "躯干直立",
            "上身直立",
            "挺胸",
            "保持",
            "停顿",
            "定势",
            "手臂",
            "肩部",
            "躯干",
            "腿部",
            "重心",
            "节奏",
            "八段锦",
            "健身气功",
            "太极",
            "云手",
            "武术",
            "马步",
            "弓步",
            "冲拳",
            "踢腿",
            "民族舞",
            "舞蹈",
        ]
        return [token for token in vocab if token in text]
