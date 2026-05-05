"""
模板动作完整描述库
包含起势、过渡、定势、收势等详细阶段描述
"""

from __future__ import annotations

# 预定义的完整模板描述
TEMPLATE_DESCRIPTIONS = {
    "八段锦-起势": {
        "name": "八段锦-起势",
        "category": "健身气功",
        "full_description": "八段锦起势动作，身体自然站立，两脚平行分开与肩同宽，双臂自然下垂，目视前方",
        "phases": [
            {
                "name": "预备姿势",
                "description": "身体自然站立，两脚平行分开与肩同宽，膝关节微屈，含胸拔背，虚灵顶劲",
                "duration_range": [2, 4],
                "key_points": ["站姿稳定", "重心居中", "呼吸自然"]
            },
            {
                "name": "起势",
                "description": "两臂慢慢向前平举至与肩同高，掌心向下，两臂平行",
                "duration_range": [3, 5],
                "key_points": ["双臂同步", "速度均匀", "肩部放松"]
            },
            {
                "name": "定势",
                "description": "两臂保持平举，停顿1-2秒，感受肩臂的力量",
                "duration_range": [1, 2],
                "key_points": ["保持稳定", "呼吸平稳", "意守丹田"]
            },
            {
                "name": "收势",
                "description": "两臂慢慢下落至体侧，恢复预备姿势",
                "duration_range": [2, 4],
                "key_points": ["动作缓慢", "保持平衡", "气沉丹田"]
            }
        ],
        "common_errors": [
            "双臂抬起不同步",
            "肩部耸起紧张",
            "重心不稳前倾或后仰",
            "呼吸不自然憋气"
        ],
        "improvement_tips": [
            "初学者：先关注站姿稳定，双脚与肩同宽，膝盖微屈",
            "进阶者：注意双臂同步抬起，保持肩部放松",
            "熟练者：配合呼吸，起势吸气，落势呼气，意念引导动作"
        ]
    },

    "太极拳-起势": {
        "name": "太极拳-起势",
        "category": "太极拳",
        "full_description": "太极拳起势，立身中正，虚领顶劲，两臂徐徐上提，气沉丹田",
        "phases": [
            {
                "name": "无极桩",
                "description": "两脚并拢站立，身体中正，精神内敛，呼吸自然",
                "duration_range": [2, 3],
                "key_points": ["立身中正", "虚领顶劲", "松静自然"]
            },
            {
                "name": "开步",
                "description": "左脚向左轻轻开步，与肩同宽，重心均匀分布",
                "duration_range": [2, 3],
                "key_points": ["轻起轻落", "重心稳定", "两脚平行"]
            },
            {
                "name": "起势上提",
                "description": "两臂慢慢向前平举，与肩同高，掌心向下",
                "duration_range": [3, 5],
                "key_points": ["沉肩坠肘", "松腰松胯", "呼吸配合"]
            },
            {
                "name": "按掌下落",
                "description": "两掌轻轻下按至腹前，气沉丹田",
                "duration_range": [3, 5],
                "key_points": ["掌心向下", "松沉自然", "意守丹田"]
            }
        ],
        "common_errors": [
            "身体僵硬不放松",
            "动作过快失去太极韵味",
            "呼吸与动作不协调",
            "重心不稳左右摇晃"
        ],
        "improvement_tips": [
            "初学者：先练习站桩，培养身体的松静状态",
            "进阶者：注意动作的连贯性，一动无有不动",
            "熟练者：体会内劲的运行，以意导气，以气运身"
        ]
    },

    "五禽戏-虎戏": {
        "name": "五禽戏-虎戏",
        "category": "健身气功",
        "full_description": "五禽戏虎戏，模仿虎的威猛姿态，两掌变虎爪，目露精光，气势威猛",
        "phases": [
            {
                "name": "虎举",
                "description": "两臂上举，掌变虎爪，目视前方，展现虎的威猛",
                "duration_range": [3, 5],
                "key_points": ["虎爪有力", "目光有神", "气势威猛"]
            },
            {
                "name": "虎扑",
                "description": "身体前倾，两爪向前扑出，如虎扑食",
                "duration_range": [2, 4],
                "key_points": ["动作迅猛", "力达爪尖", "腰背发力"]
            },
            {
                "name": "虎坐",
                "description": "身体下蹲，如虎坐山，稳如泰山",
                "duration_range": [2, 3],
                "key_points": ["重心下沉", "保持平衡", "气沉丹田"]
            },
            {
                "name": "收势",
                "description": "缓慢起身，两臂下落，恢复自然站立",
                "duration_range": [2, 4],
                "key_points": ["动作缓慢", "呼吸平稳", "精神内敛"]
            }
        ],
        "common_errors": [
            "虎爪不够有力",
            "动作缺乏威猛气势",
            "下蹲时重心不稳",
            "呼吸急促不自然"
        ],
        "improvement_tips": [
            "初学者：先练习虎爪的手型，五指分开用力",
            "进阶者：体会虎的威猛气势，目光要有神",
            "熟练者：将力量与柔韧结合，刚柔并济"
        ]
    }
}


def get_template_description(template_name: str) -> dict | None:
    """获取模板的完整描述"""
    # 精确匹配
    if template_name in TEMPLATE_DESCRIPTIONS:
        return TEMPLATE_DESCRIPTIONS[template_name]

    # 模糊匹配
    template_key = "".join(template_name.split()).lower()
    for key, desc in TEMPLATE_DESCRIPTIONS.items():
        if "".join(key.split()).lower() == template_key:
            return desc

    return None


def list_all_templates() -> list[dict]:
    """列出所有预定义模板"""
    return [
        {
            "name": desc["name"],
            "category": desc["category"],
            "description": desc["full_description"]
        }
        for desc in TEMPLATE_DESCRIPTIONS.values()
    ]


def get_phase_description(template_name: str, phase_name: str) -> dict | None:
    """获取特定阶段的描述"""
    template = get_template_description(template_name)
    if not template:
        return None

    for phase in template.get("phases", []):
        if phase["name"] == phase_name:
            return phase

    return None


def generate_training_guidance(template_name: str, learner_level: str = "进阶者") -> str:
    """生成训练指导文本"""
    template = get_template_description(template_name)
    if not template:
        return f"未找到模板 {template_name} 的详细描述"

    lines = []
    lines.append(f"【{template['name']}】训练指导")
    lines.append("")
    lines.append(f"动作说明: {template['full_description']}")
    lines.append("")

    lines.append("【动作阶段】")
    for i, phase in enumerate(template["phases"], 1):
        lines.append(f"{i}. {phase['name']}")
        lines.append(f"   {phase['description']}")
        lines.append(f"   要点: {', '.join(phase['key_points'])}")
        lines.append("")

    lines.append("【常见错误】")
    for i, error in enumerate(template["common_errors"], 1):
        lines.append(f"{i}. {error}")
    lines.append("")

    lines.append("【改进建议】")
    for tip in template["improvement_tips"]:
        if learner_level in tip or "所有" in tip:
            lines.append(f"- {tip}")

    return "\n".join(lines)
