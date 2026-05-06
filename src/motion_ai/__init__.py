"""动作分析工作台包入口。

避免在包导入阶段立即加载 OpenCV / PySide6 / Ultralytics 等重依赖，
这样 `python app.py --check-env`、打包探测和纯路径配置读取都不会被
`ModuleNotFoundError: cv2` 之类的问题提前拦截。
"""

__all__ = ["ActionAnalyzer", "DescriptionParser"]


def __getattr__(name: str):
    if name == "ActionAnalyzer":
        from .analyzer import ActionAnalyzer

        return ActionAnalyzer
    if name == "DescriptionParser":
        from .description import DescriptionParser

        return DescriptionParser
    raise AttributeError(name)
