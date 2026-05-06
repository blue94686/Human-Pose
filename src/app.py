from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from motion_ai.app_identity import APP_NAME, APP_VERSION
from motion_ai.config import DEFAULT_WEIGHTS, PROJECT_ROOT, YOLO_POSE_MODELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=APP_NAME)
    runtime_group = parser.add_argument_group("运行模式")
    runtime_group.add_argument("--desktop", action="store_true", help="启动桌面三栏应用")
    runtime_group.add_argument("--check-env", action="store_true", help="检查运行环境")
    runtime_group.add_argument("--analyze", help="分析本地视频文件")

    analysis_group = parser.add_argument_group("离线视频分析")
    analysis_group.add_argument("--template", help="标准视频或模板 JSON 文件")
    analysis_group.add_argument("--output", help="分析输出目录，例如 outputs/analysis_20260427_120000")
    analysis_group.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLOv8-Pose 权重路径")
    analysis_group.add_argument("--description", default="", help="动作描述文本")
    analysis_group.add_argument("--mode", choices=["test", "template"], default="test", help="分析模式")
    analysis_group.add_argument("--frame-stride", type=int, default=1, help="逐帧步长，默认 1")
    analysis_group.add_argument("--max-frames", type=int, help="最大处理帧数")
    analysis_group.add_argument("--evaluator-level", default="熟练者", help="评价标准：初习者/进阶者/熟练者")

    internal_group = parser.add_argument_group("桌面端内部工具")
    internal_group.add_argument("--desktop-camera-scan", action="store_true", help="内部使用：扫描摄像头")
    internal_group.add_argument("--desktop-read-text", help="内部使用：读取文本文件")
    internal_group.add_argument(
        "--desktop-video-thumb",
        nargs=2,
        metavar=("VIDEO", "OUTPUT"),
        help="内部使用：生成视频缩略图",
    )
    return parser


def ensure_stdio_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def prepare_runtime_environment() -> None:
    cache_root = PROJECT_ROOT / ".cache"
    mpl_root = PROJECT_ROOT / ".mplconfig"
    cache_root.mkdir(parents=True, exist_ok=True)
    mpl_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_root))


def check_env(weights_path: str) -> int:
    checks: list[tuple[str, bool, str]] = []
    modules = [
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Pillow", "PIL"),
        ("PySide6", "PySide6"),
        ("openpyxl", "openpyxl"),
    ]
    for label, module_name in modules:
        try:
            __import__(module_name)
            checks.append((label, True, "已安装"))
        except Exception as exc:
            checks.append((label, False, str(exc)))

    try:
        __import__("ultralytics")
        checks.append(("Ultralytics", True, "已安装"))
    except Exception as exc:
        checks.append(("Ultralytics", False, str(exc)))

    weights_file = Path(weights_path)
    checks.append(("默认权重文件", weights_file.exists(), str(weights_file)))

    downloaded = []
    missing = []
    for model_info in YOLO_POSE_MODELS:
        model_name = str(model_info["name"])
        candidates = [
            PROJECT_ROOT / model_name,
            PROJECT_ROOT / "weights" / model_name,
            Path.cwd() / model_name,
            Path.cwd() / "weights" / model_name,
        ]
        if any(path.exists() and path.stat().st_size > 1024 * 1024 for path in candidates):
            downloaded.append(model_name)
        else:
            missing.append(model_name)
    checks.append(("YOLOv8-Pose 模型", bool(downloaded), ", ".join(downloaded) if downloaded else "未找到"))

    exit_code = 0
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'MISSING'}] {name}: {detail}")
        if not ok:
            exit_code = 1
    print(f"[INFO] 缺失模型: {', '.join(missing) if missing else '无'}")
    return exit_code


def launch_desktop() -> int:
    from motion_ai.qt_gui import run_qt_gui

    return run_qt_gui(app_name=APP_NAME, app_version=APP_VERSION)


def _default_output_dir(prefix: str) -> Path:
    from datetime import datetime

    target = PROJECT_ROOT / "outputs" / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _load_template_payload(template_arg: str, weights_path: str, output_dir: Path, description_text: str, evaluator_level: str) -> dict | None:
    template_path = Path(template_arg).expanduser().resolve()
    if not template_path.exists():
        raise RuntimeError(f"模板文件不存在：{template_path}")
    if template_path.suffix.lower() == ".json":
        return json.loads(template_path.read_text(encoding="utf-8"))

    from motion_ai.analyzer import ActionAnalyzer

    template_output_dir = output_dir / "template_extract"
    template_output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = ActionAnalyzer(weights_path=weights_path)
    template_result = analyzer.analyze_video(
        source=str(template_path),
        description_text=description_text or f"{template_path.stem} 标准动作模板",
        output_dir=template_output_dir,
        frame_stride=1,
        analysis_mode="template",
        evaluator_level=evaluator_level,
    )
    baseline_path = Path(template_result.artifacts.get("template_baseline_json", ""))
    if not baseline_path.exists():
        raise RuntimeError("模板视频分析完成，但未生成模板基线 JSON")
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def run_cli_analysis(
    *,
    source: str,
    output: str | None,
    weights: str,
    description: str,
    template: str | None,
    mode: str,
    frame_stride: int,
    max_frames: int | None,
    evaluator_level: str,
) -> int:
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise RuntimeError(f"输入视频不存在：{source_path}")

    output_dir = Path(output).expanduser().resolve() if output else _default_output_dir("analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_payload = None
    if template:
        baseline_payload = _load_template_payload(template, weights, output_dir, description, evaluator_level)

    from motion_ai.analyzer import ActionAnalyzer

    analyzer = ActionAnalyzer(weights_path=weights)
    result = analyzer.analyze_video(
        source=str(source_path),
        description_text=description or f"{source_path.stem} 动作分析",
        output_dir=output_dir,
        frame_stride=max(1, int(frame_stride)),
        max_frames=max_frames,
        analysis_mode="template" if mode == "template" else "test",
        baseline_payload=baseline_payload,
        export_overlay_video=True,
        evaluator_level=evaluator_level,
    )

    print(f"[OK] 分析完成: {output_dir}")
    for key in [
        "raw_keypoints_csv",
        "smoothed_keypoints_csv",
        "raw_keypoints_cn_csv",
        "smoothed_keypoints_cn_csv",
        "inhibition_metrics_cn_xlsx",
        "frame_metrics_csv",
        "summary_metrics_json",
        "analysis_report_md",
        "overlay_video",
        "analysis_result_xlsx",
    ]:
        path = result.artifacts.get(key)
        if path:
            print(f"[OUTPUT] {key}: {path}")
    return 0


def main() -> int:
    ensure_stdio_utf8()
    prepare_runtime_environment()
    parser = build_parser()
    args = parser.parse_args()

    if args.check_env:
        return check_env(str(DEFAULT_WEIGHTS))

    if args.analyze:
        return run_cli_analysis(
            source=args.analyze,
            output=args.output,
            weights=args.weights,
            description=args.description,
            template=args.template,
            mode=args.mode,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            evaluator_level=args.evaluator_level,
        )

    if args.desktop_camera_scan:
        from motion_ai.desktop_backend import print_camera_scan

        return print_camera_scan()

    if args.desktop_read_text:
        print(Path(args.desktop_read_text).read_text(encoding="utf-8"))
        return 0

    if args.desktop_video_thumb:
        from motion_ai.desktop_backend import create_video_thumbnail

        video_path, output_path = args.desktop_video_thumb
        return create_video_thumbnail(video_path, output_path)

    if args.desktop:
        return launch_desktop()

    return launch_desktop()


if __name__ == "__main__":
    raise SystemExit(main())
