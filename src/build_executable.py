from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DIST_DIR = PROJECT_ROOT / "dist"
SPEC_PATH = PROJECT_ROOT / "motion_ai_workbench.spec"
APP_NAME = "动作分析工作台"
EXE_PATH = DIST_DIR / f"{APP_NAME}.exe"


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="构建桌面 EXE 可执行文件")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="构建前清理 PyInstaller 缓存目录",
    )
    parser.add_argument(
        "--spec",
        default=str(SPEC_PATH),
        help="指定 PyInstaller spec 文件路径",
    )
    return parser


def resolve_python_executable() -> Path:
    """解析当前 Python 解释器路径。"""
    return Path(sys.executable).resolve()


def resolve_spec_path(spec_arg: str) -> Path:
    """解析并校验 spec 文件路径。"""
    spec_path = Path(spec_arg).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"未找到 spec 文件：{spec_path}")
    return spec_path


def build_environment() -> dict[str, str]:
    """构建打包所需环境变量。"""
    env = os.environ.copy()
    env.setdefault("MOTION_AI_HOME", str(PROJECT_ROOT))
    return env


def clean_build_directories() -> None:
    """清理构建缓存目录。"""
    for target in [PROJECT_ROOT / "build", DIST_DIR]:
        if target.exists():
            print(f"[INFO] 清理目录：{target}")
            for child in sorted(target.iterdir(), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                else:
                    shutil.rmtree(child)


def build_pyinstaller_command(python_exe: Path, spec_path: Path) -> list[str]:
    """生成 PyInstaller 命令。"""
    return [str(python_exe), "-m", "PyInstaller", "--noconfirm", str(spec_path)]


def run_build(*, spec_path: Path, clean: bool) -> Path:
    """执行打包流程并返回产物路径。"""
    if clean:
        clean_build_directories()

    python_exe = resolve_python_executable()
    pyinstaller_cmd = build_pyinstaller_command(python_exe, spec_path)
    env = build_environment()

    print(f"[INFO] Python 解释器：{python_exe}")
    print(f"[INFO] 使用 Spec：{spec_path}")
    subprocess.run(pyinstaller_cmd, cwd=PROJECT_ROOT, env=env, check=True)
    if not EXE_PATH.exists():
        raise FileNotFoundError(f"未找到打包产物：{EXE_PATH}")
    return EXE_PATH


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    spec_path = resolve_spec_path(args.spec)
    output_path = run_build(spec_path=spec_path, clean=bool(args.clean))
    print(f"[OK] 打包完成：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
