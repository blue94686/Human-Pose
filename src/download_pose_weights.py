from __future__ import annotations

import shutil
import sys
import urllib.request
import argparse
from pathlib import Path

from motion_ai.config import PROJECT_ROOT, WEIGHTS_DIR, YOLO_POSE_MODELS


def resolve_weight_path(name: str) -> Path:
    candidates = [
        WEIGHTS_DIR / name,
        PROJECT_ROOT / name,
        PROJECT_ROOT / "weights" / name,
        PROJECT_ROOT.parent / name,
        PROJECT_ROOT.parent / "weights" / name,
        Path.cwd() / name,
        Path.cwd() / "weights" / name,
        Path.cwd().parent / name,
        Path.cwd().parent / "weights" / name,
    ]
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.expanduser().resolve())
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and candidate.stat().st_size > 1024 * 1024:
            return candidate
    return WEIGHTS_DIR / name


def list_status() -> None:
    print("YOLOv8-Pose model status:")
    for model_info in YOLO_POSE_MODELS:
        name = str(model_info["name"])
        path = resolve_weight_path(name)
        exists = path.exists() and path.stat().st_size > 1024 * 1024
        flag = "DOWNLOADED" if exists else "MISSING"
        size = f"{path.stat().st_size / 1024 / 1024:.1f}MB" if exists else "-"
        print(f"[{flag}] {name}\t{model_info.get('label')}\t{model_info.get('profile')}\t{size}\t{path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download or inspect YOLOv8-Pose weights.")
    parser.add_argument("--list", action="store_true", help="Only list local model status.")
    parser.add_argument("--missing-only", action="store_true", default=True, help="Download only missing models.")
    parser.add_argument("--model", action="append", default=[], help="Download only the named model. Can be repeated.")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.list:
        list_status()
        return 0

    root_weight = PROJECT_ROOT / "yolov8n-pose.pt"
    if root_weight.exists():
        target = WEIGHTS_DIR / root_weight.name
        if not target.exists():
            shutil.copy2(root_weight, target)
            print(f"copied {root_weight.name} -> {target}")

    selected = {name.strip() for name in args.model if name.strip()}
    for model_info in YOLO_POSE_MODELS:
        name = str(model_info["name"])
        if selected and name not in selected:
            continue
        target = WEIGHTS_DIR / name
        if target.exists() and target.stat().st_size > 1024 * 1024:
            print(f"exists {name}")
            continue
        url = str(model_info["url"])
        temp_path = target.with_suffix(".download")
        print(f"downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, temp_path)
        except Exception as exc:
            print(f"failed {name}: {exc}", file=sys.stderr)
            if temp_path.exists():
                temp_path.unlink()
            continue
        temp_path.replace(target)
        print(f"saved {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
