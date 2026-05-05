# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


project_root = Path.cwd()
src_root = project_root / "src"
hiddenimports = []
datas = []
datas += collect_data_files("ultralytics")
datas += [
    (str(src_root / "motion_ai" / "desktop_gui.tcl"), "motion_ai"),
    (str(project_root / "templates" / "action_templates.json"), "templates"),
    (str(project_root / "templates" / "error_rules.json"), "templates"),
    (str(project_root / "templates" / "template_library.json"), "templates"),
    (str(project_root / "fonts" / "simhei.ttf"), "fonts"),
    (str(project_root / "sample_description.txt"), "."),
]
for weights_path in sorted(project_root.glob("yolov8*-pose*.pt")):
    datas.append((str(weights_path), "."))
for weights_path in sorted((project_root / "weights").glob("yolov8*-pose*.pt")):
    datas.append((str(weights_path), "weights"))


a = Analysis(
    [str(src_root / "app.py")],
    pathex=[str(project_root), str(src_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(project_root / "hooks")],
    hooksconfig={
        "matplotlib": {
            "backends": ["Agg"],
        },
    },
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "setuptools",
        "pkg_resources",
        "IPython",
        "ipykernel",
        "jupyter_client",
        "jupyter_core",
        "traitlets",
        "prompt_toolkit",
        "pygments",
        "jedi",
        "parso",
        "tornado",
        "zmq",
        "fsspec",
        "networkx",
        "matplotlib",
        "matplotlib.tests",
        "matplotlib.testing",
        "torch.utils.tensorboard",
        "tensorboard",
        "matplotlib.backends._backend_tk",
        "matplotlib.backends.backend_tkagg",
        "matplotlib.backends.backend_tkcairo",
        "matplotlib.backends.backend_gtk3",
        "matplotlib.backends.backend_gtk3agg",
        "matplotlib.backends.backend_gtk3cairo",
        "matplotlib.backends.backend_gtk4",
        "matplotlib.backends.backend_gtk4agg",
        "matplotlib.backends.backend_gtk4cairo",
        "matplotlib.backends.backend_wx",
        "matplotlib.backends.backend_wxagg",
        "matplotlib.backends.backend_wxcairo",
        "matplotlib.backends.backend_webagg",
        "matplotlib.backends.backend_webagg_core",
        "matplotlib.backends.backend_nbagg",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="动作分析工作台",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
