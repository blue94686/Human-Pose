import os

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    get_package_paths,
    is_module_satisfies,
    logger,
)

if is_module_satisfies("PyInstaller >= 6.0"):
    from PyInstaller import compat
    from PyInstaller.utils.hooks import PY_DYLIB_PATTERNS

    module_collection_mode = "pyz+py"
    warn_on_missing_hiddenimports = False

    datas = collect_data_files(
        "torch",
        excludes=[
            "**/*.h",
            "**/*.hpp",
            "**/*.cuh",
            "**/*.lib",
            "**/*.cpp",
            "**/*.pyi",
            "**/*.cmake",
        ],
    )
    # 让 PyInstaller 依据真实导入链收集 torch 模块，避免把测试、分布式、
    # tensorboard 等大量非推理依赖整包带入 onefile 产物。
    hiddenimports = [
        "torch._dynamo.polyfills._collections",
        "torch._dynamo.polyfills.builtins",
        "torch._dynamo.polyfills.functools",
        "torch._dynamo.polyfills.fx",
        "torch._dynamo.polyfills.itertools",
        "torch._dynamo.polyfills.operator",
        "torch._dynamo.polyfills.os",
        "torch._dynamo.polyfills.struct",
        "torch._dynamo.polyfills.sys",
        "torch._dynamo.polyfills.tensor",
        "torch._dynamo.polyfills.torch_c_nn",
        "torch._dynamo.polyfills.traceback",
    ]
    binaries = collect_dynamic_libs(
        "torch",
        search_patterns=PY_DYLIB_PATTERNS + ["*.so.*"],
    )

    if compat.is_linux:
        def _infer_nvidia_hiddenimports():
            import packaging.requirements
            from _pyinstaller_hooks_contrib.compat import importlib_metadata
            from _pyinstaller_hooks_contrib.utils import nvidia_cuda as cudautils

            dist = importlib_metadata.distribution("torch")
            requirements = [packaging.requirements.Requirement(req) for req in dist.requires or []]
            requirements = [req.name for req in requirements if req.marker is None or req.marker.evaluate()]
            return cudautils.infer_hiddenimports_from_requirements(requirements)

        try:
            nvidia_hiddenimports = _infer_nvidia_hiddenimports()
        except Exception:
            logger.warning("hook-torch: failed to infer NVIDIA CUDA hidden imports!", exc_info=True)
            nvidia_hiddenimports = []
        logger.info("hook-torch: inferred hidden imports for CUDA libraries: %r", nvidia_hiddenimports)
        hiddenimports += nvidia_hiddenimports
        bindepend_symlink_suppression = ["**/torch/lib/*.so*"]

    if compat.is_win:
        def _collect_mkl_dlls():
            conda_torch_dist = None
            if compat.is_conda:
                from PyInstaller.utils.hooks import conda_support

                try:
                    conda_torch_dist = conda_support.package_distribution("torch")
                except ModuleNotFoundError:
                    conda_torch_dist = None

            if conda_torch_dist:
                if "mkl" not in conda_torch_dist.dependencies:
                    logger.info("hook-torch: this torch build (Anaconda package) does not depend on MKL...")
                    return []

                logger.info("hook-torch: collecting DLLs from MKL and its dependencies (Anaconda packages)")
                mkl_binaries = conda_support.collect_dynamic_libs("mkl", dependencies=True)
            else:
                import packaging.requirements
                from _pyinstaller_hooks_contrib.compat import importlib_metadata

                dist = importlib_metadata.distribution("torch")
                requirements = [packaging.requirements.Requirement(req) for req in dist.requires or []]
                requirements = [req.name for req in requirements if req.marker is None or req.marker.evaluate()]
                if "mkl" not in requirements:
                    logger.info("hook-torch: this torch build does not depend on MKL...")
                    return []

                try:
                    dist = importlib_metadata.distribution("mkl")
                except importlib_metadata.PackageNotFoundError:
                    return []
                requirements = [packaging.requirements.Requirement(req) for req in dist.requires or []]
                requirements = [req.name for req in requirements if req.marker is None or req.marker.evaluate()]
                requirements = ["mkl"] + requirements

                mkl_binaries = []
                logger.info("hook-torch: collecting DLLs from MKL and its dependencies: %r", requirements)
                for requirement in requirements:
                    try:
                        dist = importlib_metadata.distribution(requirement)
                    except importlib_metadata.PackageNotFoundError:
                        continue

                    for dist_file in (dist.files or []):
                        dll_file = dist.locate_file(dist_file).resolve()
                        if not dll_file.match("**/Library/bin/*.dll"):
                            continue
                        mkl_binaries.append((str(dll_file), "."))

            if mkl_binaries:
                logger.info(
                    "hook-torch: found MKL DLLs: %r",
                    sorted([os.path.basename(src_name) for src_name, dest_name in mkl_binaries]),
                )
            else:
                logger.info("hook-torch: no MKL DLLs found.")

            return mkl_binaries

        try:
            mkl_binaries = _collect_mkl_dlls()
        except Exception:
            logger.warning("hook-torch: failed to collect MKL DLLs!", exc_info=True)
            mkl_binaries = []
        binaries += mkl_binaries
else:
    datas = [(get_package_paths("torch")[1], "torch")]
