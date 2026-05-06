# Human-Pose

这个仓库已经整理为适合上传 GitHub 的结构，源码统一放在 `src/`。

## 目录结构

```text
.
├─ src/
│  ├─ app.py
│  ├─ build_executable.py
│  ├─ download_pose_weights.py
│  └─ motion_ai/
├─ templates/
├─ fonts/
├─ hooks/
├─ docs/
├─ weights/
├─ motion_ai_workbench.spec
├─ build_executable_windows.bat
├─ requirements.txt
└─ sample_description.txt
```

## 运行

安装依赖：

```bash
pip install -r requirements.txt
```

启动桌面程序：

```bash
python src/app.py --desktop
```

检查环境：

```bash
python src/app.py --check-env
```

下载模型：

```bash
python src/download_pose_weights.py
```

## 打包

Windows：

```bat
build_executable_windows.bat
```

或：

```bash
python src/build_executable.py
```

## 上传说明

- 代码已集中到 `src/`
- 历史版本、缓存、输出结果、测试素材和大文件已清理
- `.gitignore` 已配置，默认不提交权重、输出和构建产物

## 文档

- [使用指南](docs/使用指南.md)
- [功能文档](docs/功能文档.md)
- [技术方案](docs/AI动作分析系统技术栈与方案实现.md)
