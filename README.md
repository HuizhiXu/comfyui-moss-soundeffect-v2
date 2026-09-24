# MOSS SoundEffect v2

三个节点，都在右键菜单的 **audio/moss** 下面：

1. **MOSS SoundEffect v2 加载**：从 `ComfyUI/models/moss-soundeffect-v2/` 选择模型。输出一根自定义连线 `pipeline`。这里不加载权重。
2. **MOSS SoundEffect v2 生成**：把提示词交给推理进程，输出 ComfyUI 的 `AUDIO`。后面接 **Preview Audio** 或 **Save Audio**。
3. **MOSS SoundEffect v2 卸载**：把生成节点的音频输出接到这里，图跑完后停掉推理进程，把大约 11GB 显存还回去。这根线是必填的，用来把卸载排在生成之后；不接的话这个节点无法运行。生成节点的音频同时还可以接到 Preview Audio。

模型是 1.3B DiT，权重大约 11GB，输出 48 kHz 单声道。界面上的秒数最长 30 秒。短于 30 秒时，模型仍按 30 秒去噪，再裁到设定的长度，耗时和 30 秒接近。同一时间只有一个推理进程；一张图里两套不同的加载参数，会在两次生成之间卸掉再加载。第一次生成会编译 DiT，可能要几分钟，进度会先停在 0。点中断会停掉推理进程。

## 安装

把本目录放进 `ComfyUI/custom_nodes/`。用 Manager 安装时会探测推理环境；手动复制时，第一次运行加载节点也会探测。

探测顺序：

1. 加载节点上填写的 `python`，或环境变量 `MOSS_SFX_PYTHON`。留空就跳过。
2. ComfyUI 自己的解释器。能导入推理代码就直接用。
3. 节点目录里已有的 `.venv`。
4. 都不可用时，在节点目录创建 Python 3.12 环境并安装依赖。这一步只做一次，不改 ComfyUI 的包。驱动支持 CUDA 12.8 及以上时装 Torch 2.9 cu128，12.6 和 12.7 装 cu126。更低的 CUDA、ROCm、macOS 或没有 NVIDIA 驱动时不会安装，避免装进不能用的轮子。这种情况请填写已经能运行推理的 `python`。

## 准备权重

节点不下载权重。把 Hugging Face 仓库放到：

`ComfyUI/models/moss-soundeffect-v2/MOSS-SoundEffect-v2.0`

```bash
huggingface-cli download OpenMOSS-Team/MOSS-SoundEffect-v2.0 \
  --local-dir ComfyUI/models/moss-soundeffect-v2/MOSS-SoundEffect-v2.0
```

重启或刷新 ComfyUI 后，加载节点的 `model` 下拉框会列出这个目录。目录里需要有 `model_index.json`。

加载节点上的 `device` 是下拉框，默认 `auto`。主进程的 PyTorch 能看见 CUDA 时用它的设备列表；看不见时用 `nvidia-smi` 的数量，并按 `CUDA_VISIBLE_DEVICES` 重排。只露出一张卡时，列表里只有 `cuda:0`。`python` 是可选项，已有可用环境时才填。

## 发布到 Comfy Registry

本目录需要单独成为一个 Git 仓库，`__init__.py` 在仓库根目录。建好仓库不会出现在 Manager 里。

发布前在 `pyproject.toml` 里填上这些。`name` 和 `PublisherId` 第一次发布后不能改。`name` 不要包含 `ComfyUI`。`PublisherId` 在 [registry.comfy.org](https://registry.comfy.org) 注册，个人页上显示成 `@你的id`，文件里不要带 `@`。

- `[project].name`
- `[project].version`，格式 `X.Y.Z`
- `[project.urls] Repository`
- `[tool.comfy] PublisherId`

`DisplayName` 和 `LICENSE` 官方标为可选，这里已经写了。许可证是 `license = { file = "LICENSE" }`，沿用上游的 Apache-2.0。

然后任选一种：

1. 安装 [comfy-cli](https://docs.comfy.org/registry/publishing)，在仓库根目录执行 `comfy node publish`，粘贴这个 Publisher 的 API key。
2. 把同一个 key 存成 GitHub 仓库密钥 `REGISTRY_ACCESS_TOKEN`。`.github/workflows/publish_action.yml` 会在推送改过版本号的 `pyproject.toml` 时发布。工作分支如果不是 `main`，改 workflow 里的分支名。
