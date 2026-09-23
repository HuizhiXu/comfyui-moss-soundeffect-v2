# MOSS SoundEffect v2

两个节点，都在右键菜单的 **audio/moss** 下面：

1. **MOSS SoundEffect v2 加载**：在你填写的 Python 3.12 进程里加载模型。输出一根自定义连线 `pipeline`。
2. **MOSS SoundEffect v2 生成**：把提示词交给这个进程，输出 ComfyUI 的 `AUDIO`。后面接 **Preview Audio** 或 **Save Audio**。

模型是 1.3B DiT，权重大约 11GB，输出 48 kHz 单声道，最长 30 秒。第一次生成会编译 DiT，可能要几分钟。ComfyUI 的 Python 只负责收发 JSON 和读波形，不安装 `numpy==1.26.4` 或 `descript-audiotools`。

## 准备 Python 3.12

推理环境跟上游 [moss_soundeffect_v2](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_soundeffect_v2/README.md) 相同，和 ComfyUI 分开：

```bash
conda create -n moss-soundeffect-v2 python=3.12 -y
conda activate moss-soundeffect-v2
cd /path/to/MOSS-TTS/moss_soundeffect_v2
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -e ".[torch-cu128]"
```

## 接到 ComfyUI

把本目录作为 `custom_nodes` 里的一个包。它还在 MOSS-TTS 仓库里时：

```bash
ln -s /path/to/MOSS-TTS/community/comfyui-moss-soundeffect-v2 \
  /path/to/ComfyUI/custom_nodes/comfyui-moss-soundeffect-v2
```

重启 ComfyUI。加载节点上填写：

| 控件 | 填什么 |
|---|---|
| `python` | 上面那个环境的解释器，例如 `.../envs/moss-soundeffect-v2/bin/python` |
| `moss_root` | MOSS-TTS 仓库根目录。环境里已经 `pip install -e` 过可以留空 |
| `device` | `auto`、`cpu`、`cuda` 或 `cuda:1` |

同一组解释器、仓库、模型和设备会复用已经启动的进程。换其中任意一项会关掉旧进程。

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
