"""为推理准备一个可用的 Python，不改 ComfyUI 自己的环境。

顺序：

1. 调用方传入的解释器，或环境变量 ``MOSS_SFX_PYTHON``。指定了就必须能用。
2. ComfyUI 当前这个解释器。能导入推理代码就直接用。
3. 节点目录里已经建好的 ``.venv``。
4. 用 uv 在节点目录新建 Python 3.12，只往这个新环境里装依赖。
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

NODE_ROOT = Path(__file__).resolve().parent
VENV_DIR = NODE_ROOT / ".venv"
TOOLS_DIR = NODE_ROOT / ".tools"
RUNTIME_REQUIREMENTS = NODE_ROOT / "requirements-runtime.txt"
MIN_FREE_BYTES = 8 * 1024 ** 3

_PROBE_CACHE: dict[str, tuple[bool, str]] = {}

_PROBE_CODE = """
import audiotools
import torch
from moss_soundeffect_v2 import MossSoundEffectPipeline
print("ok")
"""


def venv_python(venv: Path = VENV_DIR) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def probe(python: Path) -> tuple[bool, str]:
    key = str(python)
    cached = _PROBE_CACHE.get(key)
    if cached is not None:
        return cached
    if not python.is_file() or not os.access(python, os.X_OK):
        result = (False, f"解释器不存在或不可执行：{python}")
        _PROBE_CACHE[key] = result
        return result

    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(NODE_ROOT) + (os.pathsep + previous if previous else "")
    env["PYTHONUNBUFFERED"] = "1"
    try:
        completed = subprocess.run(
            [str(python), "-c", _PROBE_CODE],
            cwd=str(NODE_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        result = (False, f"探测超时：{python}")
        _PROBE_CACHE[key] = result
        return result

    if completed.returncode == 0 and "ok" in completed.stdout:
        result = (True, "")
        _PROBE_CACHE[key] = result
        return result
    detail = (completed.stderr or completed.stdout or "").strip()
    if len(detail) > 1500:
        detail = detail[-1500:]
    result = (False, detail or f"探测失败，退出码 {completed.returncode}")
    _PROBE_CACHE[key] = result
    return result


def resolve_python(explicit: str = "") -> Path:
    chosen = str(explicit or "").strip() or os.environ.get("MOSS_SFX_PYTHON", "").strip()
    if chosen:
        path = Path(chosen).expanduser()
        ok, detail = probe(path)
        if not ok:
            raise RuntimeError(
                f"指定的解释器不能运行 MOSS-SoundEffect v2：{path}\n{detail}"
            )
        print(f"[moss-soundeffect] 使用指定解释器：{path}", flush=True)
        return path

    current = Path(sys.executable)
    ok, _detail = probe(current)
    if ok:
        print(f"[moss-soundeffect] 使用 ComfyUI 的解释器：{current}", flush=True)
        return current

    existing = venv_python()
    if existing.is_file():
        ok, _detail = probe(existing)
        if ok:
            print(f"[moss-soundeffect] 使用节点自带环境：{existing}", flush=True)
            return existing

    return bootstrap()


def _driver_cuda() -> tuple[int, int] | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", out)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def torch_packages() -> tuple[str, str, str]:
    """按本机选择 Torch 2.9 的索引和版本。对不上就直接失败，不装错轮子。"""
    if platform.system() == "Darwin":
        raise RuntimeError(
            "自动安装不支持 macOS。请自行准备带 MPS 或 CPU 版 Torch 的解释器，"
            "填到加载节点的 python，或设置 MOSS_SFX_PYTHON。"
        )
    cuda = _driver_cuda()
    if cuda is None:
        if os.environ.get("ROCM_PATH") or Path("/opt/rocm").is_dir():
            raise RuntimeError(
                "检测到 ROCm。自动安装不会下载 CUDA 版 Torch。"
                "请准备对应的解释器，填到加载节点的 python，或设置 MOSS_SFX_PYTHON。"
            )
        raise RuntimeError(
            "没有检测到 NVIDIA 驱动，自动安装不会下载 CUDA 版 Torch。"
            "已有可用解释器时，填到加载节点的 python，或设置 MOSS_SFX_PYTHON。"
        )
    major, minor = cuda
    if (major, minor) >= (12, 8):
        tag = "cu128"
    elif (major, minor) >= (12, 6):
        tag = "cu126"
    else:
        raise RuntimeError(
            f"驱动支持的 CUDA 是 {major}.{minor}。自动安装的 Torch 2.9 需要 CUDA 12.6 或更高。"
            "请升级驱动，或填写已经能运行推理的 python。"
        )
    index = f"https://download.pytorch.org/whl/{tag}"
    return index, f"torch==2.9.0+{tag}", f"torchaudio==2.9.0+{tag}"


def bootstrap() -> Path:
    free = shutil.disk_usage(NODE_ROOT).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"磁盘剩余 {free / 1024 ** 3:.1f} GB，安装推理环境至少需要 8 GB。"
            f"目录：{NODE_ROOT}"
        )

    print(
        "[moss-soundeffect] 当前解释器缺少推理依赖，正在节点目录创建 Python 3.12 环境。"
        "这一步只做一次，会下载 Torch 2.9。",
        flush=True,
    )
    if VENV_DIR.exists():
        shutil.rmtree(VENV_DIR)
    _PROBE_CACHE.pop(str(venv_python()), None)

    uv = ensure_uv()
    subprocess.run(
        [str(uv), "venv", "--python", "3.12", str(VENV_DIR)],
        cwd=str(NODE_ROOT),
        check=True,
    )
    python = venv_python()
    if not python.is_file():
        raise RuntimeError(f"虚拟环境已创建，但没有找到解释器：{python}")

    index_url, torch_spec, torchaudio_spec = torch_packages()
    print(f"[moss-soundeffect] 安装 {torch_spec}（{index_url}）", flush=True)
    env = os.environ.copy()
    env["UV_LINK_MODE"] = "copy"
    subprocess.run(
        [
            str(uv),
            "pip",
            "install",
            "--python",
            str(python),
            "--index-url",
            index_url,
            "--extra-index-url",
            "https://pypi.org/simple",
            "--index-strategy",
            "unsafe-best-match",
            "-c",
            str(NODE_ROOT / "constraints-runtime.txt"),
            "-r",
            str(RUNTIME_REQUIREMENTS),
            torch_spec,
            torchaudio_spec,
        ],
        cwd=str(NODE_ROOT),
        env=env,
        check=True,
    )

    _PROBE_CACHE.pop(str(python), None)
    ok, detail = probe(python)
    if not ok:
        raise RuntimeError(f"推理环境已安装，但探测没有通过：{python}\n{detail}")
    print(f"[moss-soundeffect] 推理环境就绪：{python}", flush=True)
    return python


def ensure_uv() -> Path:
    found = shutil.which("uv")
    if found:
        return Path(found)

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    binary = TOOLS_DIR / ("uv.exe" if os.name == "nt" else "uv")
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary

    asset = _uv_asset_name()
    url = f"https://github.com/astral-sh/uv/releases/latest/download/{asset}"
    print(f"[moss-soundeffect] 下载 uv：{url}", flush=True)
    with tempfile.TemporaryDirectory(prefix="moss-uv-") as tmp:
        archive = Path(tmp) / asset
        urllib.request.urlretrieve(url, archive)
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as packed:
                packed.extractall(tmp)
        else:
            with tarfile.open(archive) as packed:
                packed.extractall(tmp)
        extracted = next(path for path in Path(tmp).rglob("uv*") if path.is_file() and path.name in {"uv", "uv.exe"})
        shutil.copy2(extracted, binary)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return binary


def _uv_asset_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        arch = "x86_64"
    elif machine in {"arm64", "aarch64"}:
        arch = "aarch64"
    else:
        raise RuntimeError(f"没有对应的 uv 安装包：{platform.system()} {platform.machine()}")
    if system == "linux":
        return f"uv-{arch}-unknown-linux-gnu.tar.gz"
    if system == "darwin":
        return f"uv-{arch}-apple-darwin.tar.gz"
    if system == "windows":
        return f"uv-{arch}-pc-windows-msvc.zip"
    raise RuntimeError(f"没有对应的 uv 安装包：{platform.system()} {platform.machine()}")
