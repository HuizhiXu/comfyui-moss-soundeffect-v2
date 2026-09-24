"""MOSS-SoundEffect v2 的 ComfyUI 节点。

ComfyUI 启动时导入这个文件，所以这里不能加载模型，也不能导入
``descript-audiotools``。推理放在探测通过的 Python 3.12 进程里，
两边用一行一个 JSON 通信。

权重由用户放到 ``ComfyUI/models/moss-soundeffect-v2/``，加载节点只列出
里面带 ``model_index.json`` 的目录。
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path

from .bootstrap import resolve_python

_WORKER = Path(__file__).resolve().with_name("worker.py")
_NODE_ROOT = Path(__file__).resolve().parent
_MISSING = "未找到模型"
_STATE: dict[str, object] = {
    "key": None,
    "proc": None,
    "tail": deque(maxlen=40),
    "lock": threading.Lock(),
}


def models_root() -> Path:
    try:
        import folder_paths

        root = Path(folder_paths.models_dir)
    except Exception:
        root = Path(os.environ.get("COMFYUI_MODELS_DIR", "models"))
    path = root / "moss-soundeffect-v2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_choices() -> list[tuple[str, Path]]:
    root = models_root()
    choices: list[tuple[str, Path]] = []
    if not root.is_dir():
        return choices
    seen: set[Path] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        current = Path(dirpath)
        try:
            resolved = current.resolve()
        except OSError:
            dirnames.clear()
            continue
        if resolved in seen:
            dirnames.clear()
            continue
        seen.add(resolved)
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        if "model_index.json" not in filenames:
            continue
        relative = current.relative_to(root).as_posix()
        label = root.name if relative == "." else relative
        choices.append((label, current))
        dirnames.clear()
    choices.sort(key=lambda item: item[0])
    return choices


def _nvidia_smi_listing() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "-L"],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return ""


def _count_visible_devices(listing: str) -> int:
    """nvidia-smi 列出全部卡。下拉框序号要按 CUDA_VISIBLE_DEVICES 重排后的结果。"""
    physical = sum(1 for line in listing.splitlines() if line.startswith("GPU "))
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return physical
    raw = raw.strip()
    if raw == "" or raw.lower() in {"-1", "none", "void", "nodevfiles"}:
        return 0
    uuids = set()
    for line in listing.splitlines():
        if "UUID:" not in line:
            continue
        uuids.add(line.split("UUID:", 1)[1].strip().rstrip(")"))
    count = 0
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            if int(token) < physical:
                count += 1
        elif token in uuids:
            count += 1
    return count


def _cuda_device_count() -> int:
    """主进程的 PyTorch 没有 CUDA 时，改用 nvidia-smi，并遵守 CUDA_VISIBLE_DEVICES。"""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return _count_visible_devices(_nvidia_smi_listing())


def device_choices() -> list[str]:
    """序号跟着当前进程能看见的卡，不是机器上的物理卡号。"""
    choices = ["auto"]
    count = _cuda_device_count()
    choices.extend(f"cuda:{index}" for index in range(count))
    try:
        import torch

        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            choices.append("mps")
    except Exception:
        pass
    choices.append("cpu")
    return choices


def model_help() -> str:
    destination = models_root() / "MOSS-SoundEffect-v2.0"
    return (
        "把 MOSS-SoundEffect v2 的模型目录放到 "
        f"{models_root()} 。\n"
        "下载示例：\n"
        "huggingface-cli download OpenMOSS-Team/MOSS-SoundEffect-v2.0 "
        f"--local-dir {destination}"
    )


_STEP_PROGRESS = re.compile(r"(\d+)\s*/\s*(\d+)")


def _pump_stderr(proc: subprocess.Popen[str], tail: deque[str]) -> None:
    assert proc.stderr is not None
    for line in proc.stderr:
        tail.append(line)
        print(f"[moss-soundeffect] {line}", end="", flush=True)
        bar = _STATE.get("progress")
        match = _STEP_PROGRESS.search(line)
        if bar is not None and match is not None:
            bar.update_absolute(int(match.group(1)), int(match.group(2)))


def _stop(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    if proc.stdin is not None:
        try:
            proc.stdin.close()
        except OSError:
            pass
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=5)


def _processing_interrupted() -> bool:
    try:
        import comfy.model_management as model_management
    except Exception:
        return False
    return bool(model_management.processing_interrupted())


def _raise_if_interrupted(proc: subprocess.Popen[str]) -> None:
    if not _processing_interrupted():
        return
    _stop(proc)
    _STATE["key"] = None
    _STATE["proc"] = None
    import comfy.model_management as model_management

    model_management.throw_exception_if_processing_interrupted()


def _read_stdout_line(proc: subprocess.Popen[str]) -> str:
    assert proc.stdout is not None
    box: dict[str, str] = {}

    def read() -> None:
        box["line"] = proc.stdout.readline()

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(0.2)
        _raise_if_interrupted(proc)
    return box.get("line", "")


def _request(proc: subprocess.Popen[str], payload: dict, tail: deque[str]) -> dict:
    if proc.poll() is not None:
        raise RuntimeError("推理进程已退出。\n" + "".join(tail))
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    line = _read_stdout_line(proc)
    if not line:
        _raise_if_interrupted(proc)
        raise RuntimeError("推理进程没有返回结果。\n" + "".join(tail))
    data = json.loads(line)
    if not data.get("ok"):
        raise RuntimeError(str(data.get("error") or "推理失败") + "\n" + "".join(tail))
    return data


def _find_libcuda() -> Path | None:
    for candidate in (
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/lib64/libcuda.so.1"),
        Path("/usr/lib/libcuda.so.1"),
    ):
        if candidate.exists():
            return candidate.resolve()
    try:
        out = subprocess.check_output(
            ["ldconfig", "-p"],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "libcuda.so.1" not in line or "=>" not in line:
            continue
        path = Path(line.split("=>", 1)[1].strip())
        if path.exists():
            return path.resolve()
    return None


def _libcuda_link_dir() -> str | None:
    """Triton 用 -lcuda 链接，需要未带版本号的 libcuda.so。

    链接放在临时目录，不写进节点目录，避免安装目录只读时失败。
    """
    if os.name == "nt":
        return None
    real = _find_libcuda()
    if real is None:
        return None
    link_dir = Path(tempfile.gettempdir()) / "moss-soundeffect-libcuda"
    try:
        link_dir.mkdir(exist_ok=True)
        for name in ("libcuda.so", "libcuda.so.1"):
            dest = link_dir / name
            if dest.is_symlink() and dest.resolve() == real:
                continue
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            dest.symlink_to(real)
    except OSError:
        return None
    return str(link_dir)


def _windows_cuda_lib() -> Path | None:
    candidates: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path))
    root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if root.is_dir():
        candidates.extend(sorted(root.glob("v*"), reverse=True))
    for base in candidates:
        lib = base / "lib" / "x64"
        if (lib / "cuda.lib").is_file():
            return lib
    return None


def _ensure_worker(spec: dict) -> subprocess.Popen[str]:
    key = (
        spec["python"],
        spec["model"],
        spec["device"],
        spec["dtype"],
        spec["vram"],
    )
    proc = _STATE["proc"]
    if _STATE["key"] == key and isinstance(proc, subprocess.Popen) and proc.poll() is None:
        return proc

    _stop(proc if isinstance(proc, subprocess.Popen) else None)
    tail: deque[str] = deque(maxlen=40)
    _STATE["key"] = None
    _STATE["proc"] = None
    _STATE["tail"] = tail

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_NODE_ROOT) + (os.pathsep + previous if previous else "")
    link_dir = _libcuda_link_dir()
    if link_dir is not None:
        env["TRITON_LIBCUDA_PATH"] = link_dir
        library_path = env.get("LIBRARY_PATH", "")
        env["LIBRARY_PATH"] = link_dir + (os.pathsep + library_path if library_path else "")
    if os.name == "nt":
        cuda_lib = _windows_cuda_lib()
        if cuda_lib is not None:
            env["LIB"] = str(cuda_lib) + (os.pathsep + env["LIB"] if env.get("LIB") else "")
            bin_dir = cuda_lib.parent.parent / "bin"
            if bin_dir.is_dir():
                env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")

    popen_kwargs = {"start_new_session": True} if os.name != "nt" else {}
    proc = subprocess.Popen(
        [spec["python"], str(_WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        **popen_kwargs,
    )
    threading.Thread(target=_pump_stderr, args=(proc, tail), daemon=True).start()
    try:
        _request(
            proc,
            {
                "cmd": "load",
                "model": spec["model"],
                "device": spec["device"],
                "dtype": spec["dtype"],
                "vram": spec["vram"],
            },
            tail,
        )
    except Exception:
        _stop(proc)
        raise

    _STATE["key"] = key
    _STATE["proc"] = proc
    return proc


class MossSoundEffectV2Loader:
    @classmethod
    def INPUT_TYPES(cls):
        labels = [label for label, _path in model_choices()] or [_MISSING]
        return {
            "required": {
                "model": (
                    labels,
                    {
                        "tooltip": "ComfyUI/models/moss-soundeffect-v2/ 里已放好的模型目录",
                    },
                ),
                "device": (
                    device_choices(),
                    {
                        "default": "auto",
                        "tooltip": "auto 使用第一张可见的 CUDA 设备。序号跟着 CUDA_VISIBLE_DEVICES，不是机器上的物理卡号",
                    },
                ),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "vram_management": (
                    ["auto", "on", "off"],
                    {
                        "default": "auto",
                        "tooltip": "auto：显存低于 20GB 时，把暂时不用的模块卸到内存",
                    },
                ),
            },
            "optional": {
                "python": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "留空则自动探测。已有可用环境时填写该解释器路径，或设置环境变量 MOSS_SFX_PYTHON",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MOSS_SFX_PIPE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "audio/moss"
    DESCRIPTION = "选择本地模型。权重在第一次生成时加载。同一时间只有一个推理进程。"

    def load(self, model, device, dtype, vram_management, python=""):
        if model == _MISSING:
            raise FileNotFoundError(model_help())

        selected = dict(model_choices()).get(model)
        if selected is None or not (selected / "model_index.json").is_file():
            raise FileNotFoundError(f"找不到模型 {model}。\n" + model_help())

        executable = resolve_python(python)
        return ({
            "python": str(executable),
            "model": str(selected),
            "device": str(device).strip() or "auto",
            "dtype": dtype,
            "vram": vram_management,
        },)


class MossSoundEffectV2Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("MOSS_SFX_PIPE",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "The crisp, rhythmic click-clack of fast typing "
                            "on a mechanical keyboard."
                        ),
                    },
                ),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seconds": (
                    "FLOAT",
                    {"default": 10.0, "min": 0.1, "max": 30.0, "step": 0.1},
                ),
                "steps": ("INT", {"default": 100, "min": 1, "max": 250, "step": 1}),
                "cfg_scale": (
                    "FLOAT",
                    {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1},
                ),
                "sigma_shift": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/moss"
    DESCRIPTION = "取回 48 kHz 单声道音频。短于 30 秒也会先按 30 秒去噪再裁切。第一次生成会编译 DiT。"

    def generate(
        self,
        pipeline,
        prompt,
        negative_prompt,
        seconds,
        steps,
        cfg_scale,
        sigma_shift,
        seed,
    ):
        import numpy as np
        import torch

        text = str(prompt).strip()
        if not text:
            raise ValueError("prompt 不能为空")

        fd, out_path = tempfile.mkstemp(suffix=".npy", prefix="moss-sfx-")
        os.close(fd)
        print(
            "[moss-soundeffect] 开始生成。第一次会编译 DiT，可能要几分钟，进度会停在 0。",
            flush=True,
        )
        try:
            from comfy.utils import ProgressBar

            bar = ProgressBar(int(steps))
            bar.update_absolute(0, int(steps))
        except Exception:
            bar = None
        try:
            with _STATE["lock"]:
                _STATE["progress"] = bar
                proc = _ensure_worker(pipeline)
                tail = _STATE["tail"]
                assert isinstance(tail, deque)
                data = _request(
                    proc,
                    {
                        "cmd": "generate",
                        "prompt": text,
                        "negative_prompt": str(negative_prompt or ""),
                        "seconds": float(seconds),
                        "steps": int(steps),
                        "cfg_scale": float(cfg_scale),
                        "sigma_shift": float(sigma_shift),
                        "seed": int(seed),
                        "out": out_path,
                    },
                    tail,
                )
            arr = np.load(out_path)
        finally:
            _STATE["progress"] = None
            try:
                os.remove(out_path)
            except OSError:
                pass

        waveform = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=torch.float32)
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        return ({"waveform": waveform, "sample_rate": int(data["sample_rate"])},)


class MossSoundEffectV2Unload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "接生成节点的 audio。这根线用来排在生成之后，音频本身不会被改",
                    },
                ),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "unload"
    CATEGORY = "audio/moss"
    DESCRIPTION = "停掉推理进程并释放显存。把生成节点的 audio 接过来，卸载才会排在生成之后。"

    def unload(self, audio):
        with _STATE["lock"]:
            proc = _STATE["proc"]
            _stop(proc if isinstance(proc, subprocess.Popen) else None)
            _STATE["key"] = None
            _STATE["proc"] = None
        return ()


NODE_CLASS_MAPPINGS = {
    "MossSoundEffectV2Loader": MossSoundEffectV2Loader,
    "MossSoundEffectV2Generate": MossSoundEffectV2Generate,
    "MossSoundEffectV2Unload": MossSoundEffectV2Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossSoundEffectV2Loader": "MOSS SoundEffect v2 加载",
    "MossSoundEffectV2Generate": "MOSS SoundEffect v2 生成",
    "MossSoundEffectV2Unload": "MOSS SoundEffect v2 卸载",
}
