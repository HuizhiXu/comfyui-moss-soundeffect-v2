"""MOSS-SoundEffect v2 的 ComfyUI 节点。

ComfyUI 启动时导入这个文件，所以这里不能加载模型，也不能依赖
``descript-audiotools``。推理放在加载节点所填的 Python 3.12 进程里，
两边用一行一个 JSON 通信。

一个类要变成节点，需要这四样：

- ``INPUT_TYPES``：左侧插座和控件。类型名是字符串，两边对得上就能连线。
- ``RETURN_TYPES``：右侧插座。``MOSS_SFX_PIPE`` 传递解释器、仓库和模型参数。
  ``AUDIO`` 是 ComfyUI 内置波形：``{"waveform": Tensor[B, C, T], "sample_rate": int}``。
- ``FUNCTION``：真正执行的方法名。
- ``CATEGORY``：右键菜单里的分组。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path

_WORKER = Path(__file__).resolve().with_name("worker.py")
_STATE: dict[str, object] = {
    "key": None,
    "proc": None,
    "tail": deque(maxlen=40),
    "lock": threading.Lock(),
}


def _pump_stderr(proc: subprocess.Popen[str], tail: deque[str]) -> None:
    assert proc.stderr is not None
    for line in proc.stderr:
        tail.append(line)
        print(f"[moss-soundeffect] {line}", end="", flush=True)


def _stop(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    if proc.stdin is not None:
        try:
            proc.stdin.close()
        except OSError:
            pass
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


def _request(proc: subprocess.Popen[str], payload: dict, tail: deque[str]) -> dict:
    if proc.poll() is not None:
        raise RuntimeError("推理进程已退出。\n" + "".join(tail))
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("推理进程没有返回结果。\n" + "".join(tail))
    data = json.loads(line)
    if not data.get("ok"):
        raise RuntimeError(str(data.get("error") or "推理失败") + "\n" + "".join(tail))
    return data


def _ensure_worker(spec: dict) -> subprocess.Popen[str]:
    key = (
        spec["python"],
        spec["moss_root"],
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
    if spec["moss_root"]:
        previous = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = spec["moss_root"] + (os.pathsep + previous if previous else "")

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
        start_new_session=True,
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
        return {
            "required": {
                "python": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Python 3.12 解释器，例如 /path/to/envs/moss-soundeffect-v2/bin/python",
                    },
                ),
                "moss_root": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "MOSS-TTS 仓库根目录。该环境里已经 pip install -e moss_soundeffect_v2 时留空",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "OpenMOSS-Team/MOSS-SoundEffect-v2.0",
                        "tooltip": "Hugging Face 仓库名，或那个环境能读到的本地模型目录",
                    },
                ),
                "device": (
                    "STRING",
                    {
                        "default": "auto",
                        "tooltip": "auto、cpu、cuda 或 cuda:1",
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
            }
        }

    RETURN_TYPES = ("MOSS_SFX_PIPE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "audio/moss"
    DESCRIPTION = "记下独立的 Python 3.12 解释器并在那个进程里加载模型。相同参数会复用已启动的进程。"

    def load(self, python, moss_root, model, device, dtype, vram_management):
        executable = Path(str(python).strip()).expanduser()
        if not executable.is_file():
            raise FileNotFoundError(
                "填写 Python 3.12 解释器的路径。这个路径写在节点上，不读本机环境变量。"
            )
        if not os.access(executable, os.X_OK):
            raise PermissionError(f"解释器不可执行：{executable}")

        root = str(moss_root).strip()
        if root:
            root_path = Path(root).expanduser()
            package = root_path / "moss_soundeffect_v2" / "__init__.py"
            if not package.is_file():
                raise FileNotFoundError(
                    f"moss_root 里没有 moss_soundeffect_v2：{root_path}"
                )
            root = str(root_path)

        model_name = str(model).strip()
        if not model_name:
            raise ValueError("model 不能为空")

        spec = {
            "python": str(executable),
            "moss_root": root,
            "model": model_name,
            "device": str(device).strip() or "auto",
            "dtype": dtype,
            "vram": vram_management,
        }
        with _STATE["lock"]:
            _ensure_worker(spec)
        return (spec,)


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
    DESCRIPTION = "把提示词交给已经启动的 Python 3.12 进程，取回 48 kHz 单声道音频。第一次生成会编译 DiT。"

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
        try:
            with _STATE["lock"]:
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
            try:
                os.remove(out_path)
            except OSError:
                pass

        waveform = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype=torch.float32)
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        return ({"waveform": waveform, "sample_rate": int(data["sample_rate"])},)


NODE_CLASS_MAPPINGS = {
    "MossSoundEffectV2Loader": MossSoundEffectV2Loader,
    "MossSoundEffectV2Generate": MossSoundEffectV2Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossSoundEffectV2Loader": "MOSS SoundEffect v2 加载",
    "MossSoundEffectV2Generate": "MOSS SoundEffect v2 生成",
}
