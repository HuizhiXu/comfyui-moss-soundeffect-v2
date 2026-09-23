"""MOSS-SoundEffect v2 推理进程。

由加载节点里填写的 Python 3.12 解释器启动，不由 ComfyUI 的 Python 导入。
标准输出只写一行一个 JSON。模型加载时的打印转到标准错误，避免和协议混在一起。
"""

from __future__ import annotations

import json
import sys
import traceback


def _reply(protocol, payload: dict) -> None:
    protocol.write(json.dumps(payload, ensure_ascii=False) + "\n")
    protocol.flush()


def _load(req: dict):
    import torch

    from moss_soundeffect_v2 import MossSoundEffectPipeline

    vram = {"auto": None, "on": True, "off": False}[req["vram"]]
    return MossSoundEffectPipeline.from_pretrained(
        req["model"],
        torch_dtype=getattr(torch, req["dtype"]),
        device=req["device"],
        enable_vram_management=vram,
    )


def _generate(pipe, req: dict) -> None:
    import numpy as np
    import torch

    wave = pipe(
        prompt=req["prompt"],
        seconds=float(req["seconds"]),
        num_inference_steps=int(req["steps"]),
        cfg_scale=float(req["cfg_scale"]),
        sigma_shift=float(req["sigma_shift"]),
        seed=int(req["seed"]),
        negative_prompt=str(req.get("negative_prompt") or ""),
    )
    arr = wave.detach().to(device="cpu", dtype=torch.float32).numpy()
    np.save(req["out"], arr)


def main() -> None:
    protocol = sys.stdout
    sys.stdout = sys.stderr
    pipe = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply(protocol, {"ok": False, "error": f"无效的 JSON：{exc}"})
            continue

        cmd = req.get("cmd")
        try:
            if cmd == "load":
                pipe = _load(req)
                _reply(protocol, {"ok": True, "sample_rate": int(pipe.sample_rate)})
            elif cmd == "generate":
                if pipe is None:
                    raise RuntimeError("模型还没有加载")
                _generate(pipe, req)
                _reply(
                    protocol,
                    {"ok": True, "sample_rate": int(pipe.sample_rate), "path": req["out"]},
                )
            elif cmd == "shutdown":
                _reply(protocol, {"ok": True})
                return
            else:
                raise RuntimeError(f"未知命令：{cmd}")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _reply(protocol, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            if cmd == "load":
                return


if __name__ == "__main__":
    main()
