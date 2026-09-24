"""ComfyUI-Manager 安装时调用。

探测或创建推理环境。失败时不让整个节点安装失败，加载节点第一次执行会再试。
"""

from __future__ import annotations

from bootstrap import resolve_python


def install() -> None:
    try:
        python = resolve_python("")
    except Exception as exc:
        print(f"[moss-soundeffect] 推理环境还没准备好：{exc}", flush=True)
        print("[moss-soundeffect] 节点已安装。第一次运行加载节点时会再试。", flush=True)
        return
    print(f"[moss-soundeffect] 推理解释器：{python}", flush=True)


if __name__ == "__main__":
    install()
