"""Probe whether PyTorch can load (DLL / install issues vs missing package)."""

from __future__ import annotations


def pytorch_load_probe() -> tuple[bool, str | None]:
    """
    Returns ``(True, None)`` if ``import torch`` succeeds.

    On Windows Store Python builds, DLL init often fails with ``OSError: [WinError 1114]``
    loading ``c10.dll`` — dense retrieval and sentence-transformers cannot run until fixed.
    """
    try:
        import torch  # noqa: F401

        _ = torch.__version__
    except Exception as exc:
        return False, str(exc)
    return True, None
