"""Start with Windows for the current user only (HKCU\\...\\Run) – no administrator rights needed."""

from __future__ import annotations

import sys

from . import APP_ID
from .paths import IS_WINDOWS

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --background'
    exe = sys.executable
    if exe.lower().endswith("python.exe"):
        exe = exe[:-10] + "pythonw.exe"
    return f'"{exe}" -m filesorter --background'


def is_enabled() -> bool:
    if not IS_WINDOWS:
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as k:
            winreg.QueryValueEx(k, APP_ID)
            return True
    except OSError:
        return False


def set_enabled(enabled: bool) -> bool:
    """Returns True on success (always False on non-Windows systems)."""
    if not IS_WINDOWS:
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            if enabled:
                winreg.SetValueEx(k, APP_ID, 0, winreg.REG_SZ, command())
            else:
                try:
                    winreg.DeleteValue(k, APP_ID)
                except FileNotFoundError:
                    pass
        return True
    except OSError:
        return False
