"""Hard offline mode.

The app has no networking code of its own. As an extra guarantee, `enable()` patches Python's socket layer so
that *any* attempt to reach a non-local address (by the app or by any library it uses) fails immediately.
Loopback (127.0.0.1 / ::1) stays allowed because some libraries use it internally for inter-thread signalling.
"""

from __future__ import annotations

import ipaddress
import os
import socket

from . import APP_NAME


class NetworkBlockedError(ConnectionRefusedError):
    pass


_ENABLED = False

_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}


def _is_local(host) -> bool:
    if host is None:
        return False
    if isinstance(host, (bytes, bytearray)):
        host = host.decode(errors="ignore")
    host = str(host).strip().strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _check(family, address) -> None:
    if family == getattr(socket, "AF_UNIX", object()):
        return
    host = address[0] if isinstance(address, tuple) and address else address
    if not _is_local(host):
        raise NetworkBlockedError(f"{APP_NAME} runs offline: blocked a network connection to {host!r}")


def enable() -> None:
    """Block all outgoing network connections from this process (idempotent)."""
    global _ENABLED
    if _ENABLED:
        return
    os.environ.update(_OFFLINE_ENV)

    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    orig_sendto = socket.socket.sendto
    orig_getaddrinfo = socket.getaddrinfo
    orig_gethostbyname = socket.gethostbyname
    orig_gethostbyname_ex = socket.gethostbyname_ex

    def connect(self, address):
        _check(self.family, address)
        return orig_connect(self, address)

    def connect_ex(self, address):
        _check(self.family, address)
        return orig_connect_ex(self, address)

    def sendto(self, data, *args):
        if args:
            _check(self.family, args[-1])
        return orig_sendto(self, data, *args)

    def getaddrinfo(host, *args, **kwargs):
        if host is not None and not _is_local(host):
            raise socket.gaierror(socket.EAI_NONAME, f"{APP_NAME} runs offline: name lookup blocked for {host!r}")
        return orig_getaddrinfo(host, *args, **kwargs)

    def gethostbyname(host):
        if not _is_local(host):
            raise socket.gaierror(socket.EAI_NONAME, f"{APP_NAME} runs offline: name lookup blocked for {host!r}")
        return orig_gethostbyname(host)

    def gethostbyname_ex(host):
        if not _is_local(host):
            raise socket.gaierror(socket.EAI_NONAME, f"{APP_NAME} runs offline: name lookup blocked for {host!r}")
        return orig_gethostbyname_ex(host)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname
    socket.gethostbyname_ex = gethostbyname_ex
    _ENABLED = True


def is_enabled() -> bool:
    return _ENABLED


def verify() -> bool:
    """True when a connection attempt to a public address is refused by the guard."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.settimeout(1)
            s.connect(("1.1.1.1", 443))
        finally:
            s.close()
    except NetworkBlockedError:
        return True
    except OSError:
        return False
    return False
