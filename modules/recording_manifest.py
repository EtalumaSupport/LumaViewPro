# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.

"""The host provenance a recording's manifest carries.

Both recording legs write it into their manifest's ``provenance.host``
(``VideoRecordingEngine``'s manifest_extra). Its shape mirrors the char
tool's manifest provenance, so manifests across LVP recordings and char
runs are comparable. Kept Kivy-free so it can be tested directly.
"""

from __future__ import annotations

import platform
import socket


def gather_host_provenance() -> dict:
    """Return the host fingerprint dict.

    Mirrors the char tool's provenance schema. Failures fall back to
    empty strings so the manifest write doesn't abort.
    """
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = ''
    try:
        os_platform = platform.platform()
    except Exception:
        os_platform = ''
    try:
        cpu_model = platform.processor() or platform.machine()
    except Exception:
        cpu_model = ''
    try:
        py_version = platform.python_version()
    except Exception:
        py_version = ''
    return {
        'hostname': hostname,
        'os_platform': os_platform,
        'cpu_model': cpu_model,
        'python_version': py_version,
    }
