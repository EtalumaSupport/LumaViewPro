# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The host's request to stay awake: the one place LVP asks the operating
system not to sleep while the computer is idle.

One call per platform: ``hold_awake`` takes the request and returns the
``AwakeHold`` that releases it. Each hold is the operating system's own
handle, never tied to the thread that took it -- a session takes it on the
thread that builds the session and releases it on the thread that closes
it -- and the operating
system drops it when the process exits, so a crash never leaves the
computer held awake. The screen may still turn off, and the person's own
Sleep, a closed lid or an update restart still sleep the computer: every
backend asks only that idle sleep wait.

- Windows: a power request (``PowerCreateRequest`` / ``PowerSetRequest``,
  ``PowerRequestSystemRequired``). Not ``SetThreadExecutionState``, which
  holds for the calling thread only, so a release on another thread would
  leave the first thread's request standing.
- macOS: an IOKit power assertion, ``PreventUserIdleSystemSleep``, the
  assertion ``caffeinate -i`` makes, made in-process.
- Linux: a ``systemd-inhibit --what=idle`` child. logind's inhibitor is a
  held D-Bus file descriptor and no D-Bus client is among LVP's
  dependencies, so the child holds it; it runs ``tail --pid`` on this
  process, so it exits with LVP.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class AwakeNotHeld:
    """Why the operating system's request to stay awake was not taken.

    The answer ``hold_awake`` returns in place of a hold, and what it hands
    ``on_lost``: never raised, since its one caller tells it and carries on.

    Attributes:
        reason: ``'keep_awake_unsupported'`` -- this host has no way to take
            it (a platform with no backend, a Linux without
            ``systemd-inhibit``); ``'keep_awake_refused'`` -- the operating
            system declined it, or a hold it had granted ended by itself.
        detail: The operating system's own words.
    """

    reason: str
    detail: str


class AwakeHold:
    """One request to stay awake, held until ``release``.

    ``release`` is idempotent: a second call does nothing, so the request's
    handle is never released twice (IOKit answers a second release of one
    assertion with ``kIOReturnBadArgument``).
    """

    def __init__(self, release: Callable[[], None]):
        self._release = release
        self._lock = threading.Lock()
        self._released = False

    def release(self) -> None:
        """Release the request; a hold already released stays released.

        Raises:
            OSError: the operating system did not release it.
        """
        with self._lock:
            if self._released:
                return
            self._released = True
        self._release()


def hold_awake(why: str, on_lost: Callable[[AwakeNotHeld], None]) -> AwakeHold | AwakeNotHeld:
    """Ask the operating system not to sleep while the computer is idle.

    Args:
        why: The words the operating system lists beside the request
            (``powercfg /requests``, ``pmset -g assertions``,
            ``systemd-inhibit --list``).
        on_lost: Called, on a thread of its own, if a hold the operating
            system granted ends without ``release`` -- only the Linux
            child can, when logind refuses it after it started.

    Returns:
        The hold, or ``AwakeNotHeld`` saying why the request was not taken.
    """
    if sys.platform == 'win32':
        return _hold_windows(why)
    if sys.platform == 'darwin':
        return _hold_macos(why)
    if sys.platform.startswith('linux'):
        return _hold_linux(why, on_lost)
    return AwakeNotHeld(
        'keep_awake_unsupported', f'LumaViewPro has no way to keep a {sys.platform} host awake'
    )


def _hold_windows(why: str) -> AwakeHold | AwakeNotHeld:
    import ctypes
    from ctypes import wintypes

    class _Detailed(ctypes.Structure):
        _fields_ = [
            ('LocalizedReasonModule', wintypes.HMODULE),
            ('LocalizedReasonId', wintypes.ULONG),
            ('ReasonStringCount', wintypes.ULONG),
            ('ReasonStrings', ctypes.POINTER(wintypes.LPWSTR)),
        ]

    class _Reason(ctypes.Union):
        _fields_ = (('Detailed', _Detailed), ('SimpleReasonString', wintypes.LPWSTR))

    class _ReasonContext(ctypes.Structure):
        _fields_ = [('Version', wintypes.ULONG), ('Flags', wintypes.DWORD), ('Reason', _Reason)]

    power_request_context_version = 0
    power_request_context_simple_string = 0x1
    power_request_system_required = 1
    invalid_handle = wintypes.HANDLE(-1).value

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.PowerCreateRequest.argtypes = [ctypes.POINTER(_ReasonContext)]
    kernel32.PowerCreateRequest.restype = wintypes.HANDLE
    kernel32.PowerSetRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
    kernel32.PowerSetRequest.restype = wintypes.BOOL
    kernel32.PowerClearRequest.argtypes = [wintypes.HANDLE, ctypes.c_int]
    kernel32.PowerClearRequest.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    context = _ReasonContext(power_request_context_version, power_request_context_simple_string)
    context.Reason.SimpleReasonString = why
    handle = kernel32.PowerCreateRequest(ctypes.byref(context))
    if not handle or handle == invalid_handle:
        return AwakeNotHeld('keep_awake_refused', ctypes.FormatError(ctypes.get_last_error()))
    if not kernel32.PowerSetRequest(handle, power_request_system_required):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(handle)
        return AwakeNotHeld('keep_awake_refused', ctypes.FormatError(error))

    def release() -> None:
        cleared = kernel32.PowerClearRequest(handle, power_request_system_required)
        error = ctypes.get_last_error()
        kernel32.CloseHandle(handle)
        if not cleared:
            raise ctypes.WinError(error)

    return AwakeHold(release)


def _hold_macos(why: str) -> AwakeHold | AwakeNotHeld:
    import ctypes
    import ctypes.util

    iokit = ctypes.cdll.LoadLibrary(ctypes.util.find_library('IOKit'))
    cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library('CoreFoundation'))
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    iokit.IOPMAssertionCreateWithName.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    iokit.IOPMAssertionCreateWithName.restype = ctypes.c_int32
    iokit.IOPMAssertionRelease.argtypes = [ctypes.c_uint32]
    iokit.IOPMAssertionRelease.restype = ctypes.c_int32

    utf8 = 0x08000100
    assertion_level_on = 255
    kind = cf.CFStringCreateWithCString(None, b'PreventUserIdleSystemSleep', utf8)
    name = cf.CFStringCreateWithCString(None, why.encode(), utf8)
    assertion = ctypes.c_uint32(0)
    try:
        returned = iokit.IOPMAssertionCreateWithName(
            kind, assertion_level_on, name, ctypes.byref(assertion)
        )
    finally:
        cf.CFRelease(kind)
        cf.CFRelease(name)
    if returned != 0:
        return AwakeNotHeld(
            'keep_awake_refused', f'IOPMAssertionCreateWithName returned {returned:#x}'
        )

    def release() -> None:
        returned = iokit.IOPMAssertionRelease(assertion.value)
        if returned != 0:
            raise OSError(f'IOPMAssertionRelease returned {returned:#x}')

    return AwakeHold(release)


def _hold_linux(why: str, on_lost: Callable[[AwakeNotHeld], None]) -> AwakeHold | AwakeNotHeld:
    inhibit = shutil.which('systemd-inhibit')
    if inhibit is None:
        return AwakeNotHeld(
            'keep_awake_unsupported', 'systemd-inhibit is not installed on this computer'
        )
    try:
        child = subprocess.Popen(
            [
                inhibit,
                '--what=idle',
                '--who=LumaViewPro',
                f'--why={why}',
                '--mode=block',
                'tail',
                f'--pid={os.getpid()}',
                '-f',
                '/dev/null',
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            # Its own process group, so the release ends systemd-inhibit and
            # the tail it runs together: a TERM to systemd-inhibit alone
            # would leave the tail running until LVP exits.
            start_new_session=True,
        )
    except OSError as ex:
        return AwakeNotHeld('keep_awake_refused', str(ex))
    released = threading.Event()

    # logind answers after the child has started, so a refusal arrives as
    # the child's exit: watched here, never waited for on the thread that
    # took the hold, which is building a session.
    def watch() -> None:
        _, stderr = child.communicate()
        if not released.is_set():
            detail = stderr.strip() or f'systemd-inhibit exited with status {child.returncode}'
            on_lost(AwakeNotHeld('keep_awake_refused', detail))

    threading.Thread(target=watch, name='LVP-KeepAwakeWatch', daemon=True).start()

    def release() -> None:
        released.set()
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    return AwakeHold(release)
