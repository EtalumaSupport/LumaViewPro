# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The host's request to stay awake: the one place LVP asks the operating
system to keep the computer, and its screen, awake while it is idle.

One call per platform: ``hold_awake`` takes the request and returns the
``AwakeHold`` that releases it. Each hold is the operating system's own
handle, never tied to the thread that took it -- a session takes it on the
thread that builds the session and releases it on the thread that closes
it -- and the operating
system drops it when the process exits, so a crash never leaves the
computer held awake. The screen is held on as well as the system: on a
Modern Standby PC the screen's idle timeout is what enters connected
standby, whatever system request a process holds, and inside it the
process runs throttled. The person's own Sleep, a closed lid or an update
restart still sleep the computer; only idle sleep and the idle screen
timeout wait.

- Windows: one power request (``PowerCreateRequest`` / ``PowerSetRequest``)
  holding ``PowerRequestDisplayRequired`` and
  ``PowerRequestSystemRequired``. Display-required also keeps the
  screensaver and the idle auto-lock from starting; system-required stays
  beside it, so a screen turned off anyway (by a policy, the lock screen's
  own timeout) still leaves the system up. Not ``SetThreadExecutionState``,
  which holds for the calling thread only, so a release on another thread
  would leave the first thread's request standing.
- macOS: two IOKit power assertions, ``PreventUserIdleDisplaySleep`` and
  ``PreventUserIdleSystemSleep`` (``caffeinate -di``), made in-process. The
  display assertion does not light a display already off, so the system
  assertion stays beside it.
- Linux: a ``systemd-inhibit --what=idle:sleep`` child: the idle inhibitor
  blocks logind's idle action, and the sleep inhibitor the auto-suspend a
  desktop routes through logind. Whether the screen blanks is the
  desktop's, not logind's, so it is not held here. logind's inhibitor is a
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

    Attributes:
        holds: What the operating system was asked to keep awake, in the
            backend's own words, for the log: what one platform holds is
            not what another does.
    """

    def __init__(self, release: Callable[[], None], holds: str):
        self.holds = holds
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
    """Ask the operating system to keep the computer and its screen awake while it is idle.

    On Linux the screen is the desktop's, so only sleep is held there.

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
    power_request_display_required = 0
    power_request_system_required = 1
    requests = (power_request_display_required, power_request_system_required)
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
    for request in requests:
        if not kernel32.PowerSetRequest(handle, request):
            error = ctypes.get_last_error()
            # Closing the handle drops every request set on it, so a refused
            # second set leaves nothing of the first standing.
            kernel32.CloseHandle(handle)
            return AwakeNotHeld('keep_awake_refused', ctypes.FormatError(error))

    def release() -> None:
        # Both clears are tried before the handle closes, so a refused first
        # clear never leaves the second request set.
        errors = []
        for request in requests:
            if not kernel32.PowerClearRequest(handle, request):
                errors.append(ctypes.get_last_error())
        kernel32.CloseHandle(handle)
        if errors:
            raise ctypes.WinError(errors[0])

    return AwakeHold(
        release,
        'the display and the system (PowerRequestDisplayRequired, PowerRequestSystemRequired)',
    )


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
    kinds = (b'PreventUserIdleDisplaySleep', b'PreventUserIdleSystemSleep')
    assertions: list[int] = []
    for kind_name in kinds:
        kind = cf.CFStringCreateWithCString(None, kind_name, utf8)
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
            # A refused second assertion releases the first, so nothing is
            # left held when the answer is that nothing was taken; a first
            # one IOKit will not release is said, since it stays held until
            # the process exits.
            detail = f'IOPMAssertionCreateWithName({kind_name.decode()}) returned {returned:#x}'
            for taken in assertions:
                released = iokit.IOPMAssertionRelease(taken)
                if released != 0:
                    detail += f'; the assertion already taken was not released ({released:#x})'
            return AwakeNotHeld('keep_awake_refused', detail)
        assertions.append(assertion.value)

    def release() -> None:
        # Both releases are tried before either refusal is raised.
        refused = [
            f'IOPMAssertionRelease returned {returned:#x}'
            for returned in [iokit.IOPMAssertionRelease(taken) for taken in assertions]
            if returned != 0
        ]
        if refused:
            raise OSError('; '.join(refused))

    return AwakeHold(
        release,
        'the display and the system (PreventUserIdleDisplaySleep, PreventUserIdleSystemSleep)',
    )


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
                '--what=idle:sleep',
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

    return AwakeHold(
        release, "idle and sleep (systemd-inhibit --what=idle:sleep); the screen is the desktop's"
    )
