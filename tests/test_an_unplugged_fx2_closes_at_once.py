# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""An FX2 whose stream is stopped closes at once, whatever its transfers do.

The stream's stop (``_LibusbTransport.stop_stream``) closes python-libusb1's
handle, and that close waits, with no end, for every transfer still in
flight. On Windows a Classic unplugged while it streams keeps a transfer that
never completes, and its cancel is refused (``LIBUSB_ERROR_NOT_FOUND``), so
LumaViewPro never closed. The stop now waits a bounded time for its transfers
and, for one still out at the bound, abandons the whole python-libusb1 chain
instead of closing it, so neither the stop nor the process's exit waits on
it. Transfers that do come back are closed as before.

Each case runs in a fresh interpreter, on python-libusb1's real classes
(the test process has a stub in their place), with only libusb's own calls
faked: a stand-in by design, since the hang is libusb's on Windows and no
simulated scope reaches python-libusb1.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# The child stops one stream and prints what it saw as JSON; the parent
# asserts it, so a failure names every value. A child that hangs, in the stop
# or at its exit, is killed at the parent's timeout.
_CHILD = r"""
import json, logging, sys, threading, time
import usb1
from usb1 import libusb1
from _lumascope.drivers import fx2driver

mode = sys.argv[1]
records = []
class _Keep(logging.Handler):
    def emit(self, record):
        records.append(record)
logging.getLogger('LVP').addHandler(_Keep())
logging.getLogger('LVP').setLevel(logging.DEBUG)

lock = threading.Lock()
pending = []        # (transfer pointer, status) the fake libusb completes next
order = []          # 'submit' and 'cancel', in the order libusb saw them
closed = []         # handles libusb closed
first_cancel = threading.Event()
resubmit_entered = threading.Event()
deliver_one = threading.Event()

def deliver():
    with lock:
        due, pending[:] = list(pending), []
    for transfer_p, status in due:
        transfer_p.contents.status = status
        usb1.USBTransfer._USBTransfer__callbackWrapper(transfer_p)

def handle_events_timeout(ctx_p, tv):
    if deliver_one.is_set():
        deliver_one.clear()
        with lock:
            pending.append((transfers[0]._USBTransfer__transfer, usb1.TRANSFER_COMPLETED))
    deliver()
    time.sleep(0.01)
    return 0

def handle_events(ctx_p):
    deliver()
    time.sleep(0.01)
    return 0

def submit(transfer_p):
    if mode == 'race' and started:
        # The resubmit of a callback already past its streaming check.
        resubmit_entered.set()
        first_cancel.wait(0.5)
    order.append('submit')
    return 0

def cancel(transfer_p):
    order.append('cancel')
    first_cancel.set()
    if mode == 'never':
        return libusb1.LIBUSB_ERROR_NOT_FOUND
    if mode == 'after_refused_cancel':
        # Windows 2026-11-25 10:56: the cancel refused, the transfer back
        # all the same, completing with its error.
        with lock:
            pending.append((transfer_p, usb1.TRANSFER_ERROR))
        return libusb1.LIBUSB_ERROR_NOT_FOUND
    with lock:
        pending.append((transfer_p, usb1.TRANSFER_CANCELLED))
    return 0

libusb1.libusb_handle_events_timeout = handle_events_timeout
libusb1.libusb_handle_events = handle_events
libusb1.libusb_submit_transfer = submit
libusb1.libusb_cancel_transfer = cancel
libusb1.libusb_control_transfer = lambda *a: libusb1.LIBUSB_ERROR_NO_DEVICE
libusb1.libusb_release_interface = lambda *a: libusb1.LIBUSB_ERROR_NO_DEVICE
libusb1.libusb_close = lambda handle_p: closed.append(handle_p)

class _Stream:
    def packet(self, data):
        pass
    def fail(self, why):
        pass

# The chain python-libusb1 builds for openByVendorIDAndProductID (through
# USBDevice.open): the context holds the device, the device the handle, the
# handle the transfers.
ctx = usb1.USBContext().open()
class _Device(usb1._LibUSB1Finalizer):
    pass
device = _Device()
device.close = ctx._getFinalizer(
    device, usb1.USBDevice.close, device_p=None, finalizer_dict=device._finalizer_dict,
    descriptor_list=[], libusb_unref_device=None, libusb_free_config_descriptor=None,
)
handle = usb1.USBDeviceHandle(
    context=ctx, handle=libusb1.libusb_device_handle_p(), device=device,
    getFinalizer=device._getFinalizer, can_close_device=False,
)

transport = fx2driver._LibusbTransport()
transport.find = lambda pid: None   # no device for the idle reopen
started = False
transfers = []
for _ in range(3):
    xfer = handle.getTransfer(iso_packets=4)
    xfer.setIsochronous(
        0x82, 512 * 4, callback=transport._iso_callback,
        iso_transfer_length_list=[512] * 4,
    )
    xfer.submit()
    transfers.append(xfer)
started = True
transport._ctx, transport._handle, transport._transfers = ctx, handle, transfers
transport._stream, transport._on_gone, transport._streaming = _Stream(), lambda: None, True
transport._event_thread = threading.Thread(target=transport._usb_event_loop, daemon=True)
transport._event_thread.start()

if mode == 'race':
    order.clear()
    deliver_one.set()
    assert resubmit_entered.wait(5), 'the event thread never resubmitted'

t0 = time.monotonic()
transport.stop_stream()
seconds = time.monotonic() - t0
cancels = [i for i, call in enumerate(order) if call == 'cancel']
print(json.dumps({
    'seconds': seconds,
    'errors': [r.getMessage() for r in records if r.levelno >= logging.ERROR],
    'submit_after_cancel': bool(cancels) and 'submit' in order[cancels[0]:],
    'handles_closed': len(closed),
    'released': transport._ctx is None and transport._handle is None,
}), flush=True)
"""


def _stop(mode: str) -> dict:
    """Stop one stream in a fresh interpreter; what it saw, once it has exited."""
    try:
        result = subprocess.run(
            [sys.executable, '-c', _CHILD, mode],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as hung:
        pytest.fail(f'the stop or the exit after it hung ({mode}): {hung.stdout!r} {hung.stderr!r}')
    assert result.returncode == 0, (result.stdout, result.stderr)
    # A finalizer left live behind a detached one fails its own assertion at exit.
    assert 'AssertionError' not in result.stderr, result.stderr
    return json.loads(result.stdout.splitlines()[-1])


def test_a_transfer_that_never_comes_back_is_abandoned_and_the_process_exits():
    seen = _stop('never')
    assert seen['seconds'] < 3.0, seen
    assert seen['handles_closed'] == 0, seen
    assert seen['released'], seen
    assert len(seen['errors']) == 1, seen
    assert '3 of 3' in seen['errors'][0], seen


def test_transfers_back_after_a_refused_cancel_are_closed_at_once():
    seen = _stop('after_refused_cancel')
    assert seen['seconds'] < 1.5, seen
    assert seen['handles_closed'] == 1, seen
    assert seen['released'], seen
    assert seen['errors'] == [], seen


def test_transfers_back_on_their_cancel_are_closed_at_once():
    seen = _stop('on_cancel')
    assert seen['seconds'] < 1.5, seen
    assert seen['handles_closed'] == 1, seen
    assert seen['released'], seen
    assert seen['errors'] == [], seen


def test_no_transfer_is_resubmitted_after_the_stop_cancels():
    seen = _stop('race')
    assert not seen['submit_after_cancel'], seen
