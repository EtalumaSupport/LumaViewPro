# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A simulated Lumascope Classic FX2: the device behind the real FX2 drivers.

A simulated LS620 or LS560 runs the production ``FX2Camera`` and
``FX2LEDController`` over this device, so what an integrator sees is the
FX2's: the MT9P031's window, exposure and gain as its registers hold them,
LED commands as the peripheral receives them, frames at the sensor's period
arriving a transfer at a time as the wire delivers them, and a black field
unless the peripheral holds a channel lit.

The device answers only what the driver sends on the wire: the firmware
upload on the bootloader, then ``VR_I2C_WRITE`` to the sensor or the LED
peripheral and the stream's start and stop on the running firmware. Any
other request, or an I2C address it does not model, raises, so a new driver
request cannot pass unmodelled.

It imports no USB library: the transport below stands where the platform
transports stand, behind the same connection.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np

from lvp_logger import logger
from drivers.fx2driver import (
    COLUMN_SIZE_OVER_WIDTH,
    FRAME_DELIM,
    I2C_LED,
    I2C_SENSOR,
    ISO_NUM_PACKETS,
    ISO_TRANSACTION_SIZE,
    PID_APP,
    PID_BOOT,
    REG_COL_SIZE,
    REG_EXPOSURE,
    REG_GLOBAL_GAIN,
    REG_RESET,
    REG_ROW_SIZE,
    VR_ANCHOR_DLD,
    VR_I2C_WRITE,
    VR_START_STREAMING,
    VR_STOP_STREAMING,
    _ByteStream,
    _FX2Connection,
    _register_to_gain_db,
    exposure_s,
    frame_layout,
    frame_time_s,
    parse_intel_hex,
)
from drivers.simulated_specimen import specimen_frames

TIMINGS = ('instant', 'realistic')

# Upload to re-enumeration on the bench: 6.5 s on three launches (an LS620
# twice, an LS720), macOS, 2026-09-30.
REENUMERATE_S = 6.5

# The exposure at which the specimen field renders at its own grey levels
# with unity gain: the driver's default, so a simulated FX2 at its defaults
# shows the field as the other simulated camera does. A model choice, not a
# measurement.
_REFERENCE_EXPOSURE_S = 0.050

# The FX2 firmware's I2C handler writes at most this many bytes of a request.
_I2C_MAX_BYTES = 3

# An ISO transfer completes every 256 microframes of 125 us, carrying the
# bytes that arrived in that time: the unit the driver's reader appends. A
# whole frame delivered at once is not what the wire does, and it leaves the
# grab loop re-reading a frame's worth of bytes with no delimiter after it
# until the next burst; a transfer always full stretches its gap instead,
# and a small window's frames arrive seconds apart in bunches.
TRANSFER_S = ISO_NUM_PACKETS * 125e-6

_LED_PREAMBLE = 0xFF
_LED_CHANNELS = (ord('A'), ord('B'), ord('C'), ord('D'))


def bytes_per_transfer(w: int, h: int, frame_period_s: float) -> float:
    """The bytes one transfer carries, frames back to back.

    The stream is continuous: one frame's bytes fill one frame period, and
    the next frame's delimiter follows its last row.
    """
    frame_bytes = len(FRAME_DELIM) + frame_layout(w, h).frame_bytes
    return frame_bytes / frame_period_s * TRANSFER_S


# Power-on values of the registers the model reads.
_POWER_ON = {
    REG_ROW_SIZE: 0x0797,
    REG_COL_SIZE: 0x0A1F,
    REG_EXPOSURE: 0x0797,
    REG_GLOBAL_GAIN: 0x0008,
}


class _Mt9p031:
    """The sensor's registers, as the driver writes them."""

    def __init__(self):
        self.registers = dict(_POWER_ON)

    def write(self, data: bytes) -> None:
        if len(data) != 3:
            raise ValueError(
                f'MT9P031 write of {len(data)} bytes; the driver sends [reg, high, low]'
            )
        reg, high, low = data
        self.registers[reg] = (high << 8) | low
        if reg == REG_RESET and self.registers[reg] & 1:
            # A soft reset (DS p21) returns the registers to their power-on
            # values; none the model reads is among those it keeps.
            self.registers.update(_POWER_ON)

    def window(self) -> tuple[int, int]:
        """The window the frames carry: the width the driver's Column_Size is for, H less two.

        DS Table 8: H = Row_Size + 1, of which ``frame_layout`` stores all but two.
        """
        output_h = self.registers[REG_ROW_SIZE] + 1
        return self.registers[REG_COL_SIZE] - COLUMN_SIZE_OVER_WIDTH, output_h - 2

    def exposure_s(self) -> float:
        """The integration the shutter width gives at the window's row time."""
        return exposure_s(self.registers[REG_EXPOSURE], self.registers[REG_COL_SIZE])

    def frame_period_s(self) -> float:
        """The sensor's frame time for the window and shutter width it holds."""
        return frame_time_s(
            self.registers[REG_COL_SIZE],
            self.registers[REG_ROW_SIZE],
            self.registers[REG_EXPOSURE],
        )

    def gain(self) -> float:
        """The linear gain the register encodes."""
        return _register_to_gain_db(self.registers[REG_GLOBAL_GAIN])[0]


class _LedPeripheral:
    """The Classic LED peripheral: one command is ``[0xFF, channel, brightness]``.

    A brightness byte of 0xFF is a new preamble, so the command it ends is
    lost and the channel keeps what it held.
    """

    def __init__(self):
        self.brightness = dict.fromkeys(_LED_CHANNELS, 0)
        self.commands: list[tuple[str, int]] = []
        self.lost_commands = 0
        self._expecting = 'preamble'
        self._channel: int | None = None

    def receive(self, byte: int) -> None:
        if byte == _LED_PREAMBLE:
            if self._expecting == 'brightness':
                self.lost_commands += 1
            self._expecting = 'channel'
            return
        if self._expecting == 'channel':
            if byte not in self.brightness:
                raise ValueError(f'LED peripheral has no channel 0x{byte:02X}')
            self._channel = byte
            self._expecting = 'brightness'
            return
        if self._expecting == 'brightness':
            self.brightness[self._channel] = byte
            self.commands.append((chr(self._channel), byte))
            self._expecting = 'preamble'
            return
        raise ValueError(f'LED peripheral got 0x{byte:02X} outside a command')

    def lit(self) -> bool:
        return any(self.brightness.values())


class SimulatedFX2Device:
    """One simulated FX2, from its bootloader to its stream.

    Args:
        timing: ``'instant'`` re-enumerates as soon as the upload ends;
            ``'realistic'`` waits the bench's ``REENUMERATE_S``.
    """

    def __init__(self, *, timing: str = 'instant'):
        if timing not in TIMINGS:
            raise ValueError(f'timing mode {timing!r} is not one of {TIMINGS}')
        self._timing = timing
        self._lock = threading.Lock()
        self.pid: int | None = PID_BOOT
        self.uploads = 0
        self._in_reset = False
        self._image = bytearray()
        self.sensor = _Mt9p031()
        self.leds = _LedPeripheral()
        self._sink: Callable[[bytes], None] | None = None
        self._streaming = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._specimen: dict[tuple[int, int], list[np.ndarray]] = {}
        self._frame_index = 0
        self._pending: deque[bytes] = deque()
        self.extra_rows = 0
        # Every Nth frame's delimiter lost, or written as other bytes: the
        # two ways the shipped firmware fails it at some windows. 0 is never.
        self.lost_delimiter_every = 0
        self.wrong_delimiter_every = 0
        self._frames_sent = 0

    # -- the wire ------------------------------------------------------------

    def vendor_out(self, request: int, value: int, index: int, data: bytes) -> int:
        """A vendor OUT request; returns the bytes written."""
        if self.pid == PID_BOOT:
            if request != VR_ANCHOR_DLD:
                raise ValueError(f'the FX2 bootloader has no request 0x{request:02X}')
            self._load(value, bytes(data))
            return len(data)
        if self.pid != PID_APP:
            raise RuntimeError('the simulated FX2 is not on the bus')
        if request == VR_I2C_WRITE:
            written = bytes(data[:_I2C_MAX_BYTES])
            if index == I2C_SENSOR:
                self.sensor.write(written)
            elif index == I2C_LED:
                for byte in written:
                    self.leds.receive(byte)
            else:
                raise ValueError(f'the simulated FX2 models no I2C device at 0x{index:02X}')
            return len(written)
        if request == VR_START_STREAMING:
            self._start()
            return 0
        if request == VR_STOP_STREAMING:
            self.stop()
            return 0
        raise ValueError(f'the simulated FX2 firmware has no request 0x{request:02X}')

    def _load(self, address: int, data: bytes) -> None:
        """The bootloader's 0xA0: hold or release the 8051, or write its RAM."""
        if address == 0xE600:
            if data == b'\x01':
                self._in_reset = True
                self._image = bytearray()
            elif data == b'\x00':
                self._release()
            else:
                raise ValueError(f'CPUCS write {data!r}')
            return
        if not self._in_reset:
            raise RuntimeError('firmware written to the FX2 while its 8051 runs')
        if address != len(self._image):
            raise ValueError(
                f'firmware chunk at 0x{address:04X}, expected 0x{len(self._image):04X}'
            )
        self._image.extend(data)

    def _release(self) -> None:
        """Run the uploaded image, which must be the shipped firmware, whole."""
        shipped, end = parse_intel_hex(_FX2Connection.find_firmware_path())
        if bytes(self._image) != shipped[:end]:
            raise RuntimeError(
                f'the simulated FX2 received {len(self._image)} bytes that are not '
                f'the shipped firmware ({end} bytes)'
            )
        self._in_reset = False
        self.uploads += 1
        self.pid = None  # off the bus while it re-enumerates
        if self._timing == 'instant':
            self.pid = PID_APP
        else:
            timer = threading.Timer(REENUMERATE_S, self._enumerate)
            timer.daemon = True
            timer.start()

    def _enumerate(self) -> None:
        self.pid = PID_APP

    # -- the stream ----------------------------------------------------------

    def attach(self, sink: Callable[[bytes], None]) -> None:
        """Where the ISO pipe delivers, one packet a call: the transport opens it before START."""
        self._sink = sink

    def detach(self) -> None:
        self._sink = None

    def _start(self) -> None:
        if self._sink is None:
            raise RuntimeError('VR_START_STREAMING with no ISO pipe open')
        with self._lock:
            if self._streaming.is_set():
                return
            self._streaming.set()
            self._stop.clear()
            self._thread = threading.Thread(target=self._stream_loop, daemon=True)
            self._thread.start()

    def halt(self) -> None:
        """The stream stops delivering, with no word to the host: no STOP, no error.

        What the bench saw at an unplug. The frame thread ends on its own; the
        sink stays attached, as the host's pipe does.
        """
        self._streaming.clear()
        self._stop.set()

    def stop(self) -> None:
        """Stop the frame clock; the device stays on the bus."""
        with self._lock:
            self._streaming.clear()
            self._stop.set()
            thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def frame(self) -> bytes:
        """The next frame's bytes as the wire carries them, without the delimiter.

        The window, exposure and gain are read here, at the frame's start, so
        a change takes effect on the next frame. The first and last output
        rows are sensor rows the parser does not store, as on the wire.
        ``extra_rows`` rows of zeros after the frame make it the wrong
        length for its window, the shape the parser counts as shifted. The
        columns a row carries beyond the window trail it, as on the wire; the
        simulator renders the specimen only in the window and leaves them 0.
        """
        w, h = self.sensor.window()
        layout = frame_layout(w, h)
        rows = np.zeros((h + 2, layout.stride), dtype=np.uint8)
        rows[:, layout.column : layout.column + w] = self._pixels(w, h + 2)
        # The skip is one byte longer than a row. What that byte carries is
        # unmeasured; the parser never reads it.
        first = rows[0].tobytes() + bytes(layout.skip - layout.stride)
        extra = bytes(layout.stride * self.extra_rows)
        return b''.join((first, rows[1:].tobytes(), extra))

    def packets(self) -> list[bytes]:
        """The next frame as ISO packets: the delimiter, then the frame.

        The frame goes in packets of two transactions, the size the bench
        saw most, and its last bytes, fewer than that and not a whole
        transaction, go as the short packet the firmware commits at a
        frame's end. The delimiter comes alone, as on the wire; the faults
        drop it or write other bytes in its place.
        """
        self._frames_sent += 1
        n = self._frames_sent
        out = []
        if self.lost_delimiter_every and n % self.lost_delimiter_every == 0:
            pass
        elif self.wrong_delimiter_every and n % self.wrong_delimiter_every == 0:
            # A pixel byte where the delimiter's first byte belongs, as the
            # bench captured it.
            out.append(b'\x29' + FRAME_DELIM[1:])
        else:
            out.append(FRAME_DELIM)
        body = self.frame()
        step = 2 * ISO_TRANSACTION_SIZE
        out.extend(body[i : i + step] for i in range(0, len(body), step))
        return out

    def _pixels(self, w: int, h: int) -> np.ndarray:
        """The specimen field as this sensor sees it: black unless an LED is lit."""
        if not self.leds.lit():
            return np.zeros((h, w), dtype=np.uint8)
        frames = self._specimen.get((w, h))
        if frames is None:
            frames = self._specimen[(w, h)] = specimen_frames(h, w)
        field = frames[self._frame_index % len(frames)]
        self._frame_index += 1
        scale = self.sensor.exposure_s() / _REFERENCE_EXPOSURE_S * self.sensor.gain()
        return np.clip(field.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    def _stream_loop(self) -> None:
        """Deliver the frames' packets back to back, a transfer every ``TRANSFER_S``.

        Each transfer carries the packets whose bytes have come due by then;
        a packet not yet due waits for the next. Each wait runs to a deadline
        kept from the stream's start, so the time spent building a frame is
        inside its period, not added to it.

        A frame's bytes come at the rate of the window and shutter it was
        started with: the sensor finishes the frame it is reading in that
        frame's period, and a change takes effect on the next.
        """
        self._pending = deque()
        owed = 0.0
        rate = 0.0
        deadline = time.monotonic()
        while self._streaming.is_set():
            sink = self._sink
            if sink is None:
                return
            if not self._pending:
                rate = self._start_frame()
            owed += rate
            while True:
                if not self._pending:
                    rate = self._start_frame()
                if len(self._pending[0]) > owed:
                    break
                packet = self._pending.popleft()
                owed -= len(packet)
                sink(packet)
            deadline += TRANSFER_S
            if self._stop.wait(max(0.0, deadline - time.monotonic())):
                return

    def _start_frame(self) -> float:
        """Queue the next frame's packets; return the bytes a transfer carries of it."""
        w, h = self.sensor.window()
        rate = bytes_per_transfer(w, h, self.sensor.frame_period_s())
        self._pending.extend(self.packets())
        return rate


class SimulatedFX2Transport:
    """The transport the connection talks to in place of a USB library."""

    def __init__(self, device: SimulatedFX2Device):
        self.device = device
        self._stream: _ByteStream | None = None
        self._on_gone: Callable[[], None] | None = None

    def find(self, pid: int) -> SimulatedFX2Device | None:
        return self.device if self.device.pid == pid else None

    def describe(self, dev: SimulatedFX2Device) -> str:
        return f'simulated FX2 idProduct=0x{dev.pid:04X}'

    def write_to(
        self, dev: SimulatedFX2Device, request: int, value: int, index: int, data: bytes
    ) -> None:
        dev.vendor_out(request, value, index, data)

    def open(self, dev: SimulatedFX2Device) -> None:
        pass  # nothing to claim

    def control_out(self, request: int, value: int, index: int, data: bytes, timeout: int) -> int:
        return self.device.vendor_out(request, value, index, data)

    def start_stream(self, stream: _ByteStream, on_gone: Callable[[], None]) -> None:
        # A transfer fails here only through the faults below.
        self._stream = stream
        self._on_gone = on_gone
        self.device.attach(stream.packet)
        self.device.vendor_out(VR_START_STREAMING, 0, 0, b'')

    def stop_stream(self) -> None:
        # The real transports send STOP best-effort: a device that has gone
        # cannot take it, and the stop still releases the host's side.
        try:
            self.device.vendor_out(VR_STOP_STREAMING, 0, 0, b'')
        except RuntimeError:
            self.device.stop()
        self.device.detach()
        self._stream = None
        self._on_gone = None

    # -- faults --------------------------------------------------------------

    def unplug(self, *, resubmit_failures: int = 0) -> None:
        """The cable is pulled while the device streams.

        The bytes stop with no error and the device leaves the bus, as the
        bench's unplugs did. ``resubmit_failures`` transfers first fail and
        cannot be resubmitted because the device is gone -- the four the B8
        bench unplug logged; the earlier Stage 0 unplug logged none.
        """
        self.device.halt()
        self.device.pid = None
        for _ in range(resubmit_failures):
            if self._stream is not None:
                self._stream.fail()
            if self._on_gone is not None:
                self._on_gone()

    def go_silent(self) -> None:
        """The stream stops while the device stays on the bus."""
        self.device.halt()

    def misalign(self, rows: int = 1) -> None:
        """Every frame from the next one carries ``rows`` rows too many; 0 realigns.

        The device keeps streaming at its rate but the parser can frame
        nothing it sends: the shape of a stream that has lost its alignment
        to the window.
        """
        self.device.extra_rows = rows

    def lose_delimiters(self, every: int) -> None:
        """Every ``every``-th frame from now goes with no delimiter after the one before; 0 stops."""
        self.device.lost_delimiter_every = every

    def garble_delimiters(self, every: int) -> None:
        """Every ``every``-th frame from now has 4 other bytes in its delimiter's place; 0 stops."""
        self.device.wrong_delimiter_every = every

    def close(self) -> None:
        self.device.stop()
        self.device.detach()


class SimulatedFX2:
    """A simulated FX2 and the connection its two drivers share.

    Building it runs the driver's own bring-up against the device: the
    upload into the bootloader, the wait for re-enumeration, the open.
    """

    def __init__(self, *, timing: str = 'instant'):
        self.device = SimulatedFX2Device(timing=timing)
        self.connection = _FX2Connection(SimulatedFX2Transport(self.device))
        logger.info(f'[SimFX2    ] simulated FX2 up (timing={timing})')
