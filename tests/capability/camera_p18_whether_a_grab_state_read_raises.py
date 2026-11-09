"""Camera P18: whether a camera's grab-state read ever raises, through a binning change, a disconnect and a removal.

A grab-state read that fails raises `CameraReadFailedError`, with the SDK's
error as its cause, where it was once answered "not streaming". The paths
that meet the raise are traced, not seen: whether the SDK's own read
(`IsGrabbing()` on a Basler) ever raises is the camera's, and this probe
reads it. A worker samples the driver's grab state at about SAMPLE_HZ while
one event happens:

- `binning`: live view grabs, the binning goes to 2 and back to 1, each
  write stopping and restarting the grab;
- `disconnect`: live view grabs and the driver's `disconnect()` runs;
- `pull`: live view grabs and the camera leaves the bus. On hardware the
  probe prints `UNPLUG NOW` and the USB cable is pulled within WINDOW_S; on
  a scope whose camera reaches the host through the board, that is the
  scope's one cable, and the stage and LEDs leave with it. On the simulator
  `simulation.remove_camera()` stands in for the hand.

Each sample reads both the driver's `is_grabbing()`, which reads the removal
latch first and asks the SDK nothing once it is set, and its `_is_grabbing()`,
which asks the SDK whatever the latch holds: the first is what every caller
meets, the second is the SDK's own answer in the window before the latch and
after it. Each records its answer or the exception type with its cause's
type, whether the driver holds a camera, and the latch. A run of identical
samples is recorded once, with its first time and its count.

    python tests/capability/camera_p18_whether_a_grab_state_read_raises.py                     # all three, simulated
    python tests/capability/camera_p18_whether_a_grab_state_read_raises.py binning --hardware
    python tests/capability/camera_p18_whether_a_grab_state_read_raises.py disconnect --hardware
    python tests/capability/camera_p18_whether_a_grab_state_read_raises.py pull --hardware

On hardware each event is its own run, since a disconnected or removed
camera comes back only with a new session; the pull goes last. Nothing is
lit, nothing is captured, the stage does not move beyond the bring-up home,
and the binning is put back to 1.
"""

import sys
import threading
import time
import traceback

from harness import HARDWARE, check, figure, hardware_session, make_session, report

EVENTS = ('binning', 'disconnect', 'pull')
SAMPLE_HZ = 100.0
# Grab before the event, so frames are flowing, and after it, so the state
# the event left is read for a while.
BEFORE_S = 2.0
AFTER_S = 5.0
# The operator's window for the pull.
WINDOW_S = 120.0


def _answer(read):
    try:
        return repr(read())
    except Exception as e:
        cause = e.__cause__
        return f'{type(e).__name__}({type(cause).__name__ if cause else "-"})'


class _Sampler:
    """The driver's grab state, read on a worker until stopped, runs of one sample folded."""

    def __init__(self, driver):
        self._driver = driver
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self.runs = []  # [first_s, count, sample]
        self.samples = 0
        self._thread = threading.Thread(target=self._run, name='p18-sampler', daemon=True)

    def _sample(self):
        driver = self._driver
        return (
            _answer(driver.is_grabbing),
            _answer(driver._is_grabbing),
            driver.active is not None,
            driver.is_device_removed(),
        )

    def _run(self):
        period = 1.0 / SAMPLE_HZ
        while not self._stop.is_set():
            sample = self._sample()
            self.samples += 1
            if self.runs and self.runs[-1][2] == sample:
                self.runs[-1][1] += 1
            else:
                self.runs.append([round(time.monotonic() - self._t0, 3), 1, sample])
            self._stop.wait(period)

    def now(self):
        return round(time.monotonic() - self._t0, 3)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5.0)
        elapsed = time.monotonic() - self._t0
        rate = self.samples / elapsed if elapsed else 0.0
        figure('samples', self.samples)
        figure('sample rate, Hz', round(rate, 1))
        figure(
            'runs: [first s, count, (is_grabbing, _is_grabbing, holds a camera, latched)]',
            self.runs,
        )
        raised = sorted({a for _t, _n, s in self.runs for a in s[:2] if a not in ('True', 'False')})
        figure('raised', raised)
        check('the sampler ran at half its rate or more', rate >= SAMPLE_HZ / 2, f'{rate:.1f} Hz')
        return False


def _the_hand(session):
    if HARDWARE:
        print(f'UNPLUG NOW: pull the USB cable within {WINDOW_S:.0f} s', flush=True)
        return
    threading.Timer(0.5, session.scope.simulation.remove_camera).start()


def _event(session, event):
    scope = session.scope
    driver = scope._camera_driver
    figure('event', event)
    figure('camera', scope.capabilities.camera_model)
    scope.imaging.enable_streaming()
    with _Sampler(driver) as sampler:
        time.sleep(BEFORE_S)
        figure('event at s', sampler.now())
        if event == 'binning':
            try:
                scope.camera.set_binning(2)
                figure('binned 2 by s', sampler.now())
            finally:
                scope.camera.set_binning(1)
            figure('binned 1 by s', sampler.now())
        elif event == 'disconnect':
            driver.disconnect()
            figure('disconnected by s', sampler.now())
        else:
            _the_hand(session)
            until = time.monotonic() + (WINDOW_S if HARDWARE else 10.0)
            while not driver.is_device_removed() and time.monotonic() < until:
                time.sleep(0.01)
            latched = driver.is_device_removed()
            figure('latched by s', sampler.now() if latched else None)
            check('the removal was latched in the window', latched)
        time.sleep(AFTER_S)
    if event == 'binning':
        scope.imaging.disable_streaming()


def main():
    asked = [a for a in sys.argv[1:] if a in EVENTS]
    if HARDWARE:
        if len(asked) != 1:
            print(f'REFUSED: name one event of {EVENTS}; each is its own run on hardware.')
            sys.exit(2)
        with hardware_session() as (session, _runner):
            _event(session, asked[0])
        return
    for event in asked or EVENTS:
        session, _live = make_session(f'probe_grab_state_{event}')
        try:
            _event(session, event)
        finally:
            session.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        check('probe completed without an unexpected raise', False)
    sys.exit(report())
