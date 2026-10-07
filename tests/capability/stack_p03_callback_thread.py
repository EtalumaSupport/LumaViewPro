"""Stack S4: on which thread a lane task's completion callback runs, in a host
that passed no dispatcher (a script, a REST server) against one that did (the
GUI, Kivy's clock) -- and what a callback that calls another lane's member
does in each.

    python tests/capability/stack_p03_callback_thread.py
"""

import sys
import threading
import time
import traceback

import harness
from harness import check, figure, report
from modules.kivy_utils import UiDispatcher
from modules.scope_session import ScopeSession
from modules.sequential_io_executor import IOTask
from tests.settings_fixtures import complete_settings


def _session(dispatcher):
    ScopeSession.set_ui_dispatcher(
        None if dispatcher is None else UiDispatcher(schedule=dispatcher, thread=None)
    )
    live = harness.live_dir('stack_p03')
    return ScopeSession.create(
        complete_settings(live_folder=str(live), microscope='LS850T'),
        simulate=True,
    )


def _probe(session, label):
    io = session.io_executor
    seen = {}
    done = threading.Event()

    def cb():
        seen['thread'] = threading.current_thread().name
        done.set()

    io.put(IOTask(action=lambda: None, callback=cb))
    done.wait(5)
    figure(f'S4.{label}.callback_thread', seen.get('thread'))

    # The callback calls a member of ANOTHER lane (the camera's).
    result = {}
    finished = threading.Event()

    def cross_lane():
        t = time.perf_counter()
        try:
            v = session.scope.imaging.set_exposure_ms(20.0)
            result['outcome'] = f'returned {v}'
        except Exception as ex:
            result['outcome'] = f'raised {type(ex).__name__}: {str(ex)[:90]}'
        result['s'] = round(time.perf_counter() - t, 3)
        finished.set()

    io.put(IOTask(action=lambda: None, callback=cross_lane))
    if finished.wait(5):
        figure(f'S4.{label}.callback_calling_camera_lane', result)
    else:
        figure(f'S4.{label}.callback_calling_camera_lane', 'blocked > 5 s')

    # The callback calls a member of ITS OWN lane (the LED's, on the IO lane).
    result2 = {}
    finished2 = threading.Event()

    def same_lane():
        t = time.perf_counter()
        try:
            v = session.scope.illumination.led_on('BF', 10)
            result2['outcome'] = f'returned {v}'
        except Exception as ex:
            result2['outcome'] = f'raised {type(ex).__name__}: {str(ex)[:90]}'
        result2['s'] = round(time.perf_counter() - t, 3)
        finished2.set()

    io.put(IOTask(action=lambda: None, callback=same_lane))
    if finished2.wait(5):
        figure(f'S4.{label}.callback_calling_own_lane', result2)
    else:
        figure(f'S4.{label}.callback_calling_own_lane', 'blocked > 5 s')


def main():
    def host_clock(fn, _dt):
        threading.Thread(
            target=lambda: fn(0), name='host-clock'
        ).start()  # Kivy's clock calls fn(dt)

    for label, dispatcher in (('headless', None), ('with_dispatcher', host_clock)):
        session = _session(dispatcher)
        try:
            _probe(session, label)
        except BaseException:
            traceback.print_exc()
            check('probe completed without an unexpected raise', False)
        finally:
            session.shutdown()
            ScopeSession.set_ui_dispatcher(None)
    check('S4 ran', True)
    harness.assert_no_ui()
    sys.exit(report())


if __name__ == '__main__':
    main()
