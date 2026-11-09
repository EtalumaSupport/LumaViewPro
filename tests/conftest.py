# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Pytest configuration for LumaViewPro tests.

Conftest is loaded by pytest before any test module is collected, so the
mock installation here happens before any test file's imports. Test files
can therefore import driver modules at module level without each one
re-installing its own MagicMock deps.

Hardware-test opt-in flags
--------------------------
    --run-hardware        firmware/serial board hardware (legacy flag)
    --run-ids-hardware    real IDS Peak SDK + connected camera
    --run-pylon-hardware  real Pylon SDK + connected camera

When a hardware flag is set, the corresponding SDK is NOT mocked so the
real module loads. Hardware tests are gated by markers (`ids_hardware`,
`pylon_hardware`) -- see `pytest_collection_modifyitems` below.
"""

import faulthandler
import functools
import gc
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import psutil
import pytest

from _lumascope.drivers import host_clock

# Keep Kivy from writing anything to ~/.kivy/logs/ during tests. App code
# sets these in lumaviewpro.py + src/_lumascope/lvp_logger.py, but pytest may import Kivy
# before either runs, so set them here as well.
os.environ.setdefault('KIVY_NO_CONSOLELOG', '1')
os.environ.setdefault('KIVY_NO_FILELOG', '1')

# ---------------------------------------------------------------------------
# Path setup -- the checkout's root for `tests.` and `ui.`. Its src/ is first
# already (pytest's pythonpath), and every process the suite starts inherits
# it through PYTHONPATH, so a child runs this checkout's API whatever
# lumascope.pth the interpreter holds.
# ---------------------------------------------------------------------------
_CHECKOUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CHECKOUT)
os.environ['PYTHONPATH'] = os.pathsep.join(
    [os.path.join(_CHECKOUT, 'src'), *filter(None, [os.environ.get('PYTHONPATH')])]
)

# Typed pypylon stand-in (real handler bases + exception types). Needs the
# repo-root path insert above.
from tests import ids_peak_stub as _ids_peak_stub
from tests import pypylon_stub as _pypylon_stub
from tests import host_power_stub as _host_power_stub
from tests import usb_stub as _usb_stub
from tests.bench_results import bench_results_dir
from tests.scope_fakes import build_scope
from _lumascope.drivers.sim_wire.backend import runtime_platform


# ---------------------------------------------------------------------------
# Hardware-flag detection (must run before mock install)
# ---------------------------------------------------------------------------
# pytest_addoption hasn't been called yet at conftest import time, so we
# sniff sys.argv directly. Tolerant of `--flag` and `--flag=1` forms.
def _flag_in_argv(name):
    return any(a == name or a.startswith(f'{name}=') for a in sys.argv)


_HARDWARE_FLAG_MOCKS = {
    '--run-ids-hardware': [
        'ids_peak',
        'ids_peak.ids_peak',
        'ids_peak.ids_peak_ipl_extension',
        'ids_peak_ipl',
    ],
    '--run-pylon-hardware': [
        'pypylon',
        'pypylon.pylon',
        'pypylon.genicam',
    ],
    '--run-fx2-hardware': [
        'usb',
        'usb.core',
        'usb.util',
        'usb1',
    ],
}
_skip_mocks = set()
for _flag, _mods in _HARDWARE_FLAG_MOCKS.items():
    if _flag_in_argv(_flag):
        _skip_mocks.update(_mods)

# A run with any hardware flag is a bench run. It runs on the real logger, as
# it runs on the real SDK, so every log it writes, its driver log and its
# results file go to the one log folder, which the support zip carries. Any
# other run is the simulator suite, whose writes stay in temporary folders.
_BENCH_FLAGS = (
    '--run-hardware',
    '--run-ids-hardware',
    '--run-pylon-hardware',
    '--run-fx2-hardware',
    '--run-tmcm6110-hardware',
)
BENCH_RUN = any(_flag_in_argv(f) for f in _BENCH_FLAGS)
if BENCH_RUN:
    _skip_mocks.add('_lumascope.lvp_logger')


# ---------------------------------------------------------------------------
# Centralized mock installation
# ---------------------------------------------------------------------------
# Test files used to duplicate this block at module level. Now they don't
# have to -- conftest installs the union before any test is collected.
# Idempotent (uses setdefault) so files that still call install_mock_deps()
# are no-ops.


def install_mock_deps():
    """Install MagicMock stand-ins for the layers a test never has: the GUI
    (kivy), the camera SDKs, the USB bus, the network client (requests) and
    the user directories (platformdirs).

    Never a library the test runner itself reads: pytest-xdist takes its
    auto worker count from psutil.cpu_count(), and a mock there builds zero
    workers (tests/guards/test_psutil_is_real.py).

    Idempotent. Skips SDK mocks when the corresponding --run-*-hardware
    flag is set, so the real SDK can load.
    """
    deps = {
        # General heavy deps
        'platformdirs': MagicMock(),
        'requests': MagicMock(),
        'requests.structures': MagicMock(),
        'kivy': MagicMock(),
        'kivy.clock': MagicMock(),
        # EventLoop.status must read as a RUNNING loop: the pre-mainloop
        # popup detector fires on 'idle', and a bare MagicMock attribute
        # would compare unequal to every real value and false-negative --
        # or, gated the other way, mark every headless popup pre-mainloop.
        'kivy.base': MagicMock(EventLoop=MagicMock(status='started')),
        'kivy.app': MagicMock(),
        # FX2 / libusb -- skipped when --run-fx2-hardware is set. A typed
        # stub: the driver tells a timeout from a removal by these
        # libraries' exception classes.
        'usb': _usb_stub.usb,
        'usb.core': _usb_stub.core,
        'usb.util': _usb_stub.util,
        'usb1': _usb_stub.usb1,
        # Camera SDKs -- skipped when their --run-*-hardware flag is set.
        # pypylon gets a typed stub (real subclassable handler bases +
        # exception types) instead of a blanket MagicMock so the driver's
        # ImageHandler / _CameraRemovalHandler classes can be instantiated
        # and their callbacks driven directly in unit tests.
        'pypylon': _pypylon_stub.pypylon,
        'pypylon.pylon': _pypylon_stub.pylon,
        'pypylon.genicam': _pypylon_stub.genicam,
        # IDS peak gets typed node and exception classes, which the driver
        # tells a node and a failure by; the rest stays a MagicMock.
        'ids_peak': _ids_peak_stub.package,
        'ids_peak.ids_peak': _ids_peak_stub.ids_peak,
        'ids_peak.ids_peak_ipl_extension': MagicMock(),
        'ids_peak_ipl': MagicMock(),
    }
    for name, mock_mod in deps.items():
        if name in _skip_mocks:
            continue
        sys.modules.setdefault(name, mock_mod)

    _install_kivy_uix_stubs()
    # The host's sleep request: a test never holds the computer awake, and
    # every host answers alike.
    _host_power_stub.install()

    # The simulator suite runs the real logger with its files declined, so no
    # session opens the log folder, and what production saves beside the logs
    # (a diagnostics run's results) goes to the temporary folder: a simulated
    # result is never the instrument's. A bench run keeps its files and its
    # folder.
    if not BENCH_RUN:
        import _lumascope.lvp_logger

        _lumascope.lvp_logger.log_dir = tempfile.gettempdir()
        _lumascope.lvp_logger.start_logging(log_files=False)


class StubWidget:
    """Subclassable stand-in for every kivy.uix base a ui/ module inherits.

    A bare MagicMock cannot be subclassed, so widget modules would only
    import if some earlier test file had installed a real stub module --
    an import-order dependence that made whole test files pass or fail
    depending on which file ran first. Registering real ModuleType stubs
    here, before collection, removes the ordering.
    """

    def __init__(self, **kwargs):
        pass


def _install_kivy_uix_stubs():
    """Register stubs for every kivy submodule production code imports.

    The top-level `kivy` mock is not a package, so importing any
    UNregistered `kivy.x` submodule raises ModuleNotFoundError; the set
    below covers every `from kivy.<x>` in ui/, src/_lumascope/modules/, plugins/ and
    lumaviewpro.py. setdefault, like the MagicMock deps above: a test
    file that installs its own richer stub for one of these names keeps
    it.
    """
    for name in (
        'kivy.animation',
        'kivy.core',
        'kivy.core.image',
        'kivy.core.text',
        'kivy.core.window',
        'kivy.factory',
        'kivy.graphics',
        'kivy.graphics.opengl',
        'kivy.graphics.texture',
        'kivy.input',
        'kivy.lang',
        'kivy.lang.builder',
        'kivy.metrics',
        'kivy.properties',
        'kivy.uix',
    ):
        sys.modules.setdefault(name, MagicMock())

    # One name per `from kivy.uix.<mod> import <Base>` in ui/ (ui/sim_walk.py
    # included), src/_lumascope/modules/, plugins/, lumaviewpro.py -- keep in sync with that
    # import set.
    bases = {
        'accordion': ('Accordion', 'AccordionItem'),
        'behaviors': ('ButtonBehavior',),
        'boxlayout': ('BoxLayout',),
        'button': ('Button',),
        'dropdown': ('DropDown',),
        'floatlayout': ('FloatLayout',),
        'image': ('Image',),
        'label': ('Label',),
        'modalview': ('ModalView',),
        'popup': ('Popup',),
        'progressbar': ('ProgressBar',),
        'scatter': ('Scatter',),
        'scrollview': ('ScrollView',),
        'slider': ('Slider',),
        'spinner': ('Spinner', 'SpinnerOption'),
        'textinput': ('TextInput',),
        'togglebutton': ('ToggleButton',),
        'widget': ('Widget',),
    }
    for mod_name, class_names in bases.items():
        full_name = f'kivy.uix.{mod_name}'
        mod = ModuleType(full_name)
        for class_name in class_names:
            setattr(mod, class_name, StubWidget)
        sys.modules.setdefault(full_name, mod)

    # ui/sim_walk.py drives the window with Kivy's test touch; the class is a
    # stand-in here like every other Kivy name, so the real driver module
    # imports and a test patches the one member it asks.
    tests_pkg = ModuleType('kivy.tests')
    common = ModuleType('kivy.tests.common')
    common.UnitTestTouch = StubWidget
    sys.modules.setdefault('kivy.tests', tests_pkg)
    sys.modules.setdefault('kivy.tests.common', common)


# Run at conftest import time -- before any test file is collected.
install_mock_deps()

# A bench run owns its process's logging, as LumaViewPro does: importing the
# real logger configures nothing, so the run starts it, before any test.
if BENCH_RUN:
    from _lumascope.lvp_logger import start_program_logging

    start_program_logging()


def _install_host_absent_production_modules():
    """A production module this host cannot import gets an importable shell.

    ``src/_lumascope/drivers/winusb_iso.py`` imports ``ctypes.WinDLL`` at module scope, so on
    macOS and Linux the import raises; ``src/_lumascope/drivers/fx2driver.py`` reaches it
    only on Windows. A test of the WinUSB transport patches what the
    transport asks for, ``WinUsbIsoReader`` and ``device_bindings``, on the
    module, the same call on every host; here the module exists to be
    patched. The guard
    ``tests/guards/test_no_test_installs_a_module_stand_in.py`` keeps this
    the one place.
    """
    try:
        import _lumascope.drivers.winusb_iso
    except ImportError:
        import _lumascope.drivers

        shell = ModuleType('_lumascope.drivers.winusb_iso')
        shell.WinUsbIsoReader = None
        shell.device_bindings = None
        sys.modules.setdefault('_lumascope.drivers.winusb_iso', shell)
        # The dotted form of monkeypatch.setattr walks the package's attributes.
        _lumascope.drivers.winusb_iso = sys.modules['_lumascope.drivers.winusb_iso']


_install_host_absent_production_modules()


# ---------------------------------------------------------------------------
# A real serial port is unreachable from a test
# ---------------------------------------------------------------------------
# Both boards connect inside their constructors: SerialBoard enumerates with
# list_ports.comports and opens the match with serial.Serial. A test that
# reaches a real board therefore opens the bench's port when a scope is
# attached (two concurrent full-suite runs held both boards' ports and
# interrupted the motor firmware) and passes quietly through the no-port
# path when none is, so the suite behaved differently by what was plugged
# in. Unless --run-hardware is set, enumeration finds nothing and the open
# raises, and the TOUCH fails the test at its end: the driver registry's
# auto mode swallows a failed constructor into a null driver, so an
# exception alone would let the test go green on a null board, which is
# the quiet pass again.
SERIAL_REFUSAL_BANNER = 'REAL SERIAL PORT REACHED'
_serial_touches: list[tuple[str, str]] = []


def _install_serial_refusers():
    import serial
    import serial.tools.list_ports as list_ports

    real_serial = serial.Serial

    def refused_comports(*args, **kwargs):
        _serial_touches.append(('enumerate', 'serial.tools.list_ports.comports'))
        return []

    class RefusedSerial(real_serial):
        # A subclass, so a Mock(spec=serial.Serial) keeps the real methods;
        # only the open is intercepted. Refused in __new__, before an instance
        # exists, since pyserial's __del__ closes a half-built one and raises.
        def __new__(cls, *args, **kwargs):
            port = kwargs.get('port') or (args[0] if args else '?')
            _serial_touches.append(('open', str(port)))
            raise serial.SerialException(f'{SERIAL_REFUSAL_BANNER}: {port} (no --run-hardware)')

    list_ports.comports = refused_comports
    serial.Serial = RefusedSerial


if not _flag_in_argv('--run-hardware'):
    _install_serial_refusers()


# A real scope takes its installation's instance lock, on the port that
# installation's settings name. One built on the checkout's installation, or
# on a copy of its data/, takes the port a LumaViewPro, a REST server or a
# bench script on this machine holds: the shipped template's, or the
# checkout's own current.json's. The test then fails whenever one runs and
# passes when none does. Unless --run-hardware is set, such a take fails the
# test at its end, by name, whether or not the port was free; a test's real
# scope is built on `installation_of_its_own()`, whose port is 0.
SHIPPED_LOCK_BANNER = 'SHIPPED INSTANCE-LOCK PORT TAKEN'
_shipped_lock_takes: list[int] = []


def _install_shipped_lock_recorder():
    import _lumascope.modules.lumascope_api._lumascope as lumascope_module
    from _lumascope.modules.app_config import get_lvp_lock_port
    from _lumascope.modules.path_utils import get_source_root

    checkout = get_source_root()
    shipped = json.loads((checkout / 'data' / 'settings.json').read_text(encoding='utf-8'))
    template = shipped['lvp_lock_port']
    held_here = {template, get_lvp_lock_port(checkout)} - {0}
    real_take = lumascope_module.take_scope_lock

    @functools.wraps(real_take)
    def recorded_take(source_path):
        port = get_lvp_lock_port(source_path)
        if port in held_here:
            _shipped_lock_takes.append(port)
        return real_take(source_path)

    lumascope_module.take_scope_lock = recorded_take


if not _flag_in_argv('--run-hardware'):
    _install_shipped_lock_recorder()


def _shipped_lock_failure(item, call) -> str | None:
    if not _shipped_lock_takes:
        return None
    ports = sorted(set(_shipped_lock_takes))
    _shipped_lock_takes.clear()
    return (
        f'{SHIPPED_LOCK_BANNER}: {item.nodeid} built a real scope on port(s) {ports} during '
        f'{call.when}, the port a LumaViewPro or a bench session on this machine holds, so '
        'it fails whenever one runs. Build it on '
        'tests.settings_fixtures.installation_of_its_own(), whose port is 0.'
    )


def _serial_failure(item, call) -> str | None:
    if not _serial_touches:
        return None
    touches = list(_serial_touches)
    _serial_touches.clear()
    kind, detail = touches[0]
    return (
        f'{SERIAL_REFUSAL_BANNER}: {item.nodeid} reached the real serial layer during '
        f'{call.when} ({len(touches)} touch(es); first: {kind} {detail}). Build the scope '
        'with simulate=True or a simulated driver, or give a bare SerialBoard a port; '
        'a real port is opened only under --run-hardware.'
    )


# The checkout's data/ is an installation a person runs LumaViewPro from: its
# current.json is their live configuration. A test that writes there leaves
# the next launch on the test's settings, a live folder in a temporary
# directory pytest deletes among them. Every write into it, an open for
# writing or a rename onto it, fails the test that made it, by name; a test
# that needs an installation builds `installation_of_its_own()`. Subprocesses
# are not watched.
CHECKOUT_WRITE_BANNER = 'CHECKOUT INSTALLATION WRITTEN'
_checkout_writes: list[str] = []


def _install_checkout_write_recorder():
    from _lumascope.modules.path_utils import get_source_root

    data = os.path.realpath(get_source_root() / 'data') + os.sep

    def recorded(event, args):
        if event == 'open':
            path, mode = args[0], args[1]
            if not isinstance(mode, str) or not set(mode) & set('wax+'):
                return
        elif event == 'os.rename':
            path = args[1]
        else:
            return
        if isinstance(path, int):
            return
        if os.path.realpath(os.fsdecode(path)).startswith(data):
            _checkout_writes.append(f'{event} {os.fsdecode(path)}')

    sys.addaudithook(recorded)


_install_checkout_write_recorder()


def _checkout_write_failure(item, call) -> str | None:
    if not _checkout_writes:
        return None
    writes = sorted(set(_checkout_writes))
    _checkout_writes.clear()
    return (
        f"{CHECKOUT_WRITE_BANNER}: {item.nodeid} wrote into the checkout's data/ during "
        f'{call.when}: {writes}. That folder is the installation a person runs LumaViewPro '
        'from; build one for the test with tests.settings_fixtures.installation_of_its_own().'
    )


# The full (generation-2) collections run since the current test started,
# each (start, end) on the scope's steady clock, the clock a capture's
# give-up line states its wait on: a failed test's report lists them, so a
# red that waited on the simulator shows whether a collection paused the
# feed inside its wait. Cleared at each test's start.
GC_SECTION = 'full garbage collections'
_gen2_collections: list[tuple[int, int]] = []
_gen2_started = {'at_ns': None}


def _record_gen2(phase, info):
    if info['generation'] != 2:
        return
    if phase == 'start':
        _gen2_started['at_ns'] = host_clock.steady_ns()
    elif _gen2_started['at_ns'] is not None:
        _gen2_collections.append((_gen2_started['at_ns'], host_clock.steady_ns()))
        _gen2_started['at_ns'] = None


gc.callbacks.append(_record_gen2)


def _gen2_section() -> str:
    if not _gen2_collections:
        return 'none during this test'
    return '\n'.join(
        f'at steady {start / 1e9:.3f}s, {(end - start) / 1e6:.1f} ms'
        for start, end in _gen2_collections
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    failures = [
        f
        for f in (
            _serial_failure(item, call),
            _shipped_lock_failure(item, call),
            _checkout_write_failure(item, call),
        )
        if f
    ]
    if failures:
        report.outcome = 'failed'
        report.longrepr = '\n\n'.join(failures)
    if report.failed:
        report.sections.append((GC_SECTION, _gen2_section()))
        _write_failure_file(item, report, call)


# Every failed test phase leaves one file holding its whole report -- the
# failure, every log line production wrote during the test (pyproject's
# log_level keeps DEBUG), stdout and stderr -- whatever --tb or -q the run
# printed, so a red that passes on its rerun is read from what it left
# instead of chased. One folder per run, named by the controller and handed
# to each xdist worker; CI uploads it.
FAILURE_FILES_BANNER = 'failure files'


_FAILURE_RUN = pytest.StashKey[tuple[Path, str]]()


def _failure_run(config) -> tuple[Path, str]:
    """The run's failure folder and the commit it ran, the controller's on a worker.

    Named here, the first time it is asked for, and kept for the run: the
    Firmware repo's conftests take ``pytest_runtest_makereport`` from this
    one by name without its ``pytest_configure``, so what the hook needs is
    made by the hook's own path.
    """
    workerinput = getattr(config, 'workerinput', None)
    if workerinput is not None and 'lvp_failure_dir' in workerinput:
        return Path(workerinput['lvp_failure_dir']), workerinput['lvp_failure_commit']
    if _FAILURE_RUN not in config.stash:
        named = time.strftime('%Y%m%d-%H%M%S')
        config.stash[_FAILURE_RUN] = (
            Path(config.rootpath) / 'build' / 'test_failures' / f'{named}-{os.getpid()}',
            _head_commit(Path(config.rootpath)),
        )
    return config.stash[_FAILURE_RUN]


def _write_failure_file(item, report, call) -> None:
    """The whole traceback from the test's own file down, whatever the run's
    --tb, else the report's own words for a failure this conftest made of a
    passing phase."""
    config = item.config
    folder, commit = _failure_run(config)
    if call.excinfo is not None:
        excinfo = pytest.ExceptionInfo.from_exception(call.excinfo.value)
        excinfo.traceback = excinfo.traceback.cut(path=item.path)
        failure = str(excinfo.getrepr(style='long'))
    else:
        failure = report.longreprtext
    folder.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', report.nodeid)[:180]
    worker = getattr(config, 'workerinput', {}).get('workerid', 'main')
    parts = [
        f'{report.nodeid} failed in {report.when}',
        f'worker {worker}, {report.duration:.3f} s, commit {commit}',
        '',
        failure,
    ]
    parts += [f'\n----- {title} -----\n{content}' for title, content in report.sections]
    (folder / f'{name}__{report.when}.txt').write_text('\n'.join(parts), encoding='utf-8')


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    """The controller hands each xdist worker the run's failure folder.

    An xdist hook: optional, so a run without xdist loads this conftest.
    """
    folder, commit = _failure_run(node.config)
    node.workerinput['lvp_failure_dir'] = str(folder)
    node.workerinput['lvp_failure_commit'] = commit


def _head_commit(root: Path) -> str:
    """The checkout's HEAD, or what kept it unread: the files name the run they came from."""
    answer = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'], cwd=root, capture_output=True, text=True
    )
    return answer.stdout.strip() if answer.returncode == 0 else f'unread ({answer.stderr.strip()})'


@pytest.fixture(autouse=True)
def _fresh_notification_dedup():
    """Each test starts with the notification centre's show-once memory empty.

    The centre is one process-wide object, and it drops a repeat of the same
    (category, title) inside its dedup window. Two tests that each end in the
    same notice then pass alone and fail together: the second test's listener
    sees nothing because the first test's notice is still remembered. A module
    nobody has imported yet holds no memory, so it is not imported here.
    """
    center = sys.modules.get('_lumascope.modules.notification_center')
    if center is not None:
        center.notifications._dedup.clear()
    yield


class CentrePosts(list):
    """The notifications the centre posted during one test, in order.

    ``threads[i]`` names the thread that posted ``self[i]``, so a post that
    arrives from a worker after the test's own events can be told apart.
    """

    def __init__(self):
        super().__init__()
        self.threads = []

    def clear(self):
        super().clear()
        self.threads.clear()


@pytest.fixture
def centre_posts():
    """Every post the notification centre makes during the test, as a client sees it.

    A listener on the real singleton at DEBUG, the way the GUI's popup bridge
    and REST's outcome subscription observe it, so a post is seen whichever
    internal path made it and whichever module bound the centre at import. A
    test that replaced a posting method instead watched one door; when the
    reporter posted through another, a "nothing posted" assertion passed with
    nothing watching. The listener only appends: an assertion raised inside a
    listener is swallowed by the centre's listener guard.
    """
    from _lumascope.modules.notification_center import Severity, notifications

    posts = CentrePosts()

    def heard(notification):
        posts.threads.append(threading.current_thread().name)
        posts.append(notification)

    notifications.add_listener(heard, min_severity=Severity.DEBUG)
    yield posts
    notifications.remove_listener(heard)


@pytest.fixture
def post_processing():
    """The session's builds on a real lane of their own, for a test with no Session."""
    from tests.post_processing_lane import a_post_processing_api

    with a_post_processing_api() as api:
        yield api


@pytest.fixture
def a_host_in_this_process(monkeypatch):
    """A host's ``main()`` run inside the test process takes none of the process's logging or crash hooks.

    A stand-in by design: a host owns its process, so its ``main()`` starts
    the program's logging and installs the crash hooks; the test process is
    the suite's, whose logging is declined once at its start, so the real
    start is refused there and the real hooks would outlive the test.
    """
    import _lumascope.lvp_logger

    monkeypatch.setattr(_lumascope.lvp_logger, 'start_program_logging', lambda: None)
    monkeypatch.setattr(_lumascope.lvp_logger, 'install_crash_hooks', lambda: None)


@pytest.fixture
def unattended_run():
    """The real centre judges the test's posts as an unattended run's, closed at teardown."""
    from _lumascope.modules.notification_center import notifications

    notifications.open_run_scope(attended=False)
    yield
    notifications.close_run_scope()


@pytest.fixture(autouse=True)
def _no_refused_edit_outlives_its_test():
    """Each test starts with no refused edit on record for the input being handled.

    The boundary remembers the frame of the last refused request, and a
    button pressed in that frame does not act. Kivy's frame count does not
    advance under the suite, so one test's refused edit would be "this
    input" for every later test on the worker. Not imported here when no
    test has.
    """
    helpers = sys.modules.get('ui.ui_helpers')
    if helpers is not None:
        helpers._unanswered_frame = None
    yield


@pytest.fixture(autouse=True)
def _no_settings_replacement_outlives_its_test():
    """Each test starts with no stored-value replacement waiting to be reported.

    A preparation that replaced one leaves it for the next session to report;
    a test that prepared settings and built no session would hand it to the
    next test's session.
    """
    init = sys.modules.get('_lumascope.modules.settings_init')
    if init is not None:
        init.take_stored_replacements()
    yield


@pytest.fixture(autouse=True)
def _no_log_setting_outlives_its_test():
    """Each test ends with the log levels and the autofocus score log as it found them.

    A session puts its settings' level on the 'LVP' and 'lvp_logger'
    loggers and turns the score log on or off, process-wide; left in place,
    what a red's captured log holds would depend on what ran before it on
    the worker. The session's level, whether bring-up is running and the
    trace logs it sets (``_lumascope.lib.log_helpers``) are put back the same way.
    """
    from _lumascope.lib import log_helpers

    loggers = [logging.getLogger(name) for name in ('LVP', 'lvp_logger')]
    levels = [lg.level for lg in loggers]
    trace_files = dict(log_helpers._trace_files)
    session_level, bring_up_running = log_helpers._session_level, log_helpers._bring_up_running
    autofocus = sys.modules.get('_lumascope.modules.autofocus_functions')
    score_log = autofocus._enable_af_score_logging if autofocus is not None else False
    yield
    log_helpers._trace_files.clear()
    log_helpers._trace_files.update(trace_files)
    log_helpers._session_level, log_helpers._bring_up_running = session_level, bring_up_running
    for lg, level in zip(loggers, levels, strict=True):
        lg.setLevel(level)
    autofocus = sys.modules.get('_lumascope.modules.autofocus_functions')
    if autofocus is not None:
        autofocus.enable_af_score_logging(enable=score_log)


@pytest.fixture(autouse=True)
def _disconnect_the_scopes_a_test_built():
    """Each scope a test built through `build_scope` is disconnected when the test ends.

    A scope runs threads of its own until it is disconnected, so a suite of
    tests that each leave one behind accumulates them in every worker; the
    lanes `give_stub_lanes` gave a stub are shut the same way. A
    module-scoped fixture's scope was built before this mark and is stopped
    at the module's end (`_stop_what_a_module_built`), when the test asks for
    it as an argument: fetched inside the test with `request.getfixturevalue`,
    it is built after the mark, and this test's end shuts it down under the
    module's later tests.
    """
    from tests.scope_fakes import tear_down_since, teardown_mark

    mark = teardown_mark()
    yield
    tear_down_since(mark)


@pytest.fixture(scope='module', autouse=True)
def _stop_what_a_module_built():
    """What a module-scoped fixture built is stopped, and let go of, when the module ends.

    A test's own mark is taken after its module's fixtures are built, so the
    test's end leaves their scopes and sessions queued. Each queued stop
    holds what it stops, so a queue nobody ran kept every module fixture's
    session alive for the rest of the worker's run. This fixture is torn
    down after the module's own fixtures, so a fixture that stopped its
    scope sees a second stop, a no-op, and one that did not has it stopped.
    """
    from tests.scope_fakes import tear_down_since, teardown_mark

    mark = teardown_mark()
    yield
    tear_down_since(mark)


@pytest.fixture(scope='session', autouse=True)
def _every_simulated_camera_is_disconnected_by_its_test():
    """Every simulated camera, however a test built it, is disconnected when that test ends.

    A grabbing simulated camera free-runs on its own thread until it is
    disconnected. Hundreds of tests construct one directly, so the teardown
    is attached to the construction itself rather than to each site; a
    camera built by a module-scoped fixture before the test began is
    disconnected at the module's end, as a scope is.
    """
    from _lumascope.drivers.simulated_camera import SimulatedCamera
    from tests.scope_fakes import disconnect_when_the_test_ends

    built = SimulatedCamera.__init__

    @functools.wraps(built)
    def built_and_queued(self, *args, **kwargs):
        built(self, *args, **kwargs)
        disconnect_when_the_test_ends(self)

    with patch.object(SimulatedCamera, '__init__', built_and_queued):
        yield


@pytest.fixture(scope='session', autouse=True)
def _the_fixtures_live_folders_are_temporary(tmp_path_factory):
    """A settings dict ``complete_settings`` builds with no live folder named gets a fresh temporary one.

    The template's live folder is relative to the installation, and a
    session's door brings the folder up where it is, so a hand-built dict
    that named none would have the door create ``capture`` under the
    checkout, once per test.
    """
    from tests import settings_fixtures

    settings_fixtures.LIVE_FOLDER_ROOT = tmp_path_factory.mktemp('live_folders')
    yield
    settings_fixtures.LIVE_FOLDER_ROOT = None


@pytest.fixture(scope='session', autouse=True)
def _every_session_is_shut_down_by_its_test():
    """Every session, however a test built it, is shut down when that test ends.

    See `tests.scope_fakes.shut_down_when_the_test_ends`. Attached to the
    construction for the reason the simulated camera's is; a session built by
    a module-scoped fixture before the test began is shut down at the
    module's end.
    """
    from _lumascope.modules.scope_session import ScopeSession
    from tests.scope_fakes import shut_down_when_the_test_ends

    built = ScopeSession.__init__

    @functools.wraps(built)
    def built_and_queued(self, *args, **kwargs):
        built(self, *args, **kwargs)
        shut_down_when_the_test_ends(self)

    with patch.object(ScopeSession, '__init__', built_and_queued):
        yield


@pytest.fixture(autouse=True)
def _no_test_leaves_the_popups_muted():
    """A test that ends with the notification centre judging posts for a run fails.

    Starting a run opens the centre's process-wide run scope and only that
    run's cleanup closes it. A test that leaves a run live leaves it open,
    and every later test on the worker finds its notices muted or counted
    as already shown -- a failure far from its cause. It is reported here,
    on the test that left it, and closed so the next test starts clean.
    """
    yield
    center = sys.modules.get('_lumascope.modules.notification_center')
    if center is not None and center.notifications._run_scope is not None:
        center.notifications.close_run_scope()
        pytest.fail('the test left a run live: the notification centre still judges posts as its')


@pytest.fixture
def no_hardware(monkeypatch):
    """The registries answer as a machine with no scope does: null boards, no camera."""
    from _lumascope.drivers.null_ledboard import NullLEDBoard
    from _lumascope.drivers.null_motorboard import NullMotionBoard
    from _lumascope.drivers.registry import DriverFallback, NoLiveDriverError
    from _lumascope.modules.lumascope_api import _lumascope

    absent = DriverFallback('not_detected', ())
    monkeypatch.setattr(
        _lumascope.led_registry,
        'create_with_fallback',
        lambda name='auto', **kw: (NullLEDBoard(), absent),
    )
    monkeypatch.setattr(
        _lumascope.motor_registry,
        'create_with_fallback',
        lambda name='auto', **kw: (NullMotionBoard(), absent),
    )

    def no_camera(name='auto', **kw):
        if kw.get('simulate'):
            return real_create(name, **kw)
        raise NoLiveDriverError('camera', absent)

    real_create = _lumascope.camera_registry.create
    monkeypatch.setattr(_lumascope.camera_registry, 'create', no_camera)


@pytest.fixture
def no_hardware_scope(no_hardware):
    """A real scope on a machine with no scope: null boards, no camera, no session's settings.

    On an installation of its own, since a real scope takes its installation's lock.
    """
    from _lumascope.modules.lumascope_api import Lumascope
    from tests.settings_fixtures import installation_of_its_own

    instance = Lumascope(source_path=installation_of_its_own(), warn_pre_release=False)
    try:
        yield instance
    finally:
        instance.disconnect()


# ---------------------------------------------------------------------------
# Memory cap
# ---------------------------------------------------------------------------
# A test that loops or accumulates when a run it expects to start is refused
# can grow one xdist worker past the machine's RAM while its state stays R
# and its output stays quiet; three such workers reached about 114 GB on a
# 48 GB machine before a person killed them, and the run that followed
# reported a tainted result as if it were one. macOS refuses every rlimit
# form (RLIMIT_AS and RLIMIT_DATA raise, `ulimit -v` fails), so the cap is
# a watchdog: every pytest process, controller or worker, polls its own
# resident size once a second and, over the cap, writes the running test
# and every thread's stack, then exits with a status nothing reads as
# success. 5 GiB: a normal worker measures about 75 MB.
MEMORY_CAP_BYTES = 5 * 1024**3
MEMORY_CAP_EXIT_STATUS = 3
MEMORY_CAP_BANNER = 'MEMORY CAP EXCEEDED'
# Bound at import so a test that swaps psutil in sys.modules cannot blind
# the watchdog.
_MEMCAP_PROCESS = psutil.Process()
_memcap_running = {'nodeid': None}


def _memcap_watch(cap_bytes, report_dir, capman):
    while True:
        time.sleep(1.0)
        rss = _MEMCAP_PROCESS.memory_info().rss
        if rss > cap_bytes:
            _memcap_fail(rss, cap_bytes, report_dir, capman)


def _memcap_fail(rss, cap_bytes, report_dir, capman):
    line = (
        f'{MEMORY_CAP_BANNER}: pid {os.getpid()} resident {rss / 2**30:.2f} GiB, '
        f'cap {cap_bytes / 2**30:.2f} GiB, while running {_memcap_running["nodeid"]}'
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f'memcap_{os.getpid()}.txt'
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(line + '\n\n')
        faulthandler.dump_traceback(file=fh, all_threads=True)
    # pytest's fd capture owns stderr during a test; the write goes to the
    # real one or it is lost with the process.
    if capman is not None:
        with capman.global_and_fixture_disabled():
            sys.stderr.write(f'\n{line}\nthread stacks: {path}\n')
            sys.stderr.flush()
    else:
        sys.stderr.write(f'\n{line}\nthread stacks: {path}\n')
        sys.stderr.flush()
    os._exit(MEMORY_CAP_EXIT_STATUS)


def _memcap_reports_since(report_dir, started):
    if not report_dir.is_dir():
        return []
    return sorted(
        path for path in report_dir.glob('memcap_*.txt') if path.stat().st_mtime >= started
    )


def pytest_runtest_logstart(nodeid, location):
    _memcap_running['nodeid'] = nodeid
    _serial_touches.clear()
    _gen2_collections.clear()


def pytest_runtest_logfinish(nodeid, location):
    _memcap_running['nodeid'] = None


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    """Register hardware-test opt-in flags."""

    def _safe(*args, **kwargs):
        try:
            parser.addoption(*args, **kwargs)
        except (ValueError, Exception):
            pass  # already registered by another plugin/conftest

    _safe(
        '--run-hardware',
        action='store_true',
        default=False,
        help='Run hardware serial tests (firmware boards via SerialBoard)',
    )
    _safe(
        '--run-ids-hardware',
        action='store_true',
        default=False,
        help='Run IDS Peak hardware tests (real SDK + connected camera)',
    )
    _safe(
        '--run-pylon-hardware',
        action='store_true',
        default=False,
        help='Run Pylon hardware tests (real SDK + connected camera)',
    )
    _safe(
        '--run-fx2-hardware',
        action='store_true',
        default=False,
        help='Run FX2 hardware tests (pyusb/libusb1 + connected LS620/LS560)',
    )
    _safe(
        '--run-tmcm6110-hardware',
        action='store_true',
        default=False,
        help="Run TMCM-6110 hardware tests (the LS720's stage controller on USB)",
    )
    _safe(
        '--run-timing-sensitive',
        action='store_true',
        default=False,
        help='Run wall-clock timing-sensitive tests (can be flaky under load)',
    )
    _safe(
        '--memory-cap-bytes',
        type=int,
        default=MEMORY_CAP_BYTES,
        help='Resident-memory cap per pytest process (controller and each xdist '
        'worker); over it the process writes the running test and every '
        f"thread's stack to build/memcap_<pid>.txt and exits {MEMORY_CAP_EXIT_STATUS}. "
        'Lowered only by the guard test that proves the cap fires.',
    )
    _safe(
        '--driver-log',
        action='store_true',
        default=False,
        help='Route the (normally mocked) driver loggers to a REAL DEBUG log '
        'file + stdout, so a test run records the driver internals (poll / '
        'DeviceLost / wedge classification / grab detail) instead of discarding '
        'them into the MagicMock. Works on any test; essential for diagnosing a '
        '--run-ids-hardware bench-test failure.',
    )


def pytest_configure(config):
    """Register custom markers used by hardware tests."""
    config.addinivalue_line(
        'markers',
        'ids_hardware: requires real IDS Peak SDK + connected camera '
        '(only runs with --run-ids-hardware)',
    )
    config.addinivalue_line(
        'markers',
        'pylon_hardware: requires real Pylon SDK + connected camera '
        '(only runs with --run-pylon-hardware)',
    )
    config.addinivalue_line(
        'markers',
        'fx2_hardware: requires pyusb/libusb1 + connected FX2 scope '
        '(only runs with --run-fx2-hardware)',
    )
    config.addinivalue_line(
        'markers',
        "tmcm6110_hardware: requires the LS720's TMCM-6110 on USB "
        '(only runs with --run-tmcm6110-hardware)',
    )
    config.addinivalue_line(
        'markers',
        'timing_sensitive: measures wall-clock timing and can be flaky '
        'under CI/load (only runs with --run-timing-sensitive)',
    )
    config.addinivalue_line(
        'markers',
        'needs_display: opens a real Kivy window (skipped on a Linux host '
        'with no display, such as the GitHub runner)',
    )
    config.addinivalue_line(
        'markers',
        'firmware_tier: runs the firmware-backed simulator (skipped on a '
        'platform with no MicroPython runtime)',
    )

    if config.getoption('--driver-log', default=False):
        _enable_driver_logging(config)

    config._memcap_started = time.time()
    config._memcap_report_dir = Path(config.rootpath) / 'build'
    threading.Thread(
        target=_memcap_watch,
        args=(
            config.getoption('--memory-cap-bytes'),
            config._memcap_report_dir,
            config.pluginmanager.getplugin('capturemanager'),
        ),
        name='memory-cap',
        daemon=True,
    ).start()


# The loggers the drivers write to (src/_lumascope/lib/log_helpers) and the protocol lane's.
_DRIVER_LOG_NAMES = ('lvp_logger', 'LVP.camera', 'LVP.protocol')


def _enable_driver_logging(config):
    """Write the drivers' loggers, at DEBUG, to a timestamped file + stdout.

    The suite's loggers go to each test's captured log, at the level the
    test's session set; a --run-ids-hardware bench test needs the driver's
    whole narrative at DEBUG, in a file kept with the run. A logger is found
    by its name, so the handlers go on the named loggers the drivers write to
    (src/_lumascope/lib/log_helpers: 'lvp_logger', 'LVP.camera') and the protocol lane's
    ('LVP.protocol'), whoever imported them first. The bench tests that read
    this build a driver, not a session, so no session's level lowers them.
    Off by default (flag-gated).
    """
    import time

    from _lumascope.modules.log_stamp import make_formatter

    # In the one log folder, beside the bench run's results file, so the
    # support zip carries the driver's narrative with the run it explains.
    log_path = str(bench_results_dir() / f'driver_test_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(make_formatter('%(levelname)s %(message)s'))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('[driver] %(message)s'))
    loggers = [logging.getLogger(name) for name in _DRIVER_LOG_NAMES]
    for driver_logger in loggers:
        driver_logger.setLevel(logging.DEBUG)
        driver_logger.addHandler(file_handler)
        driver_logger.addHandler(stream_handler)

    config._driver_log_path = log_path
    config._driver_log_handlers = (file_handler, stream_handler)
    config._driver_loggers = loggers
    print(f'\n[--driver-log] driver logging -> {log_path}')


def pytest_unconfigure(config):
    """Close the --driver-log handlers and restate the file path."""
    handlers = getattr(config, '_driver_log_handlers', None)
    if not handlers:
        return
    for handler in handlers:
        handler.close()
        for driver_logger in config._driver_loggers:
            driver_logger.removeHandler(handler)
    print(f'\n[--driver-log] driver log written to {config._driver_log_path}')


_bench_reports = []


def pytest_runtest_logreport(report):
    """A bench run keeps each test's outcome and what it printed, for its results file."""
    if not BENCH_RUN:
        return
    if report.when != 'call' and report.passed:
        return
    _bench_reports.append(
        {
            'test': report.nodeid,
            'phase': report.when,
            'outcome': report.outcome,
            'seconds': round(report.duration, 3),
            'failure': str(report.longrepr) if report.failed else None,
            'stdout': report.capstdout,
            'stderr': report.capstderr,
        }
    )


def _write_bench_results(exitstatus):
    """One file per bench run in the log folder: what ran, on which build, and each result.

    Output printed with ``-s`` goes to the console, not here: a bench run
    is run without it so its figures reach the file.
    """
    import datetime
    import json

    import _lumascope.lvp_logger

    stamp = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%SZ')
    path = bench_results_dir() / f'pytest_{stamp}_results.json'
    path.write_text(
        json.dumps(
            {
                'command': sys.argv,
                'exit_status': int(exitstatus),
                'lvp_version': _lumascope.lvp_logger.version,
                'lvp_commit': _lumascope.lvp_logger.git_revision(),
                'log_folder': _lumascope.lvp_logger.log_dir,
                'results': _bench_reports,
            },
            indent=2,
        )
    )
    print(f'\nBENCH RESULTS: {path}', flush=True)


def pytest_sessionfinish(session, exitstatus):
    """A worker ships its ratchet observations to the controller (tests/ratchets.py);
    a bench run leaves its results file in the log folder."""
    workeroutput = getattr(session.config, 'workeroutput', None)
    if workeroutput is not None:
        from tests import ratchets

        workeroutput['ratchets'] = ratchets.ship()
    elif BENCH_RUN:
        _write_bench_results(exitstatus)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    """The controller takes each worker's ratchet observations as the worker leaves.

    An xdist hook: optional, so a run without xdist loads this conftest.
    """
    from tests import ratchets

    ratchets.take(getattr(node, 'workeroutput', {}).get('ratchets', ()))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Name the interpreter the run used, and announce every ratchet count a
    test observed, beside its pin.

    A test can pass on one supported Python and fail on the other, and
    `-q` drops pytest's own platform header, so the log states it here.
    The ratchet tests fail on a wrong count; this prints the numbers
    whether or not they failed, so every pytest log carries the progress
    of each migration. Observations: tests/ratchets.py.
    """
    from tests import ratchets

    terminalreporter.line(f'interpreter: Python {platform.python_version()} at {sys.executable}')

    failures, _ = _failure_run(config)
    if failures.is_dir():
        terminalreporter.section(FAILURE_FILES_BANNER, sep='!', red=True)
        terminalreporter.line(
            f'{len(list(failures.glob("*.txt")))} failed phase(s), each with its whole log: {failures}'
        )

    # A worker the cap killed is reported by xdist as a crash; this names the
    # cap and the test so the log cannot read as an ordinary failure.
    reports = _memcap_reports_since(config._memcap_report_dir, config._memcap_started)
    if reports:
        terminalreporter.section(MEMORY_CAP_BANNER, sep='!', red=True)
        for path in reports:
            with open(path, encoding='utf-8') as fh:
                terminalreporter.line(fh.readline().rstrip())
            terminalreporter.line(f'thread stacks: {path}')

    lines = ratchets.summary_lines()
    if not lines:
        return
    terminalreporter.section('ratchets: current count beside its pin (falling is progress)')
    for line in lines:
        terminalreporter.line(line)


def pytest_collection_modifyitems(config, items):
    """Skip hardware-marked tests unless the matching opt-in flag is set,
    window tests on a host with no display, and firmware-tier tests on a
    platform the simulator's runtime is not built for."""
    gates = [
        ('ids_hardware', '--run-ids-hardware'),
        ('pylon_hardware', '--run-pylon-hardware'),
        ('fx2_hardware', '--run-fx2-hardware'),
        ('tmcm6110_hardware', '--run-tmcm6110-hardware'),
        ('timing_sensitive', '--run-timing-sensitive'),
    ]
    for marker, flag in gates:
        if config.getoption(flag, default=False):
            continue
        skip = pytest.mark.skip(reason=f'needs {flag}')
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        skip = pytest.mark.skip(reason='needs a display, and this Linux host has none')
        for item in items:
            if 'needs_display' in item.keywords:
                item.add_marker(skip)
    if runtime_platform() is None:
        skip = pytest.mark.skip(
            reason=f'no simulator runtime for {sys.platform}/{platform.machine()}'
        )
        for item in items:
            if 'firmware_tier' in item.keywords:
                item.add_marker(skip)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sim_scope():
    """Lumascope with simulated hardware in fast timing mode.

    An LS850: the LS850T without the turret. Its users select an objective
    directly, which only a scope with no turret can do -- on a turret scope
    the objective is the slot's assignment. Bound to the template's
    settings, as a session binds the scope it brings up; a test names its
    own plate or objective with ``bind_settings_like_a_session``.
    """

    from tests.scope_fakes import bind_settings_like_a_session

    s = build_scope(simulate=True, sim_model='LS850')
    bind_settings_like_a_session(s)
    s._led_driver.set_timing_mode('fast')
    s._motion_driver.set_timing_mode('fast')
    s._camera_driver.set_timing_mode('fast')
    s.imaging.enable_streaming()
    yield s
    s.imaging.disable_streaming()
    s.disconnect()


@pytest.fixture
def pacific_time(monkeypatch):
    """The host's local zone set to one with a fall-back hour, then restored.

    Los Angeles: 2026-11-01 01:00 to 02:00 happens twice, daylight time
    (-07:00) and then standard time (-08:00).
    """
    if not hasattr(time, 'tzset'):
        # Windows sets no process zone from TZ; the code these tests pin has
        # no platform branch, so the POSIX hosts pin it for every host.
        pytest.skip('the local zone cannot be set in-process on this platform')
    monkeypatch.setenv('TZ', 'America/Los_Angeles')
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.fixture
def sim_camera(sim_scope):
    """(the camera API, the simulated camera under it) of ``sim_scope``.

    For a test that reads the camera's own state or wraps one of its
    members; a fault the camera meets is injected through
    ``sim_scope.simulation``.
    """
    return sim_scope.camera, sim_scope._camera_driver


@pytest.fixture
def sim_session(request, tmp_path):
    """A session on the simulator over the shipped template, its live folder temporary.

    Unhomed, as a scope comes up. A test names other settings by
    parametrizing it indirectly with their overrides
    (``@pytest.mark.parametrize('sim_session', [{'microscope': 'LS620'}], indirect=True)``);
    ``gui_ctx`` takes the same session.
    """
    from _lumascope.modules.scope_session import ScopeSession
    from tests.settings_fixtures import complete_settings

    overrides = getattr(request, 'param', {})
    session = ScopeSession.create(
        complete_settings(live_folder=str(tmp_path), **overrides), simulate=True
    )
    yield session
    session.shutdown()


@pytest.fixture
def sim_session_on_its_own_installation(tmp_path):
    """A session on the simulator on an installation of its own, ``tmp_path / 'installation'``.

    For what reads the session's installation folders (its ``data/``, its
    logs): the repo's own ``data/`` and ``logs/`` are not in them. Its live
    folder is ``tmp_path / 'live'``.
    """
    from _lumascope.modules.scope_session import ScopeSession
    from tests.installation_fixtures import write_installation
    from tests.settings_fixtures import complete_settings

    root = write_installation(tmp_path / 'installation')
    live = tmp_path / 'live'
    live.mkdir()
    session = ScopeSession.create(
        complete_settings(live_folder=str(live)), simulate=True, source_path=str(root)
    )
    yield session
    session.shutdown()


@pytest.fixture
def gui_ctx(sim_session, monkeypatch):
    """The GUI's app context over ``sim_session``, built as the GUI host builds it.

    The widget fields start empty: the Kivy tree never runs here, so a test
    hands in the panels its subject reads (``gui_ctx.motion_settings = ...``).
    The main widget answers the scope, as the host's does.
    """
    from types import SimpleNamespace

    import _lumascope.modules.app_context as app_context

    ctx = app_context.AppContext.over_session(
        sim_session, lumaview=SimpleNamespace(scope=sim_session.scope)
    )
    monkeypatch.setattr(app_context, 'ctx', ctx)
    return ctx


@pytest.fixture(scope='module')
def sim_turreted_session(tmp_path_factory):
    """A homed LS850T session on the simulator, with the shipped template's plate and stage offset.

    The template's 96-well plate and its 5500/4000 um stage offset give the
    plate frame its reachable band, and the turret carries the test
    objectives. Module-scoped: a refusal leaves nothing behind, and a test
    that moves an axis reads where it landed in the same test, so the
    tests of one module share one bring-up. A test that needs other
    settings builds its own session.
    """
    from _lumascope.modules.scope_session import ScopeSession
    from tests.scope_fakes import TEST_TURRET_OBJECTIVES, home_sim_scope
    from tests.settings_fixtures import complete_settings

    settings = complete_settings(
        microscope='LS850T',
        turret_objectives=dict(TEST_TURRET_OBJECTIVES),
        live_folder=str(tmp_path_factory.mktemp('turreted')),
    )
    session = ScopeSession.create(settings, simulate=True)
    home_sim_scope(session.scope)
    yield session
    session.shutdown()


@pytest.fixture
def scale_capabilities():
    """The optics of a scope that reports a known image scale.

    ``common_utils.get_pixel_size`` / ``get_field_of_view`` take the scope's
    capabilities as an argument; production has no hardcoded fallback, so a
    test that needs a real scale (scale bar, tiling, field-of-view readout)
    hands in optics that report one. The values match Etaluma's Classic optics
    so geometry assertions written against the previous default stay valid.
    The LED cap is the EL-0940 firmware's, because a Protocol built from
    capabilities carries it into validate_steps. The axes are a stage
    scope's, since a protocol built from capabilities records a position
    only on the axes the scope has.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        pixel_size_um=2.0, lens_focal_length_mm=47.8, led_max_ma=1000, axes=('X', 'Y', 'Z')
    )
