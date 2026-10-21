# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""Each enrolled member answers what its contract says, in Python and over REST alike.

A member's contract is its docstring's ``Raises:`` section (W0-R25). This is
the walk that holds a member to it, on the simulated scope, for the members
W0 changed the answers of: the path members, the motion members W0 gave
their stops and waits, ``frame_at_binning`` and
``apply``. For each:

- **A valid call** answers, and its read-back shows what it did.
- **A bad argument** answers the refusal its contract names, with the
  reason it names, and changes nothing: the settings and the live folder
  are as they were.
- **An injected fault** (``scope.simulation``) answers the class its
  contract names.
- **Over REST** the same call answers the status and the problem body the
  server makes of the Python answer (``rest.problems.answered_by_member``).

An answer its contract does not declare fails the walk: the section is
where a caller learns what it may be told, so it is fixed with the code.

The builds' valid calls build from run folders the simulator saved
(``run_folders``); the support report's is made elsewhere
(``_VALID_PENDING``).
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import math
import pathlib
import shutil
from collections.abc import Callable

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from modules import wire_encoding
from modules.api_surface import _function_of
from modules.exceptions import (
    ArgumentRefusedError,
    CameraSettingRejected,
    CameraSettingUnsupportedError,
    HardwareCommandRefusedError,
    HomingFaultError,
    LiveFolderPathRefusedError,
    MoveWaitExpiredError,
    PositionOutOfRangeError,
    PostProcessingRefusedError,
)
from modules.cell_count_method import CellCountMethod
from modules.protocol import StepNotFoundError
from modules.scope_session import ScopeSession
from rest.app import build_app
from rest.problems import answered_by_member
from tests.guards.test_every_member_states_its_contract import (
    _ENTRY,
    _ROOT,
    _SECTION,
    _multi_reason,
)
from tests.scope_fakes import home_sim_scope
from tests.settings_fixtures import complete_settings

# The members whose valid call waits on a fixture this walk does not yet have.
_VALID_PENDING = {
    # A report runs for minutes; one is made on the simulator, and read
    # back, in tests/test_a_support_report_says_what_bring_up_found.py.
    'ScopeSession.make_support_report': 'its minutes of work',
}

ENROLLED = (
    'ScopeSession.save_protocol',
    'ScopeSession.load_protocol',
    'ScopeSession.open_protocol',
    'ScopeSession.make_logs_zip',
    'ScopeSession.make_support_report',
    'ScopeSession.live_folder_listing',
    'PostProcessingAPI.stitch',
    'PostProcessingAPI.zproject',
    'PostProcessingAPI.composite',
    'PostProcessingAPI.video',
    'PostProcessingAPI.enhance',
    'PostProcessingAPI.count_cells',
    'PostProcessingAPI.graph_options',
    'PostProcessingAPI.graph',
    'MotionAPI.home',
    'MotionAPI.move_absolute',
    'MotionAPI.wait_until_finished_moving',
    'CameraAPI.frame_at_binning',
    'CameraAPI.apply',
    'ProtocolsAPI.step_targets',
    'ProtocolsAPI.tiling_config',
)


@dataclasses.dataclass
class Scope:
    """A simulated session, its live folder, its REST client and a protocol on it."""

    session: ScopeSession
    live: pathlib.Path
    client: TestClient
    protocol: object = None
    handle: str = ''
    # The run folders a build's valid call copies, by kind (``run_folders``).
    runs: dict[str, pathlib.Path] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Valid:
    case: str
    member: str
    call: Callable[[Scope], object]
    # A sentence naming what the read-back did not show, or None.
    read_back: Callable[[Scope, object], str | None]
    wire: tuple[str, Callable[[Scope], dict]] | None = None


@dataclasses.dataclass(frozen=True)
class Bad:
    case: str
    member: str
    call: Callable[[Scope], object]
    refusal: type
    reason: str
    wire: tuple[str, Callable[[Scope], dict]] | None = None


@dataclasses.dataclass(frozen=True)
class Fault:
    case: str
    member: str
    model: str
    cause: Callable[[Scope], object]
    call: Callable[[Scope], object]
    raised: type
    reason: str | None
    wire: tuple[str, Callable[[Scope], dict]] | None = None


def _contract(member: str) -> dict[str, str]:
    """``{class name: entry words}`` of the member's ``Raises:`` section."""
    owner, name = member.split('.')
    cls = wire_encoding.project_classes()[owner]
    doc = inspect.getdoc(_function_of(vars(cls)[name])) or ''
    section = _SECTION.search(doc)
    entries: dict[str, str] = {}
    # A class may have several entries, one per condition: each is read.
    for name, words in _ENTRY.findall(section.group(1)) if section else ():
        entries[name] = entries.get(name, '') + words
    return entries


def _undeclared(member: str, raised: BaseException) -> str | None:
    """Why ``raised`` is not what ``member``'s contract says it may raise, or None."""
    entries = _contract(member)
    declared = [
        words
        for name, words in entries.items()
        if name in {k.__name__ for k in type(raised).__mro__}
    ]
    if not declared:
        return f'raised {type(raised).__name__}, which its Raises section does not name'
    reason = getattr(raised, 'reason', None)
    # Only a class with several reasons must say which, as the guard holds.
    several = (
        _multi_reason(_ROOT[type(raised).__name__]) if type(raised).__name__ in _ROOT else False
    )
    if several and reason and not any(reason in words for words in declared):
        return f'raised {type(raised).__name__} {reason!r}, a reason its entry does not name'
    return None


def _state(scope: Scope) -> tuple[dict, list[str]]:
    """What a refused call must leave as it was: the settings and the live folder."""
    files = sorted(str(p.relative_to(scope.live)) for p in scope.live.rglob('*'))
    return copy.deepcopy(scope.session.settings), files


def _ask(scope: Scope, route: str, body: dict):
    """Call ``route`` as a wire client does: a read with GET, its arguments in
    the query string, any other member with POST and a JSON body; which, the
    route's own description says."""
    described = scope.client.app.openapi()['paths'][route.replace('{handle}', '{handle_id}')]
    path = _route(scope, route)
    if 'post' in described:
        return scope.client.post(path, json=body)
    query = []
    for name, value in body.items():
        if isinstance(value, dict):
            query += [(f'{name}.{field}', each) for field, each in value.items()]
        else:
            query += [(name, each) for each in (value if isinstance(value, list) else [value])]
    return scope.client.get(
        path, params=[(k, str(v).lower() if isinstance(v, bool) else str(v)) for k, v in query]
    )


def _wire_matches(scope: Scope, route: str, body: dict, expected: BaseException) -> str | None:
    """Why the REST answer differs from the problem the server makes of ``expected``, or None."""
    answer = _ask(scope, route, body)
    got = answer.json()
    if answer.status_code == 202:
        # Still running when the route stopped waiting: its job's end is
        # the answer, as a client reading the job is told.
        while got['ended'] is None:
            got = scope.client.get(
                f'/api/v1/jobs/{got["id"]}', headers={'Prefer': 'wait=60'}
            ).json()
        got = got.get('error') or {}
        status = got.get('status', 200)
    else:
        status = answer.status_code
    if status < 400:
        return f'over REST answered {answer.status_code} {got}, not a problem'
    want = answered_by_member(expected, got.get('instance', '').removeprefix('urn:uuid:'))
    if status != want.status:
        return f'over REST answered {status}, not {want.status}: {got}'
    if got != want.body:
        return f'over REST answered {got}, not {want.body}'
    return None


def _make_scope(tmp_path: pathlib.Path, model: str = 'LS850', protocol: bool = True) -> Scope:
    """``model`` is a model name, with ``/firmware`` for the firmware tier."""
    model, _, tier = model.partition('/')
    live = tmp_path / f'live_{model}'
    live.mkdir(parents=True)
    (live / 'a_file.txt').write_text('not an image, not a folder')
    (live / 'a_folder').mkdir()
    # A folder where save_protocol would write its .tsv.
    (live / 'a_folder.tsv').mkdir()
    (live / 'images').mkdir()
    tifffile.imwrite(live / 'images' / 'frame.tif', np.full((64, 64), 100, dtype=np.uint16))
    # A cell-count results table to graph: a text column and two of numbers.
    (live / 'results.csv').write_text('file,count,mean_area\na.tif,3,1.5\nb.tif,5,2.5\n')
    session = ScopeSession.create(
        complete_settings(live_folder=str(live), microscope=model, simulator_tier=tier or 'fast'),
        simulate=True,
    )
    if tier or model == 'LS720':
        # The firmware tier and the TMCM-6110 home in the simulator as on
        # the bench; home_sim_scope's shortcut is the fast board's.
        session.scope.motion.home()
    else:
        home_sim_scope(session.scope)
    # A protocol has steps only for a layer that acquires.
    session.set_layer_acquire('BF', 'image')
    client = TestClient(build_app(session))
    client.__enter__()
    scope = Scope(session, live, client)
    if protocol:
        scope.protocol = session.new_protocol()
        scope.handle = client.post('/api/v1/new_protocol').json()['handle']
    return scope


def _close(scope: Scope) -> None:
    scope.client.__exit__(None, None, None)
    scope.session.shutdown()


@pytest.fixture
def scopes(tmp_path):
    made = []

    def make(model: str = 'LS850', protocol: bool = True) -> Scope:
        made.append(_make_scope(tmp_path / str(len(made)), model, protocol))
        return made[-1]

    yield make
    for scope in made:
        _close(scope)


def _over_a_well(session: ScopeSession) -> list[dict]:
    """Park over a plate position clear of the lower Z limit, and return it as a run's one position."""
    motion = session.scope.motion
    motion.move_absolute('X', 40000.0)
    motion.move_absolute('Y', 30000.0)
    motion.move_absolute('Z', 1000.0)
    motion.wait_until_finished_moving(timeout_s=60)
    here = session.get_current_plate_position()
    return [{'x': here.x_mm, 'y': here.y_mm, 'z': here.z_um, 'name': 'A1'}]


def _saved(live: pathlib.Path, start: Callable[[], object]) -> pathlib.Path:
    """The folder of the run *start* launches, once it has completed and its files are written."""
    before = set(live.rglob('protocol_record.tsv'))
    pending = start()
    outcome = pending.result(timeout=120)
    assert outcome.status == 'completed', outcome
    assert pending.files.result(timeout=60) is not None, "the run's files never finished"
    (record,) = set(live.rglob('protocol_record.tsv')) - before
    return record.parent


@pytest.fixture(scope='module')
def run_folders(tmp_path_factory):
    """Run folders the simulator saved: a 2x2 BF and Blue scan, a three-slice BF Z-stack, a three-scan time-lapse.

    Made on a session of their own, so the walk's scope keeps its settings.
    """
    live = tmp_path_factory.mktemp('runs')
    session = ScopeSession.create(complete_settings(live_folder=str(live)), simulate=True)
    try:
        home_sim_scope(session.scope)
        runner = session.create_protocol_runner()
        session.set_layer_acquire('BF', 'image')
        session.set_layer_acquire('Blue', 'image')

        def scan(tiling: str) -> object:
            config = session.get_sequenced_capture_config(tiling=tiling, use_zstacking=False)
            config['positions'] = _over_a_well(session)
            return session.scope.protocols.create_protocol(input_config=config)

        launch = {'parent_dir': str(live), 'enable_image_saving': True}
        tiled = _saved(
            live,
            lambda: runner.start_single_scan(protocol=scan('2x2'), sequence_name='tiled', **launch),
        )
        session.update_settings('zstack.step_size', 10.0)
        session.update_settings('zstack.range', 20.0)
        _over_a_well(session)
        zstack = _saved(live, lambda: runner.start_zstack(layer='BF', sequence_name='zstack'))
        # Minutes and hours, as the protocol settings hold them: three scans a second apart.
        session.update_settings('protocol.period', 1 / 60)
        session.update_settings('protocol.duration', 3 / 3600)
        timelapse = _saved(
            live,
            lambda: runner.start_protocol(
                protocol=scan('1x1'), sequence_name='timelapse', **launch
            ),
        )
        yield {'tiled': tiled, 'zstack': zstack, 'timelapse': timelapse}
    finally:
        session.shutdown()


def _built_from(s: Scope, run: str, build: str) -> pathlib.Path:
    """A copy of run folder *run* in the scope's live folder, for *build* alone.

    A build writes its outputs into the folder it reads, so each takes the
    run as it was saved. The copy's name leads with the build's: a folder
    whose name ends in ``_video`` is read as a video recording.
    """
    own = s.live / f'{build}_{s.runs[run].name}'
    shutil.copytree(s.runs[run], own)
    return own


def _made(result, count: int) -> str | None:
    """What the build's answer does not show of *count* new outputs, each on disk, or None."""
    if result.new_count == count and all(path.is_file() for path in result.artifact_paths):
        return None
    return f'{result}'


_AUTO_GAIN = {'target_brightness': 0.5, 'min_gain_db': 0.0, 'max_gain_db': 20.0}


def _valid_rows() -> list[Valid]:
    method = CellCountMethod()
    return [
        Valid(
            'frame_at_binning 2',
            'CameraAPI.frame_at_binning',
            lambda s: s.session.camera.frame_at_binning(2),
            lambda s, frame: None if frame['width'] > 0 and frame['height'] > 0 else f'{frame}',
            ('/api/v1/scope/camera/frame_at_binning', lambda s: {'size': 2}),
        ),
        Valid(
            'apply',
            'CameraAPI.apply',
            lambda s: s.session.scope.camera.apply(2.0, 20.0),
            lambda s, applied: (
                None
                if math.isclose(s.session.scope.camera.read_gain_db(), 2.0, abs_tol=0.5)
                else f'gain read back {s.session.scope.camera.read_gain_db()}'
            ),
            (
                '/api/v1/scope/camera/apply',
                lambda s: {'gain_db': 2.0, 'exposure_ms': 20.0},
            ),
        ),
        Valid(
            'save_protocol, then load_protocol',
            'ScopeSession.save_protocol',
            lambda s: s.session.save_protocol(s.protocol, s.live / 'saved' / 'p.tsv'),
            lambda s, path: (
                None
                if s.session.load_protocol(path).num_steps() == s.protocol.num_steps()
                else 'the protocol read back has other steps'
            ),
            ('/api/v1/save_protocol', lambda s: {'protocol': s.handle, 'file_path': 'wire/p.tsv'}),
        ),
        Valid(
            'load_protocol',
            'ScopeSession.load_protocol',
            lambda s: s.session.load_protocol(
                s.session.save_protocol(s.protocol, s.live / 'loaded' / 'p.tsv')
            ),
            lambda s, loaded: (
                None
                if loaded.num_steps() == s.protocol.num_steps()
                else 'the protocol loaded has other steps'
            ),
            ('/api/v1/load_protocol', lambda s: {'file_path': 'loaded/p.tsv'}),
        ),
        Valid(
            'open_protocol',
            'ScopeSession.open_protocol',
            lambda s: s.session.open_protocol(
                s.session.save_protocol(s.protocol, s.live / 'opened' / 'p.tsv')
            ),
            lambda s, opened: (
                None
                if opened.num_steps() == s.protocol.num_steps()
                else 'the protocol opened has other steps'
            ),
        ),
        Valid(
            'live_folder_listing',
            'ScopeSession.live_folder_listing',
            lambda s: s.session.live_folder_listing('images'),
            lambda s, listing: None if 'frame.tif' in str(listing) else f'{listing}',
            ('/api/v1/live_folder_listing', lambda s: {'name': 'images'}),
        ),
        Valid(
            'make_logs_zip',
            'ScopeSession.make_logs_zip',
            lambda s: s.session.make_logs_zip(output_dir=s.live / 'logs'),
            lambda s, saved: None if any((s.live / 'logs').glob('*.zip')) else 'no zip',
        ),
        Valid(
            'enhance one image',
            'PostProcessingAPI.enhance',
            lambda s: s.session.post_processing.enhance(s.live / 'images' / 'frame.tif'),
            lambda s, result: None if result.created else f'{result}',
        ),
        Valid(
            'stitch a 2x2 scan',
            'PostProcessingAPI.stitch',
            lambda s: pp(s).stitch(_built_from(s, 'tiled', 'stitch')),
            # One stitched image a channel.
            lambda s, result: _made(result, 2),
        ),
        Valid(
            'composite a two-channel scan',
            'PostProcessingAPI.composite',
            lambda s: pp(s).composite(_built_from(s, 'tiled', 'composite')),
            # One composite a tile.
            lambda s, result: _made(result, 4),
        ),
        Valid(
            'zproject a Z-stack',
            'PostProcessingAPI.zproject',
            lambda s: pp(s).zproject(_built_from(s, 'zstack', 'zproject'), method='Max'),
            lambda s, result: _made(result, 1),
        ),
        Valid(
            'video of a time-lapse',
            'PostProcessingAPI.video',
            lambda s: pp(s).video(_built_from(s, 'timelapse', 'video')),
            # One video a channel.
            lambda s, result: _made(result, 2),
        ),
        Valid(
            'count_cells',
            'PostProcessingAPI.count_cells',
            lambda s: s.session.post_processing.count_cells(s.live / 'images', method=method),
            lambda s, result: None if (s.live / 'images' / 'results.csv').is_file() else 'no csv',
        ),
        Valid(
            'graph_options',
            'PostProcessingAPI.graph_options',
            lambda s: pp(s).graph_options(s.live / 'results.csv', x='count', y='mean_area'),
            lambda s, options: (
                None
                if (options.x_axes, options.y_axes) == (('count', 'mean_area'),) * 2
                and 'Linear' in options.trendlines
                else f'{options}'
            ),
            (
                '/api/v1/post_processing/graph_options',
                lambda s: {'results': 'results.csv', 'x': 'count', 'y': 'mean_area'},
            ),
        ),
        Valid(
            'graph',
            'PostProcessingAPI.graph',
            lambda s: pp(s).graph(
                s.live / 'results.csv',
                s.live / 'graphs' / 'g.png',
                x='count',
                y='mean_area',
                trendline='Linear',
            ),
            lambda s, _: None if (s.live / 'graphs' / 'g.png').is_file() else 'no png',
            (
                '/api/v1/post_processing/graph',
                lambda s: {
                    'results': 'results.csv',
                    'path': 'wire_graphs/g.png',
                    'x': 'count',
                    'y': 'mean_area',
                },
            ),
        ),
        Valid(
            'move_absolute',
            'MotionAPI.move_absolute',
            lambda s: s.session.scope.motion.move_absolute('Z', 1000.0),
            lambda s, _: (
                None
                if math.isclose(
                    s.session.scope.motion.get_actual_position('Z'), 1000.0, abs_tol=1.0
                )
                else f'Z at {s.session.scope.motion.get_actual_position("Z")}'
            ),
        ),
        Valid(
            'wait_until_finished_moving',
            'MotionAPI.wait_until_finished_moving',
            lambda s: s.session.scope.motion.wait_until_finished_moving(timeout_s=5.0),
            lambda s, _: None,
        ),
        Valid(
            'home',
            'MotionAPI.home',
            lambda s: s.session.scope.motion.home(),
            lambda s, _: None if s.session.scope.motion.has_homed() else 'not homed',
        ),
        Valid(
            'step_targets of the first step',
            'ProtocolsAPI.step_targets',
            lambda s: s.session.scope.protocols.step_targets(s.protocol, 0),
            lambda s, targets: (
                None
                if targets.z == s.protocol.step_row(idx=0)['Z'] and targets.turret_slot is None
                else f'{targets}'
            ),
            (
                '/api/v1/scope/protocols/step_targets',
                lambda s: {'protocol': s.handle, 'step_idx': 0},
            ),
        ),
        Valid(
            'tiling_config',
            'ProtocolsAPI.tiling_config',
            lambda s: s.session.scope.protocols.tiling_config(),
            lambda s, grids: None if grids.default in grids.available else f'{grids}',
            ('/api/v1/scope/protocols/tiling_config', lambda s: {}),
        ),
    ]


def pp(s: Scope):
    return s.session.post_processing


def _bad_rows() -> list[Bad]:
    method = CellCountMethod()
    rows = [
        Bad(
            'step_targets of a step the protocol lacks',
            'ProtocolsAPI.step_targets',
            lambda s: s.session.scope.protocols.step_targets(s.protocol, 99),
            StepNotFoundError,
            'no_such_step',
            (
                '/api/v1/scope/protocols/step_targets',
                lambda s: {'protocol': s.handle, 'step_idx': 99},
            ),
        ),
        Bad(
            'frame_at_binning 0',
            'CameraAPI.frame_at_binning',
            lambda s: s.session.camera.frame_at_binning(0),
            CameraSettingUnsupportedError,
            'binning_unsupported',
            ('/api/v1/scope/camera/frame_at_binning', lambda s: {'size': 0}),
        ),
        Bad(
            'apply, empty auto-gain settings',
            'CameraAPI.apply',
            lambda s: s.session.scope.camera.apply(1.0, 5.0, auto_gain=True, auto_gain_settings={}),
            ArgumentRefusedError,
            'missing_key',
            (
                '/api/v1/scope/camera/apply',
                lambda s: {
                    'gain_db': 1.0,
                    'exposure_ms': 5.0,
                    'auto_gain': True,
                    'auto_gain_settings': {},
                },
            ),
        ),
        Bad(
            'save_protocol into a folder',
            'ScopeSession.save_protocol',
            lambda s: s.session.save_protocol(s.protocol, s.live / 'a_folder.tsv'),
            LiveFolderPathRefusedError,
            'not_a_file',
            (
                '/api/v1/save_protocol',
                lambda s: {'protocol': s.handle, 'file_path': 'a_folder.tsv'},
            ),
        ),
        Bad(
            'save_protocol under a file',
            'ScopeSession.save_protocol',
            lambda s: s.session.save_protocol(s.protocol, s.live / 'a_file.txt' / 'p.tsv'),
            LiveFolderPathRefusedError,
            'not_a_folder',
            (
                '/api/v1/save_protocol',
                lambda s: {'protocol': s.handle, 'file_path': 'a_file.txt/p.tsv'},
            ),
        ),
        Bad(
            'save_protocol without .tsv',
            'ScopeSession.save_protocol',
            lambda s: s.session.save_protocol(s.protocol, s.live / 'plate1'),
            LiveFolderPathRefusedError,
            'wrong_extension',
            (
                '/api/v1/save_protocol',
                lambda s: {'protocol': s.handle, 'file_path': 'plate1'},
            ),
        ),
        Bad(
            'load_protocol missing',
            'ScopeSession.load_protocol',
            lambda s: s.session.load_protocol(s.live / 'nothing_here' / 'p.tsv'),
            LiveFolderPathRefusedError,
            'not_found',
            ('/api/v1/load_protocol', lambda s: {'file_path': 'nothing_here/p.tsv'}),
        ),
        Bad(
            'load_protocol a folder',
            'ScopeSession.load_protocol',
            lambda s: s.session.load_protocol(s.live / 'a_folder'),
            LiveFolderPathRefusedError,
            'not_a_file',
            ('/api/v1/load_protocol', lambda s: {'file_path': 'a_folder'}),
        ),
        Bad(
            'open_protocol missing',
            'ScopeSession.open_protocol',
            lambda s: s.session.open_protocol(s.live / 'nothing_here' / 'p.tsv'),
            LiveFolderPathRefusedError,
            'not_found',
        ),
        Bad(
            'open_protocol a folder',
            'ScopeSession.open_protocol',
            lambda s: s.session.open_protocol(s.live / 'a_folder'),
            LiveFolderPathRefusedError,
            'not_a_file',
        ),
        Bad(
            'make_logs_zip under a file',
            'ScopeSession.make_logs_zip',
            lambda s: s.session.make_logs_zip(output_dir=s.live / 'a_file.txt' / 'out'),
            LiveFolderPathRefusedError,
            'not_a_folder',
            ('/api/v1/make_logs_zip', lambda s: {'output_dir': 'a_file.txt/out'}),
        ),
        Bad(
            'make_logs_zip into a file',
            'ScopeSession.make_logs_zip',
            lambda s: s.session.make_logs_zip(output_dir=s.live / 'a_file.txt'),
            LiveFolderPathRefusedError,
            'not_a_folder',
            ('/api/v1/make_logs_zip', lambda s: {'output_dir': 'a_file.txt'}),
        ),
        Bad(
            'make_support_report into a file',
            'ScopeSession.make_support_report',
            lambda s: s.session.make_support_report(output_dir=s.live / 'a_file.txt'),
            LiveFolderPathRefusedError,
            'not_a_folder',
            ('/api/v1/make_support_report', lambda s: {'output_dir': 'a_file.txt'}),
        ),
        Bad(
            'live_folder_listing missing',
            'ScopeSession.live_folder_listing',
            lambda s: s.session.live_folder_listing('nothing_here'),
            LiveFolderPathRefusedError,
            'not_found',
            ('/api/v1/live_folder_listing', lambda s: {'name': 'nothing_here'}),
        ),
        Bad(
            'live_folder_listing a file',
            'ScopeSession.live_folder_listing',
            lambda s: s.session.live_folder_listing('a_file.txt'),
            LiveFolderPathRefusedError,
            'not_a_folder',
            ('/api/v1/live_folder_listing', lambda s: {'name': 'a_file.txt'}),
        ),
        Bad(
            'enhance missing',
            'PostProcessingAPI.enhance',
            lambda s: pp(s).enhance(s.live / 'nothing_here'),
            LiveFolderPathRefusedError,
            'not_found',
            ('/api/v1/post_processing/enhance', lambda s: {'target': 'nothing_here'}),
        ),
        Bad(
            'enhance a file that is not an image',
            'PostProcessingAPI.enhance',
            lambda s: pp(s).enhance(s.live / 'a_file.txt'),
            PostProcessingRefusedError,
            'unreadable',
            ('/api/v1/post_processing/enhance', lambda s: {'target': 'a_file.txt'}),
        ),
        Bad(
            'count_cells missing',
            'PostProcessingAPI.count_cells',
            lambda s: pp(s).count_cells(s.live / 'nothing_here', method=method),
            LiveFolderPathRefusedError,
            'not_found',
            (
                '/api/v1/post_processing/count_cells',
                lambda s: {'folder': 'nothing_here', 'method': method.to_dict()},
            ),
        ),
        Bad(
            'count_cells an unusable method',
            'PostProcessingAPI.count_cells',
            lambda s: pp(s).count_cells(
                s.live / 'images', method=CellCountMethod(pixels_per_um=0.0)
            ),
            PostProcessingRefusedError,
            'method_invalid',
            (
                '/api/v1/post_processing/count_cells',
                lambda s: {
                    'folder': 'images',
                    'method': {**method.to_dict(), 'pixels_per_um': 0.0},
                },
            ),
        ),
        Bad(
            'graph_options of a missing file',
            'PostProcessingAPI.graph_options',
            lambda s: pp(s).graph_options(s.live / 'nothing_here.csv'),
            LiveFolderPathRefusedError,
            'not_found',
            ('/api/v1/post_processing/graph_options', lambda s: {'results': 'nothing_here.csv'}),
        ),
        Bad(
            'graph_options with a text column as X',
            'PostProcessingAPI.graph_options',
            lambda s: pp(s).graph_options(s.live / 'results.csv', x='file', y='count'),
            PostProcessingRefusedError,
            'axis_not_plottable',
            (
                '/api/v1/post_processing/graph_options',
                lambda s: {'results': 'results.csv', 'x': 'file', 'y': 'count'},
            ),
        ),
        Bad(
            'graph to a name without .png',
            'PostProcessingAPI.graph',
            lambda s: pp(s).graph(
                s.live / 'results.csv', s.live / 'g.v2', x='count', y='mean_area'
            ),
            LiveFolderPathRefusedError,
            'wrong_extension',
            (
                '/api/v1/post_processing/graph',
                lambda s: {
                    'results': 'results.csv',
                    'path': 'g.v2',
                    'x': 'count',
                    'y': 'mean_area',
                },
            ),
        ),
        Bad(
            'graph of a missing file',
            'PostProcessingAPI.graph',
            lambda s: pp(s).graph(
                s.live / 'nothing_here.csv', s.live / 'g.png', x='count', y='mean_area'
            ),
            LiveFolderPathRefusedError,
            'not_found',
            (
                '/api/v1/post_processing/graph',
                lambda s: {
                    'results': 'nothing_here.csv',
                    'path': 'g.png',
                    'x': 'count',
                    'y': 'mean_area',
                },
            ),
        ),
        Bad(
            'graph with no X chosen',
            'PostProcessingAPI.graph',
            lambda s: pp(s).graph(s.live / 'results.csv', s.live / 'g.png', x=None, y='mean_area'),
            PostProcessingRefusedError,
            'graph_incomplete',
            (
                '/api/v1/post_processing/graph',
                lambda s: {'results': 'results.csv', 'path': 'g.png', 'x': None, 'y': 'mean_area'},
            ),
        ),
        Bad(
            'graph with a text column as X',
            'PostProcessingAPI.graph',
            lambda s: pp(s).graph(s.live / 'results.csv', s.live / 'g.png', x='file', y='count'),
            PostProcessingRefusedError,
            'axis_not_plottable',
            (
                '/api/v1/post_processing/graph',
                lambda s: {'results': 'results.csv', 'path': 'g.png', 'x': 'file', 'y': 'count'},
            ),
        ),
        Bad(
            'home an axis that is not one',
            'MotionAPI.home',
            lambda s: s.session.scope.motion.home('Q'),
            ArgumentRefusedError,
            'axis_unknown',
            ('/api/v1/scope/motion/home', lambda s: {'axis': 'Q'}),
        ),
        Bad(
            'move_absolute past the travel',
            'MotionAPI.move_absolute',
            lambda s: s.session.scope.motion.move_absolute('Z', 1e9),
            PositionOutOfRangeError,
            'position_out_of_range',
            ('/api/v1/scope/motion/move_absolute', lambda s: {'axis': 'Z', 'position': 1e9}),
        ),
        Bad(
            # No wire row: JSON carries no NaN.
            'wait_until_finished_moving with a NaN timeout',
            'MotionAPI.wait_until_finished_moving',
            lambda s: s.session.scope.motion.wait_until_finished_moving(timeout_s=float('nan')),
            ArgumentRefusedError,
            'not_a_number',
        ),
    ]
    for name, member, call, wire_extra, setting in (
        ('stitch', 'PostProcessingAPI.stitch', lambda s, f, **k: pp(s).stitch(f, **k), {}, 'mode'),
        (
            'zproject',
            'PostProcessingAPI.zproject',
            lambda s, f, **k: pp(s).zproject(f, **({'method': 'Max'} | k)),
            {'method': 'Max'},
            'method',
        ),
        (
            'composite',
            'PostProcessingAPI.composite',
            lambda s, f, **k: pp(s).composite(f),
            {},
            None,
        ),
        (
            'video',
            'PostProcessingAPI.video',
            lambda s, f, **k: pp(s).video(f, **k),
            {},
            'frames_per_sec',
        ),
    ):
        route = f'/api/v1/post_processing/{name}'
        rows.append(
            Bad(
                f'{name} missing',
                member,
                lambda s, call=call: call(s, s.live / 'nothing_here'),
                LiveFolderPathRefusedError,
                'not_found',
                (route, lambda s, extra=wire_extra: {'folder': 'nothing_here', **extra}),
            )
        )
        rows.append(
            Bad(
                f'{name} a file',
                member,
                lambda s, call=call: call(s, s.live / 'a_file.txt'),
                LiveFolderPathRefusedError,
                'not_a_folder',
                (route, lambda s, extra=wire_extra: {'folder': 'a_file.txt', **extra}),
            )
        )
        if setting is not None:
            bogus = 0 if setting == 'frames_per_sec' else 'bogus'
            rows.append(
                Bad(
                    f'{name} an unusable {setting}',
                    member,
                    lambda s, call=call, setting=setting, bogus=bogus: call(
                        s, s.live / 'images', **{setting: bogus}
                    ),
                    PostProcessingRefusedError,
                    'invalid_setting',
                    (
                        route,
                        lambda s, extra=wire_extra, setting=setting, bogus=bogus: {
                            'folder': 'images',
                            **extra,
                            setting: bogus,
                        },
                    ),
                )
            )
    return rows


def _fault_rows() -> list[Fault]:
    return [
        Fault(
            'frame_at_binning with the camera removed',
            'CameraAPI.frame_at_binning',
            'LS850',
            lambda s: s.session.scope.simulation.remove_camera(),
            lambda s: s.session.camera.frame_at_binning(2),
            HardwareCommandRefusedError,
            'not_connected',
            ('/api/v1/scope/camera/frame_at_binning', lambda s: {'size': 2}),
        ),
        Fault(
            'apply, the camera refusing the gain',
            'CameraAPI.apply',
            'LS850',
            lambda s: s.session.scope.simulation.refuse_camera_write('gain', 'false'),
            lambda s: s.session.scope.camera.apply(3.0, 20.0),
            CameraSettingRejected,
            None,
            (
                '/api/v1/scope/camera/apply',
                lambda s: {'gain_db': 3.0, 'exposure_ms': 20.0},
            ),
        ),
        Fault(
            'wait_until_finished_moving on a held move',
            'MotionAPI.wait_until_finished_moving',
            'LS850',
            # 1000 past where Z stands, so the move the wire's call waits on
            # has somewhere to go after the first one was released.
            lambda s: (
                s.session.scope.simulation.hold_axis('Z'),
                s.session.scope.motion.start_move_absolute(
                    'Z', s.session.scope.motion.get_actual_position('Z') + 1000.0
                ),
            ),
            lambda s: s.session.scope.motion.wait_until_finished_moving(timeout_s=0.2),
            MoveWaitExpiredError,
            'still_moving',
            ('/api/v1/scope/motion/wait_until_finished_moving', lambda s: {'timeout_s': 0.2}),
        ),
        Fault(
            'home with the Z motor stalling',
            'MotionAPI.home',
            'LS850T/firmware',
            lambda s: s.session.scope.simulation.fail_home('Z', 'stall'),
            lambda s: s.session.scope.motion.home('Z'),
            HomingFaultError,
            None,
            ('/api/v1/scope/motion/home', lambda s: {'axis': 'Z'}),
        ),
        Fault(
            'home with the lid open',
            'MotionAPI.home',
            'LS720',
            lambda s: s.session.scope.simulation.interlock(lid_open=True),
            lambda s: s.session.scope.motion.home(),
            HardwareCommandRefusedError,
            'lid_open',
            ('/api/v1/scope/motion/home', lambda s: {}),
        ),
        Fault(
            # The lid stops X and Y; Z moves with it open.
            'move_absolute X with the lid open',
            'MotionAPI.move_absolute',
            'LS720',
            lambda s: s.session.scope.simulation.interlock(lid_open=True),
            lambda s: s.session.scope.motion.move_absolute('X', 1000.0),
            HardwareCommandRefusedError,
            'lid_open',
            ('/api/v1/scope/motion/move_absolute', lambda s: {'axis': 'X', 'position': 1000.0}),
        ),
    ]


def _route(scope: Scope, route: str) -> str:
    return route.replace('{handle}', scope.handle)


def test_every_enrolled_member_has_its_rows():
    covered = {row.member for row in (*_valid_rows(), *_bad_rows(), *_fault_rows())}
    valid = {row.member for row in _valid_rows()} | set(_VALID_PENDING)
    assert set(ENROLLED) - covered == set()
    assert set(ENROLLED) - valid == set(), 'give each a valid row, or name what it waits on'


@pytest.mark.slow
def test_every_enrolled_member_answers_as_its_contract_says(scopes, run_folders):
    scope = scopes()
    scope.runs = run_folders
    wrong = []
    for row in _valid_rows():
        try:
            answered = row.call(scope)
        except Exception as raised:  # each offender is named in the one failure
            wrong.append(f'{row.case}: raised {type(raised).__name__}: {raised}')
            continue
        missing = row.read_back(scope, answered)
        if missing:
            wrong.append(f'{row.case}: {missing}')
        if row.wire is not None:
            route, body = row.wire
            answer = _ask(scope, route, body(scope))
            if answer.status_code != 200:
                wrong.append(f'{row.case}: over REST answered {answer.status_code} {answer.json()}')
    for row in _bad_rows():
        before = _state(scope)
        try:
            row.call(scope)
        except Exception as raised:  # each offender is named in the one failure
            if not isinstance(raised, row.refusal) or getattr(raised, 'reason', None) != row.reason:
                wrong.append(
                    f'{row.case}: raised {type(raised).__name__} '
                    f'{getattr(raised, "reason", None)!r}, not {row.reason!r}'
                )
                continue
            undeclared = _undeclared(row.member, raised)
            if undeclared:
                wrong.append(f'{row.case}: {undeclared}')
            if _state(scope) != before:
                wrong.append(f'{row.case}: the refusal changed the settings or the live folder')
            if row.wire is not None:
                route, body = row.wire
                differs = _wire_matches(scope, route, body(scope), raised)
                if differs:
                    wrong.append(f'{row.case}: {differs}')
        else:
            wrong.append(f'{row.case}: was not refused')
    assert wrong == [], '\n'.join(wrong)


@pytest.mark.slow
def test_every_injected_fault_answers_as_the_contract_says(scopes):
    wrong = []
    for row in _fault_rows():
        # A fault row needs no protocol, and a turreted scope with its
        # slots unassigned would refuse one.
        scope = scopes(row.model, protocol=False)
        row.cause(scope)
        try:
            row.call(scope)
        except Exception as raised:  # each offender is named in the one failure
            if not isinstance(raised, row.raised) or (
                row.reason is not None and getattr(raised, 'reason', None) != row.reason
            ):
                wrong.append(
                    f'{row.case}: raised {type(raised).__name__} '
                    f'{getattr(raised, "reason", None)!r}, not {row.raised.__name__} {row.reason!r}'
                )
                continue
            undeclared = _undeclared(row.member, raised)
            if undeclared:
                wrong.append(f'{row.case}: {undeclared}')
            if row.wire is not None:
                route, body = row.wire
                # The fault again, for the wire's call: undone first, as a
                # hold or a failed home holds until it is cleared.
                scope.session.scope.simulation.clear()
                row.cause(scope)
                differs = _wire_matches(scope, route, body(scope), raised)
                if differs:
                    wrong.append(f'{row.case}: {differs}')
        else:
            wrong.append(f'{row.case}: answered with no fault injected')
    assert wrong == [], '\n'.join(wrong)
