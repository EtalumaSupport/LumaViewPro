# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
GUI-independent configuration and state helper functions.

These functions extract configuration data from the settings dict
and scope objects without any Kivy/GUI dependencies. They can be
used by LumaViewPro, the REST API, or standalone scripts.
"""

import collections
import copy
import dataclasses
import gc
import sys
import threading
import typing

import psutil

import _lumascope.modules.binning as binning
import _lumascope.modules.common_utils as common_utils
import _lumascope.modules.image_mode as image_mode
import _lumascope.modules.wall_time as wall_time
from _lumascope.lib.real_numbers import is_finite_number
from _lumascope.lvp_logger import logger, metrics_logger
from _lumascope.modules.acquire_mode import AcquireMode
from _lumascope.modules.coord_transformations import PlateLocation
from _lumascope.modules.channel_record import ChannelSliderMaxima, Modality, release_channel
from _lumascope.modules.labware_loader import WellPlateLoader
from _lumascope.modules.objectives_loader import ObjectiveLoader
from _lumascope.modules.protocol_state_machine import SequencedCaptureRunMode
from _lumascope.modules.tiling_config import TilingGrids
from _lumascope.modules.zstack_config import ZStackReference

if typing.TYPE_CHECKING:
    # Import-time only: _lumascope.modules.protocol imports this module's siblings, so
    # a runtime import here would close a cycle.
    from _lumascope.modules.executor_registry import ExecutorBundle
    from _lumascope.modules.lumascope_api._lumascope import Lumascope
    from _lumascope.modules.protocol import Protocol
    from _lumascope.modules.lumascope_api.camera import CameraSettingLimits
    from _lumascope.modules.scope_capabilities import ScopeCapabilities

# ---------------------------------------------------------------------------
# Protocol / Step helpers
# ---------------------------------------------------------------------------


def find_nearest_step(x: float, y: float, protocol: 'Protocol | None') -> int:
    """Given a position, find the nearest step index in the protocol.

    A step with no X/Y is at no place on the plate, so it is never the
    nearest; -1 when no step has one.
    """
    if protocol is None or protocol.num_steps <= 0:
        return -1

    placed = protocol.steps()[['X', 'Y']].dropna()
    if placed.empty:
        return -1
    idx = (placed.sub([x, y]).pow(2).sum(axis=1)).idxmin()
    # int(), not the bare idxmin: it returns np.int64, and this index is
    # handed on as a step number to callers that store and display it.
    # The annotation above promises a Python int.
    return int(idx)


# ---------------------------------------------------------------------------
# Channel configuration
# ---------------------------------------------------------------------------


def get_channel_configs(settings: dict, specific_channels: list | None = None) -> dict:
    """Each channel's settings block in ``settings``, as a run or a protocol build takes it.

    One dict per channel of the release, or per channel named in
    ``specific_channels``; a name there that is no channel is skipped, not
    refused (``ScopeSession.get_channel_configs`` refuses it).
    Illumination, gain and exposure are rounded to their stated precision
    and ``auto_gain`` is read as a bool.

    The channels come in the drawers' order (``common_utils.get_channels``):
    every build takes its channels from here, so a place's steps run in that
    one order however the protocol was built.

    Returns:
        dict of channel key -> config dict with keys:
        acquire, video_config, autofocus, false_color,
        illumination_ma, gain_db, auto_gain, exposure_ms, sum, focus
    """
    channel_configs = {}
    for channel in common_utils.get_channels():
        if (specific_channels is not None) and (channel not in specific_channels):
            continue

        channel_config = settings[channel]

        acquire = channel_config['acquire']
        # A copy: the answer is the caller's, and the block is the store's.
        video_config = copy.deepcopy(channel_config['video_config'])

        autofocus = channel_config['autofocus']
        false_color = channel_config['false_color']
        illumination = round(
            channel_config['illumination_ma'], common_utils.max_decimal_precision('illumination')
        )
        sum_count = channel_config['sum']
        gain = round(channel_config['gain_db'], common_utils.max_decimal_precision('gain'))
        auto_gain = common_utils.to_bool(channel_config['auto_gain'])
        exposure = round(
            channel_config['exposure_ms'], common_utils.max_decimal_precision('exposure')
        )
        focus = channel_config['focus']

        channel_configs[channel] = {
            'acquire': acquire,
            'video_config': video_config,
            'autofocus': autofocus,
            'false_color': false_color,
            'illumination_ma': illumination,
            'gain_db': gain,
            'auto_gain': auto_gain,
            'exposure_ms': exposure,
            'sum': sum_count,
            'focus': focus,
        }

    return channel_configs


def get_auto_gain_settings(settings: dict) -> dict:
    """A copy of the stored auto-gain settings, the dictionary the camera's auto-gain takes."""
    return settings['protocol']['autogain'].copy()


def get_sequenced_run_settings(settings: dict, *, run_mode: SequencedCaptureRunMode) -> dict:
    """Resolve the settings-derived kwargs a sequenced-capture run forwards.

    The single source for the run-parameters a run reads from the user
    settings, spread into SequencedCaptureRunner.prepare() at every call
    site. The GUI starters and the API protocol path all build the run
    kwargs, and each forwarding a separate hand-picked subset is how they
    drift: a key present on one path but missing on the other silently
    changes hardware behavior (LED hold, output layout, AF ceiling) for
    whichever path forgot it. Routing every path through here makes the
    per-path subset one definition, so adding a new run-param cannot reach
    the runner on only one path -- and a run that has no settings dict at
    all (headless) cannot silently fall back to defaults the engine
    invented, because the engine no longer reads settings itself.

    The values are read once here, by the caller that owns the settings,
    and frozen on the run plan: a mid-run toggle must never make some
    steps of one scan behave differently from others.

    Two of the reads are nested (``video.timestamp_overlay``,
    ``video.max_fps``) while the run-value names are flat; the key paths
    are what the GUI writers use, so a typo here returns the default on
    every install without failing anything -- keep them exactly as the
    writers spell them.

    ``run_mode`` carries the two run kinds whose values are not the user's.
    An autofocus run, at one position or every step, must NOT hold the
    excitation LED across focus moves (photobleaching the sample) and saves
    nothing, so it never keeps the LED between steps and never makes
    per-channel folders, whatever the settings say. A composite's merge
    reads only some sequenced formats back, so its image config is the
    composite one, which departs from a scan's in that format alone. Both
    guarantees live here, once, rather than at each call site.
    """
    autofocus_scan = run_mode.is_autofocus
    if run_mode == SequencedCaptureRunMode.SINGLE_COMPOSITE:
        image_capture_config = get_composite_image_capture_config_from_settings(settings)
    else:
        image_capture_config = get_image_capture_config_from_settings(settings)
    return {
        'image_capture_config': image_capture_config,
        'keep_led_between_steps': (False if autofocus_scan else settings['keep_led_between_steps']),
        'recording_format': image_mode.RecordingFormat(settings['video_recording_format']),
        'separate_folder_per_channel': (
            False if autofocus_scan else settings['separate_folder_per_channel']
        ),
        'timestamp_overlay': settings['video']['timestamp_overlay'],
        'video_max_fps': settings['video']['max_fps'],
    }


def get_manual_video_max_duration(settings: dict) -> float:
    """Return the max recording duration in seconds (video.max_duration_s).

    Manual recording is stop-when-I-say bounded by this cap. The store is
    prepared, so the key is read; its shipped value is the template's.
    """
    return settings['video']['max_duration_s']


def get_ag_ae_max_exposure_ms(channel: str, ceilings: dict) -> float:
    """Return the AG/AE exposure upper bound (ms) for a channel's class.

    AG/AE is capped per channel class so auto-exposure cannot drive the
    sensor toward its native max on dim scenes (especially with the
    MinimizeGain profile), washing out brightfield and making the live-view
    auto loop hunt. Distinct from the manual exposure-slider limits.
    Transmitted light is bright (short exposures); fluorescence needs more;
    luminescence integrates long. Resolves the channel to its class and
    returns that class's ceiling. Unknown channels take the fluorescence cap.

    ``ceilings`` is the per-class map the settings hold under
    ``ag_ae_max_exposure_ms`` (``{'fluorescence': 200.0, ...}``), handed in
    by the caller that owns the settings -- a run reads it at its prepare,
    the live view off the session. It is NOT the whole settings dict.
    """
    return float(ceilings[_ag_ae_channel_class(channel)])


def log_resolved_optics(
    objective_id: str,
    focal_length: float,
    binning_size: int | None,
    *,
    capabilities: 'ScopeCapabilities | None',
) -> None:
    """Record the optics behind image scale, at the moment they are chosen.

    Scale is written into every frame, hyperstack and still as a real
    PhysicalSizeX, but the values producing it are read off the scope and
    consumed in process -- so a returned support bundle could only show the
    shipped default templates, which describe what ships rather than what this
    scope is set to. On a bench unit the two disagreed in the fourth decimal,
    which is the size of error a wrong-measurement report is about.

    The um/px comes from the resolver rather than being recomputed, so the
    logged number cannot drift from the one written into the images. A scope
    that cannot report its optics still logs, naming the missing input: "no
    scale" is the condition a returned bundle most needs explained.

    The caller hands in the capabilities of the scope it images with, the
    way the engine's producers do: resolving them off a GUI context here
    would make the record silent in exactly the headless session whose
    images it is meant to explain.
    """
    tube_focal_length = None if capabilities is None else capabilities.lens_focal_length_mm
    pixel_width = None if capabilities is None else capabilities.sensor_pitch_um

    um_per_pixel = (
        None
        if capabilities is None
        else common_utils.get_pixel_size(
            focal_length=focal_length, binning_size=binning_size, capabilities=capabilities
        )
    )

    if um_per_pixel is None:
        missing = [
            name
            for name, value in (
                ('active scope', capabilities),
                ('tube lens focal length', tube_focal_length),
                ('sensor pixel size', pixel_width),
                ('camera binning', binning_size),
            )
            if value is None
        ]
        logger.warning(
            f'[Optics   ] objective={objective_id} objective_focal_length={focal_length}mm '
            f'binning={binning_size} -- no image scale available, missing: '
            f'{", ".join(missing)}. Images from this scope carry no PhysicalSizeX.'
        )
        return

    logger.info(
        f'[Optics   ] objective={objective_id} objective_focal_length={focal_length}mm '
        f'tube_focal_length={tube_focal_length}mm sensor_pixel_size={pixel_width}um '
        f'binning={binning_size} -> {um_per_pixel}um/px'
    )


def get_ag_ae_min_exposure_ms(channel: str) -> float:
    """Return the usable exposure floor (ms) for a channel's class.

    Auto-exposure may drive the sensor below this on a bright scene (Pylon
    bottoms out near 0.03 ms); the floor is the lowest exposure the class
    treats as usable: a setting written below it produces near-black
    protocol steps on ordinary scenes, and fluorescence / luminescence
    sliders cannot show sub-millisecond values at all. An auto-gain lock
    at or below this floor reports AT_MINIMUM, and the exposure a caller
    leaving auto-gain stores is floored to it (``stored_exposure_ms``).
    Unknown channels fall back to the
    fluorescence floor, as the ceiling sibling does.
    """
    return DEFAULT_AG_AE_MIN_EXPOSURE_MS[_ag_ae_channel_class(channel)]


def _ag_ae_channel_class(channel: str) -> Modality:
    if channel in common_utils.get_transmitted_channels():
        return Modality.TRANSMITTED
    if channel in common_utils.get_luminescence_channels():
        return Modality.LUMINESCENCE
    return Modality.FLUORESCENCE


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------


def get_current_plate_position(
    scope: 'Lumascope', labware_id: str, axes: 'typing.Iterable[str]' = ('X', 'Y', 'Z')
) -> PlateLocation:
    """Get current plate position in plate coordinates.

    Args:
        labware_id: The plate the position is stated on. A step is stored
            on its protocol's plate, which need not be the scope's
            selection; converting through another plate puts it in a frame
            the run will not drive it in.
        axes: The stage axes the caller keeps; an axis not named is None and
            is not read, so a move on it refuses nothing. X and Y are one
            conversion to the plate, so naming either reads both.

    Returns:
        The position on the plate: X and Y in mm, Z in um. An axis this
        scope has no motor for is None: there is nothing to read there, and
        a number in its place (the origin, or the stage origin converted to
        the plate) would be saved as a place the scope never measured.

    Raises:
        HardwareCommandRefusedError: ``'not_connected'`` -- this scope's
            model has a motor controller and none is connected, so there is
            no position to read. A step or a run recorded at a made-up
            position would image the wrong place under the right name.
        AxisMovingError: ``'axis_moving'``, an axis it reads is moving, so
            its number is a place it is passing through.
        CatalogueNameRefusedError: ``'labware_unknown'``, ``labware_id``
            is not a plate the catalogue has. Converting through a different
            plate would put every position
            in the wrong frame.
    """
    scope.motion.refuse_controller_not_connected('plate_position')
    labware_id = scope.wellplate_loader.resolve_plate_key(labware_id)
    pos = scope.motion.settled_positions(_plate_axes(scope, axes))
    return _on_plate(scope, labware_id, pos)


def kept_plate_position(
    scope: 'Lumascope', labware_id: str, axes: 'typing.Iterable[str]', *, then: str
) -> PlateLocation:
    """The current plate position, as a position to keep: every axis read at rest and known.

    ``plate_position`` for a caller that keeps the position
    itself and has no owner to refuse an unknown axis after the read: the
    states and the numbers are one snapshot (``positions_to_keep``), so a
    home that starts between a separate state question and the read cannot
    pass the number it left behind.

    Args:
        labware_id: The plate the position is stated on.
        axes: The stage axes the caller keeps, as ``plate_position``
            takes them.
        then: What the user does once the scope knows its position, ending
            the refusal (e.g. ``'save the bookmark'``).

    Returns:
        The position on the plate, as ``plate_position`` answers.

    Raises:
        HardwareCommandRefusedError: ``'not_connected'`` -- this scope's
            model has a motor controller and none is connected.
        AxisMovingError: ``'axis_moving'``, an axis it reads is moving.
        AxisStateUnknownError: ``'position_unknown'``, an axis it reads is
            homing or lost its reference.
        CatalogueNameRefusedError: ``'labware_unknown'``, ``labware_id``
            is not a plate the catalogue has.
    """
    # The cable before the reference: with the controller gone every axis
    # is unknown, and the remedy is the cable, not a home.
    scope.motion.refuse_controller_not_connected('plate_position')
    labware_id = scope.wellplate_loader.resolve_plate_key(labware_id)
    pos = scope.motion.positions_to_keep(_plate_axes(scope, axes), then=then)
    return _on_plate(scope, labware_id, pos)


def _plate_axes(scope: 'Lumascope', axes: 'typing.Iterable[str]') -> tuple[str, ...]:
    """The axes a plate read reads: those asked that the scope has, X and Y together."""
    wanted = set(axes)
    if wanted & {'X', 'Y'}:
        wanted |= {'X', 'Y'}
    present = scope.capabilities.axes
    return tuple(axis for axis in ('X', 'Y', 'Z') if axis in wanted and axis in present)


def _on_plate(
    scope: 'Lumascope', labware_id: str, pos: 'typing.Mapping[str, float]'
) -> PlateLocation:
    """Stage positions, as a position on the plate ``labware_id`` names; an axis not read is None."""
    x_mm = y_mm = z_um = None
    if 'X' in pos and 'Y' in pos:
        plate = scope.runtime_state.stage_to_plate(pos['X'], pos['Y'], labware_id=labware_id)
        x_mm = round(plate.x_mm, common_utils.max_decimal_precision('x'))
        y_mm = round(plate.y_mm, common_utils.max_decimal_precision('y'))
    if 'Z' in pos:
        z_um = round(pos['Z'], common_utils.max_decimal_precision('z'))
    return PlateLocation(x_mm=x_mm, y_mm=y_mm, z_um=z_um)


# ---------------------------------------------------------------------------
# System / logging helpers
# ---------------------------------------------------------------------------


def log_environment_once() -> None:
    """Log fixed environment fingerprint -- system boot time, uptime, OS build,
    Pylon SDK version, Defender state. Once-per-startup pairs with the
    per-tick log_system_metrics surface so post-mortem can correlate a
    runtime metric trace against the host's exact version state.

    Call once at startup before the periodic log_system_metrics schedule.
    Long-uptime hosts can be in pathological states (e.g. memory
    exhaustion at 632h that contaminated one perf-investigation run); without recording boot time, that contamination is
    invisible in post-hoc log analysis.
    """
    import platform as _platform

    boot = common_utils.host_boot()
    boot_dt = wall_time.local_time(round(boot.boot_epoch_s * 1e9)).isoformat(timespec='seconds')

    try:
        os_release = _platform.platform()
    except Exception:
        os_release = 'NA'

    try:
        ncores = psutil.cpu_count(logical=True)
    except Exception:
        ncores = -1

    pylon_ver = 'NA'
    try:
        from pypylon import pylon as _pylon

        pylon_ver = getattr(_pylon, '__version__', 'NA')
    except Exception:
        pass

    defender_state = 'NA'
    if common_utils._IS_WINDOWS:
        try:
            import subprocess as _sub

            out = (
                _sub.check_output(
                    [
                        'powershell',
                        '-Command',
                        '(Get-MpComputerStatus | '
                        'Select-Object -Property RealTimeProtectionEnabled,'
                        'AntivirusSignatureLastUpdated,'
                        'QuickScanStartTime | ConvertTo-Json -Compress)',
                    ],
                    stderr=_sub.DEVNULL,
                    creationflags=0x08000000,  # CREATE_NO_WINDOW
                    timeout=15,
                )
                .decode('utf-8', errors='ignore')
                .strip()
            )
            defender_state = out or 'NA'
        except Exception:
            pass

    metrics_logger.info(
        f'[ENV METRICS] boot={boot_dt} | uptime_hr={boot.uptime_hr:.1f} | '
        f'os={os_release} | cores={ncores} | pylon={pylon_ver} | '
        f'defender={defender_state}',
    )


@typing.runtime_checkable
class DisplayRates(typing.Protocol):
    """A live display's own rates, which only a host with a display has.

    The GUI hands its display down through ``ScopeSession.enable_metrics``;
    a module never reaches up to it.
    """

    def display_fps(self) -> float:
        """The frames drawn in the last second.

        Raises:
            Nothing.
        """
        ...

    def frame_interval_percentiles_ms(self) -> dict[str, float]:
        """``p50``, ``p95``, ``p99`` and ``max`` intervals between drawn
        frames, in ms, and ``n`` the samples; empty before the first.

        Raises:
            Nothing.
        """
        ...


class _Tick(typing.NamedTuple):
    """What one metrics tick reads besides the host itself."""

    live_folder: str
    collect_open_files: bool
    scope: 'Lumascope'
    executors: 'ExecutorBundle'
    display: DisplayRates | None


class _Warned(str):
    """A line one of whose fields failed its probe: written at WARNING, so
    it reaches the errors log as a line that failed whole does."""


_MIB = 1024 * 1024


def _named(error: BaseException) -> str:
    """The error's type, qualified by its module unless it is a builtin."""
    kind = type(error)
    if kind.__module__ == 'builtins':
        return kind.__qualname__
    return f'{kind.__module__}.{kind.__qualname__}'


def _failed(value) -> bool:
    return isinstance(value, (BaseException, common_utils.CounterFailed))


def _field(value, spec: str, unit: str = '') -> str:
    """One field of a line: its reading, ``NA`` where there is none yet or
    this host lacks its counter, or ``unavailable`` where its probe failed."""
    if value is None:
        return f'NA{unit}'
    if isinstance(value, common_utils.CounterFailed):
        return f'unavailable: 0x{value.status:08x}'
    if isinstance(value, BaseException):
        return f'unavailable: {_named(value)}'
    return f'{value:{spec}}{unit}'


def _per(value, unit: float):
    """A reading in ``unit``; anything but a number passes through."""
    return value / unit if is_finite_number(value) else value


def _read(probe: typing.Callable[[], object]) -> object:
    """The probe's reading, or the error it raised: for a field whose
    failure must not take its line's other fields with it."""
    try:
        return probe()
    except Exception as failed:
        return failed


def _leveled(text: str, *values) -> str:
    return _Warned(text) if any(_failed(v) for v in values) else text


# Each line below answers its text, or None where its group does not apply
# on this host or this tick; a raise is the line's failure.


def _system_line(tick: _Tick) -> str:
    # System uptime per-tick -- pairs with [ENV METRICS] boot timestamp
    # logged once at startup. Lets post-hoc log analysis tag each tick
    # with "system has been up for N hours" to filter out runs
    # contaminated by long-uptime memory exhaustion.
    cpu = common_utils.system_cpu_percent()
    vmem = psutil.virtual_memory()
    uptime_hr = common_utils.host_boot().uptime_hr
    return (
        f'CPU Usage: {_field(cpu, ".1f", "%")} | '
        f'RAM Available: {vmem.available / 1e9:.1f} GB | '
        f'RAM Usage: {vmem.percent:.1f}% | uptime_hr={uptime_hr:.1f}'
    )


def _disk_line(tick: _Tick) -> str:
    # The live folder as stored: the tick never creates it, and a folder it
    # cannot read fails this line rather than reporting another volume.
    disk = psutil.disk_usage(tick.live_folder)
    free_mb = disk.free / _MIB
    if free_mb < 1024:  # Less than 1 GB
        logger.error(
            f'Low disk space: {free_mb:.1f} MB remaining',
        )
    return f'Disk Free: {disk.free / 1024**3:.1f} GB | Disk Usage: {disk.percent:.1f}%'


def _process_line(tick: _Tick) -> str:
    proc = common_utils.self_process()
    cpu = common_utils.process_cpu_percent()
    mem = proc.memory_info()
    # Private memory bytes (Windows-specific; falls back to RSS).
    # Working Set / RSS underreports because Windows can trim it while the
    # process still holds committed virtual memory. Private bytes is what
    # Task Manager calls "Memory (private working set)" and is the actual
    # leak indicator.
    private = getattr(mem, 'private', mem.rss)
    return (
        f'Process CPU Usage: {_field(cpu, ".1f", "%")} | '
        f'Process RAM Usage: {mem.rss / 1e6:.1f} MB, '
        f'{proc.memory_percent():.1f}% | '
        f'Private: {private / 1e6:.1f} MB'
    )


def _extra_disks_line(tick: _Tick) -> str | None:
    return common_utils.get_extra_disks_info(exclude_path=tick.live_folder)


# --- Long-run stability metrics ---
# Each line is grep-able and routed through metrics_logger to
# metrics.log so errors.log stays signal-only. Each count levels off
# in the first minutes of a run (threads within 30 s); a leak is a
# steady climb across successive hourly lines.


def _gdi_line(tick: _Tick) -> str | None:
    # GDI / USER objects (Windows only). The #1 cause of "Windows feels
    # slow after 24 hours" -- process limit is 10k, desktop degrades at ~5k.
    counts = common_utils.gui_resources()
    if counts is None:
        return None
    gdi, user = counts
    return f'gdi={gdi} | user={user}'


def _handle_line(tick: _Tick) -> str:
    # OS handles (Windows) / file descriptors (POSIX), and the open files.
    # Windows caps process handles around 16M but typical apps run
    # 500-2000. A steady climb of a few handles per minute is a leak
    # of file/socket/thread handles. POSIX fds equivalent.
    #
    # The open-files count is opt-in, off unless a bench operator asks.
    # It tells you which files are leaked when handles climb, but it costs
    # one os.stat per open handle -- so it scales with the very quantity
    # it exists to watch, and is most expensive during the leak it is
    # meant to catch. On Windows it also enumerates through a
    # C-extension call that holds the GIL, stalling every thread in the
    # process, and a handle whose name query times out is silently
    # skipped -- so the count reads LOW exactly when it matters. The
    # flat-cost handle count is the trend indicator; this is the
    # follow-up you switch on deliberately.
    #
    # Three states, three renderings: a count, unavailable for a probe that
    # ran and failed, and "off" for one nobody asked to run. Each field
    # fails alone, so neither reading hides the other.
    proc = common_utils.self_process()
    count = proc.num_handles if hasattr(proc, 'num_handles') else proc.num_fds
    handles = _read(count)
    if tick.collect_open_files:
        open_files = _read(lambda: len(proc.open_files()))
        open_files_field = _field(open_files, 'd')
    else:
        open_files = None
        open_files_field = 'off'
    return _leveled(
        f'handles={_field(handles, "d")} | open_files={open_files_field}', handles, open_files
    )


def _thread_line(tick: _Tick) -> str:
    # Thread count. Should plateau ~20-25 (8 executors * 2 threads +
    # camera + main + a few Kivy); growth means an executor/handler is
    # spawning without joining.
    count = threading.active_count()
    # Compact name summary: dedupe by stem (e.g. "ThreadPoolExecutor-3_4")
    # so 8 executor pool threads collapse to one entry with count.
    name_summary = collections.Counter(
        t.name.split('-')[0] if '-' in t.name else t.name for t in threading.enumerate()
    )
    names_str = ', '.join(f'{k}={v}' for k, v in sorted(name_summary.items()))
    return f'count={count} | {names_str}'


def _thread_cpu_line(tick: _Tick) -> str | None:
    # Per-thread CPU over the interval -- which thread is hot (protocol vs
    # scope_display vs camera). Pairs with [THREAD METRICS] counts above; same
    # names. Absent until there is a baseline, and on a tick where no thread
    # had one. Percentages are per-core (100 = one full core) so total tracks
    # [PROCESS METRICS] CPU.
    thread_cpu = common_utils.thread_cpu_percentages()
    if not thread_cpu:
        return None
    ranked = sorted(thread_cpu.items(), key=lambda kv: kv[1], reverse=True)
    hot = ' '.join(f'{name}={pct:.1f}%' for name, pct in ranked if pct >= 0.1)
    total = sum(thread_cpu.values())
    return f'total={total:.1f}% | {hot if hot else "(all idle)"}'


def _gpu_line(tick: _Tick) -> str | None:
    # GPU utilization + memory (Windows, vendor-agnostic via PDH GPU Engine
    # counters -- AMD / Intel / NVIDIA alike). 3D is the OpenGL/Kivy render
    # engine; shared memory is the meaningful figure on an integrated GPU
    # (no dedicated VRAM).
    gpu = common_utils.query_gpu_metrics()
    if gpu is None:
        return None
    return _leveled(
        f'util={_field(gpu.util_total_percent, ".1f", "%")} '
        f'(3d={_field(gpu.util_3d_percent, ".1f", "%")}) | '
        f'shared_mem={_field(gpu.shared_mem_mb, ".0f", " MB")} | '
        f'dedicated_mem={_field(gpu.dedicated_mem_mb, ".0f", " MB")}',
        *gpu,
    )


def _gc_line(tick: _Tick) -> str:
    # Python heap (catches reference-cycle / closure-capture leaks).
    # The count is of the memory blocks Python's allocator holds, which it
    # keeps as a running total. Not len(gc.get_objects()): that list hands
    # this thread every object the collector tracks, a tuple another thread
    # is still building among them, and that thread's tuple() then raises
    # SystemError -- a protocol run ended that way.
    # Steady linear growth indicates accumulation, typically from
    # observers/callbacks holding refs; the growth-rate thresholds are
    # the allocated_blocks_growth_per_hr row in PERFORMANCE_BUDGETS.md.
    blocks = sys.getallocatedblocks()
    stats = gc.get_stats()
    return (
        f'blocks={blocks} | '
        f'gen0={stats[0]["collections"]} '
        f'gen1={stats[1]["collections"]} '
        f'gen2={stats[2]["collections"]}'
    )


def _swap_line(tick: _Tick) -> str:
    # Swap pressure. "RAM looks low" doesn't catch page-file thrashing.
    swap = psutil.swap_memory()
    return f'used={swap.used / 1e9:.1f} GB ({swap.percent:.1f}%)'


def _process_io_line(tick: _Tick) -> str | None:
    # Per-process I/O bytes (cumulative + per-second rates).
    # Cumulative distinguishes "we wrote 50 GB this hour" from "Windows
    # Defender did"; both counters reset only when the process restarts.
    # Rates give the steady-state save rate to compare against
    # camera_data_rate in [BUFFER METRICS] and Defender's read rate.
    # The rates are averages over the gap since the previous tick, so they
    # mean different things at different cadences: at the 60 s engineering
    # interval they read as MB/sec of TIFF writes; at the hourly production
    # interval they are hour-averages and will flatten any burst.
    proc = common_utils.self_process()
    if not hasattr(proc, 'io_counters'):  # psutil has none on macOS
        return None
    io = proc.io_counters()
    read_rate = _per(common_utils.delta_rate('proc_io_read', io.read_bytes), 1e6)
    write_rate = _per(common_utils.delta_rate('proc_io_write', io.write_bytes), 1e6)
    return (
        f'read={io.read_bytes / 1e6:.1f} MB | write={io.write_bytes / 1e6:.1f} MB | '
        f'read_rate={_field(read_rate, ".2f", " MB/s")} | '
        f'write_rate={_field(write_rate, ".2f", " MB/s")}'
    )


# The memory record's page-fault count, by its name on each platform:
# Windows, then macOS. Linux's record has none.
_PAGE_FAULT_FIELDS = ('num_page_faults', 'pfaults')


def _page_fault_line(tick: _Tick) -> str | None:
    # Page-fault rate.
    # Sustained > 1000 pf/sec on a desktop = real memory pressure (paging
    # working set in/out). If pf/sec stays low while standby grows, the
    # slowdown is allocator/standby-cache, not real paging. If pf/sec
    # spikes during slow state, real paging.
    # That threshold only means anything at the sub-minute engineering
    # cadence: averaged over an hour, a paging storm lasting minutes
    # disappears into the mean.
    mem = common_utils.self_process().memory_info()
    field = next((f for f in _PAGE_FAULT_FIELDS if f in mem._fields), None)
    if field is None:
        return None
    total = getattr(mem, field)
    rate = common_utils.delta_rate('page_faults', total)
    return f'total={total} | rate={_field(rate, ".1f", "/s")}'


def _summed(parts: list):
    """The parts' sum: a part's failure where one failed, None where one
    has no reading."""
    failed = next((p for p in parts if _failed(p)), None)
    if failed is not None:
        return failed
    if any(p is None for p in parts):
        return None
    return sum(parts)


def _pdh_line(tick: _Tick) -> str | None:
    # --- Windows PDH counters ---
    # Standby cache + nonpaged pool are the specific signals for
    # diagnosing buffer-churn / slow-onset memory growth on Windows.
    #
    # Standby split: Normal + Reserve + Core = total standby cache.
    #   - Standby growing while RAM available stays high -> mapped-file
    #     accumulation (the slowdown signal).
    #   - Nonpaged pool growing -> kernel-side leak (Pylon DMA, drivers).
    #   - System cache (\Memory\Cache Bytes) is the file-system cache --
    #     overlaps with standby on Windows; track both for cross-check.
    pdh = common_utils.query_windows_perf_counters()
    if pdh is None:
        return None

    def mb(key: str, unit: str = '') -> str:
        return _field(_per(pdh.get(key), _MIB), '.0f', unit)

    standby_total = _summed(
        [
            pdh.get(k)
            for k in ('standby_normal_bytes', 'standby_reserve_bytes', 'standby_core_bytes')
        ]
    )
    return _leveled(
        f'standby_total={_field(_per(standby_total, _MIB), ".0f", " MB")} '
        f'(normal={mb("standby_normal_bytes")} '
        f'reserve={mb("standby_reserve_bytes")} '
        f'core={mb("standby_core_bytes")}) | '
        f'nonpaged_pool={mb("pool_nonpaged_bytes", " MB")} | '
        f'paged_pool={mb("pool_paged_bytes", " MB")} | '
        f'sys_cache={mb("system_cache_bytes", " MB")} | '
        f'modified={mb("modified_page_bytes", " MB")} | '
        f'free_zero={mb("free_zero_bytes", " MB")} | '
        f'available={mb("available_bytes", " MB")} | '
        f'commit={mb("commit_bytes")}/{mb("commit_limit_bytes", " MB")}',
        *pdh.values(),
    )


def _buffer_line(tick: _Tick) -> str:
    # --- Buffer-churn signals from the live capture path ---
    # What the camera delivered over the last second and the bytes it put on
    # the link, from the imaging API in every host. Each frame currently
    # allocates ~3 fresh OS-level buffers (camera copy, 12->8 LUT, tobytes()).
    # The standby-cache growth in [PDH METRICS] should track the data rate
    # roughly. The display's rate is the GUI's, logged only where one exists.
    delivered = tick.scope.imaging.delivered_rate
    wire_frame_bytes = (
        delivered.bytes_per_s / delivered.frame_rate_fps if delivered.frame_rate_fps else 0.0
    )
    display_field = (
        f'display_fps={tick.display.display_fps():.1f} | ' if tick.display is not None else ''
    )
    return (
        f'capture_fps={delivered.frame_rate_fps:.1f} | '
        f'{display_field}'
        f'camera_data_rate={delivered.megabytes_per_s:.1f} MB/s | '
        f'frame_size={wire_frame_bytes / 1000:.0f} KB'
    )


def _frame_interval_line(tick: _Tick) -> str | None:
    # Frame-interval percentiles -- consumer-stall detection.
    # Spikes in p99/max correlate with main-thread congestion
    # or worker-thread blocks; tracking these surfaces UI lock
    # contention and IO scheduling issues.
    if tick.display is None:
        return None
    pcts = tick.display.frame_interval_percentiles_ms()
    if not pcts:
        return None
    return (
        f'p50={pcts["p50"]:.1f} ms | '
        f'p95={pcts["p95"]:.1f} ms | '
        f'p99={pcts["p99"]:.1f} ms | '
        f'max={pcts["max"]:.1f} ms | '
        f'n={pcts["n"]}'
    )


def _defender_line(tick: _Tick) -> str | None:
    # --- Defender (MsMpEng.exe) metrics ---
    # Direct signal for the "Defender memory-maps every TIFF write"
    # interaction. If Defender's io_read_rate tracks our [PROCESS IO]
    # write_rate x ~1, Defender is the slowdown source. Its private
    # memory growing alongside [PDH METRICS] standby_total is also
    # implicating.
    defender = common_utils.query_defender_metrics()
    if defender is None:
        return None
    return _leveled(
        f'private={defender.private_mb:.0f} MB | '
        f'rss={defender.rss_mb:.0f} MB | '
        f'io_read_total={_field(defender.io_read_total_mb, ".0f", " MB")} | '
        f'io_read_rate={_field(defender.io_read_mbps, ".2f", " MB/s")}',
        defender.io_read_total_mb,
        defender.io_read_mbps,
    )


def _gc_pressure_line(tick: _Tick) -> str:
    # --- GC pressure ---
    # gc.get_count() = current uncollected objects per generation (depth).
    # [GC METRICS]' genN = collections-since-start (rate). Both together
    # separate "lots of churn but clean steady state" from "real
    # generation-2 leak."
    g0, g1, g2 = gc.get_count()
    return f'gen0_depth={g0} | gen1_depth={g1} | gen2_depth={g2}'


def _queue_line(tick: _Tick) -> str:
    # --- Queue depth ---
    # Monotonic queue growth = save can't keep up, frames pile up
    # retaining 16-48 MB each. Direct mechanism for slow-onset
    # multi-hour memory growth on the save path. The session's own
    # executors, so every host logs it.
    states = {**tick.executors.snapshot(), **tick.executors.thread_states()}
    return ' | '.join(f'{name}={value}' for name, value in states.items())


def _futures_line(tick: _Tick) -> str:
    # Per-executor caller_futures alloc/pop/live counters. Sustained
    # drift between allocs and pops means Future entries are
    # accumulating in caller_futures; on Windows each orphan Future
    # leaves a Lock/Semaphore kernel handle open. Steady-state
    # invariant: alloc == pop within a few per cadence (each
    # in-flight task is one of the few "live" entries).
    parts = []
    for name, executor in tick.executors.queued_executors():
        allocs, pops, live = executor.caller_futures_stats()
        parts.append(f'{name}: alloc={allocs} pop={pops} live={live}')
    return ' | '.join(parts)


# The tick's lines, in the order they are written, each under the tag a
# reader greps.
_METRICS_LINES: tuple[tuple[str, typing.Callable[[_Tick], str | None]], ...] = (
    ('SYSTEM METRICS', _system_line),
    ('DISK METRICS', _disk_line),
    ('PROCESS METRICS', _process_line),
    ('EXTRA DISKS', _extra_disks_line),
    ('GDI METRICS', _gdi_line),
    ('HANDLE METRICS', _handle_line),
    ('THREAD METRICS', _thread_line),
    ('THREAD CPU', _thread_cpu_line),
    ('GPU METRICS', _gpu_line),
    ('GC METRICS', _gc_line),
    ('SWAP METRICS', _swap_line),
    ('PROCESS IO', _process_io_line),
    ('PAGE FAULTS', _page_fault_line),
    ('PDH METRICS', _pdh_line),
    ('BUFFER METRICS', _buffer_line),
    ('FRAME INTERVAL', _frame_interval_line),
    ('DEFENDER METRICS', _defender_line),
    ('GC PRESSURE', _gc_pressure_line),
    ('QUEUE METRICS', _queue_line),
    ('FUTURES METRICS', _futures_line),
)


def log_system_metrics(
    settings: dict,
    *,
    scope: 'Lumascope',
    executors: 'ExecutorBundle',
    display: DisplayRates | None = None,
) -> None:
    """Log CPU, RAM, and disk metrics, the scope's camera delivery, the
    session's executor queues and futures, and the display's rates when
    the host has a display.

    Each line is written on its own: a line whose probe fails is written
    as ``[<TAG>] unavailable: <type>: <message>`` at WARNING, and every
    other line is still written. A line whose group does not apply on
    this host is left out.
    """
    # `is True`, not truthiness: nothing validates the settings files, so a
    # hand-edited value of the STRING "false" would otherwise read truthy and
    # switch on a probe that can stall the process for hundreds of ms a tick.
    # Any non-boolean reads as off, which is the safe direction.
    tick = _Tick(
        live_folder=settings['live_folder'],
        collect_open_files=settings['profiling']['open_files_enabled'] is True,
        scope=scope,
        executors=executors,
        display=display,
    )
    for tag, line in _METRICS_LINES:
        try:
            text = line(tick)
        except Exception as failed:
            message = str(failed)
            metrics_logger.warning(
                f'[{tag}] unavailable: {_named(failed)}' + (f': {message}' if message else '')
            )
            continue
        if text is None:
            continue
        write = metrics_logger.warning if isinstance(text, _Warned) else metrics_logger.info
        write(f'[{tag}] {text}')


# ---------------------------------------------------------------------------
# Headless config getters
#
# These read from the settings dict (or scope object), never from Kivy
# widgets, so the REST API, the GUI and any non-GUI context share them.
# ---------------------------------------------------------------------------

# The usable exposure FLOOR per channel class (ms): see
# get_ag_ae_min_exposure_ms. Transmitted light can legitimately run at a
# tenth of a millisecond; sub-millisecond fluorescence or luminescence is
# never realistic and its sliders start at 1 ms.
DEFAULT_AG_AE_MIN_EXPOSURE_MS = {
    Modality.TRANSMITTED: 0.1,
    Modality.FLUORESCENCE: 1.0,
    Modality.LUMINESCENCE: 1.0,
}


def camera_max_gain_for_ui(limits: 'CameraSettingLimits | None') -> float | None:
    """The gain-slider upper bound from the camera's limits; None with no
    maximum known (no camera, its range unread, or none declared), where a
    slider keeps its own.

    Normalised to the same precision every other gain number in the app
    carries. A GenICam float node reports its maximum in continuous units and
    that value can carry a float tail past the last real step -- a 48 dB gain
    node reports 48.00000004350822 -- which the driver publishes raw whenever
    the node declares no fixed increment, deliberately, because inventing a
    step would narrow a range the camera did not narrow. Raw, it becomes the
    slider maximum, the ceiling a typed entry is clamped to, and the value
    written into the store when a channel is reconciled down to the cap, so the
    tail reaches current.json and the gain box. This resolver is the one point
    the whole chain passes through, so it is the one place to normalise it.

    The precision comes from the existing owner rather than a number chosen
    here: a second authority on the same quantity could disagree with the
    first. Rounding rather than flooring matches how every other gain value is
    normalised, and the overshoot it can introduce is orders below the
    driver's own short-circuit tolerance, so it cannot produce a request the
    camera treats as a change.
    """
    if limits is None or limits.gain_max_db is None:
        return None
    return round(limits.gain_max_db, common_utils.max_decimal_precision('gain'))


# The manual ceilings are each channel's catalogue record's
# (`manual_max_exposure_ms`, `manual_max_illumination_ma` in scopes.json).
#
# Illumination: the transmitted LEDs (BF / PC / DF) are far brighter per mA
# than the fluorescence channels, so their slider stops well under the
# board's ceiling; the text entry reaches past it for the rare sample that
# needs it (Eric, 2026-09-12: "slider at 50, but you can type up to 500
# still"), up to what the connected board can be asked for, which the
# settings writer refuses to exceed.
#
# Exposure: transmitted light is bright enough that the useful manual range
# sits far under the sensor's maximum, BF's lowest of all. Fluorescence
# exposures past a second stop being a usable manual range: the Pylon sensors
# reach ten seconds, which would put every useful slider position in the
# first tenth of the track. Luminescence has no ceiling -- integrating as long
# as the sensor allows is that channel's purpose, so its slider goes to the
# camera's own cap (Eric, 2026-09-12).
#
# Deliberately NOT settings['ag_ae_max_exposure_ms']: that bounds what the AUTO
# loop may drive to and is set per install, so reusing it would let auto-exposure
# tuning silently resize the manual slider.


def _narrowed(cap: float | None, ceiling: float | None) -> float | None:
    """A connected part's cap narrowed to a channel's manual ceiling: None with
    no part to cap it, the cap where the channel has no ceiling."""
    if cap is None or ceiling is None:
        return cap
    return min(ceiling, cap)


def channel_slider_maxima(
    channel: str,
    *,
    exposure_cap_ms: float | None,
    illumination_cap_ma: int | None,
    gain_limits: 'CameraSettingLimits | None',
) -> ChannelSliderMaxima:
    """One channel's manual slider maxima: the connected camera's exposure cap
    and the current cap of the channel's LED, each narrowed to the channel's
    manual ceiling where it has one; None where there is no cap (no camera,
    or its range not known; no LED board, or no LED for the channel). The
    gain slider's is the camera's gain maximum (``camera_max_gain_for_ui``).

    Raises:
        ArgumentRefusedError: ``channel`` is not a channel of this release.
    """
    catalogue = release_channel(channel)
    return ChannelSliderMaxima(
        slider_max_exposure_ms=_narrowed(exposure_cap_ms, catalogue.manual_max_exposure_ms),
        slider_max_illumination_ma=_narrowed(
            illumination_cap_ma, catalogue.manual_max_illumination_ma
        ),
        slider_max_gain_db=camera_max_gain_for_ui(gain_limits),
    )


def get_binning_from_settings(settings: dict) -> int:
    """Read binning size from settings dict (no UI needed).

    Reads the key the GUI actually writes and the template actually ships,
    ``settings['binning']['size']``, which holds the selector's label ('2x2')
    rather than a factor. This used to read a top-level ``binning_size``
    integer that no code writes and no shipped template carries, so it
    answered 1 for every configuration -- a headless caller silently captured
    unbinned while the screen showed 2x2. Scope bring-up already reads the
    label key this way, so this makes the two agree instead of adding a
    second convention.
    """
    return binning.binning_size_str_to_int(settings['binning']['size'])


def get_frame_dimensions_from_settings(settings: dict) -> dict:
    """Read frame dimensions from settings dict (no UI needed)."""
    frame = settings['frame']
    return {
        'width': int(frame['width']),
        'height': int(frame['height']),
    }


def get_protocol_time_params_from_settings(settings: dict) -> dict:
    """The stored default period and duration, as a new protocol takes them.

    Returns dict with 'period' and 'duration' as timedelta objects, converted
    from the store's minutes and hours and otherwise unchanged.

    Raises:
        ProtocolScheduleRefusedError: a stored period or duration is not one the
            protocol can run. The store's two writers refuse one, so this
            means a caller built the dict by hand.
    """
    from _lumascope.modules.protocol import schedule_from_units

    protocol = settings['protocol']
    return {key: schedule_from_units(key, protocol[key]) for key in ('period', 'duration')}


def get_image_capture_config_from_settings(settings: dict) -> image_mode.ImageCaptureConfig:
    """Read image capture config from settings dict (no UI needed).

    Reads the stored image_mode (and its capture_depth / save_encoding); the
    load folds the legacy keys of an older install into it before the store
    is prepared.
    """
    output_format = settings['image_output_format']
    mode = settings['image_mode']
    return image_mode.ImageCaptureConfig.from_image_mode(
        mode,
        output_format_live=output_format['live'],
        output_format_sequenced=output_format['sequenced'],
        jpg_quality=settings['jpg_quality'],
    )


def get_selected_labware_from_settings(
    settings: dict,
    wellplate_loader: WellPlateLoader,
) -> tuple[str, object]:
    """The selected plate, as ``(labware_id, wellplate_object)``; never None.

    The id is the catalogue's key for the stored name, so a plate renamed
    since the settings named it is found under its old spelling.

    Raises:
        CatalogueNameRefusedError: ``'labware_unknown'``, the settings name
            a plate the catalogue does not have; ``offered`` carries the
            plates it has. No other plate is substituted: a different plate's geometry puts
            every well position in the wrong place while the protocol reads
            as if it ran normally. Bring-up (``ScopeSession.create``)
            replaces a stored plate the catalogue cannot resolve; this
            reader still refuses one in settings no bring-up has read.
    """
    labware_id = wellplate_loader.resolve_plate_key(settings['protocol']['labware'])
    return labware_id, wellplate_loader.get_plate(labware_id=labware_id)


def get_zstack_params_from_settings(settings: dict) -> dict:
    """Read z-stack params from the settings store.

    The store keeps the stack in a TOP-LEVEL ``zstack`` container -- not under
    ``protocol`` -- and keeps its reference in ``position``. Both are written
    by the z-stack UI. Reading a ``protocol.zstack`` container and a
    ``z_reference`` leaf found neither and answered with invented values
    instead of failing.

    The store is prepared, so every key is read, and each holds what its
    writer and the load allow: ``range`` and ``step_size`` a finite number,
    ``position`` a ``ZStackReference`` value. The shipped ``step_size`` is 0,
    so an unconfigured store reports no stack.
    """
    zstack = settings['zstack']
    return {
        'range': float(zstack['range']),
        'step_size': float(zstack['step_size']),
        'z_reference': ZStackReference(zstack['position']),
    }


def build_sequenced_capture_config(values: dict) -> dict:
    """Assemble the canonical sequenced-capture input_config from a flat values
    dict.

    Single source for the key set Protocol.from_config consumes. Every source
    lane (UI / settings / zstack) routes its values through here, so no lane can
    silently omit a key -- in particular tiling_overlap_percent, whose omission
    used to force the settings/headless lane to a silent 0% overlap. The key is
    always present here (defaulting to 0.0) rather than relying on each lane to
    remember it. `positions` is included only when a lane supplies explicit
    positions (the z-stack single-position case); otherwise Protocol.from_config
    derives positions from the labware.
    """
    config = {
        'labware_id': values['labware_id'],
        'objective_id': values['objective_id'],
        'zstack_params': values['zstack_params'],
        'use_zstacking': values['use_zstacking'],
        'tiling': values['tiling'],
        'tiling_overlap_percent': values.get('tiling_overlap_percent', 0.0),
        'channel_configs': values['channel_configs'],
        'period': values['period'],
        'duration': values['duration'],
        'frame_dimensions': values['frame_dimensions'],
        'binning_size': values['binning_size'],
        'current_z': values['current_z'],
    }
    if 'positions' in values:
        config['positions'] = values['positions']
    return config


# The step Label / Well every composite channel shares. Protocol.from_config
# requires a name on an explicit position, and the merge groups its inputs by
# it -- one constant name is what puts every channel in one group.
COMPOSITE_POSITION_NAME = 'Composite'

# A merge needs at least two channels to combine: the merge skips groups of
# one, so a single-channel composite produces no artifact at all.
COMPOSITE_MIN_CHANNELS = 2


def get_composite_channels(settings: dict) -> list:
    """The channels a composite run captures, in capture order.

    Every channel set to capture contributes one step, an image or a video,
    in the drawers' order, the order every build takes its channels in
    (``get_channel_configs``). None is dropped here: a composite has one
    transmitted plane and takes no video, and the run engine refuses a
    second transmitted channel and a Video channel in prepare(), where every
    refusal a run can meet is raised, so the user is told rather than given
    a composite without them.

    Read from the settings snapshot rather than the open accordion: a
    composite must assemble with every channel collapsed, and a headless
    caller has no accordion at all.

    Fewer than COMPOSITE_MIN_CHANNELS is returned as it is, for prepare()
    to refuse the same way.
    """
    return [
        channel
        for channel in common_utils.get_channels()
        if settings[channel]['acquire'] is not None
    ]


def get_composite_blend_thresholds(settings: dict) -> dict:
    """Per-channel blend thresholds the post-run merge needs, as percentages.

    Snapshotted from settings by the caller that has them, because the
    merge runs on a worker thread after the run: reading settings there is
    unavailable headless and would in any case be a later value than the
    run was configured with.
    """
    return {
        channel: settings[channel]['composite_brightness_threshold']
        for channel in common_utils.get_channels()
        if 'composite_brightness_threshold' in settings[channel]
    }


def get_composite_image_capture_config_from_settings(
    settings: dict,
) -> image_mode.ImageCaptureConfig:
    """The capture/save intent for a composite run: a scan's, but for the format.

    The per-channel captures are taken and saved in the user's image mode,
    as a scan's are: they are the data the user keeps, and the merge reads
    each at its own depth; only the merged image is 8-bit RGB.

    The one departure is forced by what the merge consumes: the format
    follows the user's sequenced preference whenever the merge can read it
    back, and is otherwise overridden to OME-TIFF with the substitution
    logged -- a silent one leaves a user wondering why their composite is
    not the format they picked.
    """
    output_format = settings['image_output_format']
    requested = output_format['sequenced']
    if requested in image_mode.MERGEABLE_SEQUENCED_FORMATS:
        resolved = requested
    else:
        resolved = image_mode.OutputFormat.OME_TIFF
        logger.info(
            f'[Composite] Sequenced output format {requested!r} cannot be merged; '
            f'this run saves {resolved!r} instead.'
        )

    return dataclasses.replace(
        get_image_capture_config_from_settings(settings), output_format_sequenced=resolved
    )


def get_composite_capture_config_from_settings(
    settings: dict,
    objective_helper: ObjectiveLoader,
    position: PlateLocation,
) -> dict:
    """Build the input_config for a composite run at *position*.

    A composite is the degenerate sequenced run: one position, one step per
    acquiring channel, no z-stack and no tiling -- either would multiply the
    steps and leave the merge several frames per channel to choose between.

    The position's z is deliberately None. Protocol.from_config falls
    through a missing z to each channel's own stored focus, which is what puts
    every channel in ITS focal plane; a numeric z would pin all of them to
    whatever plane the stage happened to be at and silently discard the
    per-channel focus the user set. The stage's z rides along as the
    current Z, which is where a channel whose focus was never saved is
    imaged.

    """
    channels = get_composite_channels(settings)
    objective_id = settings['objective_id']
    objective_helper.get_objective_info(objective_id=objective_id)

    composite_position = {
        'x': position.x_mm,
        'y': position.y_mm,
        'z': None,
        'name': COMPOSITE_POSITION_NAME,
    }

    return build_sequenced_capture_config(
        {
            # No default: bring-up and the settings writers admit only a
            # plate the catalogue has, and an empty name would be saved into
            # the run's record as its plate.
            'labware_id': settings['protocol']['labware'],
            'objective_id': objective_id,
            'zstack_params': {},
            'use_zstacking': False,
            'tiling': TilingGrids.no_tiling_label(),
            'tiling_overlap_percent': 0.0,
            'channel_configs': get_channel_configs(settings, specific_channels=channels),
            'period': None,
            'duration': None,
            'frame_dimensions': get_frame_dimensions_from_settings(settings),
            'binning_size': get_binning_from_settings(settings),
            'current_z': position.z_um,
            'positions': [composite_position],
        }
    )


def get_standalone_capture_config_from_settings(
    settings: dict,
    objective_helper: ObjectiveLoader,
    wellplate_loader: WellPlateLoader,
    *,
    channel: str,
    position: PlateLocation,
    position_name: str,
    autofocus: bool,
    use_zstacking: bool,
) -> dict:
    """Build the input_config for a one-position, one-channel run at *position*.

    The third settings lane into the canonical builder, beside the
    plate-wide and composite ones. It serves the degenerate case both
    standalone buttons drive -- a single field, a single channel, with
    autofocus or z-stacking turned on -- which until now had no named
    selector, so each starter chose the same thirteen values inline.

    Not expressed as the plate-wide selector plus arguments on purpose:
    this case does not subset that one. It forces tiling off, period and
    duration to nothing, names its position, and overrides the channel's
    stored autofocus flag -- arguments the plate-wide callers could not
    use, in combinations that would mean nothing to them.

    The position keeps its z, unlike the composite lane which nulls it so
    each channel falls to its own focus. Here the z IS the input: it is
    where the autofocus sweep starts, and the plane a z-stack is built
    around.

    Args:
        channel: Which channel to capture. Named by the caller because a GUI
            reads it from the open drawer and a script has none.
        position: Plate coordinates for the single step, from
            get_current_plate_position.
        position_name: What the step is called in the saved data.
        autofocus: Whether the step runs autofocus. Overrides the channel's
            stored flag either way, so a caller gets what it asked for
            rather than what the user last left switched on.
        use_zstacking: Whether the step expands into a z-stack. The
            z-stack parameters are read from settings only when this is
            set; a caller that is not stacking gets none rather than
            stale ones.

    *channel* is a channel of this release: the members that build this
    (``ProtocolRunner.start_autofocus``, ``start_zstack``) judge it at their
    door, before they read the scope. Unjudged, an unknown name would not
    fail here -- the channel selector below skips what the catalogue lacks
    -- and the run would be refused several steps later as "Protocol has
    no steps", naming a cause that has nothing to do with the mistake.
    """
    objective_id = settings['objective_id']
    objective_helper.get_objective_info(objective_id=objective_id)
    labware_id, _ = get_selected_labware_from_settings(settings, wellplate_loader)

    channel_configs = get_channel_configs(settings, specific_channels=[channel])
    channel_config = channel_configs[channel]
    # Both starters do exactly this: the step captures an image, and the
    # caller's intent decides autofocus rather than the channel's stored
    # flag. Assembling a config is not the place to honour a leftover
    # switch the caller said nothing about.
    channel_config['acquire'] = AcquireMode.IMAGE
    channel_config['autofocus'] = autofocus

    step_position = {
        'x': position.x_mm,
        'y': position.y_mm,
        'z': position.z_um,
        'name': position_name,
    }

    return build_sequenced_capture_config(
        {
            'labware_id': labware_id,
            'objective_id': objective_id,
            'zstack_params': get_zstack_params_from_settings(settings) if use_zstacking else {},
            'use_zstacking': use_zstacking,
            'tiling': TilingGrids.no_tiling_label(),
            'tiling_overlap_percent': 0.0,
            'channel_configs': {channel: channel_config},
            'period': None,
            'duration': None,
            'frame_dimensions': get_frame_dimensions_from_settings(settings),
            'binning_size': get_binning_from_settings(settings),
            'current_z': position.z_um,
            'positions': [step_position],
        }
    )


def get_empty_protocol_config_from_settings(
    settings: dict,
    wellplate_loader: WellPlateLoader,
) -> dict:
    """The config an empty-steps protocol takes: labware, timing, geometry.

    No objective: an empty protocol has no step to stamp one into, so it
    can be built while the objective in the light path is unknown. Reached
    through ScopeSession.create_empty_protocol.
    """
    labware_id, _ = get_selected_labware_from_settings(settings, wellplate_loader)
    time_params = get_protocol_time_params_from_settings(settings)
    return {
        'labware_id': labware_id,
        'period': time_params['period'],
        'duration': time_params['duration'],
        'frame_dimensions': get_frame_dimensions_from_settings(settings),
        'binning_size': get_binning_from_settings(settings),
    }


def get_sequenced_capture_config_from_settings(
    settings: dict,
    objective_helper: ObjectiveLoader,
    wellplate_loader: WellPlateLoader,
    *,
    current_z: float | None,
    tiling: str = '1x1',
    use_zstacking: bool = False,
) -> dict:
    """Build a sequenced capture config from settings (no UI needed).

    The single builder for this config; every caller, the GUI included,
    reaches it through ScopeSession.get_sequenced_capture_config, and the
    GUI supplies the two authoring choices below from its widgets.

    tiling and use_zstacking are ARGUMENTS, not settings reads. They are
    authoring inputs with no settings home: tiling's store is the
    protocol itself (its steps carry a Tile column, which is where a
    saved tiling is recovered from), and neither survives a restart by
    design. They used to be read as settings['protocol'] keys that
    nothing writes, so this builder answered '1x1' and False for every
    caller regardless of what the user had chosen.

    wellplate_loader is required: labware goes through the accessor that
    falls back to the shipped default and warns, so this lane and the
    GUI's resolve a missing or unloadable plate the same way instead of
    handing a bare '' to the protocol.
    """
    objective_id = settings['objective_id']
    objective_helper.get_objective_info(objective_id=objective_id)
    time_params = get_protocol_time_params_from_settings(settings)
    labware_id, _ = get_selected_labware_from_settings(settings, wellplate_loader)

    return build_sequenced_capture_config(
        {
            'labware_id': labware_id,
            'objective_id': objective_id,
            'zstack_params': get_zstack_params_from_settings(settings),
            'use_zstacking': use_zstacking,
            'tiling': tiling,
            # Overlap is stored top-level, not under protocol. Reading it from
            # under protocol found nothing and silently gave every headless
            # run 0% overlap regardless of what the user had configured.
            'tiling_overlap_percent': settings['tiling_overlap_percent'],
            'channel_configs': get_channel_configs(settings),
            'period': time_params['period'],
            'duration': time_params['duration'],
            'frame_dimensions': get_frame_dimensions_from_settings(settings),
            'binning_size': get_binning_from_settings(settings),
            'current_z': current_z,
        }
    )
