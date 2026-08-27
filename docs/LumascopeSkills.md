# LumaViewPro — API & Integration Reference

**What is public.** The public surface is defined in code: a member is
public unless its docstring marks it internal ("not part of the L2 API
surface") or an engineering ruling places it on the bench/tech-support
surface. This document is the reference for that surface, and the test
suite holds the two in lockstep both ways -- every call form here must
resolve on the live API, and every public member must appear here. A
member absent from this document is internal or engineering surface:
it may be renamed, moved, or removed in any release without notice,
and code that calls it is unsupported.

## PRE-RELEASE API

The Lumascope SDK API documented in this file is **subject to breaking changes** in 4.1 / 4.1.5 / 4.2. Specifically:

- 4.1.5 ships the sub-API decomposition (Wave 7): hardware-direct methods on `Lumascope` move to sub-APIs (`scope.motion.*`, `scope.illumination.*`, `scope.imaging.*`, `scope.diagnostics.*`, `scope.capabilities.*`, `scope.io.*`). The `Lumascope` class becomes a thin facade; L2 entry point shifts to `ScopeSession`.
- 4.2 ships the capability + wire contract changes that may rename or restructure protocol-level surfaces.
- The REST endpoint convention is **deferred** to a dedicated design session; do not assume current shapes are final.

If you are using this API before stabilization, **contact Etaluma support** so we know to consult you before structural changes. Internal LumaViewPro use does not trigger this requirement.

The warning retires when the first non-`-beta` LumaViewPro release ships (the `4.0.0` git tag on the `4.0.0-beta` lineage). At that point the L2-callable surface freezes for the 4.x major version and every subsequent change is recorded in the [Changelog](#changelog) section below (additive / behavior-change / rename / removal, with version + justification). Until `4.0.0` ships, the API surface stays structurally fluid -- methods may be renamed, moved into sub-APIs, or retired without a deprecation cycle.

---

## Overview

LumaViewPro controls Etaluma microscopes: LED illumination, XYZ stage + turret motion, and camera image acquisition. This document is the integration reference for developers building scripts, headless automation, or external control applications on top of LumaViewPro.

**Repository**: `EtalumaSupport/LumaViewPro`
**Platform**: Python 3.12–3.13, Windows / macOS / Linux

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Your Application                               │
│  (MATLAB, Python script, LabVIEW, web app)      │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  REST surface  (HTTP/JSON, any language)        │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  ScopeSession session layer  (Python, headless) │
│  └─ executor-routed commands, protocol runner   │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Lumascope composition root  (Python)           │
│  ├─ scope.motion        ├─ scope.diagnostics    │
│  ├─ scope.illumination  ├─ scope.capabilities   │
│  ├─ scope.imaging       └─ scope.io             │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  modules/  (image_save, coord_transformations,  │
│             composite_builder, autofocus, …)    │
└─────────────────────────────────────────────────┘
```

Each layer wraps the one below. Higher = easier. Lower = more control.

For internal serial-protocol details (firmware updates, bring-up tooling), see **Appendix A** at the end of this document — not intended for application integration.

---

## Integration Levels

Pick the layer that fits your use case:

| Layer | Interface | Language | Best for |
|---|---|---|---|
| **REST surface** | HTTP (JSON) | Any | External apps, cross-language control |
| **ScopeSession session layer** | Python | Python | Headless scripts, automation, tests |
| **Lumascope + sub-APIs** | Python | Python | Full hardware control, custom applications |

The remainder of this document is organized as the sub-API reference (one section per sub-API), then the modules layer, plugin platform pointers, REST surface, and finally practical patterns + appendices.

### Sentinel-return vs raise contract

Methods on the L2 surface follow one of two contracts; if a method's docstring has a `Raises:` section it follows the raise contract, otherwise the sentinel contract.

- **Hardware-state queries** (capability probes, status reads, getters like `get_led_ma`, `get_target_position`, `get_led_states`, `max_gain_db_cached`, `read_motor_fan_rpm`) return a sentinel value -- `None`, `False`, or an empty container -- when the value cannot be read (no hardware, channel not set, firmware does not implement the probe). No exception is raised. The caller branches on the sentinel.
- **Camera value getters** (`get_gain_db`, `get_exposure_ms`, `get_width`/`get_height`, `get_binning_size`) are a stricter subclass of the sentinel contract: a **transient read failure is invisible** -- the getter answers with the validated last-known-good value, so a momentary USB/SDK glitch can never hand you a failure code where a physical value belongs (no `-1` gain into arithmetic, no `None` frame size into a subscript). The documented camera-absent defaults (`get_gain_db` -1.0, `get_exposure_ms` 0.0, width/height getters 0, `get_binning_size` 1) occur **only** when no camera is active or the value has never been successfully read -- stable states you can see coming via `camera_connected`, not something a transient failure produces mid-session. Callers that must record what the hardware was at a specific moment (file metadata, logs of record) use `get_live_camera_settings()` instead: it returns only fields whose driver read succeeded right now (`gain_db`, `exposure_ms`, `frame_size`, `pixel_format`) and omits the rest -- there, unknown stays unknown by design.
- **Naming convention -- `*_cached` vs `get_*`**: a property ending in `_cached` (`gain_db_cached`, `exposure_ms_cached`, `frame_size_cached`, `pixel_format_cached`, `active_cached`, `min_frame_size_cached`, `max_exposure_ms_cached`, `max_gain_db_cached`) reads the host-side camera cache and performs **no driver I/O** -- safe to read at any frequency from any thread. A `get_*` method is a **live driver read** under the last-known-good contract above. The name carries the contract, so a call site's I/O behavior is visible without opening the implementation.
- **State-changing operations** (setters like `set_gain_db`, `move_absolute`, `led_on`, etc.) typically return `True` on success and `False` for "couldn't do it" (no driver, mode invalid, driver does not implement, etc.). A `Raises:` section in the docstring documents the typed exception (`HardwareError`, `CaptureError`, `ConfigError` from `modules.exceptions`) that propagates when the underlying SDK call itself fails. The API layer logs (`logger.error`) and fires a user-facing notification (`notifications.error`) before re-raising at the driver boundary; the typed exception is what L2 callers should catch.
- **Hardware-command dispatch** (LED, motion, and camera commands): each command submits to its executor and blocks until the hardware has it. While a protocol run owns the executors (or an executor is disabled), the blocking form raises `HardwareCommandRefusedError` (`modules.exceptions`), carrying the machine-readable `reason` (`exclusive_activity_running`) and the refused member; the `*_async` forms drop the command with a logged warning instead of raising. With no executors registered at all (a bare `Lumascope()` in a script), every command -- blocking and `*_async` alike -- runs directly on the calling thread.
- **Sentinel-return methods log** at `logger.warning` or `logger.info` per Rule 5; they do **not** fire user notifications (no actionable failure occurred -- the value is just unknown).
- **`camera_connected` is an instantaneous, non-latching poll.** A `False` can be transient (a single flaky connectivity query on an otherwise healthy camera). Consumers may skip work on `False` and re-poll on their next cycle; they must never latch, self-cancel, or tear anything down on it -- one transient `False` on a multi-day run should cost one skipped cycle, not the rest of the session.

If you are writing a new wrapper, the `Raises:` section is the canonical declaration of which contract applies.

---

## Lumascope composition root

The `Lumascope` class is the **hardware-composition-root**. It constructs and holds the six sub-APIs (`scope.motion`, `scope.illumination`, `scope.imaging`, `scope.diagnostics`, `scope.capabilities`, `scope.io`), wires them together, and owns lifecycle (connect / disconnect / emergency shutdown).

**When to use directly:** you need fine-grained control beyond ScopeSession, or you're building a custom application. The GUI, ScopeSession, and REST surface all go through this class.

### Initialization

```python
from modules.lumascope_api import Lumascope
from modules.scope_init_config import ScopeInitConfig

scope = Lumascope()                       # real hardware (auto-detect camera)
scope = Lumascope(simulate=True)          # simulated (no hardware)
scope = Lumascope(camera_type='pylon')    # force Basler Pylon
scope = Lumascope(camera_type='ids')      # force IDS
```

Valid `camera_type` values: `'auto'` (default), `'pylon'`, `'ids'`, `'sim'`.

Then apply runtime configuration (frame size, objective, binning, stage offset). The preferred factory is `ScopeInitConfig.from_settings(settings, labware, scope_config=...)`, which reads from your LVP settings dict; you can also construct one directly:

```python
config = ScopeInitConfig(
    labware=labware_obj,
    objective_id='10x Oly',
    turret_config=None,
    binning_size=1,
    frame_width=3840,
    frame_height=2160,
    acceleration_pct=100,
    stage_offset={'x': 0, 'y': 0},
    scale_bar_enabled=False,
    # expects_motion / expects_led default to True; override for
    # models that legitimately have no motor / no LED (e.g. LS620
    # has no motor, so expects_motion=False avoids a spurious
    # "Partial Hardware Detected" popup).
)
scope.initialize(config)
```

### Connection

```python
scope.are_all_connected()                 # LED + motor + camera all up
scope.motor_connected                     # motor board (property)
scope.led_connected                       # LED board (property)
scope.camera_connected                    # camera (property)
scope.no_hardware                         # True if all-null (no real hardware found)
scope.disconnect()
```

### Objective management

Objective / labware / turret-config / stage-offset are runtime-mutable microscope configuration (not live hardware), so they live on the `scope.runtime_state` sub-API (Wave 7 split them off the composition root). L2 callers reach it through the composition root the Session exposes: `session.scope.runtime_state.*`.

```python
scope.runtime_state.set_objective('10x Oly')           # the one public setter (session.scope.runtime_state from L2)

scope.runtime_state.get_current_objective_id()
scope.runtime_state.get_objective_info('10x Oly')      # {focal_length, magnification, NA, ...}
scope.runtime_state.get_available_objectives()
scope.runtime_state.get_current_objective()

# Turret integration
scope.runtime_state.set_turret_config({1: '4x Oly', 2: '10x Oly', 3: '20x Oly', 4: '40x w/collar'})
scope.runtime_state.get_turret_config()
scope.motion.get_turret_position_for_objective_id('10x Oly')   # returns 2 (turret position is motion state)
scope.motion.is_current_turret_position_objective_set()        # False when the CURRENT turret slot has no configured objective

# Labware + stage offset -- the plate-coordinate inputs
scope.runtime_state.set_labware(labware_obj)           # LabWare object (see Coordinate transformations)
scope.runtime_state.get_labware()
scope.runtime_state.set_stage_offset({'x': 0.0, 'y': 0.0})
scope.runtime_state.get_stage_offset()
scope.runtime_state.get_well_label()                   # 'A1' for the current stage XY; '' when the labware has no wells

# Stage µm → plate mm using the registered labware + stage offset
# (the bound form of CoordinateTransformer.stage_to_plate; raises
# NoLabwareSelectedError when no labware is registered)
px, py = scope.runtime_state.stage_to_plate(sx=60000, sy=40000)
```

---

## ScopeSession session layer

GUI-free session container. All hardware commands route through executor threads for thread safety. Use this for scripts and automation.

**When to use:** You want to write a Python script that controls the microscope without the GUI.

### Setup

For **real hardware** with settings loaded from disk:

```python
import modules.settings_init as settings_init
from lvp_logger import logger
from modules.scope_session import ScopeSession

# Takes a logger and the appdata DIRECTORY; reads data/current.json
# itself (settings.json is the corrupt-file fallback + defaults-merge
# source) and populates the module-global settings dict.
settings_init.load_lvp_settings(logger, '.')
session = ScopeSession.create(settings=settings_init.settings, source_path='.')
session.start_executors()
```

For **simulated** (no hardware needed, development / CI):

```python
from modules.scope_session import ScopeSession

session = ScopeSession.create_headless()
session.start_executors()
```

`create_headless()` is the supported factory for simulated / headless sessions — it wires up simulated drivers for you. Don't hand-construct a `Lumascope(simulate=True)` + `ScopeSession.create(...)` pair unless you have a specific reason.

### Application startup sequence

```python
session.start_application_session()                  # home ALL axes, then position turret
session.start_application_session(disable_homing=True)  # skip homing, still position turret
```

`start_application_session()` is the single source of truth for the standard startup orchestration the GUI runs on launch: it queues an all-axis `move_home` on the io_executor (firmware homes Z/T/X/Y in one routine; Z-only boards home what they have), then, when the scope has a turret, moves the T-axis to the position matching `settings['objective_id']` (falling back to position 1). Headless / REST callers should use this rather than open-coding the home + turret sequence. `disable_homing=True` skips the home step (matches the App's `--no-home` flag) but still positions the turret.

### Periodic metrics logging (optional)

```python
session.start_metrics()   # start periodic runtime-health logging
session.stop_metrics()    # stop it (idempotent; shutdown() calls this too)
```

The session owns the metrics-logger lifecycle. Metrics start only when the
host injected a scheduler at construction (`ScopeSession(...,
metrics_scheduler=ThreadingTimerScheduler())` for REST / headless hosts;
the GUI injects a Kivy-clock scheduler) — with no scheduler,
`start_metrics()` is a no-op, so factory-built sessions keep metrics off
unless the host opts in. `settings['profiling']['metrics_interval_s']`
overrides the default hourly cadence. `start_metrics()` raises
`RuntimeError` if metrics are already running; `stop_metrics()` is
idempotent. After a scope rebind (`set_scope`), running metrics move to
the new scope automatically — same scheduler, same cadence. These members
are host-serialized: call them from one thread (the GUI uses its main
thread only).

### Hardware commands from L2

The Session carries no hardware-command forwarders: every command has
exactly one public spelling, on the sub-APIs of the composition root the
Session exposes as `session.scope`. The dispatch contract (blocking form
submits and blocks; refusal raises `HardwareCommandRefusedError` while a
protocol run owns the executors; `*_async` drops with a logged warning)
is documented once in the contract section above.

```python
# LED
session.scope.illumination.led_on('Blue', 200)      # blocks until the write has landed
session.scope.illumination.led_on_async('Blue', 200)  # fire-and-forget
session.scope.illumination.led_off_async('Blue')
session.scope.illumination.leds_off_async()

# Motion
session.scope.motion.move_home_async('ALL')
session.scope.motion.move_absolute_async('Z', 5000, wait_until_complete=True)
session.scope.motion.move_relative_async('X', 500)

# Imaging (blocking-only -- no imaging *_async forms)
session.scope.imaging.set_gain_db(8.0)                 # dB; blocks until applied
session.scope.imaging.set_exposure_ms(50.0)       # ms; blocks until applied
image = session.scope.imaging.capture_and_wait()    # returns frame-valid grab

# The dark-floor expectation is DERIVED from commanded LED state: with a
# channel lit (strictly positive current), a frame with no lit pixel is
# rejected (retried, then None) instead of returned as data; with nothing
# commanded -- or a channel at 0 mA -- a dark frame is by-design and
# accepted. accept_dark=True overrides a lit rejection for callers whose
# dark frames are legitimate (custom focus sweeps, benchmark probes).
# timeout_s is the retry budget for the content checks (dark floor,
# saturation, chunk verify); leave it 0.0 to judge the first grab only.
# The executor wait is bounded internally.
image = session.scope.imaging.capture_and_wait(timeout_s=2.0)
```

### Capture

```python
from modules.image_save import save_image

# The capture derives the dark-floor expectation itself from commanded
# LED state -- there is no illumination fact to pass.
image = session.scope.imaging.capture_and_wait()
save_image(
    session.scope,
    array=image, save_folder='./output',
    file_root='capture', append='_BF',
    channel='BF', false_color_on=False,
    save_encoding='right_aligned',
    significant_bits=session.scope.imaging.capture_frame_depth(image),
)

# Live-view tap: the latest buffered frame, no new exposure forced
# (the capture calls above always force one). (None, None) when unavailable.
frame, timestamp = session.scope.imaging.get_image_from_buffer()

# Payload bit depth (8 / 12 / 16) of a frame this scope just produced --
# needed to interpret or rescale full-depth payloads before saving.
session.scope.imaging.capture_frame_depth(image)
```

### Running protocols

```python
runner = session.create_protocol_runner()
protocol = session.scope.protocols.load_protocol('my_protocol.tsv')
# or build one in-memory (config= | input_config= | empty_config=):
protocol = session.scope.protocols.create_protocol(input_config=config)

# image_capture_config is REQUIRED: the caller states the run's image mode
# (bit depth + on-disk encoding) explicitly -- there is no silent default.
# Modes: '8bit', '12bit_scientific', '12bit_scaled', '12bit_false_color_rgb'.
runner.run_single_scan(
    protocol,
    image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
)
runner.wait_for_completion()

# Or abort at any time:
runner.abort()
```

`run_single_scan()` runs one scan; `run_protocol()` runs the full multi-scan protocol. Both raise `ConfigError` if `image_capture_config` is omitted, and `ProtocolRunRefusedError` (`modules.exceptions`) when the run is refused before any state is committed -- already running, files still writing, empty protocol, a validation failure, or hardware not connected. The refusal is already logged and shown to the user, so an L2 caller catches it to branch on its `reason` / `title` / `message` attributes (they map cleanly to a REST status code or a UI message) without re-notifying. See the `ProtocolRunner` source for optional callbacks, image-output config, etc.

A refusal with reason `files_writing` means the previous run's files are still draining -- wait and retry. Reason `files_writing_stalled` means the file writer has stopped making progress entirely (a wedged write, e.g. an unresponsive save drive); waiting will not clear it. Recover with:

```python
session.recover_file_writer()   # discards pending unsaved writes, unlocks the writer
```

Recovery is deliberate data loss: pending writes from the wedged run are discarded (they were never going to finish), and a partial file from the stuck write may remain on disk. Returns `False` when the session holds no file-IO executor (a GUI-hosted session -- use the GUI's recovery popup instead).

**Canonical entry points.** Build the runner with `session.create_protocol_runner()`. Build the `Protocol` it runs with one of the two constructors on the protocols sub-API -- `scope.protocols.load_protocol(file_path)` (from a `.tsv` on disk) or `scope.protocols.create_protocol(config=... | input_config=... | empty_config=...)` (in-memory). Both resolve `data/tiling.json` from the session's registered `source_path`, so prefer them over calling `Protocol.from_file(...)` directly (which makes you pass `tiling_configs_file_loc` by hand).

### Video steps and recordings

A protocol step with `Acquire` = `video` records through the session's recording engine for the step's configured duration. This is the supported video path for L2 / headless callers; the GUI's manual Record button is a GUI-hosted convenience on the same engine.

Per recording (one per well per scan), the run produces:

- a frames folder (`<step>_video/`) of per-frame TIFFs, numbered in capture order;
- `recording_manifest.json` in that folder -- the measured truth: delivered frame count, measured frame rate, per-frame timestamps, and the recording's end reason. Downstream consumers (including Create Video's `auto` rate) read the manifest, not the configured rate;
- one variable-frame-rate MP4 per recording;
- after the run completes, one OME-TIFF hyperstack per (well, scan): `T` = frame capture order, `C` = channel, per-plane `DeltaT` from the frames' own timestamps. Hyperstacks build at run completion on every host -- headless and REST runs included, no GUI involved.

Rate and duration come from the run's settings snapshot at start: `video.max_fps` (0 = uncapped; the effective rate is measured, not assumed) and `video.max_duration_seconds`. Mid-run settings edits do not affect a run in flight.

Recording starts are guarded like protocol starts: `RecordingRefusedError` (`modules.exceptions`) mirrors the `ProtocolRunRefusedError` shape, with machine-readable `reason` codes `recording_active` (another recording is live) and `exclusive_activity_running` (a protocol run or other exclusive activity holds the session's activity claim).

Both refusal errors say busy-with-what: `holder` carries the exclusive-activity owner at refusal time (`'protocol'` or `'recording'`, None for refusals that are not claim-shaped), and `holder_trigger` carries the holding run's `run_trigger_source` when the holder is a run (`'protocol'`, `'autofocus_scan'`, `'zstack'`, `'autofocus'`, `'api_scan'`, ...). A recording holder has no trigger -- its kind is the whole answer. File-drain refusals (`files_writing*`) carry the just-finished run's trigger so a poller can report whose files are draining.

**Opening hyperstacks in Fiji:** the container is OME-TIFF; channel color travels as OME `Channel.Color`. Open via `Plugins > Bio-Formats > Importer` with **Color mode = Composite** (the choice persists per user through that dialog). A plain `File > Open` renders ImageJ's default LUTs, not the file's channel colors.

**Run-state semantics:** `session.is_protocol_running` (a property, not a call) reports True while a run holds the session's exclusive-activity claim -- protocol runs, single scans, z-stacks, autofocus scans, and the standalone Autofocus button's run included. It releases at run-cleanup end; the short post-run file-drain window (files still writing after the run finished) reads False here and True on `session.run_lockout` / `session.protocol_files_draining`, so a poller that must wait for the disk to settle checks those. A live video recording is not a run: it reads False here and is visible on `session.exclusive_activity == 'recording'`.

### Run state and locks

The session derives all run and lock state from its activity claim.
These are the members a GUI-quality client binds its widget state to --
the same derivations LVP's own GUI mirrors into kv properties
(alongside the run predicates shown under Running protocols):

```python
session.run_lockout              # True during a run OR its post-run file drain
session.is_protocol_running      # True while a protocol-class run holds the claim
session.protocol_files_draining  # run files still writing after a run finished
session.exclusive_activity       # None | 'protocol' | 'recording'
session.controls_locked          # full control-surface lock (any run lockout, or a live recording)
session.motion_enabled           # user stage motion allowed right now
session.recording_capturing     # a manual recording is LIVE (not its file drain)

def on_run_state():              # called on EVERY run-state transition;
    print(session.run_lockout)   # re-read the derivations (level semantics, no payload)
session.add_run_state_listener(on_run_state)
session.notify_run_state()       # force a level-sync of all listeners
```

### Configuration queries

```python
session.get_layer_configs()              # all layer settings
session.get_current_objective_info()     # active objective
session.get_current_plate_position()     # current XY in plate coords
session.get_auto_gain_settings()         # auto-gain config
session.get_stim_configs()               # stim settings per layer
session.get_enabled_stim_configs()       # only the enabled ones
```

### Reconnect

```python
# After a hardware reconnect, rewire the SAME session onto the new scope --
# executors, metrics, and the run machinery follow automatically:
session.set_scope(new_scope)
```

### Cleanup

```python
session.shutdown()               # full teardown of everything the session constructed
# or piecewise:
session.shutdown_executors()
session.scope.disconnect()
```

---

## scope.motion

Axes available depend on the scope — always check `scope.capabilities.axes`.

```python
# Homing (required before movement)
scope.motion.home()                              # home everything the board has (axis='ALL' default)
scope.motion.home(axis='Z')                      # Z only
scope.motion.home(axis='T')                      # turret only (parks Z at 0, homes T, restores Z)
# Unknown axis raises ValueError; the async twin move_home_async(axis)
# and move_home_and_wait(axis) share the same 'Z' | 'T' | 'ALL' vocabulary.
scope.motion.move_home_and_wait('ALL')           # blocks; True only if the home ran AND succeeded
scope.motion.has_homed()                         # True if the stage/focus axes know where they are
scope.motion.has_turret_homed()                  # turret-specific

# Homing is REQUIRED, not advisory. A commanded move on an axis whose
# position is unknown raises AxisStateUnknownError instead of driving --
# there is no reference frame for it to be absolute in. An axis is
# unknown before its first successful home, and again after a home
# fails, the board disconnects mid-move, or a move stalls out. So a
# headless or REST caller homes first and checks the result:
#
#   if not scope.motion.move_home_and_wait('ALL'):
#       ...  # do not command moves; the reference frame is not established
#
# has_homed() / has_turret_homed() answer from that same live state, so they
# report False after a fault revokes a reference that was previously
# good -- not merely "a home once succeeded".

# Position queries (µm for XYZ, 1–4 for turret). Read cache, no serial I/O.
scope.motion.get_current_position('Z')           # predicted position during motion, confirmed when idle
scope.motion.get_current_position()              # dict of all axes
scope.motion.get_target_position('Z')            # target µm
scope.motion.get_actual_position('Z')            # hardware position via serial (slow; use sparingly)

# Stop + tuning
scope.motion.stop_motion()                       # stop all in-flight moves (the app-level abort for the move_* family)
scope.motion.set_acceleration_limit(50)          # motor acceleration cap, percent of max

# Absolute moves (µm)
scope.motion.move_absolute('Z', 5000)
scope.motion.move_absolute('X', 60000, wait_until_complete=True)

# Relative moves (µm)
scope.motion.move_relative('Z', 100)

# Status
scope.motion.get_target_status('Z')              # True if target reached
scope.motion.is_moving()                         # any axis moving?
scope.motion.wait_until_finished_moving()        # block until all idle

# Limit switches -- why a move stopped short. Reaching a limit is reported,
# not raised, so a move that ran out of travel and one that arrived look the
# same until you ask.
scope.motion.get_limit_switch_status('X')        # (left, right); 1 engaged, 0 clear, -1 unreadable
scope.motion.get_limit_switch_status_all_axes()  # dict of axis -> that pair, for the axes the board has

# Turret
scope.capabilities.has_turret                    # turret presence probe
scope.motion.move_turret(2)                      # turret position 2

# Stage
scope.motion.get_axis_limits('Z')                # {'min': 0, 'max': 14000}
scope.motion.get_axes_config()                   # per-axis config dict: limits + ustep-conversion funcs (motion-driver shape)
```

**Axes: two different questions, two different surfaces.** Asking *what
axes does this scope have* uses `scope.capabilities.axes` (tuple of
names; immutable identity). Asking *what is the per-axis runtime config*
(travel limits, ustep-per-mm conversion functions) uses
`scope.motion.get_axes_config()` (dict of dicts; driver-level config).
The first is frozen at boot and answers UI-gating questions; the second
exposes the motor-board's per-axis configuration for tiling /
coordinate-transform work. They are not redundant.

**Z overshoot:** firmware moves below target then approaches from below, eliminating leadscrew backlash for consistent focus.

**Axis state model:**

```python
from modules.lumascope_api import AxisState

scope.motion.get_axis_state('Z')          # 'idle', 'moving', 'homing', or 'unknown'
scope.motion.is_any_axis_moving()
```

**Position listeners** (push-based):

```python
def on_position(axis: str, target: float, state: str):
    print(f"{axis} → {target:.1f}µm ({state})")

scope.motion.add_position_listener(on_position)
scope.motion.remove_position_listener(on_position)
```

---

## scope.illumination

Channels available depend on the scope — always check `scope.capabilities.led_colors`.

**A channel is named, never numbered.** `'BF'`, `'PC'`, `'DF'`, `'Blue'`,
`'Green'`, `'Red'` — the name is the portable identity. The number behind it is
a driver detail and is NOT portable: an FX2 board carries four channels and an
RP2040 board six, so the same integer means different LEDs, or none, depending
on the board. Ask `caps.led_colors` for what a given scope can actually drive.
Passing an integer still works today as a legacy compatibility form, but it is
not the supported contract and will not be part of the REST surface.

**Colours the scope cannot drive**: `led_on` with a colour name the scope has no LED channel for raises `ConfigError` naming the colour (it never maps to a substitute channel); `led_off` with such a colour is an idempotent no-op — a channel the scope does not have is already off. Numeric channel arguments are always range-checked and raise `ValueError` when invalid.

**Luminescence** (`Lumi`): not an LED channel. In luminescence mode, all LEDs must be off — the image captures emitted light only.

```python
scope.illumination.led_on('Blue', 200)                 # Blue LED at 200 mA
scope.illumination.led_on('Blue', 200, block=True)     # wait for firmware confirmation
scope.illumination.led_off('Blue')
scope.illumination.leds_off()                          # turn off all LEDs

# Fire-and-forget: returns immediately, the write lands on the io worker.
# Dropped with a logged warning (never an exception) while a protocol run
# owns the executors -- see "Hardware-command dispatch" above.
scope.illumination.led_on_async('Red', 100)
scope.illumination.led_off_async('Red')
scope.illumination.leds_off_async()

# Channel mapping. Numbers are a DRIVER detail -- these exist to read the
# board's own wire vocabulary, not to address channels from L2.
scope.illumination.color2ch('Blue')                    # 0  (or None if the scope doesn't have this color)
scope.illumination.ch2color(0)                         # 'Blue'

```

**Safety limits** (enforced by firmware on RP2040 boards): per-channel max 1000 mA, board total max 3000 mA. FX2 boards have their own per-channel cap declared in the camera profile.

### State queries — read from the API, never the driver

Lumascope holds the authoritative LED state in an internal cache. The API layer's `get_led_state()` / `led_enabled()` / `get_led_ma()` read from that cache. **Never call the driver's state methods directly** — for FX2 scopes the driver is a pure command translator and its state queries return sentinels.

```python
scope.illumination.led_enabled('Blue')                 # True / False
scope.illumination.get_led_ma('Blue')                  # current mA, or None if off / no LED board
scope.illumination.get_led_state('Blue')               # {'enabled': True, 'illumination_ma': 200, 'owner': '…'} when on; {'enabled': False, 'illumination_ma': None, 'owner': ''} when off
scope.illumination.get_led_states()                    # all channels, same per-channel shape as get_led_state
```

### Ownership — prevents subsystems from clobbering each other

Tag each LED operation with a subsystem name. Only an owner can turn off a channel they own.

```python
scope.illumination.led_on('BF', 200, owner='autofocus')

scope.illumination.led_off('BF', owner='protocol')     # no-op — wrong owner
scope.illumination.led_off('BF', owner='autofocus')    # works

scope.illumination.leds_off_owned('autofocus')         # turn off only channels owned by this subsystem
scope.illumination.leds_off()                          # unconditional off (shutdown / cleanup)
```

### Save / restore — the autofocus pattern

Preserve the user's LED state while a subsystem does its own work, then restore:

```python
# User has Red on at 150 mA. Autofocus needs BF:
snapshot = scope.illumination.save_led_state('autofocus')        # capture current state
scope.illumination.led_on('BF', 100, owner='autofocus')
# ... autofocus runs: changes Z, captures frames, evaluates focus ...
scope.illumination.restore_led_state(snapshot, owner='autofocus')  # Red back on at 150 mA, BF off
```

`save_led_state(tag)` returns a snapshot dict; `restore_led_state(snapshot, owner='…')` reverts. The owner must match the subsystem that did the save.

### Listeners — push-based notifications

Prefer listeners over polling. Listeners fire on every LED state change (enable, disable, illumination change, ownership change) with no serial I/O cost:

```python
def on_led(channel: str, enabled: bool, illumination_ma: float, owner: str):
    print(f"{channel} {'ON' if enabled else 'OFF'} {illumination_ma}mA owner={owner!r}")

scope.illumination.add_led_listener(on_led)
# ... later ...
scope.illumination.remove_led_listener(on_led)
```

Use polling only when you specifically need the current value at a moment in time (e.g., settling a UI field to match hardware after a reconnect). For "did anything change?" questions, always use listeners.

---

## scope.imaging

Camera capture and configuration live on the `scope.imaging` sub-API
namespace. The methods below are the L2-stable surface; the underlying
driver is `scope.imaging._driver` (private; reach through the API).

```python
# Streaming control. connect() returns the camera CONFIGURED but NOT
# grabbing; capture/get_image need a live feed, so start it first. In the
# GUI this happens automatically at startup; headless callers do it
# explicitly after constructing the scope.
scope.imaging.start_streaming()   # begin the live feed (idempotent; also
                                  # restarts a feed stopped via stop_streaming)
scope.imaging.stop_streaming()    # stop the feed (get_image then times out)
scope.imaging.is_streaming()      # True while acquiring (queries the driver)
```

```python
# Raw frame grab (no validity wait — use capture_and_wait instead in most cases)
image = scope.imaging.get_image()
image = scope.imaging.get_image(force_to_8bit=False)   # keep native 12/16-bit
# Returns numpy.ndarray on success, None on failure (camera inactive,
# frame drain failed, timeout). Per the sentinel-return contract:
#   if image is None: ...
#
# Shape is (H, W) 2D mono for mono-native cameras and (H, W, 3) RGB
# for color-native cameras (see scope.capabilities.is_color_native).
# Layer false-color is NOT applied here -- apply at the display /
# encode boundary via image_utils.mono_to_rgb_falsecolor(img, layer).
# Dtype is uint8 with force_to_8bit=True (default) or for 8-bit
# cameras; uint16 with force_to_8bit=False for 12/16-bit cameras
# (see scope.capabilities.native_bit_depth).
#
# Payload depth of a frame you just captured (for scaling / saving a
# uint16 frame): scope.imaging.last_significant_bits -- the per-frame
# delivery stamp (e.g. 12 for Mono12 in a uint16 container). Prefer it
# over scope.imaging.significant_bits (derived from the current pixel
# format) when you are holding the frame -- the stamp cannot describe a
# newer format than the frame was captured under.

# Frame-validity capture — PREFERRED for all real captures.
# Waits for all pending changes (LED, gain, exposure, motion) to settle,
# drains stale frames, returns a valid frame. Returns None on failure.
# The capture honors invalidation across its WHOLE window: a state change
# landing after the drain — during the grab itself — is detected, and the
# capture re-drains, re-derives its expectations, and grabs again, so the
# returned frame always reflects the state you last commanded. A capture
# contended by state changes is therefore slower than an uncontended one.
# The settle-and-recheck work is bounded by a deadline sized from the
# pending work at entry (frames to settle, exposure, sum window): if
# invalidation keeps arriving faster than frames can settle it, the
# capture returns None in bounded seconds with "DEADLINE EXPIRED" named
# in the log, instead of holding indefinitely. The deadline suspends
# while commanded stage motion is still physically settling — a capture
# issued during a long move waits for the move, as it should.
# The dark-floor expectation is DERIVED from commanded LED state: a
# channel counts as lit only at strictly positive current, so a channel
# commanded at 0 mA is dark by design, as are luminescence captures and
# any capture with nothing commanded. With a channel lit, a frame with
# essentially no lit pixel is rejected (retrying until timeout_s, then
# None) so a stale pre-LED or starved black frame is never returned as
# data. accept_dark=True (keyword-only, default False) overrides a lit
# rejection for the callers whose dark frames are legitimate: custom
# focus sweeps (an out-of-focus fluorescence plane can carry no signal)
# and benchmark probes.
image = scope.imaging.capture_and_wait()
image = scope.imaging.capture_and_wait(
    force_to_8bit=True,
    accept_dark=False,                     # True admits a dark frame while lit
    all_ones_check=True,                   # detect saturated frames
    sum_count=4,                           # SUM 4 frames (not an average); a
                                           # summed capture is promoted to a
                                           # 16-bit container and clipped there
    sum_delay_s=0.05,                      # delay between sum frames
    exclude_sources=('z_move',),           # don't wait for this source (AF uses this)
    earliest_image_ts=None,                # optional wall-clock lower bound on returned frame
)

# Exposure (milliseconds) + gain (dB)
scope.imaging.set_exposure_ms(exposure_ms=50)
scope.imaging.get_exposure_ms()                  # last-known-good on transient read failure; 0.0 camera-absent
scope.imaging.set_gain_db(gain_db=10.0)
scope.imaging.get_gain_db()                           # last-known-good on transient read failure; -1.0 camera-absent

# Live-confirmed readings for metadata / records: only fields whose
# driver read succeeded RIGHT NOW; a field whose read failed is omitted
# (the value getters above would answer last-known-good instead).
scope.imaging.get_live_camera_settings()           # any of: gain_db, exposure_ms,
                                                   #   frame_size {'width','height'}, pixel_format;
                                                   #   {} when no camera is active

# `set_exposure_ms` warns + logs a stack trace at < 0.005 ms (the
# common L1 failure is typing 0.05 thinking microseconds and getting
# a black image).

# Every camera-settings setter in this section dispatches to the camera
# lane and BLOCKS until applied (returns the body's own result). While a
# protocol run or recording owns the hardware, these raise
# HardwareCommandRefusedError instead of interleaving with the run --
# same refusal contract as the motion and LED commands.

# Batched settings (gain + exposure + auto-gain in one call)
scope.imaging.apply_layer_camera_settings(
    gain_db=5.0, exposure_ms=50,
    auto_gain=False, auto_gain_settings=None,
)

# Auto-gain: the continuous toggle, the one-shot settle, and the setpoint
scope.imaging.set_auto_gain(True, settings={'target_brightness': 0.3, 'min_gain_db': 0.0, 'max_gain_db': 20.0})
scope.imaging.auto_gain_once(True, target_brightness=0.3, min_gain_db=0.0, max_gain_db=20.0)
scope.imaging.update_auto_gain_target_brightness(0.5)   # live setpoint tweak while auto-gain runs

# Camera-model-specific tuning knobs. Probe support first:
# scope.capabilities.camera_supports_conversion_gain_mode / _line_noise_reduction.
scope.imaging.set_conversion_gain_mode('High')     # True when applied; False when unsupported / no camera
scope.imaging.set_line_noise_reduction(True)       # same contract

# Frame size (getters answer last-known-good on a transient read
# failure; None / 0 only when no camera is active or never read)
delivered = scope.imaging.set_frame_size(2048, 2048)
# Returns the DELIVERED {'width','height'} -- the clamped/snapped
# geometry actually in effect, which may differ from the request
# (drivers clamp to the sensor max and floor to the alignment grid).
# Raises CameraSettingRejected (modules.exceptions) when a live camera
# refuses the apply; returns None (no-op) when no camera is active.
# Base geometry code on the returned dict, never on the request.
scope.imaging.frame_size_cached                    # {'width': ..., 'height': ...} -- cache read, no driver I/O
scope.capabilities.camera_max_frame_size           # (width, height) sensor ceiling -- static structure
scope.imaging.get_native_resolution()              # {'width','height'} unbinned sensor ceiling
scope.imaging.get_pixel_alignment()                # {'width','height'} deliverable frame-size granularity (even on IDS; camera grid on floor-only drivers)

# Binning
scope.imaging.set_binning_size(2)
# True when applied; raises CameraSettingRejected when a live camera
# refuses; False (no-op) only when no camera is active. Same contract
# for set_pixel_format. Success is observed by the return value,
# rejection by the typed raise -- a dropped return cannot silently
# record a rejected apply.
scope.imaging.get_binning_size()                   # always >= 1 (last-known-good on failed read)
scope.imaging.get_available_binning_sizes()        # e.g. [1, 2, 4]
scope.imaging.set_pixel_format('Mono12')           # True when applied; raises CameraSettingRejected on refusal
scope.imaging.get_supported_pixel_formats()        # e.g. ('Mono8', 'Mono12') -- the enumeration for set_pixel_format

# Geometry value getters (last-known-good on a transient failed read; 0 camera-absent)
scope.imaging.get_width()
scope.imaging.get_height()

# Cache reads, no driver I/O (the *_cached family)
scope.imaging.gain_db_cached
scope.imaging.exposure_ms_cached
scope.imaging.pixel_format_cached
scope.imaging.min_frame_size_cached                # dict, or None when no camera is connected

# Scale bar overlay (burned into frames the imaging paths return when enabled;
# skipped -- with one warning logged -- while no objective is selected)
scope.imaging.set_scale_bar(True, color='red')
scope.imaging.scale_bar_config                     # snapshot dict: {'enabled', 'color', ...}
```

The acquisition frame-rate cap lives on the camera driver and clamps frame production regardless of sensor-readout capability. It is a driver-level control with no public API member -- described here only to explain the behavior. Used by the manual-record path to match user-requested video FPS, and by characterization tools to bound capture rate during long-running probes. No-op on drivers that do not implement the underlying setter (warning logged). Distinct from `set_exposure_ms` (per-frame integration time) and from any host-side throttling.

### Dynamic camera capabilities

Cameras advertise their real limits at connect time. Use these to size UI sliders and clamp auto-exposure / auto-gain:

```python
scope.imaging.max_exposure_ms_cached                  # ms, None if no camera connected
scope.imaging.max_gain_db_cached                      # dB, None if no camera connected
```

These are derived from the camera's profile, which is populated at connect via `_query_dynamic_capabilities()` — live SDK queries for Pylon / IDS, hardcoded-from-datasheet for FX2. Per-camera values observed in practice: LS620 FX2 = 42.1 dB gain / 178 ms exposure cap; Pylon/IDS ranges are driver-reported.

### Save / restore camera state

```python
snapshot = scope.imaging.save_camera_state('autofocus')
# ... change gain/exposure ...
scope.imaging.restore_camera_state(snapshot)
```

Symmetric to the LED version, but `restore_camera_state` takes only the snapshot (no `owner` arg — camera state is single-owner by nature).

The snapshot is **omit-if-unknown**: it always carries `tag`, and carries `gain_db` / `exposure_ms` only when a usable value existed at save time (a missing field means that value was never successfully read from the camera; `save_camera_state` logs a warning when it omits one). Use `.get(...)` rather than indexing if you read snapshot fields directly. `restore_camera_state` restores the fields present, quietly skips absent ones (callers may deliberately trim fields they want left at current values), and leaves the camera unchanged for anything it skips.

### Camera listeners

```python
def on_camera(param: str, value: float):
    print(f"Camera {param} = {value}")

scope.imaging.add_camera_listener(on_camera)       # fires on set_gain_db / set_exposure
scope.imaging.remove_camera_listener(on_camera)
```

### Live frame listeners

Sync per-frame handlers fire on every successful camera grab (Pylon `PylonImageGrab` thread / IDS grab loop / simulated pump). This is the canonical entry point for live image-processing plugins (see `ctx.plugins.live_processing`) and the manual-record path.

```python
def on_frame(image, timestamp, chunks):
    # Runs on the SDK callback thread. MUST NOT block. Heavy work
    # belongs on an executor.
    queue_write(image)

scope.imaging.add_frame_listener(on_frame, name='my_recorder')
# ...
scope.imaging.remove_frame_listener(on_frame)
```

- **Don't-mutate contract.** The `image` array is shared across all listeners. Write to your own output buffer if you need to keep results; mutating the array affects later listeners + downstream display / capture consumers.
- **Budget.** Each handler must complete within ~24 ms (anchored to a 30 fps target, half the inter-frame window). Over-budget invocations log a WARNING. After 30 consecutive over-budget hits, the handler is auto-removed and the user sees a notification.
- **Re-entrancy.** A handler will not be re-entered on the same thread; the driver's fire-site is single-threaded.
- **Plugin authors**: use `ctx.plugins.live_processing.register(spec, handler)` rather than calling `add_frame_listener` directly. The registry forwards through to this API and surfaces the plugin name in the budget-violation log.
- **Tutorial**: `docs/LIVE_PROCESSING_TUTORIAL.md` -- minimum-viable plugin example + failure-injection example + common pitfalls.

### Listener callback signatures (overview)

The four listener families each pass a different callback signature -- register a callable matching the row for the listener you subscribe to:

| Listener | Register via | Callback signature |
|---|---|---|
| Motion / position | `scope.motion.add_position_listener` | `on_position(axis: str, target: float, state: str)` |
| LED / illumination | `scope.illumination.add_led_listener` | `on_led(channel: str, enabled: bool, illumination_ma: float, owner: str)` |
| Camera params | `scope.imaging.add_camera_listener` | `on_camera(param: str, value: float)` |
| Live frame | `scope.imaging.add_frame_listener` | `on_frame(image, timestamp, chunks)` |

Each has a matching `remove_*_listener(callback)`. The frame listener additionally takes a `name=` kwarg and carries the don't-mutate + 24 ms budget contract documented above; the other three are lightweight state-change notifications.

### Camera info

```python
scope.camera_connected                             # bool property (mirror of motor_connected / led_connected)
scope.imaging.active_cached                        # True if grabbing
scope.diagnostics.get_camera_temperatures_degc()        # temperature sensors (SDK-dependent)
scope.diagnostics.get_camera_info()                # model, serial, firmware
scope.imaging.camera_identity                      # {'model','serial','timestamp_tick_frequency_hz'} for provenance records; all None camera-absent
scope.diagnostics.get_camera_profile_info()        # sensor specs + dynamic ranges; returns:
# {
#   'model': 'MT9P031-LS620', 'sensor': 'Aptina MT9P031',
#   'pixel_size_um': 2.2, 'shutter': 'rolling',
#   'resolution': (2592, 1944),
#   'gain_min_db': 0.0, 'gain_max_db': 42.1,
#   'max_exposure_ms': 178.0,
#   'binning_sizes': (1, 2, 4),
# }
```

### Frame validity

Frame validity is the single source of truth for "is the next frame still what I asked for?" Every hardware state change invalidates pending frames. `capture_and_wait()` drains stale frames until all sources settle, detects any invalidation that lands mid-grab (re-draining and re-grabbing so the result reflects the newest commanded state), bounds the whole settle-and-recheck loop with a deadline (loud None when invalidation outruns it), and verifies the returned frame's own chunk metadata (exposure / gain) against the requested values on cameras with chunk support -- the saved frame proves its own settings.

```python
scope.imaging.frame_is_valid                       # True if next frame is valid
scope.imaging.frames_until_valid()                 # 0 = ready, >0 = keep draining
# To record a frame you grabbed yourself, call the frame_validity instance
# directly (see below) -- capture_and_wait handles it internally.
```

For deeper introspection (diagnostic tooling, plugin authors writing custom capture loops, advanced timing analysis), the underlying `FrameValidity` instance is available as `scope.imaging.frame_validity` and is part of the L2-stable surface:

```python
fv = scope.imaging.frame_validity

fv.is_valid                                # bool property -- next frame valid right now?
fv.is_valid_for(exclude_sources=('z_move',))  # bool -- valid if you don't care about Z motion
fv.frames_until_valid()                    # int -- drains remaining
fv.frames_until_valid(exclude_sources=('z_move',))
fv.pending_sources                         # dict {source: target_frame_counter} (snapshot)
fv.invalidation_counts                     # dict {source: total invalidate() calls} — monotone
                                           # history frames can never erase; snapshot before a
                                           # grab and compare (!=) after to detect a mid-window
                                           # invalidation even when frames already settled it
fv.invalidate('led')                       # mark a source dirty (usually called by API setters)
fv.count_frame(chunk_data=None, frame_ts=None)  # mark a frame as drained (capture_and_wait does this)
                                           # pass the grab timestamp as frame_ts so the same
                                           # buffered frame polled twice counts once; chunk_data
                                           # (ChunkExposureTime/ChunkGain) clears gain/exposure
                                           # deterministically when it matches the requested target
```

`set_settle_check(fn)` is the API-only registration hook for motion-completion gating and is not used by L2 callers directly. Everything else is fair game for plugin / SDK consumers.

Invalidation is automatic for normal flows — you don't need to call `invalidate()` yourself unless you're writing a custom hardware setter outside the API. The sources that invalidate frames are:

```
led        — LED turn on/off or illumination change
gain       — gain change
exposure   — exposure change
z_move     — Z axis motion
xy_move    — X or Y axis motion
turret     — turret move
```

When you need to capture *during* a source's active motion (e.g., autofocus captures while Z is moving), pass that source to `exclude_sources` in `capture_and_wait()`.

---

## scope.diagnostics

Hardware diagnostic probes and identity getters live on the `scope.diagnostics` sub-API. Per-call (no persistent state); meant for tech-support reports, bench tooling, and bring-up scripts that want one-shot snapshots of camera / motor / LED state.

```python
scope.diagnostics.get_microscope_model()   # 'LS850'
scope.diagnostics.get_motor_info()         # model, serial, firmware, axis config
scope.diagnostics.get_led_info()           # firmware, cal status
scope.diagnostics.get_system_info()        # combined summary
scope.capabilities.pixel_size_um           # raw per-installation um/pixel, or None if the scope cannot report it (unknown camera / no declared optics). For an objective-adjusted effective um/pixel, call common_utils.get_pixel_size(focal_length, binning_size).
scope.capabilities.lens_focal_length_mm    # tube-lens focal length, mm, or None if the scope cannot report it
```

```python
# Camera diagnostic snapshot. Returns dict with model, resolution,
# pixel_format, gain, exposure_ms, max_gain, max_exposure_ms,
# temperatures (Celsius), and per-field error strings when a probe
# fails. Returns {'connected': False} when no camera is active.
info = scope.diagnostics.get_camera_diagnostic_info()

# Camera temperature sensors. Returns dict {sensor_name: degC} or
# empty when the camera lacks temperature sensors or is inactive.
temps = scope.diagnostics.get_camera_temperatures_degc()

# Cross-host / cross-camera / cross-firmware diagnostic probe.
# Captures camera identity, current config, temperatures, and stream
# stats deltas over duration_s, stamped with the active camera SDK
# (Basler pylon, IDS peak, ...). Writes JSON to data/camera_probe/.
# A driver that does not implement the probe returns the driver's
# {'supported': False, ...} shape unchanged. Does NOT change grab state.
probe = scope.diagnostics.run_pylon_diagnostic_probe(
    duration_s=3.0, drain_camera_side_errors=True,
)

# Engineering-mode firmware diagnostic commands. Routes through the
# canonical driver path (Rule 13 logging, Rule 14 error visibility).
# target is 'led' or 'motor'.
resp = scope.diagnostics.send_diagnostic_command('led', 'INFO')
lines = scope.diagnostics.send_diagnostic_command_multiline(
    'led', 'SELFTEST', timeout_s=60,
)

# Motor-board driver / fan diagnostics (already on
# DiagnosticsAPI pre-Phase-5; documented here for completeness).
status = scope.diagnostics.read_motor_drv_status('Z')       # int register or None
rpm = scope.diagnostics.read_motor_fan_rpm()              # RPM or None
ok = scope.diagnostics.set_motor_fan_duty(50)              # bool

# LED engineering-mode handshake (FACTORY / Y / Q with post-Q drain).
# Use these in place of open-coded send_diagnostic_command sequences.
ok = scope.diagnostics.enter_led_engineering_mode(timeout_s=5.0)
scope.diagnostics.exit_led_engineering_mode()
```

---

## scope.capabilities

`scope.capabilities` is a `ScopeCapabilities` dataclass populated at connect time. **Use this to learn what the connected hardware can do** — don't hardcode axis lists, LED channel counts, or camera caps.

```python
caps = scope.capabilities

# Motion
caps.axes                       # ('X', 'Y', 'Z', 'T') on LS850T; ()         on LS620
caps.has_focus                  # True if Z is motorized
caps.has_xy_stage               # True if X/Y are motorized
caps.has_turret                 # True if the turret axis is present
caps.motor_model                # e.g. 'RP2040' or '' if no motor
caps.axis_travel_limits_um      # {'X': 120000.0, 'Y': 80000.0, 'Z': 14000.0} -- present axes only

# LED
caps.led_channels               # e.g. (0, 1, 2, 3) for FX2 scopes; (0..5) for RP2040
caps.led_colors                 # e.g. ('BF', 'Blue', 'Green', 'Red') — what THIS scope can do
caps.led_max_ma                 # per-channel current cap
caps.has_firmware_stim          # firmware-timed stim support on the LED board

# Optics -- resolved from the first real source, never a hardcoded default:
#   motorconfig.json Optics (LS820/850/850T) -> scopes.json Optics (Classic)
#   -> camera SDK-reported pitch (pixel size only) -> None
caps.pixel_size_um              # um/pixel, or None if the scope cannot report it
caps.lens_focal_length_mm       # tube lens focal length mm, or None if unavailable

# Feature probe (cross-surface, by token)
caps.supports('turret')         # searches has_X and camera_supports_X fields; unknown tokens -> False

# Camera
caps.camera_model               # 'MT9P031-LS620', 'acA2500-60um', etc.
caps.is_color_native            # True for color-native sensors; False for mono-native (default)
caps.native_bit_depth           # 8 (e.g. IDS) or 16 (uint16 container; holds 12/16-bit native)
caps.camera_supports_auto_gain
caps.camera_supports_auto_exposure
caps.camera_supports_conversion_gain_mode
caps.camera_supports_line_noise_reduction
caps.camera_pixel_formats       # e.g. ('Mono8',) or ('Mono8', 'Mono12')
caps.camera_binning_sizes       # e.g. (1, 2, 4)
caps.camera_max_frame_size      # (width, height) tuple in pixels; (0, 0) if no camera
# Exposure ceiling: scope.imaging.max_exposure_ms_cached (ms; None if no camera) -- see scope.imaging
```

Important consequences:

- **`camera_max_frame_size` is `(0, 0)` when no camera is connected** -- that is a sentinel meaning "unknown / no camera," not a usable size. Check `scope.camera_connected` (or that the tuple is non-zero / `caps.camera_model` is non-empty) before using it as a `scope.imaging.set_frame_size(w, h)` target; `set_frame_size` returns `None` (no-op) when no camera is active, so a naive `set_frame_size(*caps.camera_max_frame_size)` does nothing rather than erroring. With a live camera it returns the DELIVERED geometry and raises `CameraSettingRejected` if the apply is refused.
- **LED channel count varies by scope.** LS560/LS620 (FX2 driver) expose 4 channels (`BF`, `Blue`, `Green`, `Red`); RP2040-based scopes expose 6 (`BF`, `PC`, `DF`, `Blue`, `Green`, `Red`). Don't iterate over a hardcoded list — iterate over `caps.led_colors`.
- **Some scopes have no motor at all.** LS560/LS620 have `caps.axes == ()`. Calling `scope.motion.move_absolute('X', …)` against such a scope is a no-op, not an error — but your UI should hide motion controls based on `caps.has_xy_stage` etc.
- **`axis_travel_limits_um` is populated only for present axes.** On a Z-only scope, `'X' in caps.axis_travel_limits_um` is `False`; indexing `caps.axis_travel_limits_um['X']` raises `KeyError`. Check `caps.has_xy_stage` (or `axis in caps.axes`) before reading. The mapping is read-only (`MappingProxyType`); mutation attempts raise `TypeError`.

---

## scope.io

**Reserved.** Not populated in LumaViewPro 4.0.x.

The `scope.io` sub-API is named in the locked sub-API decomposition per `docs/PLUGIN_API_DESIGN_2026-05-09.md` §6.6. It will document future I/O surfaces (trigger devices, USB-to-IO trigger boards, external sync) once those surfaces ship; the feature flags that gate them will ride `scope.runtime_state` when they exist.

---

## scope.runtime_state

Mutable counterpart to `scope.capabilities`. Where `capabilities` holds the immutable per-scope identity (axes, led channels, camera model), `runtime_state` holds the runtime-mutable facts that legitimately change mid-session — firmware versions (after a reflash), firmware feature flags (after FW4.0 ships), and future reconnect-aware fields.

The split exists because firmware version is not a frozen scope identity — a tech-support engineer can reflash mid-session, and the version surface should reflect that. A single frozen capabilities surface would lie post-flash.

```python
scope.runtime_state.firmware_versions       # dict[str, str]
scope.runtime_state.firmware_features       # dict[str, frozenset[str]]
```

**Status in 4.0.x: empty placeholder.** Both fields ship as empty dicts. Real content lands when FW4.0 populates `INFO.features` (firmware_features) and when reconnect-aware versioning hooks are added to the driver layer (firmware_versions). Callers treat empty as "feature unknown" per the Rule 8 capability-probe contract — `scope.runtime_state.firmware_features.get('motor', frozenset())` returns the empty set today, never `KeyError`.

Until the real content ships, query firmware version via `scope.diagnostics.get_motor_info()['firmware_version']` / `scope.diagnostics.get_led_info()['firmware_version']`. The diagnostic-getter path is the live query; `runtime_state` will become the cached snapshot once reflash hooks fire it.

---

## modules

The `modules/` package holds helpers that ride alongside the API surface but are not sub-API methods. Two patterns:

- **Take `scope` as first argument**: orchestration helpers that compose sub-API calls (image-save, composite capture, protocol runner).
- **Pure functions**: stateless utilities that take frame arrays or geometry parameters (coord transformations, optical calculations, focus scoring).

### Image saving (`modules.image_save`)

Image-save helpers are free functions in `modules.image_save` (extracted
from the Lumascope class in Wave 7 Phase 6, 2026-05). Each function
takes the `scope` (a `Lumascope` instance) as its first argument; the
remaining arguments are the per-call settings:

```python
from modules.image_save import save_image

save_image(
    scope,
    array=image,
    save_folder='/path/to/output',
    file_root='experiment1',
    append='_BF_A1',
    channel='BF',                          # what was imaged -- recorded as identity
    false_color_on=False,                  # how it is displayed -- never recorded as identity
    tail_id_mode='increment',              # auto-number files
    output_format='TIFF',                  # 'TIFF' or 'OME-TIFF'
    save_encoding='right_aligned',         # from the image-mode config layer
    significant_bits=scope.imaging.capture_frame_depth(image),
    x=60000, y=40000, z=5000,              # stage position metadata (µm)
)
```

The full set of free functions in `modules.image_save`:

| Function | Purpose |
|---|---|
| `save_image(scope, array, ...)` | Save a numpy array to TIFF / OME-TIFF with metadata. |
| `save_live_image(scope, save_folder, ...)` | Grab the current live frame from the camera and save (composes `capture_and_wait` + `save_image`). |
| `prepare_image_for_saving(scope, array, ...)` | Flip / bit-convert / build metadata + path; returns `{'image', 'metadata'}`. |
| `generate_image_metadata(scope, channel, x, y, z)` | Build the TIFF metadata dict for the current capture settings + position. |
| `generate_image_save_path(scope, save_folder, ...)` | Generate the next unused file path under `tail_id_mode`. |
| `get_next_save_path(scope, path)` | Increment the trailing numeric ID on an existing path. |

### Image utilities (`modules.image_utils`)

Boundary helpers that ride alongside the mono-native pipeline. Two
patterns matter to L2 callers:

```python
from modules import image_utils

# Map a mono frame to RGB false-color at the display / encode boundary.
# Use this when you have a mono fluorescence frame from get_image() and
# need a 3-channel array for display, video encode, or a downstream
# tool that expects RGB. Mono pipeline saves do NOT call this -- the
# layer is recorded as TIFF metadata instead.
rgb = image_utils.mono_to_rgb_falsecolor(mono_frame, layer='Blue')
# layer in {'Blue', 'Green', 'Red', 'BF', 'Lumi', ...}
# Returns 3-channel ndarray, same dtype as input.

# Read a TIFF and collapse legacy 3-channel false-color-replica files
# to mono on the fly. Use this when reading any TIFF that may have
# been written by a pre-mono-native LumaViewPro: the 3-channel files
# with one populated channel auto-collapse to 2D mono; true color
# composites (multiple non-zero channels) pass through unchanged.
img = image_utils.read_tiff_with_legacy_collapse(path)
# Returns 2D mono ndarray for mono and collapsed-legacy files;
# 3D RGB ndarray for real color composites.
```

The save pipeline emits mono fluorescence TIFFs with layer metadata
in the TIFF ImageDescription field; the legacy reader bridges that
to consumers that previously assumed a 3-channel shape. FIJI, MATLAB
``imread``, and tifffile all handle mono 2D natively; the false-
color is purely a display-time concern.

Full-pixel-depth frames store raw, right-aligned sensor values (a 12-bit
frame is ``0..4095``) and declare the true depth in the OME-TIFF
``SignificantBits`` tag (e.g. ``SignificantBits=12`` inside a 16-bit
container). To render or scale such a file to 8-bit, divide by
``(1 << SignificantBits) - 1`` -- treating the values as full 16-bit will
render a 12-bit frame ~16x too dark. ``image_utils.read_tiff_significant_bits``
returns the tag (falling back to the container width for older files that were
left-justified into the 16-bit range and carry no payload-depth tag).

### Coordinate transformations (`modules.coord_transformations`)

```python
from modules.coord_transformations import CoordinateTransformer
ct = CoordinateTransformer()

# Stage µm → plate mm (top-left origin)
plate_x, plate_y = ct.stage_to_plate(
    labware=labware_obj, stage_offset=offset, sx=60000, sy=40000,
)

# Plate mm → stage µm
stage_x, stage_y = ct.plate_to_stage(
    labware=labware_obj, stage_offset=offset, px=50.0, py=30.0,
)
```

`labware` is a `LabWare` object loaded from `data/labware.json` via `WellPlateLoader`, not a raw dict. `stage_offset` is a dict like `{'x': 0.0, 'y': 0.0}`.

### Optical calculations (`modules.common_utils`)

```python
import modules.common_utils as common_utils

# Pixel size (µm per pixel)
px_um = common_utils.get_pixel_size(focal_length=4.78, binning_size=1)

# Field of view (µm)
fov = common_utils.get_field_of_view(
    focal_length=4.78,
    frame_size={'width': 2048, 'height': 2048},
    binning_size=1,
)
# Returns: {'width': ..., 'height': ...} in µm, or None if scale is unknown
```

These helpers read `scope.capabilities.pixel_size_um` / `scope.capabilities.lens_focal_length_mm` from the active scope. Both return `None` when there is no active scope, or when the scope cannot report its optics (unknown camera, no declared optics) — `get_pixel_size` and `get_field_of_view` then return `None` rather than an invented scale, and callers degrade honestly (no scale bar, no field of view, no `PhysicalSizeX`). There is deliberately no hardcoded fallback: a guessed pixel size is written into every image and cannot be told from a measured one. Note: `capabilities.pixel_size_um` is the raw per-installation pixel pitch; for an effective µm/px adjusted for current objective + binning, call `common_utils.get_pixel_size(focal_length, binning_size)`.

### Composite capture (`modules.composite_builder`)

`build_composite()` composes multi-channel frames into a single false-color image. See the [Multi-channel composite](#multi-channel-composite) pattern under Common patterns for a runnable example.

```python
from modules.composite_builder import build_composite
```

### Autofocus (`modules.autofocus_functions`)

`focus_function(image=...)` computes a Brenner-gradient focus score from a frame array. Pure function; no scope state needed.

```python
from modules.autofocus_functions import focus_function

score = focus_function(image=frame, skip_score_logging=True)
```

Used by autofocus iteration code paths (was previously available as `scope.compute_focus_score(image)`; retired in Wave 7 Phase 7 per the rule that frame-analysis functions are pure helpers, not API methods).

### Protocol (`modules.protocol`)

`Protocol.from_file(...)` loads multi-step acquisition sequences. See the [Headless protocol run](#headless-protocol-run) pattern for a runnable example.

```python
from modules.protocol import Protocol
```

---

## plugin platform reference

Plugin platform spec and live-processing tutorial both live alongside LumaViewPro.

- **Design**: `docs/PLUGIN_API_DESIGN_2026-05-09.md` — the locked platform spec (PluginSpec, namespaces, registry contracts, loading sequence).
- **Live-processing tutorial**: `docs/LIVE_PROCESSING_TUTORIAL.md` — walkthrough for writing a `ctx.plugins.live_processing` plugin.
- **Namespaces (4.x)**: `ctx.plugins.ui`, `ctx.plugins.post_processing`, `ctx.plugins.live_processing`, `ctx.plugins.rest`.

A worked plugin example ships in `etaluma-engineering/`; see its `pyproject.toml` `entry_points` for how a plugin declares itself.

---

## REST surface reference

REST is **not implemented**. There is no server, no endpoints, and no wire
format — nothing in this repo answers HTTP. Endpoints will be documented here
when REST actually ships, against the real implementation.

Nothing about a future REST surface is specified anywhere in this document. An
earlier revision carried a sketch of endpoint shapes and a MATLAB client; both
predated the 4.0 API-surface work, were never built, and were removed once they
started being read as a contract that constrained that work. `git log` has them
if the ideas are ever wanted.
---

## Common patterns

### Basic capture

```python
from modules.lumascope_api import Lumascope

scope = Lumascope()
scope.motion.home()
scope.motion.wait_until_finished_moving()

scope.runtime_state.set_objective('10x Oly')
scope.imaging.set_exposure_ms(50)
scope.imaging.set_gain_db(5.0)

scope.motion.move_absolute('X', 60000, wait_until_complete=True)
scope.motion.move_absolute('Y', 40000, wait_until_complete=True)
scope.motion.move_absolute('Z', 5000, wait_until_complete=True)

from modules.image_save import save_image

scope.illumination.led_on('BF', 100)
image = scope.imaging.capture_and_wait()
scope.illumination.leds_off()

save_image(
    scope,
    array=image, save_folder='./output',
    file_root='capture', append='_BF',
    channel='BF', false_color_on=False,
    save_encoding='right_aligned',
    significant_bits=scope.imaging.capture_frame_depth(image),
    output_format='TIFF', x=60000, y=40000, z=5000,
)
scope.disconnect()
```

### Multi-channel composite

```python
from modules.composite_builder import build_composite
from modules.image_save import save_image

channel_images = {}
for channel, illumination_ma, exposure_ms, gain_db in [
    ('Blue',  200, 100, 15),
    ('Green', 150,  80, 12),
    ('Red',   180,  90, 10),
]:
    scope.imaging.set_exposure_ms(exposure_ms)
    scope.imaging.set_gain_db(gain_db)
    scope.illumination.led_on(channel, illumination_ma)
    channel_images[channel] = scope.imaging.capture_and_wait()
    scope.illumination.led_off(channel)

# Transmitted (brightfield) base image
scope.imaging.set_exposure_ms(2.0)
scope.imaging.set_gain_db(1.0)
scope.illumination.led_on('BF', 100)
bf_image = scope.imaging.capture_and_wait()
scope.illumination.leds_off()

composite = build_composite(
    channel_images=channel_images,
    transmitted_image=bf_image,
    brightness_thresholds={'Blue': 20, 'Green': 15, 'Red': 10},
)

save_image(scope, array=composite, save_folder='./output',
           file_root='composite', channel='Composite', false_color_on=False,
           save_encoding='8bit', significant_bits=8, output_format='TIFF')
```

`build_composite` accepts fluorescence keys `'Red'`, `'Green'`, `'Blue'`, `'Lumi'`.

### Z-stack

```python
from modules.image_save import save_image

z_start, z_end, z_step = 4000, 6000, 50    # µm

scope.illumination.led_on('BF', 100)
z = z_start
while z <= z_end:
    scope.motion.move_absolute('Z', z, wait_until_complete=True)
    image = scope.imaging.capture_and_wait()
    save_image(
        scope,
        array=image, save_folder='./zstack',
        file_root='z', append=f'_{int(z)}',
        channel='BF', false_color_on=False,
        save_encoding='right_aligned',
        significant_bits=scope.imaging.capture_frame_depth(image),
        output_format='TIFF', z=z,
    )
    z += z_step
scope.illumination.leds_off()
```

### Well-plate scan

```python
from modules.coord_transformations import CoordinateTransformer
from modules.image_save import save_image
ct = CoordinateTransformer()

wells = [('A1', 10.0, 20.0), ('A2', 19.0, 20.0), ('A3', 28.0, 20.0)]

scope.illumination.led_on('BF', 100)
for well_name, px, py in wells:
    sx, sy = ct.plate_to_stage(labware=labware_obj, stage_offset=offset, px=px, py=py)
    scope.motion.move_absolute('X', sx, wait_until_complete=True)
    scope.motion.move_absolute('Y', sy, wait_until_complete=True)

    image = scope.imaging.capture_and_wait()
    save_image(
        scope,
        array=image, save_folder='./scan',
        file_root=f'{well_name}_BF',
        channel='BF', false_color_on=False,
        save_encoding='right_aligned',
        significant_bits=scope.imaging.capture_frame_depth(image),
        output_format='TIFF', x=sx, y=sy,
    )
scope.illumination.leds_off()
```

### Headless protocol run

```python
from modules.scope_session import ScopeSession
from modules.protocol import Protocol

session = ScopeSession.create_headless()    # simulated; use create(settings=…) for hardware
session.start_executors()

protocol = Protocol.from_file(
    file_path='./my_protocol.tsv',
    tiling_configs_file_loc='./data/tiling.json',
)

runner = session.create_protocol_runner()
runner.run_single_scan(
    protocol,
    image_capture_config=runner.build_image_capture_config(image_mode='8bit'),
)
runner.wait_for_completion()

session.shutdown_executors()
session.scope.disconnect()
```

---

## Simulated mode

Use for development, CI, and unit tests without hardware.

```python
scope = Lumascope(simulate=True)
scope._camera_driver.start_grabbing()   # simulator test setup; see note below

# All API calls work identically:
scope.illumination.led_on('Blue', 200)
scope.motion.move_absolute('Z', 5000)
image = scope.imaging.get_image()
```

The `start_grabbing()` call reaches through to the private camera driver because the simulator does not auto-start streaming (production camera drivers do). This is the only direct private-driver access an L2 caller needs in simulator-mode test setup — a future release may add a public `scope.imaging.start_grabbing()` for symmetry.

**Only in `simulate=True`**: `set_timing_mode('fast')` lets simulator tests run faster by skipping artificial serial / motor / camera delays. Same private-driver access pattern: timing-mode control is a simulator test-infrastructure feature, not an L2 surface.

```python
scope._led_driver.set_timing_mode('fast')
scope._motion_driver.set_timing_mode('fast')
scope._camera_driver.set_timing_mode('fast')
```

These attributes only exist on the simulated drivers. Don't call them on a real-hardware `Lumascope` — you'll get `AttributeError`.

---

## Protocol file format

Tab-separated file defining multi-step acquisition sequences.

```
LumaViewPro Protocol
Version	5
Period	1.0
Duration	0.002778
Labware	96-well
Capture Root

Steps
Name	X	Y	Z	Auto_Focus	Color	...
A1_BF	60000	40000	5000	False	BF	...
A1_Green	60000	40000	5000	False	Green	...
```

**Step fields:**

| Field | Type | Description |
|---|---|---|
| Name | string | Step label (e.g. `A1_BF`) |
| X, Y, Z | float | Position in µm |
| Auto_Focus | bool | Run autofocus at this step |
| Color | string | `Blue`, `Green`, `Red`, `BF`, `PC`, `DF`, `Lumi` |
| False_Color | bool | Apply false-color mapping |
| Illumination | float | LED current in mA |
| Gain | float | Camera gain in dB |
| Auto_Gain | bool | Enable auto-gain |
| Exposure | float | Exposure time in ms |
| Sum | int | Frame averaging count (≥1) |
| Objective | string | Must match `data/objectives.json` |
| Well | string | Well label (e.g. `A1`) |
| Acquire | string | `image` or `video` |

Consult `Protocol.from_file` in `modules/protocol.py` for the canonical field list — additions happen over time.

---

## Color channel reference

```python
from modules.common_utils import ColorChannel

ColorChannel.Blue   # 0  — blue-excitation fluorescence
ColorChannel.Green  # 1  — green-excitation fluorescence
ColorChannel.Red    # 2  — red-excitation fluorescence
ColorChannel.BF     # 3  — brightfield (white LED)
ColorChannel.PC     # 4  — phase contrast (on scopes with separate PC hardware)
ColorChannel.DF     # 5  — darkfield
ColorChannel.Lumi   # 6  — luminescence (all LEDs off, sensitive mode)
```

**Fluorescence excitation wavelengths depend on the installed filterset** — the stock filterset is 405 / 488 / 589 nm, but OEM customers may have custom filtersets at different wavelengths.

**Not every scope has every channel.** Always check `scope.capabilities.led_colors` before using a color — for example, LS560/LS620 expose only `{'BF', 'Blue', 'Green', 'Red'}`. Phase contrast on those models is brightfield with a mechanical phase slider installed, not a separate illumination channel.

---

## Appendix A: Internal serial-protocol interfaces (firmware tooling only)

This appendix documents direct serial commands used by firmware update tools, board bring-up scripts, and factory calibration. **These are not intended for integration code** — they bypass safety limits, depend on chip-internal register semantics that can change across firmware versions, and can leave the hardware in unsafe states if misused. Application code should stay at the ScopeSession or Lumascope sub-API layer.

<details>
<summary>Show internal interfaces</summary>

### Direct board drivers

```python
from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard

led = LEDBoard()                           # auto-detect by VID:PID
led.exchange_command('LED3_200')           # set BF LED to 200 mA
led.exchange_command('LEDS_OFF')

motor = MotorBoard()                       # auto-detect by VID:PID
motor.exchange_command('HOME')
motor.exchange_command('TARGET_WZ682666')  # move Z (µsteps)
pos = motor.exchange_command('ACTUAL_RZ')
```

### Connection parameters

| Parameter | LED board | Motor board |
|---|---|---|
| VID:PID | 0x0424:0x704C | 0x2E8A:0x0005 |
| Transport | UART via USB hub bridge, 115200 baud | USB CDC native |
| Line ending (send) | `\r\n` | `\n` |
| Line ending (recv) | `\r\n` | `\n` |
| Command timeout | 100 ms default | 5 s default (homing: 15–30 s) |

### Raw REPL (firmware file transfer)

```python
motor.enter_raw_repl()
motor.repl_list_files()
content = motor.repl_read_file('motorconfig.json')
motor.repl_write_file('main.py', new_source)
motor.exit_raw_repl()
```

`SerialBoard` (the shared base class) implements raw REPL for both boards.

### LED board application commands (safe-mode)

| Command | Description |
|---|---|
| `INFO` | Board info (firmware version, calibration status, heap) |
| `LEDS_ENT` / `LEDS_ENF` | Enable / disable LED driver |
| `LEDS_OFF` | Turn off all LEDs |
| `LED{ch}_{mA}` | Set channel 0–7 to `mA` (float ok: `LED3_200`, `LED0_0.5`) |
| `LED{ch}_OFF` | Turn off channel |
| `LEDREAD{ch}` | Read I_SENS + LED_K ADC feedback |

Engineering-mode commands (`FACTORY`, `RAW…`, `ADCREAD`, `CALIBRATE`, `CALSAVE`, `CALCLEAR`, `SELFTEST`, `I2CSCAN`, `FWUPDATE`) bypass safety limits and are **not documented here** — they exist for factory bring-up and firmware development only.

### Motor board application commands

| Command | Description |
|---|---|
| `INFO` / `FULLINFO` | Firmware and board info |
| `HOME` / `ZHOME` / `THOME` | Home all / Z only / turret only |
| `CENTER` | Move stage to center |
| `STOP` | Stop all motors immediately |
| `TARGET_W{axis}{steps}` | Set target position (µsteps) |
| `TARGET_R{axis}` | Read target position |
| `ACTUAL_R{axis}` | Read current position |
| `STATUS_R{axis}` | Read status register (32-bit) |
| `VOLTAGE` | Rail status |
| `CURRENT` | Per-axis motor current telemetry |

Axes: `X`, `Y`, `Z`, `T`. Position conversion (µsteps ↔ µm) is in `motorconfig.json`; prefer `scope.motion.get_axes_config()` over reading that file directly.

During homing, `STOP` aborts. `INFO`, `ACTUAL_R`, `STATUS_R`, `VOLTAGE` respond normally. Other commands return `BUSY`.

Direct SPI access to the TMC5072 (register-level motor configuration) and the associated status-register bit semantics are intentionally omitted — those are firmware-internal.

</details>


---

## Changelog

Post-freeze record of every change to the L2-callable surface (methods documented above). Entries land in the SAME commit as the underlying change. Pre-`4.0.0` (the freeze trigger) is fluid by design and not recorded here; consult `LVP_4.0.0_CHANGELOG.md` for pre-freeze prose history.

Entry format:

| Version | Date | Type | Method / surface | Change |
|---|---|---|---|---|

- **Type**: `additive` (new optional param / new return-dict key / new method) — no version bump beyond patch.
- **Type**: `behavior-change` — requires minor version bump.
- **Type**: `rename` / `removal` — requires deprecation cycle (old name retained with `FutureWarning`, retired in next major) OR a major version bump.

(No entries yet -- 4.0.0 has not shipped. Pre-freeze structural changes do not appear here.)
