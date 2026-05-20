# LumaViewPro — API & Integration Reference

## PRE-RELEASE API

The Lumascope SDK API documented in this file is **subject to breaking changes** in 4.1 / 4.1.5 / 4.2. Specifically:

- 4.1.5 ships the sub-API decomposition (Wave 7): hardware-direct methods on `Lumascope` move to sub-APIs (`scope.motion.*`, `scope.illumination.*`, `scope.imaging.*`, `scope.diagnostics.*`, `scope.capabilities.*`, `scope.io.*`). The `Lumascope` class becomes a thin facade; L2 entry point shifts to `ScopeSession`.
- 4.2 ships the capability + wire contract changes that may rename or restructure protocol-level surfaces.
- The REST endpoint convention is **deferred** to a dedicated design session; do not assume current shapes are final.

If you are using this API before stabilization, **contact Etaluma support** so we know to consult you before structural changes. Internal LumaViewPro use does not trigger this requirement.

The warning retires when (1) a tagged release publishes to PyPI / a public binary distribution channel AND (2) we have at least one named external consumer on record. See `Firmware/docs/CLAUDE.md` Rule 30 for the internal freeze trigger.

---

## Overview

LumaViewPro controls Etaluma microscopes: LED illumination, XYZ stage + turret motion, and camera image acquisition. This document is the integration reference for developers building scripts, headless automation, or external control applications on top of LumaViewPro.

**Repository**: `EtalumaSupport/LumaViewPro`
**Platform**: Python 3.11–3.13, Windows / macOS / Linux

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
scope.motor_connected                     # motor board
scope.led_connected                       # LED board
scope.imaging.camera_is_connected()       # camera
scope.no_hardware                         # True if all-null (no real hardware found)
scope.disconnect()
```

### Objective management

Objective / labware / turret-config / stage-offset stay on the composition root for now (microscope configuration, not live hardware).

```python
scope.set_objective('10x Oly')
scope.get_current_objective_id()
scope.get_objective_info('10x Oly')        # {focal_length, magnification, NA, ...}
scope.get_available_objectives()
scope.get_current_objective()

# Turret integration
scope.set_turret_config({1: '4x Oly', 2: '10x Oly', 3: '20x Oly', 4: '40x w/collar'})
scope.get_turret_config()
scope.get_turret_position_for_objective_id('10x Oly')   # returns 2
```

---

## ScopeSession session layer

GUI-free session container. All hardware commands route through executor threads for thread safety. Use this for scripts and automation.

**When to use:** You want to write a Python script that controls the microscope without the GUI.

### Setup

For **real hardware** with settings loaded from disk:

```python
from modules.settings_init import load_lvp_settings
from modules.scope_session import ScopeSession

settings = load_lvp_settings('./data/current.json')
session = ScopeSession.create(settings=settings, source_path='.')
session.start_executors()
```

For **simulated** (no hardware needed, development / CI):

```python
from modules.scope_session import ScopeSession

session = ScopeSession.create_headless()
session.start_executors()
```

`create_headless()` is the supported factory for simulated / headless sessions — it wires up simulated drivers for you. Don't hand-construct a `Lumascope(simulate=True)` + `ScopeSession.create(...)` pair unless you have a specific reason.

### LED control

```python
session.led_on('Blue', 200)              # non-blocking
session.led_on_sync('Blue', 200)         # blocks until firmware confirms
session.led_off('Blue')
session.leds_off()
```

### Motion

```python
session.move_home('ALL')
session.move_absolute('Z', 5000, wait_until_complete=True)
session.move_relative('X', 500)
```

### Capture

```python
from modules.image_save import save_image

image = session.scope.imaging.capture_and_wait()
save_image(
    session.scope,
    array=image, save_folder='./output',
    file_root='capture', append='_BF', color='BF',
)
```

### Running protocols

```python
from modules.protocol import Protocol

runner = session.create_protocol_runner()
protocol = Protocol.from_file(
    file_path='my_protocol.tsv',
    tiling_configs_file_loc='./data/tiling.json',
)

runner.run_single_scan(protocol)
runner.wait_for_completion()

# Or abort at any time:
runner.abort()
```

`run_single_scan()` runs one scan; `run_protocol()` runs the full multi-scan protocol. See the `ProtocolRunner` source for optional callbacks, image-output config, etc.

### Configuration queries

```python
session.get_layer_configs()              # all layer settings
session.get_current_objective_info()     # active objective
session.get_current_plate_position()     # current XY in plate coords
session.get_auto_gain_settings()         # auto-gain config
```

### Cleanup

```python
session.shutdown_executors()
session.scope.disconnect()
```

---

## scope.motion

Axes available depend on the scope — always check `scope.capabilities.axes`.

```python
# Homing (required before movement)
scope.motion.home()                              # home everything the board has
scope.motion.zhome()                             # Z only
scope.motion.thome()                             # turret only
scope.motion.has_homed()                         # True if home() has ever succeeded
scope.motion.has_thomed()                        # turret-specific

# Position queries (µm for XYZ, 1–4 for turret). Read cache, no serial I/O.
scope.motion.get_current_position('Z')           # predicted position during motion, confirmed when idle
scope.motion.get_current_position()              # dict of all axes
scope.motion.get_target_position('Z')            # target µm
scope.motion.get_actual_position('Z')            # hardware position via serial (slow; use sparingly)

# Absolute moves (µm)
scope.motion.move_absolute_position('Z', 5000)
scope.motion.move_absolute_position('X', 60000, wait_until_complete=True)

# Relative moves (µm)
scope.motion.move_relative_position('Z', 100)

# Status
scope.motion.get_target_status('Z')              # True if target reached
scope.motion.is_moving()                         # any axis moving?
scope.motion.wait_until_finished_moving()        # block until all idle
scope.motion.get_overshoot()                     # Z overshoot in progress?

# Turret
scope.motion.has_turret()
scope.motion.tmove(2)                            # turret position 2

# Stage
scope.motion.xycenter()                          # move to stage center
scope.motion.get_axis_limits('Z')                # {'min': 0, 'max': 14000}
scope.motion.get_axes_config()                   # all axes with limits + conversions
scope.motion.axes_present()                      # e.g. ['X', 'Y', 'Z', 'T']
scope.motion.has_axis('T')
```

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

**Luminescence** (`Lumi`): not an LED channel. In luminescence mode, all LEDs must be off — the image captures emitted light only.

```python
scope.illumination.leds_enable()
scope.illumination.led_on('Blue', 200)                 # Blue LED at 200 mA
scope.illumination.led_on(0, 200)                      # same, by channel number
scope.illumination.led_on('Blue', 200, block=True)     # wait for firmware confirmation
scope.illumination.led_off('Blue')
scope.illumination.leds_off()                          # turn off all LEDs
scope.illumination.leds_disable()

# Fast path (no response wait — timing-critical code only)
scope.illumination.led_on_fast('Red', 100)
scope.illumination.led_off_fast('Red')
scope.illumination.leds_off_fast()

# Channel mapping
scope.illumination.color2ch('Blue')                    # 0  (or -1 if the scope doesn't have this color)
scope.illumination.ch2color(0)                         # 'Blue'
```

**Safety limits** (enforced by firmware on RP2040 boards): per-channel max 1000 mA, board total max 3000 mA. FX2 boards have their own per-channel cap declared in the camera profile.

### State queries — read from the API, never the driver

Lumascope holds the authoritative LED state in an internal cache. The API layer's `get_led_state()` / `led_enabled()` / `led_illumination()` read from that cache. **Never call the driver's state methods directly** — for FX2 scopes the driver is a pure command translator and its state queries return sentinels.

```python
scope.illumination.led_enabled('Blue')                 # True / False
scope.illumination.led_illumination('Blue')            # current mA, or -1 if off
scope.illumination.get_led_state('Blue')               # {'enabled': True, 'illumination': 200, 'owner': '…'}
scope.illumination.get_led_states()                    # all channels
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
def on_led(color: str, enabled: bool, mA: float, owner: str):
    print(f"{color} {'ON' if enabled else 'OFF'} {mA}mA owner={owner!r}")

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
# Raw frame grab (no validity wait — use capture_and_wait instead in most cases)
image = scope.imaging.get_image()
image = scope.imaging.get_image(force_to_8bit=False)   # keep native 12/16-bit

# Frame-validity capture — PREFERRED for all real captures.
# Waits for all pending changes (LED, gain, exposure, motion) to settle,
# drains stale frames, returns a valid frame.
image = scope.imaging.capture_and_wait()
image = scope.imaging.capture_and_wait(
    force_to_8bit=True,
    all_ones_check=True,                   # detect saturated frames
    sum_count=4,                           # average 4 frames
    sum_delay_s=0.05,                      # delay between sum frames
    exclude_sources=('z_move',),           # don't wait for this source (AF uses this)
    earliest_image_ts=None,                # optional wall-clock lower bound on returned frame
)

# Exposure (milliseconds) + gain (dB)
scope.imaging.set_exposure_time(50)
scope.imaging.get_exposure_time()
scope.imaging.set_gain(10.0)
scope.imaging.get_gain()

# `set_exposure_time` warns + logs a stack trace at < 0.1 ms (the
# common L1 failure is typing 0.05 thinking microseconds and getting
# a black image). Internal sweep callers that walk that range
# deliberately wrap their loop in `suppress_value_warnings()`:
with scope.imaging.suppress_value_warnings():
    for exp_ms in (0.05, 0.1, 0.5, 5.0, 50.0):
        scope.imaging.set_exposure_time(exp_ms)
        # ... grab + measure ...
# Flag is restored on context exit (incl. exception).

# Batched settings (gain + exposure + auto-gain in one call)
scope.imaging.apply_layer_camera_settings(
    gain=5.0, exposure_ms=50,
    auto_gain=False, auto_gain_settings=None,
)

# Frame size
scope.imaging.set_frame_size(2048, 2048)
scope.imaging.get_frame_size()                     # {'width': ..., 'height': ...}
scope.imaging.get_max_width()
scope.imaging.get_max_height()

# Binning
scope.imaging.set_binning_size(2)
scope.imaging.get_binning_size()

# Acquisition frame-rate cap (camera-side; clamps sensor-readout pace)
scope.imaging.set_max_acquisition_frame_rate(enabled=True, fps=10.0)
scope.imaging.set_max_acquisition_frame_rate(enabled=False)   # remove cap
```

The acquisition frame-rate cap lives on the camera driver and clamps frame production regardless of sensor-readout capability. Used by the manual-record path to match user-requested video FPS, and by characterization tools to bound capture rate during long-running probes. No-op on drivers that do not implement the underlying setter (warning logged). Distinct from `set_exposure_time` (per-frame integration time) and from any host-side throttling.

### Dynamic camera capabilities

Cameras advertise their real limits at connect time. Use these to size UI sliders and clamp auto-exposure / auto-gain:

```python
scope.imaging.camera_max_exposure                  # ms, None if no camera connected
scope.imaging.camera_max_gain                      # dB, None if no camera connected
```

These are derived from the camera's profile, which is populated at connect via `_query_dynamic_capabilities()` — live SDK queries for Pylon / IDS, hardcoded-from-datasheet for FX2. Per-camera values observed in practice: LS620 FX2 = 42.1 dB gain / 178 ms exposure cap; Pylon/IDS ranges are driver-reported.

### Save / restore camera state

```python
snapshot = scope.imaging.save_camera_state('autofocus')
# ... change gain/exposure ...
scope.imaging.restore_camera_state(snapshot)
```

Symmetric to the LED version, but `restore_camera_state` takes only the snapshot (no `owner` arg — camera state is single-owner by nature).

### Camera listeners

```python
def on_camera(param: str, value: float):
    print(f"Camera {param} = {value}")

scope.imaging.add_camera_listener(on_camera)       # fires on set_gain / set_exposure
scope.imaging.remove_camera_listener(on_camera)
```

### Camera info

```python
scope.imaging.camera_is_connected()
scope.imaging.camera_active                        # True if grabbing
scope.imaging.get_camera_temps()                   # temperature sensors (SDK-dependent)
scope.diagnostics.get_camera_info()                # model, serial, firmware
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

Frame validity is the single source of truth for "is the next frame still what I asked for?" Every hardware state change invalidates pending frames. `capture_and_wait()` drains stale frames until all sources settle.

```python
scope.imaging.frame_is_valid                       # True if next frame is valid
scope.imaging.frames_until_valid()                 # 0 = ready, >0 = keep draining
scope.imaging.count_frame()                        # record that you grabbed a frame
                                           # (advances the drain count;
                                           # only callers who run their own
                                           # grab loop need this; capture_and_wait
                                           # handles it internally)
```

`pending_sources` (mapping of `{source: frames_remaining}`) is currently
accessed as `scope.imaging.frame_validity.pending_sources` -- this is an internal
diagnostic and not part of the L2-stable API surface; use it for debug,
not for production control flow.

Invalidation is automatic — you don't need to call it yourself. The sources that invalidate frames are:

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
scope.pixel_size()                         # um per pixel (method -- depends on objective; stays on composition root)
scope.lens_focal_length()                  # current tube-lens focal length (method; stays on composition root)
```

```python
# Camera diagnostic snapshot. Returns dict with model, resolution,
# pixel_format, gain, exposure_ms, max_gain, max_exposure_ms,
# temperatures (Celsius), and per-field error strings when a probe
# fails. Returns {'connected': False} when no camera is active.
info = scope.diagnostics.get_camera_diagnostic_info()

# Camera temperature sensors. Returns dict {sensor_name: degC} or
# empty when the camera lacks temperature sensors or is inactive.
temps = scope.diagnostics.get_camera_temperatures()

# Camera bandwidth + grab-cycle benchmarks. Both write a JSON
# artifact to data/camera_timing/ keyed on model + SDK + delay so a
# sweep across delays / num_cycles produces one file per data point.
bw = scope.diagnostics.run_camera_bandwidth_test(num_frames=1000)
gc = scope.diagnostics.run_grab_lifecycle_benchmark(
    num_cycles=100, inter_cycle_delay_ms=200, vary_settings=False,
)

# Pylon-specific cross-host / cross-camera / cross-firmware probe.
# Captures camera identity, current config, stream-grabber stats
# deltas over duration_s. Writes JSON to data/pylon_probe/. Returns
# the driver's {'supported': False, ...} shape unchanged for IDS or
# other non-Pylon drivers. Does NOT change grab state.
probe = scope.diagnostics.run_pylon_diagnostic_probe(
    duration_s=3.0, drain_camera_side_errors=True,
)

# Engineering-mode firmware diagnostic commands. Routes through the
# canonical driver path (Rule 13 logging, Rule 14 error visibility).
# target is 'led' or 'motor'.
resp = scope.diagnostics.send_diagnostic_command('led', 'INFO')
lines = scope.diagnostics.send_diagnostic_command_multiline(
    'led', 'SELFTEST', timeout=60,
)

# Motor-board power / driver / fan diagnostics (already on
# DiagnosticsAPI pre-Phase-5; documented here for completeness).
voltages = scope.diagnostics.read_motor_voltages()         # dict {rail: V} or None
status = scope.diagnostics.read_motor_drv_status('Z')       # int register or None
rpm = scope.diagnostics.read_motor_fanspeed()              # RPM or None
ok = scope.diagnostics.set_motor_fan_duty(50)              # bool

# LED engineering-mode handshake (FACTORY / Y / Q with post-Q drain).
# Use these in place of open-coded send_diagnostic_command sequences.
ok = scope.diagnostics.enter_led_engineering_mode(timeout=5.0)
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

# LED
caps.led_channels               # e.g. (0, 1, 2, 3) for FX2 scopes; (0..5) for RP2040
caps.led_colors                 # e.g. ('BF', 'Blue', 'Green', 'Red') — what THIS scope can do
caps.led_max_ma                 # per-channel current cap

# Camera
caps.camera_model               # 'MT9P031-LS620', 'acA2500-60um', etc.
caps.camera_supports_auto_gain
caps.camera_supports_auto_exposure
caps.camera_pixel_formats       # e.g. ('Mono8',) or ('Mono8', 'Mono12')
caps.camera_binning_sizes       # e.g. (1, 2, 4)
caps.camera_max_exposure_ms     # per-camera exposure ceiling (e.g. 178 ms on FX2)
caps.camera_pixel_size_um       # physical sensor pixel size
```

Two important consequences:

- **LED channel count varies by scope.** LS560/LS620 (FX2 driver) expose 4 channels (`BF`, `Blue`, `Green`, `Red`); RP2040-based scopes expose 6 (`BF`, `PC`, `DF`, `Blue`, `Green`, `Red`). Don't iterate over a hardcoded list — iterate over `caps.led_colors`.
- **Some scopes have no motor at all.** LS560/LS620 have `caps.axes == ()`. Calling `scope.motion.move_absolute_position('X', …)` against such a scope is a no-op, not an error — but your UI should hide motion controls based on `caps.has_xy_stage` etc.

---

## scope.io

**Reserved.** Not populated in LumaViewPro 4.0.x.

The `scope.io` sub-API is named in the locked sub-API decomposition per `docs/PLUGIN_API_DESIGN_2026-05-09.md` §6.6. It will document future I/O surfaces (trigger devices, USB-to-IO trigger boards, external sync) once those surfaces ship. See `caps.hardware_features` for the hardware-capability tokens that gate trigger-device features today.

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
    color='BF',
    tail_id_mode='increment',              # auto-number files
    output_format='TIFF',                  # 'TIFF' or 'OME-TIFF'
    x=60000, y=40000, z=5000,              # stage position metadata (µm)
)
```

The full set of free functions in `modules.image_save`:

| Function | Purpose |
|---|---|
| `save_image(scope, array, ...)` | Save a numpy array to TIFF / OME-TIFF with metadata. |
| `save_live_image(scope, save_folder, ...)` | Grab the current live frame from the camera and save (composes `capture_and_wait` + `save_image`). |
| `prepare_image_for_saving(scope, array, ...)` | Flip / bit-convert / build metadata + path; returns `{'image', 'metadata'}`. |
| `generate_image_metadata(scope, color, x, y, z)` | Build the TIFF metadata dict for the current capture settings + position. |
| `generate_image_save_path(scope, save_folder, ...)` | Generate the next unused file path under `tail_id_mode`. |
| `get_next_save_path(scope, path)` | Increment the trailing numeric ID on an existing path. |

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
# Returns: {'width': ..., 'height': ...} in µm
```

These helpers read `scope.pixel_size()` / `scope.lens_focal_length()` when an LVP context is active, and fall back to defaults (47.8 mm, 2.0 µm/px) otherwise. In a bare script that never constructs a `Lumascope`, you'll get the defaults — pass your objective's focal length explicitly.

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

Plugin platform spec lives alongside LumaViewPro; the live-processing tutorial lives in the Firmware repo internal docs.

- **Design**: `docs/PLUGIN_API_DESIGN_2026-05-09.md` — the locked platform spec (PluginSpec, namespaces, registry contracts, loading sequence).
- **Live-processing tutorial**: `Firmware/docs/LIVE_PROCESSING_TUTORIAL.md` — walkthrough for writing a `ctx.plugins.live_processing` plugin.
- **Namespaces (4.x)**: `ctx.plugins.ui`, `ctx.plugins.post_processing`, `ctx.plugins.live_processing`, `ctx.plugins.rest`.

The engineering plugin (`etaluma-engineering/`) is the first production consumer of the plugin platform; see its `pyproject.toml` `entry_points` for a concrete example of how a plugin declares itself.

---

## REST surface reference

> **Status (2026-04):** In development on `4.1.0-dev`. When it ships it will be **disabled by default** — customers enable per-deployment via a feature flag. Treat the example below as design preview, not yet-callable code.

HTTP endpoints wrap the Python API. Control the microscope from any language — MATLAB, LabVIEW, JavaScript, curl.

```
GET  /api/status                    → system status
POST /api/led/on    {color, mA}     → turn on LED
POST /api/led/off                   → turn off all LEDs
POST /api/move      {axis, pos}     → move stage
POST /api/capture                   → capture image, returns file path
GET  /api/live/frame                → grab live frame (binary)
POST /api/protocol/run              → run a protocol file
POST /api/protocol/abort            → abort running protocol
```

**MATLAB example (preview — API not yet live):**

```matlab
url = "http://localhost:8000/api";

webwrite(url + "/move", struct('axis','Z','pos',5000,'wait',true));
webwrite(url + "/led/on", struct('color','BF','mA',100));
result = webwrite(url + "/capture", struct('format','tiff'));

img = imread(result.file_path);
imshow(img);

webwrite(url + "/led/off", struct());
```

---

## Common patterns

### Basic capture

```python
from modules.lumascope_api import Lumascope

scope = Lumascope()
scope.motion.home()
scope.motion.wait_until_finished_moving()

scope.set_objective('10x Oly')
scope.imaging.set_exposure_time(50)
scope.imaging.set_gain(5.0)

scope.motion.move_absolute_position('X', 60000, wait_until_complete=True)
scope.motion.move_absolute_position('Y', 40000, wait_until_complete=True)
scope.motion.move_absolute_position('Z', 5000, wait_until_complete=True)

from modules.image_save import save_image

scope.illumination.led_on('BF', 100)
image = scope.imaging.capture_and_wait()
scope.illumination.leds_off()

save_image(
    scope,
    array=image, save_folder='./output',
    file_root='capture', append='_BF', color='BF',
    output_format='TIFF', x=60000, y=40000, z=5000,
)
scope.disconnect()
```

### Multi-channel composite

```python
from modules.composite_builder import build_composite
from modules.image_save import save_image

channel_images = {}
for color, mA, exp_ms, gain_db in [
    ('Blue',  200, 100, 15),
    ('Green', 150,  80, 12),
    ('Red',   180,  90, 10),
]:
    scope.imaging.set_exposure_time(exp_ms)
    scope.imaging.set_gain(gain_db)
    scope.illumination.led_on(color, mA)
    channel_images[color] = scope.imaging.capture_and_wait()
    scope.illumination.led_off(color)

# Transmitted (brightfield) base image
scope.imaging.set_exposure_time(2.0)
scope.imaging.set_gain(1.0)
scope.illumination.led_on('BF', 100)
bf_image = scope.imaging.capture_and_wait()
scope.illumination.leds_off()

composite = build_composite(
    channel_images=channel_images,
    transmitted_image=bf_image,
    brightness_thresholds={'Blue': 20, 'Green': 15, 'Red': 10},
)

save_image(scope, array=composite, save_folder='./output',
           file_root='composite', color=None, output_format='TIFF')
```

`build_composite` accepts fluorescence keys `'Red'`, `'Green'`, `'Blue'`, `'Lumi'`.

### Z-stack

```python
from modules.image_save import save_image

z_start, z_end, z_step = 4000, 6000, 50    # µm

scope.illumination.led_on('BF', 100)
z = z_start
while z <= z_end:
    scope.motion.move_absolute_position('Z', z, wait_until_complete=True)
    image = scope.imaging.capture_and_wait()
    save_image(
        scope,
        array=image, save_folder='./zstack',
        file_root='z', append=f'_{int(z)}', color='BF',
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
    scope.motion.move_absolute_position('X', sx, wait_until_complete=True)
    scope.motion.move_absolute_position('Y', sy, wait_until_complete=True)

    image = scope.imaging.capture_and_wait()
    save_image(
        scope,
        array=image, save_folder='./scan',
        file_root=f'{well_name}_BF', color='BF',
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
runner.run_single_scan(protocol)
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
scope.motion.move_absolute_position('Z', 5000)
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
