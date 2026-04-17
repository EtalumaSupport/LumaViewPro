# LumaViewPro — API & Integration Reference

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
│  Level 1: REST API  (HTTP/JSON, any language)   │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Level 2: ScopeSession  (Python, headless)      │
│  └─ Executor-routed commands, protocol runner   │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Level 3: Lumascope API  (Python)               │
│  ├─ capabilities  ├─ LED / motion / camera      │
│  ├─ observers     ├─ save / restore state       │
│  └─ frame validity                              │
└─────────────────────────────────────────────────┘
```

Each level wraps the one below. Higher = easier. Lower = more control.

For internal serial-protocol details (firmware updates, bring-up tooling), see **Appendix A** at the end of this document — not intended for application integration.

---

## Integration Levels

Pick the level that fits your use case:

| Level | Interface | Language | Best for |
|---|---|---|---|
| **1. REST API** | HTTP (JSON) | Any | External apps, cross-language control |
| **2. ScopeSession** | Python | Python | Headless scripts, automation, tests |
| **3. Lumascope API** | Python | Python | Full hardware control, custom applications |

---

## Level 1: REST API

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

## Level 2: ScopeSession (Headless Python)

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
image = session.scope.capture_and_wait()
session.scope.save_image(
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

## Level 3: Lumascope API (Direct Hardware)

The `Lumascope` class is the hardware abstraction layer. **All hardware state lives here** — LED on/off and illumination, motor positions, camera settings. The GUI, ScopeSession, and REST API all go through this class.

**When to use:** You need fine-grained control beyond ScopeSession, or you're building a custom application.

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
scope.camera_is_connected()               # camera
scope.no_hardware                         # True if all-null (no real hardware found)
scope.disconnect()
```

### Capabilities — **query, don't assume**

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
- **Some scopes have no motor at all.** LS560/LS620 have `caps.axes == ()`. Calling `scope.move_absolute_position('X', …)` against such a scope is a no-op, not an error — but your UI should hide motion controls based on `caps.has_xy_stage` etc.

### LED control

Channels available depend on the scope — always check `scope.capabilities.led_colors`.

**Luminescence** (`Lumi`): not an LED channel. In luminescence mode, all LEDs must be off — the image captures emitted light only.

```python
scope.leds_enable()
scope.led_on('Blue', 200)                 # Blue LED at 200 mA
scope.led_on(0, 200)                      # same, by channel number
scope.led_on('Blue', 200, block=True)     # wait for firmware confirmation
scope.led_off('Blue')
scope.leds_off()                          # turn off all LEDs
scope.leds_disable()

# Fast path (no response wait — timing-critical code only)
scope.led_on_fast('Red', 100)
scope.led_off_fast('Red')
scope.leds_off_fast()

# Channel mapping
scope.color2ch('Blue')                    # 0  (or -1 if the scope doesn't have this color)
scope.ch2color(0)                         # 'Blue'
```

**Safety limits** (enforced by firmware on RP2040 boards): per-channel max 1000 mA, board total max 3000 mA. FX2 boards have their own per-channel cap declared in the camera profile.

#### State queries — read from the API, never the driver

Lumascope holds the authoritative LED state in an internal cache. The API layer's `get_led_state()` / `led_enabled()` / `led_illumination()` read from that cache. **Never call the driver's state methods directly** — for FX2 scopes the driver is a pure command translator and its state queries return sentinels.

```python
scope.led_enabled('Blue')                 # True / False
scope.led_illumination('Blue')            # current mA, or -1 if off
scope.get_led_state('Blue')               # {'enabled': True, 'illumination': 200, 'owner': '…'}
scope.get_led_states()                    # all channels
```

#### Ownership — prevents subsystems from clobbering each other

Tag each LED operation with a subsystem name. Only an owner can turn off a channel they own.

```python
scope.led_on('BF', 200, owner='autofocus')

scope.led_off('BF', owner='protocol')     # no-op — wrong owner
scope.led_off('BF', owner='autofocus')    # works

scope.leds_off_owned('autofocus')         # turn off only channels owned by this subsystem
scope.leds_off()                          # unconditional off (shutdown / cleanup)
```

#### Save / restore — the autofocus pattern

Preserve the user's LED state while a subsystem does its own work, then restore:

```python
# User has Red on at 150 mA. Autofocus needs BF:
snapshot = scope.save_led_state('autofocus')        # capture current state
scope.led_on('BF', 100, owner='autofocus')
# ... autofocus runs: changes Z, captures frames, evaluates focus ...
scope.restore_led_state(snapshot, owner='autofocus')  # Red back on at 150 mA, BF off
```

`save_led_state(tag)` returns a snapshot dict; `restore_led_state(snapshot, owner='…')` reverts. The owner must match the subsystem that did the save.

#### Listeners — push-based notifications

Prefer listeners over polling. Listeners fire on every LED state change (enable, disable, illumination change, ownership change) with no serial I/O cost:

```python
def on_led(color: str, enabled: bool, mA: float, owner: str):
    print(f"{color} {'ON' if enabled else 'OFF'} {mA}mA owner={owner!r}")

scope.add_led_listener(on_led)
# ... later ...
scope.remove_led_listener(on_led)
```

Use polling only when you specifically need the current value at a moment in time (e.g., settling a UI field to match hardware after a reconnect). For "did anything change?" questions, always use listeners.

### Motion control

Axes available depend on the scope — always check `scope.capabilities.axes`.

```python
# Homing (required before movement)
scope.home()                              # home everything the board has
scope.zhome()                             # Z only
scope.thome()                             # turret only
scope.has_homed()                         # True if home() has ever succeeded
scope.has_thomed()                        # turret-specific

# Position queries (µm for XYZ, 1–4 for turret). Read cache, no serial I/O.
scope.get_current_position('Z')           # predicted position during motion, confirmed when idle
scope.get_current_position()              # dict of all axes
scope.get_target_position('Z')            # target µm
scope.get_actual_position('Z')            # hardware position via serial (slow; use sparingly)

# Absolute moves (µm)
scope.move_absolute_position('Z', 5000)
scope.move_absolute_position('X', 60000, wait_until_complete=True)

# Relative moves (µm)
scope.move_relative_position('Z', 100)

# Status
scope.get_target_status('Z')              # True if target reached
scope.is_moving()                         # any axis moving?
scope.wait_until_finished_moving()        # block until all idle
scope.get_overshoot()                     # Z overshoot in progress?

# Turret
scope.has_turret()
scope.tmove(2)                            # turret position 2

# Stage
scope.xycenter()                          # move to stage center
scope.get_axis_limits('Z')                # {'min': 0, 'max': 14000}
scope.get_axes_config()                   # all axes with limits + conversions
scope.axes_present()                      # e.g. ['X', 'Y', 'Z', 'T']
scope.has_axis('T')
```

**Z overshoot:** firmware moves below target then approaches from below, eliminating leadscrew backlash for consistent focus.

**Axis state model:**

```python
from modules.lumascope_api import AxisState

scope.get_axis_state('Z')                 # 'idle', 'moving', 'homing', or 'unknown'
scope.is_any_axis_moving()
```

**Position listeners** (push-based):

```python
def on_position(axis: str, target: float, state: str):
    print(f"{axis} → {target:.1f}µm ({state})")

scope.add_position_listener(on_position)
scope.remove_position_listener(on_position)
```

### Camera control

```python
# Raw frame grab (no validity wait — use capture_and_wait instead in most cases)
image = scope.get_image()
image = scope.get_image(force_to_8bit=False)   # keep native 12/16-bit

# Frame-validity capture — PREFERRED for all real captures.
# Waits for all pending changes (LED, gain, exposure, motion) to settle,
# drains stale frames, returns a valid frame.
image = scope.capture_and_wait()
image = scope.capture_and_wait(
    force_to_8bit=True,
    all_ones_check=True,                   # detect saturated frames
    sum_count=4,                           # average 4 frames
    sum_delay_s=0.05,                      # delay between sum frames
    exclude_sources=('z_move',),           # don't wait for this source (AF uses this)
)

# Exposure (milliseconds) + gain (dB)
scope.set_exposure_time(50)
scope.get_exposure_time()
scope.set_gain(10.0)
scope.get_gain()

# Batched settings (gain + exposure + auto-gain in one call)
scope.apply_layer_camera_settings(
    gain=5.0, exposure_ms=50,
    auto_gain=False, auto_gain_settings=None,
)

# Frame size
scope.set_frame_size(2048, 2048)
scope.get_frame_size()                     # {'width': ..., 'height': ...}
scope.get_max_width()
scope.get_max_height()

# Binning
scope.set_binning_size(2)
scope.get_binning_size()
```

#### Dynamic camera capabilities

Cameras advertise their real limits at connect time. Use these to size UI sliders and clamp auto-exposure / auto-gain:

```python
scope.camera_max_exposure                  # ms, None if no camera connected
scope.camera_max_gain                      # dB, None if no camera connected
```

These are derived from the camera's profile, which is populated at connect via `_query_dynamic_capabilities()` — live SDK queries for Pylon / IDS, hardcoded-from-datasheet for FX2. Per-camera values observed in practice: LS620 FX2 = 42.1 dB gain / 178 ms exposure cap; Pylon/IDS ranges are driver-reported.

#### Save / restore camera state

```python
snapshot = scope.save_camera_state('autofocus')
# ... change gain/exposure ...
scope.restore_camera_state(snapshot)
```

Symmetric to the LED version, but `restore_camera_state` takes only the snapshot (no `owner` arg — camera state is single-owner by nature).

#### Camera listeners

```python
def on_camera(param: str, value: float):
    print(f"Camera {param} = {value}")

scope.add_camera_listener(on_camera)       # fires on set_gain / set_exposure
scope.remove_camera_listener(on_camera)
```

#### Camera info

```python
scope.camera_is_connected()
scope.camera_active                        # True if grabbing
scope.get_camera_temps()                   # temperature sensors (SDK-dependent)
scope.get_camera_info()                    # model, serial, firmware
scope.get_camera_profile_info()            # sensor specs + dynamic ranges; returns:
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
scope.frame_validity.is_valid              # True if next frame is valid
scope.frame_validity.pending_sources       # {'z_move': 5, 'led': 3}
scope.frame_validity.frames_until_valid()  # 0 = ready, >0 = keep draining
```

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

### Objective management

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

### Image saving

```python
scope.save_image(
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

### System info

```python
scope.get_microscope_model()               # 'LS850'
scope.get_motor_info()                     # model, serial, firmware, axis config
scope.get_led_info()                       # firmware, cal status
scope.get_system_info()                    # combined summary
scope.pixel_size()                         # µm per pixel (method — depends on objective)
scope.lens_focal_length()                  # current tube-lens focal length (method)
```

### Coordinate transformations

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

### Optical calculations

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

---

## Common patterns

### Basic capture

```python
from modules.lumascope_api import Lumascope

scope = Lumascope()
scope.home()
scope.wait_until_finished_moving()

scope.set_objective('10x Oly')
scope.set_exposure_time(50)
scope.set_gain(5.0)

scope.move_absolute_position('X', 60000, wait_until_complete=True)
scope.move_absolute_position('Y', 40000, wait_until_complete=True)
scope.move_absolute_position('Z', 5000, wait_until_complete=True)

scope.led_on('BF', 100)
image = scope.capture_and_wait()
scope.leds_off()

scope.save_image(
    array=image, save_folder='./output',
    file_root='capture', append='_BF', color='BF',
    output_format='TIFF', x=60000, y=40000, z=5000,
)
scope.disconnect()
```

### Multi-channel composite

```python
from modules.composite_builder import build_composite

channel_images = {}
for color, mA, exp_ms, gain_db in [
    ('Blue',  200, 100, 15),
    ('Green', 150,  80, 12),
    ('Red',   180,  90, 10),
]:
    scope.set_exposure_time(exp_ms)
    scope.set_gain(gain_db)
    scope.led_on(color, mA)
    channel_images[color] = scope.capture_and_wait()
    scope.led_off(color)

# Transmitted (brightfield) base image
scope.set_exposure_time(2.0)
scope.set_gain(1.0)
scope.led_on('BF', 100)
bf_image = scope.capture_and_wait()
scope.leds_off()

composite = build_composite(
    channel_images=channel_images,
    transmitted_image=bf_image,
    brightness_thresholds={'Blue': 20, 'Green': 15, 'Red': 10},
)

scope.save_image(array=composite, save_folder='./output',
                 file_root='composite', color=None, output_format='TIFF')
```

`build_composite` accepts fluorescence keys `'Red'`, `'Green'`, `'Blue'`, `'Lumi'`.

### Z-stack

```python
z_start, z_end, z_step = 4000, 6000, 50    # µm

scope.led_on('BF', 100)
z = z_start
while z <= z_end:
    scope.move_absolute_position('Z', z, wait_until_complete=True)
    image = scope.capture_and_wait()
    scope.save_image(
        array=image, save_folder='./zstack',
        file_root='z', append=f'_{int(z)}', color='BF',
        output_format='TIFF', z=z,
    )
    z += z_step
scope.leds_off()
```

### Well-plate scan

```python
from modules.coord_transformations import CoordinateTransformer
ct = CoordinateTransformer()

wells = [('A1', 10.0, 20.0), ('A2', 19.0, 20.0), ('A3', 28.0, 20.0)]

scope.led_on('BF', 100)
for well_name, px, py in wells:
    sx, sy = ct.plate_to_stage(labware=labware_obj, stage_offset=offset, px=px, py=py)
    scope.move_absolute_position('X', sx, wait_until_complete=True)
    scope.move_absolute_position('Y', sy, wait_until_complete=True)

    image = scope.capture_and_wait()
    scope.save_image(
        array=image, save_folder='./scan',
        file_root=f'{well_name}_BF', color='BF',
        output_format='TIFF', x=sx, y=sy,
    )
scope.leds_off()
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
scope.camera.start_grabbing()

# All API calls work identically:
scope.led_on('Blue', 200)
scope.move_absolute_position('Z', 5000)
image = scope.get_image()
```

**Only in `simulate=True`**: `set_timing_mode('fast')` lets simulator tests run faster by skipping artificial serial / motor / camera delays:

```python
scope.led.set_timing_mode('fast')
scope.motion.set_timing_mode('fast')
scope.camera.set_timing_mode('fast')
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

This appendix documents direct serial commands used by firmware update tools, board bring-up scripts, and factory calibration. **These are not intended for integration code** — they bypass safety limits, depend on chip-internal register semantics that can change across firmware versions, and can leave the hardware in unsafe states if misused. Application code should stay at Level 2 or Level 3.

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

Axes: `X`, `Y`, `Z`, `T`. Position conversion (µsteps ↔ µm) is in `motorconfig.json`; prefer `scope.get_axes_config()` over reading that file directly.

During homing, `STOP` aborts. `INFO`, `ACTUAL_R`, `STATUS_R`, `VOLTAGE` respond normally. Other commands return `BUSY`.

Direct SPI access to the TMC5072 (register-level motor configuration) and the associated status-register bit semantics are intentionally omitted — those are firmware-internal.

</details>
