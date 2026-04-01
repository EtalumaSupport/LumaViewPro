# LumaViewPro — API & Integration Reference

## Overview

LumaViewPro controls Etaluma microscopes: LED illumination, XYZ stage + turret motion, and camera image acquisition. This document covers the Python API, serial protocol, and key conventions needed to write integrations.

**Repository**: `EtalumaSupport/LumaViewPro`
**Platform**: Python 3.11–3.13, Kivy GUI, Windows/macOS/Linux

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  GUI (lumaviewpro.py + Kivy)                     │
│  or REST API Server                              │
│  or External Script                              │
└──────────────┬───────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────┐
│  ScopeSession + ProtocolRunner  (GUI-free)       │
│  ├─ scope_commands (executor-routed wrappers)    │
│  ├─ config_helpers (configuration queries)       │
│  └─ SequencedCaptureExecutor (protocol engine)   │
└──────────────┬───────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────┐
│  lumascope_api.py  (Lumascope class)             │
│  ├─ LED control    ├─ Camera control             │
│  ├─ Motion control ├─ Image I/O                  │
│  └─ Coordinate transforms                       │
└──┬────────────┬─────────────┬────────────────────┘
   │            │             │
┌──▼──┐   ┌────▼────┐   ┌────▼─────┐
│ LED │   │  Motor  │   │  Camera  │
│Board│   │  Board  │   │ (Pylon/  │
│(USB)│   │  (USB)  │   │  IDS)    │
└─────┘   └─────────┘   └──────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `lumascope_api.py` | Hardware abstraction — `Lumascope` class |
| `modules/scope_session.py` | GUI-independent session container (headless + REST API) |
| `modules/scope_commands.py` | Executor-routed hardware command wrappers |
| `modules/protocol_runner.py` | Protocol execution orchestration |
| `modules/sequenced_capture_executor.py` | Protocol execution engine |
| `modules/protocol.py` | Protocol file format (load/save/validate) |
| `modules/config_helpers.py` | Configuration query helpers |
| `modules/composite_builder.py` | Composite image generation |
| `modules/motorconfig.py` | Per-unit hardware config (axis limits, conversion factors) |
| `modules/common_utils.py` | Optical calculations, naming, utilities |
| `modules/coord_transformations.py` | Stage ↔ plate ↔ pixel coordinate conversion |
| `modules/objectives_loader.py` | Objective lens database (`data/objectives.json`) |
| `modules/image_utils.py` | Image conversion, timestamps, colormaps |
| `modules/stitcher.py` | Position-based image stitching |
| `modules/stack_builder.py` | Hyperstack / Z-stack builder |
| `drivers/serialboard.py` | Serial base class (connect, exchange, reconnect) |
| `drivers/ledboard.py` | LED control driver (up to 8 channels, 0–1000 mA) |
| `drivers/motorboard.py` | Motion control driver (X, Y, Z, Turret) |
| `drivers/camera.py` | Camera base class + ImageHandlerBase |
| `drivers/pyloncamera.py` | Basler Pylon camera driver |
| `drivers/idscamera.py` | IDS camera driver |
| `drivers/camera_profiles.py` | Per-model hardware specs (gain/exposure ranges, binning) |
| `drivers/firmware_updater.py` | UF2 firmware flash orchestration |
| `drivers/raw_repl.py` | MicroPython raw REPL for config backup/restore |
| `drivers/simulated_camera.py` | Simulated camera for testing |
| `drivers/simulated_ledboard.py` | Simulated LED board for testing |
| `drivers/simulated_motorboard.py` | Simulated motor board for testing |
| `drivers/null_motorboard.py` | No-op motor board (no hardware connected) |
| `drivers/null_ledboard.py` | No-op LED board (no hardware connected) |
| `modules/frame_validity.py` | Single source of truth for capture readiness |
| `modules/notification_center.py` | Thread-safe notification bus (error/warning/info) |
| `modules/video_writer.py` | H.264/MP4 video writer (PyAV primary, cv2 fallback) |
| `modules/video_capture.py` | Protocol video capture session management |

---

## Python API — Lumascope Class

### Initialization

```python
from lumascope_api import Lumascope
from modules.scope_init_config import ScopeInitConfig

# Real hardware
scope = Lumascope()

# Simulated (no hardware needed)
scope = Lumascope(simulate=True)

# Auto-detect camera (default — tries Pylon first, then IDS)
scope = Lumascope(camera_type="auto")

# Force specific camera SDK
scope = Lumascope(camera_type="pylon")
scope = Lumascope(camera_type="ids")
```

### Scope Configuration (initialize)

Call `scope.initialize(config)` once after construction to go from "connected" to "ready-to-use". Sets all scope-level hardware in a single call. Does NOT set per-layer camera settings (gain, exposure, auto-gain).

```python
# From LVP settings dict (used by GUI startup and reconnect):
config = ScopeInitConfig.from_settings(settings, labware)
scope.initialize(config)

# Or construct directly (REST API / scripts):
config = ScopeInitConfig(
    labware=labware_obj,
    objective_id="10x Oly",
    turret_config=None,
    binning_size=1,
    frame_width=3840,
    frame_height=2160,
    acceleration_pct=100,
    stage_offset={'x': 0, 'y': 0},
    scale_bar_enabled=False,
)
scope.initialize(config)
```

### Connection

```python
scope.are_all_connected()       # True if LED + motor + camera connected
scope.no_hardware                # True if no real hardware detected
scope.disconnect()               # Disconnect all hardware
```

### LED Control

Channels: `Blue` (0), `Green` (1), `Red` (2), `BF` (3), `PC` (4), `DF` (5)

**Luminescence** (`Lumi`, channel 6): Not an LED channel. In luminescence mode, all LEDs must be fully off — the image captures emitted light only with no excitation.

```python
scope.leds_enable()              # Enable LED driver
scope.led_on('Blue', 200)        # Blue LED at 200 mA
scope.led_on(0, 200)             # Same, by channel number
scope.led_off('Blue')            # Turn off Blue
scope.leds_off()                 # Turn off all LEDs
scope.leds_disable()             # Disable LED driver

# Fast path (write-only, no response wait — for timing-critical code)
scope.led_on_fast('Red', 100)
scope.led_off_fast('Red')
scope.leds_off_fast()

# State queries (read driver cache, no serial I/O)
scope.led_enabled('Blue')        # True if channel is on
scope.led_illumination('Blue')   # Current mA, or -1 if off
scope.led_states                 # All channels: {color: {enabled, illumination}}
scope.get_led_ma('Blue')         # Current mA setting
scope.get_led_state('Blue')      # {'enabled': True, 'illumination': 200}
scope.get_led_states()           # All channels

# Channel mapping
scope.color2ch('Blue')           # 0
scope.ch2color(0)                # 'Blue'

# Wait for confirmation
scope.led_on('Blue', 200, block=True)  # Blocks until firmware confirms
scope.wait_until_led_on()              # Block until any LED confirms on
```

**Safety limits** (enforced by firmware):
- Per-channel max: 1000 mA
- Board total max: 3000 mA across all channels

### Motion Control

Axes: `X` (stage left-right), `Y` (stage front-back), `Z` (focus), `T` (turret)

```python
# Homing (required before movement)
scope.xyhome()                   # Home XY stage (also homes Z and T)
scope.zhome()                    # Home Z only
scope.thome()                    # Home turret only
scope.has_xyhomed()              # True if XY homed
scope.has_thomed()               # True if turret homed

# Position queries (µm for XYZ, position 1-4 for T)
# Returns predicted position during motion (trapezoidal ramp model),
# confirmed position when idle. Zero serial I/O — reads from cache.
scope.get_current_position('Z')          # Current Z in µm
scope.get_current_position(axis=None)    # Dict: {'X': ..., 'Y': ..., 'Z': ..., 'T': ...}
scope.get_target_position('Z')           # Target Z in µm
scope.get_actual_position('Z')           # Hardware position via serial (use sparingly)

# Absolute moves (µm)
scope.move_absolute_position('Z', 5000)  # Move Z to 5000 µm
scope.move_absolute_position('X', 60000, wait_until_complete=True)  # Blocking move

# Relative moves (µm)
scope.move_relative_position('Z', 100)   # Move Z up 100 µm
scope.move_relative_position('X', -500)  # Move X left 500 µm

# Status
scope.get_target_status('Z')             # True if Z reached target
scope.is_moving()                        # True if any axis moving
scope.wait_until_finished_moving()       # Block until all axes done
scope.get_overshoot()                    # True if Z overshoot in progress

# Turret
scope.has_turret()               # True if turret installed
scope.tmove(2)                   # Move turret to position 2 (1-4)

# Stage center
scope.xycenter()                 # Move to stage center

# Limits
scope.get_axis_limits('Z')       # {'min': 0, 'max': 14000}
scope.get_axes_config()          # All axes with limits and conversion functions
```

**Axis limits**: Axis ranges are defined in `motorconfig.json` and vary by hardware model. Use `scope.get_axis_limits('Z')` or `scope.get_axes_config()` to query limits at runtime. Turret positions are typically 1–4.

**Z overshoot**: When moving Z downward, the firmware first moves below the target then approaches from below. This eliminates backlash for consistent focus. Controlled by `overshoot_enabled` parameter.

**Position listeners** (push-based UI updates):
```python
# Register for position/state changes on any axis.
# Called from the thread that caused the change — schedule UI work via Clock.
def on_position(axis, target_pos, state):
    print(f"{axis} → {target_pos:.1f}µm ({state})")

scope.add_position_listener(on_position)
scope.remove_position_listener(on_position)
```

**Axis state model**:
```python
from modules.lumascope_api import AxisState

scope.get_axis_state('Z')       # AxisState.IDLE, MOVING, HOMING, or UNKNOWN
scope.is_any_axis_moving()      # True if any axis is MOVING
```

### Frame Validity

Frame validity is the **single source of truth** for capture readiness. Every hardware state change (LED, gain, exposure, motion) invalidates the frame. `capture_and_wait()` drains stale frames until all sources have settled, then returns a valid frame.

```python
# Check validity (for debugging / engineering mode)
scope.frame_validity.is_valid               # True if next frame is valid
scope.frame_validity.pending_sources        # {'z_move': 5, 'led': 3} — pending thresholds
scope.frame_validity.frames_until_valid()   # 0 = ready, >0 = keep draining

# Invalidation happens automatically inside:
#   scope.led_on()                → invalidate('led')
#   scope.set_gain()              → invalidate('gain')
#   scope.set_exposure_time()     → invalidate('exposure')
#   scope.move_absolute_position('Z', ...) → invalidate('z_move')
#   scope.move_absolute_position('X', ...) → invalidate('xy_move')

# For motion sources, frames_until_valid() also checks that the axis has
# physically stopped (AxisState == IDLE), not just the frame count.
```

### Camera Control

```python
# Image capture
image = scope.get_image()                    # Grab frame (numpy array, uint8)
image = scope.get_image(force_to_8bit=False) # Keep native bit depth (12/16-bit)
image = scope.get_image(sum_count=4)         # Average 4 frames

# Frame-validity capture — PREFERRED for all protocol/script captures.
# Waits until ALL pending state changes have settled:
#   - Camera pipeline flushed (2+ frames after LED/gain/exposure change)
#   - Motion physically complete (axis state == IDLE for X/Y/Z/T)
# Only then grabs a valid frame. This is the single gate for capture readiness.
image = scope.capture_and_wait()
image = scope.capture_and_wait(exclude_sources=('z_move',))  # AF: OK during Z motion

# Exposure (milliseconds)
scope.set_exposure_time(50)      # 50 ms exposure
scope.get_exposure_time()        # Current exposure (ms)

# Gain
scope.set_gain(10.0)             # Set gain
scope.get_gain()                 # Current gain

# Frame size
scope.get_width()                # Current frame width (pixels)
scope.get_height()               # Current frame height
scope.get_max_width()            # Sensor max width
scope.get_max_height()           # Sensor max height
scope.set_frame_size(2048, 2048) # Set ROI
scope.get_frame_size()           # {'width': ..., 'height': ...}

# Binning
scope.set_binning_size(2)        # 2x2 binning
scope.get_binning_size()         # Current binning factor

# Batched layer settings (gain + exposure + auto-gain in one call)
scope.apply_layer_camera_settings(
    gain=5.0, exposure_ms=50,
    auto_gain=False, auto_gain_settings=None
)

# Camera info
scope.camera_is_connected()      # True if camera connected
scope.get_camera_temps()         # Temperature sensors dict
```

### Image Saving

```python
# Save with metadata
scope.save_image(
    array=image,
    save_folder='/path/to/output',
    file_root='experiment1',
    append='_BF_A1',
    color='BF',
    output_format='tiff',        # 'tiff' or 'ome-tiff'
    x=60000, y=40000, z=5000,    # Stage position (µm)
)

# Generate metadata separately
metadata = scope.generate_image_metadata(color='Blue', x=60000, y=40000, z=5000)
# Returns: pixel_size, plate_x, plate_y, well_label, etc.
```

### Objective Management

```python
scope.set_objective('10x Oly')
scope.get_objective_info('10x Oly')
# Returns: {focal_length, magnification, NA, working_distance_mm, ...}

# Turret integration
scope.set_turret_config({1: '4x Oly', 2: '10x Oly', 3: '20x Oly', 4: '40x w/collar'})
scope.get_turret_position_for_objective_id('10x Oly')  # Returns 2
```

**Available objectives** (from `data/objectives.json`):
`1.25x Oly`, `2x Oly`, `2.5x Meiji`, `4x Oly`, `10x Oly`, `10x Phase`, `20x Oly`, `20x w/collar`, `20x Phase`, `40x w/collar`, `40x Phase`, `60x w/collar`, `60x Meiji`, `100x U Plan Oly`, `100x M Plan Oly`

---

## Headless API — ScopeSession + ProtocolRunner

For scripting, REST API, and headless operation, use `ScopeSession` instead of accessing `Lumascope` directly. ScopeSession is GUI-free and provides executor-routed command wrappers.

### Creating a Headless Session

```python
from modules.settings_init import load_lvp_settings
from modules.scope_session import ScopeSession

# Load settings
settings = load_lvp_settings('./data/current.json')

# Create session with real hardware
session = ScopeSession.create(settings=settings, source_path='./output')
session.start_executors()

# Or with simulated hardware
from lumascope_api import Lumascope
scope = Lumascope(simulate=True)
session = ScopeSession.create(settings=settings, scope=scope)
session.start_executors()
```

### Session Commands

```python
# LED control (routed through io_executor)
session.led_on('Blue', 200)
session.led_on_sync('Blue', 200, timeout=5)  # Blocking
session.led_off('Blue')
session.leds_off()

# Motion control (routed through io_executor)
session.move_absolute('Z', 5000, wait_until_complete=True)
session.move_relative('X', 500)
session.move_home('XY')

# Configuration queries
session.get_layer_configs()
session.get_current_objective_info()
session.get_current_plate_position()
```

### Running Protocols

```python
from modules.protocol import Protocol

# Create protocol runner from session
runner = session.create_protocol_runner()

# Load and run a protocol
protocol = Protocol.from_file('my_protocol.csv')

# Single scan (one pass through all steps)
runner.run_single_scan(protocol)
runner.wait_for_completion()

# Timed protocol (repeating scans over duration)
runner.run_protocol(protocol)
runner.wait_for_completion()

# Abort a running protocol
runner.abort()
```

### Session Lifecycle

```python
# When done
session.shutdown_executors()
session.scope.disconnect()
```

### Coordinate Transformations

```python
from modules.coord_transformations import CoordinateTransformer
ct = CoordinateTransformer()

# Stage µm → plate mm (top-left origin)
plate_x, plate_y = ct.stage_to_plate(labware=labware_dict, stage_offset=offset, sx=60000, sy=40000)

# Plate mm → stage µm
stage_x, stage_y = ct.plate_to_stage(labware=labware_dict, stage_offset=offset, px=50.0, py=30.0)
```

### Optical Calculations

```python
import modules.common_utils as common_utils

# Pixel size (µm per pixel)
pixel_size = common_utils.get_pixel_size(focal_length=4.78, binning_size=1)
# Formula: pixel_width / (tube_focal_length / objective_focal_length) * binning
# pixel_width = 2.0 µm (Basler sensor), tube_focal_length = 47.8 mm

# Field of view (µm)
fov = common_utils.get_field_of_view(
    focal_length=4.78,
    frame_size={'width': 2048, 'height': 2048},
    binning_size=1,
)
# Returns: {'width': ..., 'height': ...} in µm
```

---

## Serial Protocol

Communication with the RP2040 controllers is over USB CDC serial. Each board has independent firmware with its own command set.

### Connection Parameters

| Parameter | LED Board | Motor Board |
|-----------|-----------|-------------|
| VID:PID | 0x0424:0x704C | 0x2E8A:0x0005 |
| Baud | 115200 | 115200 |
| Line ending (send) | `\n` | `\n` |
| Line ending (recv) | `\r\n` | `\r\n` |
| Timeout | 100 ms | 30 s |
| Echo | Yes (`RE: <cmd>`) | No |

**Important**: LED firmware echoes every command as `RE: <command>` before sending the actual response. The host driver auto-detects and strips this echo.

### LED Commands

| Command | Response | Description |
|---------|----------|-------------|
| `INFO` | Board info string | Firmware version, cal status, reset cause |
| `STATUS` | Status string | Board status |
| `LEDS_ENT` | Confirmation | Enable all LED channels |
| `LEDS_ENF` | Confirmation | Disable all LED channels |
| `LEDS_OFF` | Confirmation | Turn off all LEDs |
| `LED{ch}_{mA}` | Confirmation | Set channel 0-7 to mA (float OK: `LED3_200`, `LED0_0.5`) |
| `LED{ch}_OFF` | Confirmation | Turn off channel |
| `LEDREAD{ch}` | Multi-line: I_SENS + LED_K | Read ADC current feedback (v2.0+, engineering mode) |

**Engineering mode** (entered via `FACTORY` command):
| Command | Description |
|---------|-------------|
| `FACTORY` | Enter engineering mode (bypasses safety limits) |
| `Q` | Return to safe mode |
| `RAW{ch}_{val}` | Set raw DAC value |
| `ADCREAD` | Read all 16 ADC channels |
| `ADC{ch}` | Read single ADC channel 0-15 |
| `CALIBRATE` | Per-channel DAC calibration |
| `CALSAVE` | Save calibration to flash |
| `CALCLEAR` | Delete calibration, restore defaults |
| `SELFTEST` | Ramp channels with ADC readback |
| `I2CSCAN` | Scan I2C bus |
| `I2CR addr reg [n]` | Read I2C register(s) |
| `I2CW addr reg bb` | Write I2C register |
| `FWUPDATE` | Reboot into UF2 bootloader |

### Motor Commands

| Command | Response | Description |
|---------|----------|-------------|
| `INFO` | Info string | Firmware version |
| `FULLINFO` | Model + serial + axis info | Extended info (v2.0+) |
| `CONFIG` | Motor configuration | Current config display (v2.0+) |
| `HOME` | Completion msg | Home all axes (XY, Z, T) |
| `ZHOME` | Completion msg | Home Z only |
| `THOME` | Completion msg | Home turret only |
| `CENTER` | Completion msg | Move stage to center |
| `STOP` | Confirmation | Stop all motors immediately (v2.0.4+) |
| `TARGET_W{axis}{steps}` | Confirmation | Set target position (µsteps) |
| `TARGET_R{axis}` | Integer (µsteps) | Read target position |
| `ACTUAL_R{axis}` | Integer (µsteps) | Read current position |
| `STATUS_R{axis}` | Integer (32-bit) | Read status register |
| `DRVSTAT` | All axes | TMC5072 driver status (v2.0+) |
| `DRVSTAT_{axis}` | Single axis | Per-axis driver status (v2.0+) |
| `MOTORDETECT` | Open load flags | Motor presence detection (v2.0+) |
| `VOLTAGE` | Rail status | 24V presence + voltage rail status (v2.0+) |
| `CURRENT` | All axes | CS_ACTUAL, IRUN, IHOLD, SG_RESULT (v2.0+) |
| `AMAX{axis}` | Integer | Read acceleration limit |
| `DMAX{axis}` | Integer | Read deceleration limit |
| `FAN:{duty}` | Confirmation | Set fan PWM duty cycle |
| `SPI{axis}0x{addr}{payload}` | Response | Direct SPI read/write to TMC5072 |

**Axes**: `X`, `Y`, `Z`, `T`

**Responsive homing** (v2.0.4+): During homing, the firmware checks for serial input. `STOP` aborts homing mid-sequence. `INFO`, `ACTUAL_R`, `STATUS_R`, and `VOLTAGE` respond normally. Other commands return `BUSY`.

**Position conversion** (µsteps ↔ µm): Conversion factors are defined in `motorconfig.json` and vary by hardware configuration. Use `scope.get_axes_config()` to retrieve the current axis scaling and limits at runtime rather than relying on hardcoded values.

**Status register** (from `STATUS_R{axis}`):
- Bit 22: Target reached (1 = at target position)
- Bit 0: Left limit switch
- Bit 1: Right limit switch

---

## Simulated Mode

For testing without hardware:

```python
scope = Lumascope(simulate=True)
scope.led.set_timing_mode('fast')      # Skip delays
scope.camera.set_timing_mode('fast')
scope.camera.start_grabbing()

# All API calls work identically
scope.led_on('Blue', 200)
scope.move_absolute_position('Z', 5000)
image = scope.get_image()
```

The simulators (`SimulatedLEDBoard`, `SimulatedMotorBoard`, `SimulatedCamera`) replicate the full API surface with in-memory state tracking.

---

## Protocol File Format

Protocols define multi-step acquisition sequences. Tab-separated file format:

```
LumaViewPro Protocol
Version	5
Period	1.0
Duration	0.002778
Labware	96-well
Capture Root

Steps
Name	X	Y	Z	Auto_Focus	Color	False_Color	Illumination	Gain	Auto_Gain	Exposure	Sum	Objective	Well	Tile	Z-Slice	Custom Step	Tile Group ID	Z-Stack Group ID	Acquire	Video Config	Stim_Config
A1_BF	60000	40000	5000	False	BF	False	100	10	False	50	1	10x Oly	A1		-1	False	-1	-1	image	{'fps': 5, 'duration': 5}	{}
```

**Header fields**:
- `Period`: Scan interval in minutes (>0)
- `Duration`: Total run time in hours (>0, 6 decimal precision)
- `Labware`: Plate type ID
- `Capture Root`: Optional output directory name

**Step fields**:

| Field | Type | Valid Values |
|-------|------|-------------|
| Name | string | Well + color label (e.g., `A1_BF`) |
| X, Y, Z | float | Position in µm (see axis limits) |
| Auto_Focus | bool | `True` / `False` |
| Color | string | `Blue`, `Green`, `Red`, `BF`, `PC`, `DF`, `Lumi` |
| False_Color | bool | Apply false color mapping |
| Illumination | float | LED current in mA (0–1000) |
| Gain | float | Camera gain (typically 0–48 dB) |
| Auto_Gain | bool | Enable auto-gain |
| Exposure | float | Exposure time in ms (>0) |
| Sum | int | Frame averaging count (≥1) |
| Objective | string | Must match `data/objectives.json` key |
| Well | string | Well label (e.g., `A1`, `H12`) |
| Tile | string | Tile label or empty |
| Z-Slice | int | Z-stack index (-1 = none) |
| Custom Step | bool | User-defined step |
| Tile Group ID | int | Tile group (-1 = none) |
| Z-Stack Group ID | int | Z-stack group (-1 = none) |
| Acquire | string | `image` or `video` |
| Video Config | dict | `{'fps': 5, 'duration': 5}` |
| Stim_Config | dict | Stimulation parameters |

---

## Testing

**~1100+ tests** across 14+ test files.

```bash
# Run all tests (no hardware needed)
python -m pytest tests/ --ignore=tests/test_hardware_serial.py -v

# Individual test suites
python -m pytest tests/test_serial_safety.py -v       # Serial driver (96 tests)
python -m pytest tests/test_simulators.py -v          # Simulator fidelity (86 tests)
python -m pytest tests/test_protocol_execution.py -v  # Protocol execution (83 tests)
python -m pytest tests/test_scope_api.py -v           # Scope API (60 tests)
python -m pytest tests/test_firmware_updater.py -v    # Firmware updater (45 tests)
python -m pytest tests/test_validate_steps.py -v      # Protocol validation (43 tests)
python -m pytest tests/test_frame_validity.py -v      # Frame validity (29 tests)
python -m pytest tests/test_time_estimator.py -v      # Time estimation (23 tests)
python -m pytest tests/test_integration.py -v         # Integration (22 tests)
python -m pytest tests/test_stitcher.py -v            # Image stitcher (19 tests)
python -m pytest tests/test_composite_builder.py -v   # Composite builder (18 tests)
python -m pytest tests/test_regression_p2.py -v       # P2 bug regressions (16 tests)
python -m pytest tests/test_motorconfig.py -v         # Motorconfig integration (15 tests)

# Hardware-only tests (requires microscope connected)
python -m pytest tests/test_hardware_serial.py --run-hardware -v
```

**Mock pattern**: All tests mock heavy dependencies (Kivy, camera SDKs, userpaths) before importing modules under test. See any test file header for the standard mock block.

---

## Color Channel Reference

```python
from modules.color_channels import ColorChannel

# Enum values (standard filterset)
ColorChannel.Blue   # 0  — 405nm excitation
ColorChannel.Green  # 1  — 488nm excitation
ColorChannel.Red    # 2  — 589nm excitation
ColorChannel.BF     # 3  — Brightfield (white LED)
ColorChannel.PC     # 4  — Phase contrast
ColorChannel.DF     # 5  — Darkfield
ColorChannel.Lumi   # 6  — Luminescence (all LEDs off, captures emitted light only)
```

**Note**: Excitation wavelengths above are for the standard filterset. A second filterset with different excitation wavelengths is available, and OEM customers may have fully custom filtersets with different wavelengths and/or a different number of excitation channels.

---

## Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `data/objectives.json` | Repo | Objective lens specs (focal length, NA, magnification) |
| `data/labware.json` | Repo | Well plate definitions (dimensions, well layout) |
| `data/settings.json` | Repo | Default application settings |
| `data/scopes.json` | Repo | Microscope model definitions |
| `data/current.json` | AppData | User's current settings (created at runtime) |

---

## Common Patterns

### Basic Capture Script

```python
from lumascope_api import Lumascope

scope = Lumascope()
scope.xyhome()
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
    array=image,
    save_folder='./output',
    file_root='capture',
    append='_BF',
    color='BF',
    output_format='tiff',
    x=60000, y=40000, z=5000,
)

scope.disconnect()
```

### Multi-Channel Composite

```python
channels = [
    {'color': 'Blue', 'mA': 200, 'exposure': 100, 'gain': 15},
    {'color': 'Green', 'mA': 150, 'exposure': 80, 'gain': 12},
    {'color': 'Red', 'mA': 180, 'exposure': 90, 'gain': 10},
]

images = {}
for ch in channels:
    scope.set_exposure_time(ch['exposure'])
    scope.set_gain(ch['gain'])
    scope.led_on(ch['color'], ch['mA'])
    images[ch['color']] = scope.capture_and_wait()
    scope.led_off(ch['color'])
```

### Z-Stack

```python
z_start = 4000   # µm
z_end = 6000     # µm
z_step = 50      # µm

scope.led_on('BF', 100)

z = z_start
while z <= z_end:
    scope.move_absolute_position('Z', z, wait_until_complete=True)
    image = scope.capture_and_wait()
    scope.save_image(array=image, save_folder='./zstack', file_root='z',
                     append=f'_{int(z)}', color='BF', output_format='tiff',
                     x=60000, y=40000, z=z)
    z += z_step

scope.leds_off()
```

---

## Development Conventions

- **Branch policy**: Work on feature branches. Do NOT commit directly to `main` without explicit agreement.
- **Serial protocol**: LED commands use CR+LF response endings, motor commands use LF only. Both accept LF-only on input.
- **Test suite**: pytest, no hardware required for most tests. Hardware tests require `--run-hardware` flag.
- **Executors**: All hardware commands must go through `SequentialIOExecutor` instances — never call hardware directly from GUI code.
- **Frame validity**: After any state change (LED, gain, exposure, Z move), use `capture_and_wait()` to drain stale frames before grabbing a fresh one.
- **Image bit depth**: Camera captures 12-bit natively. `force_to_8bit=True` (default) converts to uint8. Full-depth mode produces uint16 (12-bit values in 16-bit container).
- **Coordinate system**: Stage coordinates are in µm, origin at bottom-right (after homing). Plate coordinates are in mm, origin at top-left.
