# LumaViewPro — API & Integration Reference

## Overview

LumaViewPro controls Etaluma microscopes: LED illumination, XYZ stage + turret motion, and camera image acquisition. This document covers the Python API, serial protocol, and key conventions needed to write integrations.

**Repository**: `EtalumaSupport/LumaViewPro`
**Platform**: Python 3.10–3.12, Kivy GUI, Windows primary (macOS/Linux experimental)

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  GUI (lumaviewpro.py + Kivy)                     │
│  or External Script                              │
└──────────────┬───────────────────────────────────┘
               │  Python API
┌──────────────▼───────────────────────────────────┐
│  lumascope_api.py  (Lumascope class)             │
│  ├─ LED control                                  │
│  ├─ Motion control                               │
│  ├─ Camera control                               │
│  ├─ Image I/O                                    │
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
| `lumascope_api.py` | Main Python API — `Lumascope` class |
| `serialboard.py` | Serial base class (connect, exchange, reconnect) |
| `ledboard.py` | LED control driver (8 channels, 0–1000 mA) |
| `motorboard.py` | Motion control driver (X, Y, Z, Turret) |
| `camera/camera.py` | Camera base class (abstract) |
| `camera/pyloncamera.py` | Basler Pylon camera driver |
| `camera/idscamera.py` | IDS camera driver |
| `camera/simulated_camera.py` | Simulated camera for testing |
| `simulated_ledboard.py` | Simulated LED board for testing |
| `simulated_motorboard.py` | Simulated motor board for testing |
| `modules/common_utils.py` | Optical calculations, naming, utilities |
| `modules/coord_transformations.py` | Stage ↔ plate ↔ pixel coordinate conversion |
| `modules/objectives_loader.py` | Objective lens database (`data/objectives.json`) |
| `modules/protocol.py` | Protocol file format (load/save/validate) |
| `modules/sequenced_capture_executor.py` | Protocol execution engine |
| `image_utils.py` | Image conversion, timestamps, colormaps |

---

## Python API — Lumascope Class

### Initialization

```python
from lumascope_api import Lumascope

# Real hardware
scope = Lumascope()

# Simulated (no hardware needed)
scope = Lumascope(simulate=True)

# IDS camera instead of Basler Pylon
scope = Lumascope(camera_type="ids")
```

### Connection

```python
scope.are_all_connected()       # True if LED + motor + camera connected
scope.no_hardware                # True if no real hardware detected
scope.disconnect()               # Disconnect all hardware
```

### LED Control

Channels: `Blue` (0), `Green` (1), `Red` (2), `BF` (3), `PC` (4), `DF` (5), `Lumi` (6)

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

# State queries
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
scope.get_current_position('Z')          # Current Z in µm
scope.get_current_position(axis=None)    # Dict: {'X': ..., 'Y': ..., 'Z': ..., 'T': ...}
scope.get_target_position('Z')           # Target Z in µm

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

**Axis limits** (µm):

| Axis | Min | Max | Notes |
|------|-----|-----|-------|
| X | 0 | 120,000 | 120 mm travel |
| Y | 0 | 80,000 | 80 mm travel |
| Z | 0 | 14,000 | 14 mm focus travel |
| T | 1 | 4 | Turret positions |

**Z overshoot**: When moving Z downward, the firmware first moves below the target then approaches from below. This eliminates backlash for consistent focus. Controlled by `overshoot_enabled` parameter.

### Camera Control

```python
# Image capture
image = scope.get_image()                    # Grab frame (numpy array, uint8)
image = scope.get_image(force_to_8bit=False) # Keep native bit depth (12/16-bit)
image = scope.get_image(sum_count=4)         # Average 4 frames

# Frame-validity capture (waits for fresh frame after state changes)
image = scope.capture_and_wait()

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
| `FULLINFO` | Model + serial | Extended info |
| `HOME` | Completion msg | Home all axes (XY, Z, T) |
| `ZHOME` | Completion msg | Home Z only |
| `THOME` | Completion msg | Home turret only |
| `CENTER` | Completion msg | Move stage to center |
| `TARGET_W{axis}{steps}` | Confirmation | Set target position (µsteps) |
| `TARGET_R{axis}` | Integer (µsteps) | Read target position |
| `ACTUAL_R{axis}` | Integer (µsteps) | Read current position |
| `STATUS_R{axis}` | Integer (32-bit) | Read status register |
| `AMAX{axis}` | Integer | Read acceleration limit |
| `DMAX{axis}` | Integer | Read deceleration limit |
| `SPI{axis}0x{addr}{payload}` | Response | Direct SPI read/write to TMC5072 |

**Axes**: `X`, `Y`, `Z`, `T`

**Position conversion** (µsteps ↔ µm):

| Axis | Scale | Formula |
|------|-------|---------|
| Z | 170,667 µsteps/mm | `µm = µsteps / 170.667` |
| X, Y | 20,157 µsteps/mm | `µm = µsteps / 20.157` |
| T | 80,000 µsteps/90° | Position 1-4 mapped to angles |

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

```bash
# Run all tests (no hardware needed)
python -m pytest tests/ --ignore=tests/test_hardware_serial.py -v

# Individual test suites
python -m pytest tests/test_serial_safety.py -v       # Serial driver (96 tests)
python -m pytest tests/test_protocol_execution.py -v  # Protocol execution (78 tests)
python -m pytest tests/test_simulators.py -v          # Simulator fidelity (91 tests)
python -m pytest tests/test_scope_api.py -v           # Scope API (60 tests)
python -m pytest tests/test_integration.py -v         # Integration (23 tests)
python -m pytest tests/test_frame_validity.py -v      # Frame validity (29 tests)
python -m pytest tests/test_regression_p2.py -v       # P2 bug regressions (16 tests)

# Hardware-only tests (requires microscope connected)
python -m pytest tests/test_hardware_serial.py --run-hardware -v
```

**Mock pattern**: All tests mock heavy dependencies (Kivy, camera SDKs, userpaths) before importing modules under test. See any test file header for the standard mock block.

---

## Color Channel Reference

```python
from modules.color_channels import ColorChannel

# Enum values
ColorChannel.Blue   # 0  — 470nm excitation
ColorChannel.Green  # 1  — 530nm excitation
ColorChannel.Red    # 2  — 625nm excitation
ColorChannel.BF     # 3  — Brightfield (white LED)
ColorChannel.PC     # 4  — Phase contrast
ColorChannel.DF     # 5  — Darkfield
ColorChannel.Lumi   # 6  — Luminescence (no excitation)
```

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
