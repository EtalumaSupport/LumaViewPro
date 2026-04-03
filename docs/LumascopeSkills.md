# LumaViewPro — API & Integration Reference

## Overview

LumaViewPro controls Etaluma microscopes: LED illumination (up to 8 channels), XYZ stage + turret motion, and camera image acquisition. This document covers every level of integration — from high-level REST calls to raw serial commands.

**Repository**: `EtalumaSupport/LumaViewPro`
**Platform**: Python 3.11–3.13, Windows/macOS/Linux

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Your Application                               │
│  (MATLAB, Python script, LabVIEW, Web app)      │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Level 1: REST API  (HTTP/JSON, any language)   │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Level 2: ScopeSession  (Python, GUI-free)      │
│  ├─ Executor-routed commands (thread-safe)      │
│  ├─ Protocol runner                             │
│  └─ Configuration helpers                       │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│  Level 3: Lumascope API  (Python)               │
│  ├─ LED control       ├─ Camera control         │
│  ├─ Motion control    ├─ Image I/O              │
│  ├─ Observers         └─ Frame validity         │
│  └─ Save/restore (LED + camera state)           │
└──┬────────────┬─────────────┬───────────────────┘
   │            │             │
┌──▼──┐   ┌────▼────┐   ┌────▼─────┐
│Level│   │ Level 4 │   │ Level 4  │
│4:LED│   │  Motor  │   │  Camera  │
│Board│   │  Board  │   │ (Pylon/  │
│(USB)│   │  (USB)  │   │  IDS)    │
└──┬──┘   └────┬────┘   └──────────┘
   │            │
   │   Level 5: Serial Protocol
   └────────────┘  (USB CDC, raw ASCII commands)
```

Each level wraps the one below. Higher = easier. Lower = more control.

---

## Integration Levels

Choose the level that fits your use case:

| Level | Interface | Language | Best For |
|-------|-----------|----------|----------|
| **1. REST API** | HTTP (JSON) | Any (MATLAB, Python, LabVIEW, etc.) | External apps, cross-language control |
| **2. ScopeSession** | Python | Python | Headless scripts, automation, testing |
| **3. Lumascope API** | Python | Python | Full hardware control, custom applications |
| **4. Drivers** | Python | Python | Direct board communication, firmware tools |
| **5. Serial Protocol** | USB CDC | Any | Raw hardware control, custom drivers |

Each level wraps the one below it. Higher levels are easier; lower levels give more control.

---

## Level 1: REST API (Coming in 4.1)

HTTP endpoints that wrap the Python API. Control the microscope from any language — MATLAB, LabVIEW, JavaScript, curl.

```
GET  /api/status                    → system status
POST /api/led/on    {color, mA}    → turn on LED
POST /api/led/off                  → turn off all LEDs
POST /api/move      {axis, pos}    → move stage
POST /api/capture                  → capture image, returns file path
GET  /api/live/frame               → grab live frame (binary)
POST /api/protocol/run             → run a protocol file
POST /api/protocol/abort           → abort running protocol
```

**MATLAB example** (preview — API not yet live):
```matlab
url = "http://localhost:8000/api";

% Move to position and capture
webwrite(url + "/move", struct('axis','Z','pos',5000,'wait',true));
webwrite(url + "/led/on", struct('color','BF','mA',100));
result = webwrite(url + "/capture", struct('format','tiff'));

% Read captured image
img = imread(result.file_path);
imshow(img);

% Cleanup
webwrite(url + "/led/off", struct());
```

---

## Level 2: ScopeSession (Headless Python)

GUI-free session container. All hardware commands are routed through executor threads for thread safety. Use this for scripts and automation.

**When to use**: You want to write a Python script that controls the microscope without the GUI.

### Setup

```python
from modules.settings_init import load_lvp_settings
from modules.scope_session import ScopeSession

# Real hardware
settings = load_lvp_settings('./data/current.json')
session = ScopeSession.create(settings=settings, source_path='.')
session.start_executors()

# Or simulated (no hardware needed)
from modules.lumascope_api import Lumascope
scope = Lumascope(simulate=True)
session = ScopeSession.create(settings=settings, scope=scope)
session.start_executors()
```

### LED Control

```python
session.led_on('Blue', 200)              # Non-blocking
session.led_on_sync('Blue', 200)         # Blocking (waits for confirmation)
session.led_off('Blue')
session.leds_off()
```

### Motion

```python
session.move_home('XY')                  # Home XY (also homes Z, T)
session.move_absolute('Z', 5000, wait_until_complete=True)
session.move_relative('X', 500)
```

### Capture

```python
# Access the underlying scope for capture
image = session.scope.capture_and_wait()
session.scope.save_image(
    array=image, save_folder='./output',
    file_root='capture', append='_BF', color='BF',
)
```

### Running Protocols

```python
from modules.protocol import Protocol

runner = session.create_protocol_runner()
protocol = Protocol.from_file(
    'my_protocol.tsv',
    tiling_configs_file_loc='./data/tiling.json',
)

runner.run_single_scan(protocol)
runner.wait_for_completion()

# Or abort
runner.abort()
```

### Configuration Queries

```python
session.get_layer_configs()              # All layer settings
session.get_current_objective_info()     # Active objective
session.get_current_plate_position()     # Current XY in plate coords
session.get_auto_gain_settings()         # Auto-gain config
```

### Cleanup

```python
session.shutdown_executors()
session.scope.disconnect()
```

---

## Level 3: Lumascope API (Direct Hardware)

The `Lumascope` class is the hardware abstraction layer. All hardware state lives here — LED on/off, motor positions, camera settings. The GUI, ScopeSession, and REST API all go through this class.

**When to use**: You need fine-grained hardware control beyond what ScopeSession provides, or you're building a custom application.

### Initialization

```python
from modules.lumascope_api import Lumascope
from modules.scope_init_config import ScopeInitConfig

scope = Lumascope()                      # Real hardware (auto-detect camera)
scope = Lumascope(simulate=True)         # Simulated (no hardware)
scope = Lumascope(camera_type="pylon")   # Force Basler Pylon SDK
scope = Lumascope(camera_type="ids")     # Force IDS SDK

# Configure scope hardware (frame size, objective, binning, etc.)
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
scope.are_all_connected()                # True if LED + motor + camera connected
scope.motor_connected                    # Motor board connected?
scope.led_connected                      # LED board connected?
scope.camera_is_connected()              # Camera connected?
scope.no_hardware                        # True if no real hardware detected
scope.disconnect()                       # Disconnect all hardware
```

### LED Control

Channels: `Blue` (0), `Green` (1), `Red` (2), `BF` (3), `PC` (4), `DF` (5)

**Luminescence** (`Lumi`, channel 6): Not an LED channel. In luminescence mode, all LEDs must be fully off — the image captures emitted light only.

```python
# Basic on/off
scope.leds_enable()                      # Enable LED driver
scope.led_on('Blue', 200)               # Blue LED at 200 mA
scope.led_on(0, 200)                    # Same, by channel number
scope.led_on('Blue', 200, block=True)   # Wait for firmware confirmation
scope.led_off('Blue')                   # Turn off Blue
scope.leds_off()                        # Turn off all LEDs
scope.leds_disable()                    # Disable LED driver

# Fast path (write-only, no response wait — timing-critical code)
scope.led_on_fast('Red', 100)
scope.led_off_fast('Red')
scope.leds_off_fast()

# State queries (read cache, no serial I/O)
scope.led_enabled('Blue')               # True if channel is on
scope.led_illumination('Blue')          # Current mA, or -1 if off
scope.get_led_state('Blue')             # {'enabled': True, 'illumination': 200}
scope.get_led_states()                  # All channels

# Channel mapping
scope.color2ch('Blue')                  # 0
scope.ch2color(0)                       # 'Blue'
```

**Safety limits** (enforced by firmware):
- Per-channel max: 1000 mA
- Board total max: 3000 mA across all channels

**LED ownership** (prevents subsystems from clobbering each other):

```python
# Turn on with ownership — only the owner can turn it off
scope.led_on('BF', 200, owner='autofocus')

# No-op (wrong owner):
scope.led_off('BF', owner='protocol')

# Works (matching owner):
scope.led_off('BF', owner='autofocus')

# Turn off only channels owned by a specific subsystem
scope.leds_off_owned('autofocus')

# Nuclear off (shutdown/cleanup — ignores ownership)
scope.leds_off()
```

**LED save/restore** (for AF, protocol, camera pause):

```python
snapshot = scope.save_led_state('my_operation')
# ... do work with LEDs ...
scope.restore_led_state(snapshot, owner='my_operation')
```

**LED listeners** (push-based state notifications):

```python
def on_led(color, enabled, mA, owner):
    print(f"{color} {'ON' if enabled else 'OFF'} {mA}mA")

scope.add_led_listener(on_led)
scope.remove_led_listener(on_led)
```

### Motion Control

Axes: `X` (stage left-right), `Y` (stage front-back), `Z` (focus), `T` (turret)

```python
# Homing (required before movement)
scope.xyhome()                           # Home XY + Z + T
scope.zhome()                            # Home Z only
scope.thome()                            # Home turret only
scope.has_xyhomed()                      # True if XY homed
scope.has_thomed()                       # True if turret homed

# Position queries (µm for XYZ, position 1-4 for T)
# Returns predicted position during motion, confirmed when idle. No serial I/O.
scope.get_current_position('Z')          # Current Z in µm
scope.get_current_position()             # Dict: {'X': ..., 'Y': ..., 'Z': ..., 'T': ...}
scope.get_target_position('Z')           # Target Z in µm
scope.get_actual_position('Z')           # Hardware position via serial (slow, use sparingly)

# Absolute moves (µm)
scope.move_absolute_position('Z', 5000)
scope.move_absolute_position('X', 60000, wait_until_complete=True)

# Relative moves (µm)
scope.move_relative_position('Z', 100)
scope.move_relative_position('X', -500)

# Status
scope.get_target_status('Z')             # True if Z reached target
scope.is_moving()                        # True if any axis moving
scope.wait_until_finished_moving()        # Block until all axes done
scope.get_overshoot()                    # True if Z overshoot in progress

# Turret
scope.has_turret()                       # True if turret installed
scope.tmove(2)                           # Move turret to position 2

# Stage center
scope.xycenter()                         # Move to stage center

# Axis info
scope.get_axis_limits('Z')               # {'min': 0, 'max': 14000}
scope.get_axes_config()                  # All axes with limits and conversions
scope.axes_present()                     # ['X', 'Y', 'Z'] or ['X', 'Y', 'Z', 'T']
scope.has_axis('T')                      # True if turret axis exists
```

**Z overshoot**: When moving Z downward, firmware moves below target then approaches from below, eliminating backlash for consistent focus.

**Axis limits**: Defined in `motorconfig.json`, vary by hardware model. Always query at runtime.

**Position listeners** (push-based updates):

```python
def on_position(axis, target_pos, state):
    print(f"{axis} → {target_pos:.1f}µm ({state})")

scope.add_position_listener(on_position)
scope.remove_position_listener(on_position)
```

**Axis state model**:

```python
from modules.lumascope_api import AxisState

scope.get_axis_state('Z')               # 'idle', 'moving', 'homing', or 'unknown'
scope.is_any_axis_moving()              # True if any axis is MOVING
```

### Camera Control

```python
# Image capture — basic
image = scope.get_image()                # Grab frame (numpy uint8)
image = scope.get_image(force_to_8bit=False)  # Keep native 12/16-bit

# Frame-validity capture — PREFERRED for all captures
# Waits for all pending changes (LED, gain, exposure, motion) to settle,
# drains stale frames, then returns a valid frame.
image = scope.capture_and_wait()
image = scope.capture_and_wait(
    force_to_8bit=True,                  # Convert to uint8 (default)
    all_ones_check=True,                 # Detect saturated frames
    sum_count=4,                         # Average 4 frames
    sum_delay_s=0.05,                    # Delay between sum frames
    exclude_sources=('z_move',),         # OK during Z motion (AF use)
)

# Exposure (milliseconds)
scope.set_exposure_time(50)
scope.get_exposure_time()

# Gain (dB)
scope.set_gain(10.0)
scope.get_gain()

# Batched settings (gain + exposure + auto-gain in one call)
scope.apply_layer_camera_settings(
    gain=5.0, exposure_ms=50,
    auto_gain=False, auto_gain_settings=None,
)

# Frame size
scope.set_frame_size(2048, 2048)
scope.get_frame_size()                   # {'width': ..., 'height': ...}
scope.get_max_width()
scope.get_max_height()

# Binning
scope.set_binning_size(2)                # 2x2 binning
scope.get_binning_size()

# Camera info
scope.camera_is_connected()
scope.camera_active                      # True if camera is grabbing
scope.get_camera_temps()                 # Temperature sensors dict
scope.get_camera_info()                  # Model, serial, firmware, etc.
scope.get_camera_profile_info()          # Sensor specs, gain/exposure ranges
```

**Camera save/restore** (for AF, channel switching):

```python
snapshot = scope.save_camera_state('my_operation')
# ... change gain/exposure ...
scope.restore_camera_state(snapshot)
```

**Camera listeners** (push-based gain/exposure notifications):

```python
def on_camera(param, value):
    print(f"Camera {param} = {value}")

scope.add_camera_listener(on_camera)     # Fires on set_gain/set_exposure only
scope.remove_camera_listener(on_camera)
```

### Frame Validity

Frame validity is the single source of truth for capture readiness. Every hardware state change invalidates the frame. `capture_and_wait()` drains stale frames until all sources settle.

```python
scope.frame_validity.is_valid            # True if next frame is valid
scope.frame_validity.pending_sources     # {'z_move': 5, 'led': 3}
scope.frame_validity.frames_until_valid()  # 0 = ready, >0 = keep draining

# Invalidation happens automatically inside:
#   scope.led_on()                → invalidate('led')
#   scope.set_gain()              → invalidate('gain')
#   scope.set_exposure_time()     → invalidate('exposure')
#   scope.move_absolute_position('Z', ...) → invalidate('z_move')
```

### Objective Management

```python
scope.set_objective('10x Oly')
scope.get_current_objective_id()         # '10x Oly'
scope.get_objective_info('10x Oly')      # {focal_length, magnification, NA, ...}
scope.get_available_objectives()         # ['1.25x Oly', '2x Oly', ...]
scope.get_current_objective()            # Full info dict for current objective

# Turret integration
scope.set_turret_config({1: '4x Oly', 2: '10x Oly', 3: '20x Oly', 4: '40x w/collar'})
scope.get_turret_config()
scope.get_turret_position_for_objective_id('10x Oly')  # Returns 2
```

### Image Saving

```python
scope.save_image(
    array=image,
    save_folder='/path/to/output',
    file_root='experiment1',
    append='_BF_A1',
    color='BF',
    tail_id_mode='increment',            # Auto-number files
    output_format='TIFF',                # 'TIFF' or 'OME-TIFF'
    x=60000, y=40000, z=5000,           # Stage position metadata (µm)
)
```

### System Info

```python
scope.get_microscope_model()             # 'LS850'
scope.get_motor_info()                   # Model, serial, firmware, axis config
scope.get_led_info()                     # Firmware, calibration status
scope.get_system_info()                  # Combined system summary
scope.pixel_size                         # µm per pixel (depends on objective)
scope.lens_focal_length                  # Current objective focal length
```

### Coordinate Transformations

```python
from modules.coord_transformations import CoordinateTransformer
ct = CoordinateTransformer()

# Stage µm → plate mm (top-left origin)
plate_x, plate_y = ct.stage_to_plate(
    labware=labware_dict, stage_offset=offset, sx=60000, sy=40000,
)

# Plate mm → stage µm
stage_x, stage_y = ct.plate_to_stage(
    labware=labware_dict, stage_offset=offset, px=50.0, py=30.0,
)
```

### Optical Calculations

```python
import modules.common_utils as common_utils

# Pixel size (µm per pixel)
pixel_size = common_utils.get_pixel_size(focal_length=4.78, binning_size=1)

# Field of view (µm)
fov = common_utils.get_field_of_view(
    focal_length=4.78,
    frame_size={'width': 2048, 'height': 2048},
    binning_size=1,
)
# Returns: {'width': ..., 'height': ...} in µm
```

---

## Level 4: Drivers (Direct Board Communication)

Use the production drivers for direct board control. These handle connection, reconnection, echo stripping, and line ending differences.

**When to use**: Firmware tools, board testing, custom utilities that need direct serial access without the full Lumascope abstraction.

```python
from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard

# Connect to LED board
led = LEDBoard()                         # Auto-detect by VID:PID
led.exchange_command('LED3_200')         # Set BF LED to 200mA
led.exchange_command('LEDS_OFF')         # All LEDs off

# Connect to motor board
motor = MotorBoard()                     # Auto-detect by VID:PID
motor.exchange_command('HOME')           # Home all axes
motor.exchange_command('TARGET_WZ682666')  # Move Z to position (µsteps)
pos = motor.exchange_command('ACTUAL_RZ')  # Read Z position (µsteps)

# Raw REPL access (firmware file transfer)
motor.enter_raw_repl()
motor.repl_list_files()
motor.repl_read_file('motorconfig.json')
motor.repl_write_file('main.py', content)
motor.exit_raw_repl()
```

### Connection Parameters

| Parameter | LED Board | Motor Board |
|-----------|-----------|-------------|
| VID:PID | 0x0424:0x704C | 0x2E8A:0x0005 |
| Transport | UART via USB hub (115200 baud) | USB CDC native |
| Line ending (send) | `\n` | `\n` |
| Line ending (recv) | `\r\n` | `\n` |
| Timeout (default) | 100 ms | 5 s (homing: 15–30 s) |

---

## Level 5: Serial Protocol (Raw Commands)

For controlling the hardware from any language without the Python drivers. Send commands as ASCII text over USB CDC serial.

**When to use**: Custom drivers in C/C++/LabVIEW/MATLAB, or direct debugging via a serial terminal.

### LED Board Commands

| Command | Response | Description |
|---------|----------|-------------|
| `INFO` | Board info (6 lines) | Firmware version, cal status, heap |
| `LEDS_ENT` | Confirmation | Enable all LED channels |
| `LEDS_ENF` | Confirmation | Disable all LED channels |
| `LEDS_OFF` | Confirmation | Turn off all LEDs |
| `LED{ch}_{mA}` | Confirmation | Set channel 0-7 to mA (float OK: `LED3_200`, `LED0_0.5`) |
| `LED{ch}_OFF` | Confirmation | Turn off channel |
| `LEDREAD{ch}` | I_SENS + LED_K | Read ADC current feedback |

**Engineering mode** (entered via `FACTORY`, exit via `Q`):

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
| `FWUPDATE` | Reboot into UF2 bootloader |

### Motor Board Commands

| Command | Response | Description |
|---------|----------|-------------|
| `INFO` | Info string | Firmware version |
| `FULLINFO` | Extended info | Model, serial, axis status |
| `CONFIG` | Config display | Current motor configuration |
| `HOME` | Completion msg | Home all axes |
| `ZHOME` | Completion msg | Home Z only |
| `THOME` | Completion msg | Home turret only |
| `CENTER` | Completion msg | Move stage to center |
| `STOP` | Confirmation | Stop all motors immediately |
| `TARGET_W{axis}{steps}` | Confirmation | Set target position (µsteps) |
| `TARGET_R{axis}` | Integer | Read target position (µsteps) |
| `ACTUAL_R{axis}` | Integer | Read current position (µsteps) |
| `STATUS_R{axis}` | Integer (32-bit) | Read status register |
| `DRVSTAT` | All axes | TMC5072 driver status |
| `MOTORDETECT` | Open load flags | Motor presence detection |
| `VOLTAGE` | Rail status | 24V + voltage rail status |
| `CURRENT` | All axes | CS_ACTUAL, IRUN, IHOLD, SG_RESULT |
| `SPI{axis}0x{addr}{payload}` | Response | Direct SPI to TMC5072 |

**Axes**: `X`, `Y`, `Z`, `T`

**Position conversion**: µsteps ↔ µm factors are in `motorconfig.json` and vary by hardware. Query via `CONFIG` command or `scope.get_axes_config()`.

**Status register bits**: Bit 22 = target reached, Bit 0 = left limit, Bit 1 = right limit.

**Responsive homing**: During homing, `STOP` aborts. `INFO`, `ACTUAL_R`, `STATUS_R`, `VOLTAGE` respond normally. Other commands return `BUSY`.

---

## Common Patterns

### Basic Capture Script

```python
from modules.lumascope_api import Lumascope

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
    array=image, save_folder='./output',
    file_root='capture', append='_BF', color='BF',
    output_format='TIFF', x=60000, y=40000, z=5000,
)
scope.disconnect()
```

### Multi-Channel Composite

```python
from modules.composite_builder import build_composite

# Capture fluorescence channels
channel_images = {}
for ch in [('Blue', 200, 100, 15), ('Green', 150, 80, 12), ('Red', 180, 90, 10)]:
    color, mA, exp, gain = ch
    scope.set_exposure_time(exp)
    scope.set_gain(gain)
    scope.led_on(color, mA)
    channel_images[color] = scope.capture_and_wait()
    scope.led_off(color)

# Capture transmitted (BF) base image
scope.set_exposure_time(2.0)
scope.set_gain(1.0)
scope.led_on('BF', 100)
bf_image = scope.capture_and_wait()
scope.leds_off()

# Build composite RGB image (H, W, 3)
composite = build_composite(
    channel_images=channel_images,
    transmitted_image=bf_image,
    brightness_thresholds={'Blue': 20, 'Green': 15, 'Red': 10},
)

scope.save_image(array=composite, save_folder='./output',
                 file_root='composite', color=None, output_format='TIFF')
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
    scope.save_image(
        array=image, save_folder='./zstack',
        file_root='z', append=f'_{int(z)}', color='BF',
        output_format='TIFF', z=z,
    )
    z += z_step

scope.leds_off()
```

### Well Plate Scan

```python
from modules.coord_transformations import CoordinateTransformer
ct = CoordinateTransformer()

wells = [('A1', 10.0, 20.0), ('A2', 19.0, 20.0), ('A3', 28.0, 20.0)]

scope.led_on('BF', 100)

for well_name, px, py in wells:
    sx, sy = ct.plate_to_stage(labware=labware, stage_offset=offset, px=px, py=py)
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

### Headless Protocol Run

```python
from modules.scope_session import ScopeSession
from modules.settings_init import load_lvp_settings
from modules.protocol import Protocol

settings = load_lvp_settings('./data/current.json')
session = ScopeSession.create(settings=settings)
session.start_executors()

protocol = Protocol.from_file(
    './my_protocol.tsv',
    tiling_configs_file_loc='./data/tiling.json',
)

runner = session.create_protocol_runner()
runner.run_single_scan(protocol)
runner.wait_for_completion()

session.shutdown_executors()
session.scope.disconnect()
```

---

## Simulated Mode

For development and testing without hardware:

```python
scope = Lumascope(simulate=True)
scope.led.set_timing_mode('fast')        # Skip serial delays
scope.motion.set_timing_mode('fast')     # Skip motor delays
scope.camera.set_timing_mode('fast')     # Skip camera delays
scope.camera.start_grabbing()

# All API calls work identically
scope.led_on('Blue', 200)
scope.move_absolute_position('Z', 5000)
image = scope.get_image()
```

---

## Protocol File Format

Tab-separated file defining multi-step acquisition sequences:

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

**Step fields**:

| Field | Type | Description |
|-------|------|-------------|
| Name | string | Step label (e.g., `A1_BF`) |
| X, Y, Z | float | Position in µm |
| Auto_Focus | bool | Run autofocus at this step |
| Color | string | `Blue`, `Green`, `Red`, `BF`, `PC`, `DF`, `Lumi` |
| False_Color | bool | Apply false color mapping |
| Illumination | float | LED current in mA (0–1000) |
| Gain | float | Camera gain (typically 0–48 dB) |
| Auto_Gain | bool | Enable auto-gain |
| Exposure | float | Exposure time in ms |
| Sum | int | Frame averaging count (≥1) |
| Objective | string | Must match `data/objectives.json` |
| Well | string | Well label (e.g., `A1`) |
| Acquire | string | `image` or `video` |

---

## Color Channel Reference

```python
from modules.color_channels import ColorChannel

ColorChannel.Blue   # 0  — 405nm excitation
ColorChannel.Green  # 1  — 488nm excitation
ColorChannel.Red    # 2  — 589nm excitation
ColorChannel.BF     # 3  — Brightfield (white LED)
ColorChannel.PC     # 4  — Phase contrast
ColorChannel.DF     # 5  — Darkfield
ColorChannel.Lumi   # 6  — Luminescence (all LEDs off)
```

Excitation wavelengths are for the standard filterset. OEM customers may have custom filtersets.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `data/objectives.json` | Objective lens specs (focal length, NA, magnification) |
| `data/labware.json` | Well plate definitions (dimensions, well layout) |
| `data/settings.json` | Default application settings |
| `data/scopes.json` | Microscope model definitions |
| `data/current.json` | User's current settings (runtime) |
| `data/tiling.json` | Tiling configuration definitions |
| `data/camera_timing/*.json` | Per-model camera timing profiles |
