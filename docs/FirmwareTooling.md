# LumaViewPro — internal serial-protocol interfaces (firmware tooling only)

Nothing here is part of the API that [`LumascopeSkills.md`](LumascopeSkills.md) documents.

This document describes direct serial commands used by firmware update tools, board bring-up scripts, and factory calibration. **These are not intended for integration code** — they bypass safety limits, depend on chip-internal register semantics that can change across firmware versions, and can leave the hardware in unsafe states if misused. Application code should stay at the ScopeSession or Lumascope sub-API layer.

<details>
<summary>Show internal interfaces</summary>

### Direct board drivers

```python
from drivers.ledboard import LEDBoard
from drivers.motorboard import MotorBoard
from modules.path_utils import read_installation_file, resolve_data_file

led = LEDBoard()                           # auto-detect by VID:PID
led.exchange_command('LED3_200')           # set BF LED to 200 mA
led.exchange_command('LEDS_OFF')

# The motor driver takes the shipped defaults (travel limits, microsteps
# per mm) as a required argument; a missing or unusable file raises
# InstallationFileError, naming it.
defaults = read_installation_file(resolve_data_file('motorconfig_defaults.json'))
motor = MotorBoard(motorconfig_defaults=defaults)  # auto-detect by VID:PID
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

Axes: `X`, `Y`, `Z`, `T`. Position conversion (µsteps ↔ µm) is in `motorconfig.json`; the API takes and answers micrometres, and an axis's travel is `scope.motion.get_axis_limits(axis)`, so a caller never reads that file.

During homing, `STOP` aborts. `INFO`, `ACTUAL_R`, `STATUS_R`, `VOLTAGE` respond normally. Other commands return `BUSY`.

Direct SPI access to the TMC5072 (register-level motor configuration) and the associated status-register bit semantics are intentionally omitted — those are firmware-internal.

</details>
