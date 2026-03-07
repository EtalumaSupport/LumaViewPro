# LumaViewPro Roadmap

## Testing Improvements

### Z-Dependent Blur for Autofocus Testing (DONE)
- SimulatedCamera generates focus-target images with Z-dependent blur
- Vollath F4 scores peak correctly at the configurable focal Z position
- `z_position_func` callback wired to motor board in simulate mode
- 8 tests verify focus curve shape, symmetry, and peak location

### Real Lumascope Integration Tests
- Replace mock-based protocol execution tests with tests using `Lumascope(simulate=True)`
- Backed by `SimulatedLEDBoard`, `SimulatedMotorBoard`, and `SimulatedCamera`
- Verify end-to-end protocol flow through real code paths
- Eliminates risk of mocks hiding real bugs

### State Assertion Tests
- After protocol runs, verify simulator state matches expectations
- Check LED states (channels on/off, current levels)
- Check motor positions (moved to expected coordinates)
- Check camera settings (exposure, gain set correctly per channel)
- Verify image files written with correct metadata

### GUI Action Tests
131+ user-facing actions identified across 13 functional areas:

| Area | Actions | Examples |
|------|---------|---------|
| Camera & Live View | 15 | Start/stop live, toggle channels, adjust exposure/gain |
| Capture & Save | 12 | Save snapshot, configure save path, toggle auto-save |
| Autofocus | 8 | Run AF, set AF method/range, enable continuous AF |
| Protocol Builder | 20 | Add/remove/reorder steps, set tiling, configure z-stack |
| Protocol Execution | 10 | Run/pause/cancel protocol, view progress |
| Motion Control | 15 | Jog axes, set speed, home axes, go to position |
| Objective Management | 8 | Select objective, set magnification/NA, configure parfocal |
| Well Plate | 12 | Select plate type, click well, define scan region |
| Image Display | 10 | Zoom, pan, toggle overlay, adjust LUT |
| Settings/Config | 8 | Load/save settings, change units, toggle features |
| Histogram | 5 | Adjust min/max, toggle auto-stretch, select channel |
| Composite/Overlay | 4 | Enable composite, set channel colors, adjust blend |
| File Management | 4 | Browse folders, set output directory, open file |

Testing approach: Drive GUI actions against `Lumascope(simulate=True)` and verify hardware state changes through simulator introspection.

## Serial Communication

### Unified Serial Protocol
- Align LED and motor board serial communication patterns (see `docs/SERIAL_ANALYSIS.md`)
- Create shared `SerialBoard` base class used by both `LEDBoard` and `MotorBoard`
- Remove `RE:` echo from LED firmware (Phase 2 — requires firmware update)
- Maintain backwards compatibility: new host tolerates both echo and no-echo firmware
- Hardware timing benchmarks to validate optimized serial code is reliable

### Error Logging
- Ensure all firmware-reported errors (homing failures, SPI errors, etc.) are captured in the error log
- Motor board `exchange_command` errors should propagate to the UI or log, not silently fail
- LED board errors (overcurrent, board max exceeded) should be logged and surfaced

## Engineering Mode

### Turret Position in Filenames
- When engineering mode is active, append turret position (T1, T2, T3, T4) to captured image filenames
- Helps track which objective was used for each capture during testing

### Engineering Mode Testing
- Audit all features that engineering mode enables/changes
- Build test coverage for engineering-mode-specific behavior
- Verify engineering mode toggle doesn't affect normal operation

## Simulate Mode UX

### Auto-Detect and Prompt
- When `Lumascope.__init__` fails to find any hardware (LED, motor, camera all missing), prompt user: "No hardware detected. Run in simulator mode?"
- Kivy popup dialog with Yes/No
- If Yes, restart initialization with `simulate=True`

## Cross-Platform Support

### README Update
- Comprehensive installation instructions for:
  - Windows 10/11
  - macOS (Intel and Apple Silicon)
  - Linux x86_64 (Ubuntu/Debian, Fedora)
  - Linux ARM64 (Raspberry Pi, Jetson)
- Platform-specific dependency notes (pypylon availability, Kivy backends)
- Troubleshooting section for common platform issues

### Linux Deployment
- Research installer options: AppImage, Flatpak, snap, .deb/.rpm
- Evaluate PyInstaller on Linux (including ARM)
- Handle platform-specific dependencies (camera SDKs, serial permissions)
- Systemd service file for kiosk/headless operation
- udev rules for USB device permissions

## Resolved Issues
- **Settings dict race** (fixed): Autofocus states snapshotted on UI thread, restored via callback
- **Protocol serial safety** (analyzed): Safe by design — protocol thread serialized, `protocol_running_global` blocks UI, serial drivers have `RLock()`
- **ctypes.windll crash on non-Windows** (fixed): Platform guard added to `sequenced_capture_executor.py`
- **Composite capture serial race** (fixed): LED commands routed through `io_executor`
- **LED serial 11ms overhead** (fixed): Removed unnecessary `flushInput()`, `flush()`, and two `sleep()` calls from `exchange_command`. Now reads both echo and result lines.
- **LED host never read actual response** (fixed): Old code only read the `RE:` echo line; actual firmware result was discarded by `flushInput()`. Now reads and returns the result line.
- **Simulator API parity** (audited): LED and motor simulators are 100% API-compatible. Camera simulator was missing only `init_auto_gain_focus()` — now added.
- **Simulator timing modes** (added): All three simulators support `set_timing_mode('fast'|'realistic')` for test speed vs realistic behavior.
