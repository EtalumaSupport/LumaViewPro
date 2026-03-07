# LumaViewPro Roadmap

## Testing Improvements

### Z-Dependent Blur for Autofocus Testing
- Add Z-position-dependent blur to `SimulatedCamera` image generation
- Generate images with varying sharpness based on distance from a focal point
- Allows Vollath F4 autofocus algorithm to converge in simulation
- Enables true autofocus integration tests without hardware

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
