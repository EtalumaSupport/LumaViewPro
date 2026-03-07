# LumaViewPro Development Guide

## Architecture

LumaViewPro is a Kivy-based Python application for controlling Etaluma fluorescence microscopes.

### Core Files
- `lumaviewpro.py` — Main application (~10K lines), Kivy GUI
- `lumaviewpro.kv` — Kivy layout (3840+ lines)
- `lumascope_api.py` — Hardware abstraction layer (Lumascope class)

### Hardware Drivers
- `ledboard.py` / `motorboard.py` — Serial USB drivers (real hardware)
- `simulated_ledboard.py` / `simulated_motorboard.py` — Drop-in simulated drivers
- `camera/pylon_camera.py` / `camera/ids_camera.py` — Camera drivers
- `camera/simulated_camera.py` — Simulated camera (synthetic image generation)

### Modules
- `modules/sequenced_capture_executor.py` — Protocol execution engine
- `modules/sequential_io_executor.py` — Thread-safe IO task executor
- `modules/autofocus_functions.py` — Vollath F4 autofocus (numba-accelerated)
- `modules/autofocus_executor.py` — Autofocus state machine
- `modules/contrast_stretcher.py` — Contrast enhancement
- `modules/config_helpers.py` — Pure config/state functions (no globals)
- `modules/scope_commands.py` — LED/motion IOTask dispatch
- `modules/scope_session.py` — ScopeSession state container
- `modules/protocol_runner.py` — Headless protocol execution

### Threading Model
- `camera_executor` — Camera grab operations
- `io_executor` (`SequentialIOExecutor`) — All LED/motor serial commands
- All LED serial commands **must** go through `io_executor` to prevent interleaving
- Serial drivers use `threading.RLock()` for thread safety

### Simulate Mode
`Lumascope(simulate=True)` uses simulated hardware for testing without physical devices:
- `SimulatedLEDBoard` — Full LED state tracking, current feedback simulation
- `SimulatedMotorBoard` — Position tracking, homing simulation, limit switches
- `SimulatedCamera` — Synthetic image generation (gradient, black, white, noise patterns)

### Headless API
```python
session = ScopeSession.create(settings=settings)
session.start_executors()
runner = session.create_protocol_runner()
runner.run_single_scan(protocol, sequence_name="test")
runner.wait_for_completion()
```

## Serial Protocol

| Board | VID | PID | Line ending |
|-------|-----|-----|-------------|
| LED | 0x0424 | 0x704C | CR+LF |
| Motor | 0x2E8A | 0x0005 | LF only |

**LED commands:** `LED{ch}_{mA}`, `LED{ch}_OFF`, `LEDS_OFF`, `LEDS_ENT`, `LEDS_ENF`
**Motor commands:** `TARGET_W{axis}{steps}`, `ACTUAL_R{axis}`, `STATUS_R{axis}`, `HOME`, `ZHOME`, `THOME`

## Test Suite (289 tests)

Run all tests:
```bash
python3 -m pytest tests/ -v
```

### Test Files
| File | Tests | Coverage |
|------|-------|----------|
| `test_serial_safety.py` | 77 | LED/motor driver logic with mocked serial |
| `test_simulators.py` | 71 | LED, motor, and camera simulator tests |
| `test_scope_api.py` | 60 | config_helpers, scope_commands, ScopeSession |
| `test_protocol_execution.py` | 81 | SequencedCaptureExecutor integration |

### Protocol Execution Test Tiers
- **Tier 1 (Core paths):** Basic image capture, auto-gain, auto-focus, video, multi-channel
- **Tier 2 (Feature combos):** Tiling (1x3, 3x5), z-stack, multi-well, stimulation
- **Tier 3 (Edge cases):** Cancellation, back-to-back runs, disconnected scope, boundary values

### Test Infrastructure
- Mock pattern: `sys.modules.setdefault()` for heavy dependencies (`lvp_logger`, `userpaths`, `requests`, `pypylon`, `ids_peak`)
- Simulators are full drop-in replacements with state tracking
- Known issue: back-to-back protocol runs need ~0.2s for `file_io_executor` to drain between runs
- Known issue: `test_scope_api::test_logs_exceptions` can fail when run with protocol tests (mock leakage)

## Branches
- `main` — Production release
- `beta-3.0` — Development base
- `stability-fixes` — Performance and stability improvements (branched from `beta-3.0`)
- `refactor/scope-api` — GUI-independent API extraction (branched from `stability-fixes`)
- `beta-3.0-ids-camera` — IDS camera abstraction
- `beta-3.0-restapi-daniel` — REST API
- `feature/motorconfig2` — Motor configuration module

## Stability Fixes (branch: stability-fixes)
1. Bullseye LUT optimization (20-50x faster), contrast stretching 4x subsampling
2. Kivy texture reuse, `.tobytes()` instead of `.flatten()`, removed `gc.collect()`
3. Rounded button corners, spinner differentiation, slider label alignment
4. Histogram: single `Mesh(mode='triangles')` replacing 128 Rectangles
5. Autofocus saturated pixel masking
6. LED race fix: all LED commands routed through `io_executor`
7. Driver logic test suite (63 tests)
8. Slider track alignment, spinner down arrow, disable protocol accordions
9. Composite capture serial race fix

## API Extraction (branch: refactor/scope-api)
1. `modules/config_helpers.py` (233 lines) — Pure config/state functions
2. `modules/scope_commands.py` (193 lines) — LED/motion dispatch with explicit params
3. `modules/scope_session.py` (225 lines) — ScopeSession state container
4. `modules/protocol_runner.py` (260 lines) — Headless protocol execution
5. `camera/simulated_camera.py` — SimulatedCamera for simulate mode
6. Cross-platform fix: `ctypes.windll` guarded with `sys.platform` check
