# Plugin Platform + API Namespace Design (2026-05-09)

## Status (single source of truth -- update on every state change)

| Phase | Status | Branch | Notes |
|---|---|---|---|
| Plugin platform Phase A -- registry + namespaces + test harness | **shipped (LVP `143fec1` cherry-pick onto `4.0.0-beta`)** | `4.0.0-beta` | `modules/plugins/__init__.py` + AppContext.plugins field + startup/shutdown wire-up + plugin test harness fixture. Originally on `4.0.0-plugin-platform` as `b5f9354`; cherry-picked onto beta 2026-05-17 evening via `4.0.0-plugin-consolidated` then merged. |
| Plugin platform Phase B1 -- entry-points config in etaluma-engineering | **shipped (Firmware `b95624c`)** | `3.0-firmware` | `[project.entry-points."lvp.plugins"]` block added to `etaluma-engineering/pyproject.toml` 2026-05-17 evening. Both legacy import-by-name and new entry-points discovery paths coexisted briefly; B2 then retired the legacy path. Operational requirement: every dev/bench machine must re-run `pip install -e Firmware/etaluma-engineering/` once to register the entry-point. |
| Plugin platform Phase B2 -- LVP retires import-by-name fallback + engineering plugin refactor | **shipped (Firmware `d3a7d57` + LVP `87cf33f` integrated landing)** | beta + `3.0-firmware` | etaluma_engineering 0.7.0: module-level `PluginSpec`, `register(ctx)` calls `ctx.plugins.ui.register(spec, 'left_sidebar.accordion', builder)`, owns the engineering-mode auto-enable side effect (gated on `ctx.no_engineering`). LVP-side: deleted `try: import etaluma_engineering` block at lumaviewpro.py:770-797, added UI-mount consumer that iterates `ctx.plugins.ui.mounts()` and attaches widgets, switched `enable_engineering_logs` to read `ctx.engineering_mode` (the plugin may have flipped it). End-to-end verified: uninstalled vs reinstalled launches behave correctly (no plugin discovered/mounted vs `[Plugins ] etaluma_engineering v0.7.1 loaded` + `[LVP Main  ] Mounted etaluma_engineering at left_sidebar.accordion`). |
| Plugin platform -- auto-run on protocol complete | **shipped (LVP `312d755`)** | `4.0.0-beta` | New `PluginSpec.auto_run_on_protocol_complete: bool = False` field + `PostProcessingRegistry.handlers()` iterator + `run_protocol_complete_processors()` dispatcher + UI hook in `_dispatch_post_processing_auto_run()` (called from both completion paths). Per-plugin exceptions caught + logged + recorded via `record_runtime_error`. UI-trigger only today; REST-trigger expansion deferred until orchestration-layer relocation. Default False -- stitcher canary opted out so behavior at startup is identical. |
| Post-processing canary (Stitcher) | **shipped (LVP `c83e3ec` cherry-pick onto `4.0.0-beta`)** | `4.0.0-beta` | Stitcher canary as worked example for the intern tutorial. Cherry-picked onto beta 2026-05-17 evening via `4.0.0-plugin-consolidated`. Canary stays at `auto_run_on_protocol_complete=False` so it remains a registration validator, not a workhorse that would stitch every protocol unexpectedly. |
| Plugin tutorial doc | **shipped (LVP `c2dd87f` cherry-pick onto `4.0.0-beta`)** | `4.0.0-beta` | `docs/PluginTutorial.md` (413 lines) -- hello-world post-processor tutorial. Cherry-picked onto beta 2026-05-17 evening. Docs update for the new `auto_run_on_protocol_complete` field deferred to Eric (2026-05-18+). |
| Wave 7 Phase 1 -- sub-API namespace setup (delegating facades) | **accepted (LVP `009f1fe`)** | merged to `4.0.0-beta` | Subagent draft accepted 2026-05-14. Sub-APIs ship as delegating facades; method bodies stay on `_lumascope.py` and relocate during Phase 2-7 caller migrations. Eric green-lit the inversion to lock the namespace + freeze the API shape now without taking the 6000-line body-relocation risk. **Namespace LOCKED for 4.x**: `scope.motion / .illumination / .imaging / .diagnostics / .capabilities / .io`. |
| Wave 7 Phase 2a -- motion migration inventory + plan doc | **done (Firmware `e0e3e61`)** | `3.0-firmware` | `docs/WAVE7_PHASE_2_PLAN.md` -- 40-method classification (22 stateless / 18 stateful), 152 caller sites, 5-commit migration sequence (2b-2f), risk/rollback. |
| Wave 7 Phase 2b -- relocate stateless motion methods | **done (LVP `7c41cd4`)** | merged to `4.0.0-beta` | 22 driver-delegating methods relocated from `_lumascope.py` to `motion.py` 2026-05-14 evening. Three deviations documented in the commit body: `safe_turret_mover` renamed to `safe_turret_move`; `get_target_pos` rename deferred (later retired in LVP `689daad` as dead code); `_driver` made a @property re-resolving `self._scope._motion_driver` (survives `disconnect()` driver swap). |
| Wave 7 Phase 2c -- stateful body + state owner relocation | **done (LVP `b21ef5d`)** | merged to `4.0.0-beta` | 18 method bodies + 11 state slots + 2 events (`_homing_event` / `_turreting_event`) + 3 private helpers moved to MotionAPI 2026-05-14 late evening. Transient @property shim on Lumascope for state slots + production-code band-aids -- both retired in 2c.5/2f. |
| Wave 7 Phase 2c.5 -- test caller migration | **done (LVP `7e859db`)** | merged to `4.0.0-beta` | 148 method-call sites + 30 attribute mutations across 9 test files migrated from `scope.<method>` -> `scope.motion.<method>`. Plan-revised 2026-05-14 to put tests BEFORE production callers per `feedback_test_first_migration_order`. |
| Wave 7 Phase 2d -- production caller migration | **done (LVP `7e859db` + post-Wave-7 audit save)** | merged to `4.0.0-beta` | Initial 2d shipped with Wave 7 merge; post-merge audit caught 23 incomplete sites + 7 broken tests (ship-blocker) -- all fixed in `7e859db`. AST guard `test_no_motion_method_calls_on_bare_scope_in_production` added (MOTION_ONLY_METHODS derived from `dir(MotionAPI) - dir(Lumascope)`). |
| Wave 7 Phase 2e/2f -- retire Lumascope forwarders + @property shim | **done (LVP `7e859db`)** | merged to `4.0.0-beta` | Wave 7 Phase 2 fully closed per `docs/RULE_35A_BASELINES.md`. `_lumascope.py` line/method count: 5147 LOC / 182 methods (-373 LOC / -66 methods vs 2026-05-14 baseline). |
| Wave 7 Phase 3 -- illumination migration | pending | -- | ~29 forwarder methods on `illumination.py` await bodies; pairs with the STATE-LED-1 LED-state-machine consolidation embedded in the relocation per design doc §3. Pre-work: read `WAVE7_PHASE_2_PLAN.md` retrospective + design doc Phase 3 section, then spawn an illumination-inventory subagent + write `WAVE7_PHASE_3_PLAN.md` before relocating any bodies (test-first per `feedback_test_first_migration_order`). |
| Wave 7 Phase 4 -- imaging migration + `live_processing` infra | pending | -- | ~72 forwarder methods on `imaging.py` (larger than Phase 2) plus the `live_processing` per-frame listener infrastructure lands here. Design doc estimate revised upward to 6-8 days. |
| Wave 7 Phases 5-7 | pending | -- | Diagnostics body relocation + cleanup-composition-root + retire forwarders. |
| Phase C -- intern's first post-processing plugin | active starting 2026-05-18 | -- | Intern arrives 2026-05-18 morning. Day-one workflow: standalone Python (`__main__.py` in her plugin) or REPL for analysis logic; LVP integration via `auto_run_on_protocol_complete=True` once her processor is solid. The "her plugin appears alongside the 5 built-ins in the post-processing accordion list" experience is D9 work, scheduled for post-Wave-7. |
| D9 -- retire 5 built-in post-processors into `ctx.plugins.post_processing` | post-Wave-7 (4.1.5 per roadmap) | -- | Five built-ins (`composite_generation`, `zprojector`, `stitcher`, `stack_builder`, `video_builder`) migrate to the plugin namespace; UI accordion becomes registration-driven; intern's plugin slots in alongside built-ins. Sequenced AFTER Phase 3-7 because the built-ins use `scope.motion / scope.imaging` APIs that are mid-migration -- doing D9 first means rewriting against an in-flight surface. |

---

**Design status**: Locks the API namespace and plugin platform shape
for LVP 4.x. Decisions confirmed by Eric 2026-05-09. Phase A landed
2026-05-14; remaining phases per the Status table above.

**Inputs**:
- `docs/META_AUDIT_2026-05-09.md` (synthesis, Wave 7 pulled forward + Q6 plugin platform)
- `docs/META_AUDIT_RULES_AND_STRUCTURE_2026-05-09.md` (Track A; lumascope_api decomposition)
- `docs/META_AUDIT_OBSERVABILITY_UX_EXTENSIONS_2026-05-09.md` (Track D; one-platform plugin recommendation)
- `docs/PLUGIN_API_RESEARCH_2026-05-09.md` (research note; cluster table, options A-D, 13 open questions)

**Repo state pinned**:
- Firmware `3.0-firmware @ 62c16d9`
- LVP `layer-audit-with-instrumentation @ 8d96eae`

ASCII only. No code edits made by this doc; this is the spec the rule-change
commit and the migration phases reference.

---

## 1. What this doc locks

1. **Sub-API namespace shape**: Option A (pure-hardware split). `Lumascope`
   becomes a thin composition root holding **six** public sub-APIs:
   `scope.motion`, `scope.illumination`, `scope.imaging`, `scope.diagnostics`,
   `scope.capabilities`, **`scope.io`**.

   **Pass-3 amendment (2026-05-11) -- six sub-APIs (per Eric locked decision D2; resolves pass-3 future-features F-C-2)**: `scope.io` is reserved for USB-to-IO trigger device support (roadmap feature F9). Ships as an empty placeholder in Wave 7 Phase 1; methods land in a later wave when the third-board trigger driver is built. Reserving the name pre-freeze keeps the platform-change cost zero. EL-0940's existing TRIGI/TRIGO on the LED controller surfaces here too when LVP-side trigger work begins.
2. **Driver-handle naming**: Option 2a. Driver instances become private
   (`scope._motion_driver`, `scope._led_driver`, `scope._camera_driver`);
   sub-APIs take the public names. Tests using driver-direct access either
   migrate to sub-API calls (Rule 22) or keep private-handle access with a
   comment justifying it.
3. **Plugin platform**: one platform with namespace-scoped registries:
   `ctx.plugins.{ui, post_processing, live_processing, rest}`. Hardcoded
   namespace set for 4.x; future categories require a deliberate platform
   change. **Pass-3 amendment (2026-05-11) -- live_processing infrastructure ships during Wave 7 Phase 4 (per Eric locked decision D1; resolves pass-2 C-8 + pass-3 F-C-1)**: the `ctx.plugins.live_processing` namespace name is LOCKED in the 4.x set but ships EMPTY until Phase 4. Per-frame listener registry + driver-side fire sites (Pylon / IDS / FX2 / Sim) + thread contract + drop policy land paired with the ImagingAPI relocation in Phase 4. Wave 7 Phase 4 cost includes this listener infrastructure (+3-4 days).
4. **Sub-API and plugin namespaces are orthogonal by design.** Plugins
   extend the layer they live in (UI tab, post-processing pipeline, REST
   endpoint), not the hardware they touch.
5. **REST URL tree mirrors the sub-API tree**:
   `/motion/...`, `/illumination/...`, `/imaging/...`, `/diagnostics/...`,
   `/capabilities`, `/io/...`, `/plugins/<name>/...`. **Pass-2 C-6 DEFERRED**: REST URL convention conflicts with `REST_API_PLAN.md` flat `/api/v1/{move,led,camera,focus}` and is deferred to a dedicated REST design session. §5 is design-only-not-locked until that session ships.
6. **Backcompat freedom window (pass-1 amendment)**: Rule 30 (L2 API stability) edit ships with this design + the rule-change commit. Pre-freeze, structural changes are FREE. Freeze triggers when BOTH: (a) a tagged release published to PyPI / a public binary distribution channel, AND (b) at least one external L2 consumer named on `docs/L2_CONSUMERS.md` (created with the rule-change commit; initially empty). Until both conditions hold, this migration runs without forwarders-for-public-API constraint. (Intra-phase forwarders per §6 are still required for code-review reasons, not L2 compat reasons.)
7. **Pass-3 amendment (2026-05-11) -- Session layer promotion (per Eric locked decision D4; resolves pass-2 C-7 + pass-3 layer-audit §2.1 Option 2)**: `ScopeSession` (`modules/scope_session.py`, 355 lines, 21 methods today) is promoted to its own **Session layer** in the canonical layer chain. Rule 1 edits to: `GUI / Engineering plugin / REST -> Session -> Lumascope API -> Modules -> Drivers -> Hardware`. ScopeSession is the **session-composition-root** (composes Lumascope + settings + executors + protocol runner); Lumascope remains the **hardware-composition-root**. See §2.6 + §2.10 for placement. The rule-edit text itself ships in the rule-change commit; this design records the architectural decision.

---

## 2. Sub-API specification

Each sub-API is a class in `modules/lumascope_api/<name>.py`. Constructor
takes the relevant private driver handle(s) plus a back-reference to the
composition root for cross-sub-API queries (capability reads, executor
access).

### 2.1 `scope.motion`

**Module**: `modules/lumascope_api/motion.py`
**Class**: `MotionAPI`
**Owns**: `_pos_cache`, `_axis_state`, `_arrival_events`, `_move_profile`,
`_motion_wake`, `_motion_monitor_thread`, position-listener registry +
lock.

**Public methods** (~40 -- moves from `Lumascope` clusters 9, 10, 18, 19):
- Movement: `move_absolute_position`, `move_relative_position`,
  `move_absolute_async`, `move_relative_async`, `move_home_async`
- Homing: `home`, `zhome`, `thome`, `xycenter`, `has_homed`, `has_thomed`
- State: `get_axis_state`, `get_current_position`, `get_target_position`,
  `get_actual_position`, `is_moving`, `is_any_axis_moving`,
  `is_homing`, `is_turreting`, `wait_until_finished_moving`
- Listeners: `add_position_listener`, `remove_position_listener`,
  `_fire_position_listeners`
- Configuration: `get_axes_config`, `get_axis_limits`,
  `set_motor_precision_mode`, `set_acceleration_limit`,
  `refresh_position_cache`, `_predicted_position`
- Limits / status: `get_home_status`, `get_target_status`,
  `get_reference_status`, `get_limit_switch_status`,
  `get_limit_switch_status_all_axes`, `get_overshoot`
- Turret: `safe_turret_mover`, `tmove`, `has_turret`,
  `is_current_turret_position_objective_set`,
  `get_turret_position_for_objective_id`
- Motion monitor lifecycle: `_motion_monitor_loop`,
  `_stop_motion_monitor` (private, called by composition root)

**Driver handle**: `_motion_driver` (was `scope.motion`, class `MotorBoard`)

**Cross-sub-API calls**: reads `scope.capabilities.axes`,
`scope.capabilities.travel_limit_um`. None outbound to other sub-APIs.

### 2.2 `scope.illumination`

**Module**: `modules/lumascope_api/illumination.py`
**Class**: `IlluminationAPI`
**Owns**: `_led_state` (single source of truth per Wave 2 STATE-LED-1),
`_led_owners`, `_led_owner_lock`, LED-listener registry + lock.

**Public methods** (~30 -- moves from `Lumascope` clusters 7, 11, 12):
- **Canonical channel-spec surface (pass-3 amendment)**: `set_channel(channel_spec, intensity)`, `clear_channel(channel_spec)`, `get_illuminator_state(illuminator_id)`, `illuminator_states`
- Backcompat forwarders (for current LED illuminator): `led_on(channel, mA)`, `led_off(channel)`, `leds_off()`
- Sync control: `led_on`, `led_off`, `leds_off`, `led_on_fast`,
  `led_off_fast`, `leds_off_fast`
- Async control: `led_on_async`, `led_off_async`, `leds_off_async`,
  `led_on_sync`, `leds_off_sync`
- State: `get_led_ma`, `led_enabled`, `led_illumination`, `led_states`,
  `get_led_state`, `get_led_states`, `get_led_status`
- Save/restore: `save_led_state`, `restore_led_state`, `leds_off_owned`,
  `leds_enable`, `leds_disable`
- Wait: `wait_until_led_on`
- Channel mapping: `ch2color`, `color2ch`
- Listeners: `add_led_listener`, `remove_led_listener`,
  `_fire_led_listeners`

**Driver handle**: `_led_driver` (was `scope.led`, class `LEDBoard` /
`FX2Driver`)

**Cross-sub-API calls**: reads `scope.capabilities.illuminators`,
`scope.capabilities.led_colors`, `scope.capabilities.led_max_ma`.

**Q2 note**: per Eric 2026-05-09, LED state is API-primary (default
SoT). The driver's `led_ma` mirror is retired; `_led_state` in
`IlluminationAPI` is authoritative. Driver becomes a thin translator. This
collapse happens during Phase 3 of the migration below, so the sub-API
split and the Rule 2 cleanup land in one motion.

**Pass-3 amendment (2026-05-11) -- channel-spec widening (per Eric locked decision D3; resolves pass-3 F-C-3)**: the canonical surface widens to `set_channel(channel_spec, intensity)` where `channel_spec` is a typed dict carrying kind + identifier. Two forms supported initially: `{kind: 'led', color: <name>}` (today's LED channels) and `{kind: 'pattern', name: <name>}` (future Skylight pattern-illuminators per roadmap feature F1 path b/c). `led_on(channel, mA)` / `led_off(channel)` / `leds_off()` survive as Rule 30 backcompat forwarders for the current LED illuminator -- they call `set_channel({'kind':'led','color':color}, mA)` and `clear_channel(...)` under the hood. `ScopeCapabilities` correspondingly adds `illuminators: tuple[IlluminatorSpec, ...]` alongside today's `led_colors` / `led_channels` / `led_max_ma` (those become derived properties read from the first LED illuminator for backcompat). The rename + widening land during Wave 7 Phase 3 alongside the STATE-LED-1 collapse; pass-3 future-features audit estimates +3-5 days on top of the existing Phase 3 cost.

### 2.3 `scope.imaging`

**Module**: `modules/lumascope_api/imaging.py`
**Class**: `ImagingAPI`
**Owns**: `_camera_cache`, `_camera_cache_lock`, `_frame_buffer`,
camera-listener registry + lock, `frame_validity` instance, `_scale_bar`
config.

**Public methods** (~50 -- moves from `Lumascope` clusters 4, 8, 13, 14,
15, 16, 19, 20):
- Setters: `set_gain`, `set_exposure_time`, `set_auto_gain`,
  `set_auto_exposure_time`, `set_frame_size`, `set_binning_size`,
  `set_pixel_format`, `set_acquisition_stop_mode`,
  `set_bandwidth_reserve_mode`, `set_device_link_throughput_limit`,
  `set_max_transfer_size`, `set_num_max_queued_urbs`,
  `set_gev_packet_size`, `set_gev_inter_packet_delay`,
  `set_gain_sync`, `set_exposure_sync`
- Getters: `get_gain`, `get_exposure_time`, `get_frame_size`,
  `get_pixel_format`, `get_max_width`, `get_max_height`, `get_width`,
  `get_height`, `get_binning_size`, `get_supported_pixel_formats`,
  `get_available_binning_sizes`
- Capture: `capture`, `capture_complete`, `capture_blocking`,
  `capture_and_wait`, `capture_and_wait_sync`, `get_image`,
  `get_image_with_chunks_from_buffer`, `get_image_from_buffer`,
  `_get_latest_chunks`
- State + lifecycle: `camera_active`, `camera_is_connected`,
  `camera_gain`, `camera_exposure_ms`, `camera_frame_size`,
  `camera_max_frame_size`, `camera_min_frame_size`, `camera_max_exposure`,
  `camera_max_gain`, `camera_pixel_format`, `_load_camera_timing`,
  `_populate_camera_cache`, `_invalidate_camera_cache`
- Save/restore: `save_camera_state`, `restore_camera_state`
- Camera-config orchestration: `apply_layer_camera_settings`,
  `update_auto_gain_target_brightness`, `auto_gain_once`,
  `update_camera_config`, `suppress_value_warnings`
- Operation flags: `is_capturing`, `is_focusing`, `capture_return`,
  `autofocus_return` (via `_capturing_event`, `_focusing_event`)
- Frame validity: `frame_is_valid`, `frames_until_valid`, `count_frame`
- Scale bar: `scale_bar_config`, `scale_bar_enabled`, `set_scale_bar`
- Camera diagnostics: `get_camera_temps`, `log_camera_temps`,
  `start_camera_temp_logging`, `stop_camera_temp_logging`
- Frame-flow listeners: `add_frame_listener`, `remove_frame_listener`,
  `_fire_camera_listeners`
**Driver handle**: `_camera_driver` (was `scope.camera`, class
`PylonCamera` / `IDSCamera` / `FX2Camera` / `SimulatedCamera`)

**Cross-sub-API calls**: reads `scope.capabilities.camera_model`. Calls
`scope.illumination.leds_off()` at capture-time when configured (capture
contract: lights off before grab, restore after).

**Pass-1 amendment (Q12 reversed)**: `compute_focus_score` does NOT live in
ImagingAPI. It's a pure function on a frame array (no camera state needed),
parallel to image-save helpers. Moves to `modules/focus.py` as a free
function instead. See §2.7.

### 2.4 `scope.diagnostics`

**Module**: `modules/lumascope_api/diagnostics.py`
**Class**: `DiagnosticsAPI`
**Owns**: no persistent state (per-call probes).

**Public methods** (~10 -- moves from `Lumascope` cluster 22):
- Camera probes: `get_camera_temperatures`, `get_camera_diagnostic_info`,
  `run_camera_bandwidth_test`, `run_grab_lifecycle_benchmark`,
  `run_pylon_diagnostic_probe`
- Serial probes: `send_diagnostic_command`,
  `send_diagnostic_command_multiline`
- Helpers: `_human_os_version`, `_safe_pylon_versions`,
  `_dltl_filename_token`, `_diagnostic_target_board`

**Driver handle**: none directly; reads `scope._motion_driver`,
`scope._led_driver`, `scope._camera_driver` for per-driver probes via the
composition root.

**Cross-sub-API calls**: many (it's the cross-cutting probe surface).

### 2.5 `scope.capabilities`

**Module**: `modules/lumascope_api/capabilities.py`
**Class**: `Capabilities` (already exists; expand per Track C R3)

**Pass-1 amendment (immutable vs runtime split)**: the original proposal
mixed frozen scope identity with runtime-mutable values. Split into two
surfaces:

**Immutable identity** (`scope.capabilities`, frozen at connect, never
mutated without explicit reconnect):
- `axes: tuple[str, ...]`
- `axis_travel_limits_um: dict[str, float]`
- `illuminators: tuple[IlluminatorSpec, ...]` **(pass-3 amendment per D3 + future-features F-C-3 / F-I-2)**: each spec carries `kind` (`'led'` | `'pattern_panel'` | `'screen'`), `channels_or_patterns: tuple[str, ...]`, `unit` (`'mA'` | `'lumens'` | `'intensity_0_to_1'`), `max_intensity: float`. Future-friendly N-of-each shape.
- `led_colors: tuple[str, ...]` (derived property: reads colors from the first LED illuminator; preserved for Rule 30 backcompat)
- `led_channels: tuple[int, ...]` (derived property)
- `led_max_ma: float` (derived property)
- `layers: tuple[str, ...]` (derived from led_colors + Lumi)
- `transmitted_layers: tuple[str, ...]`
- `fluorescence_layers: tuple[str, ...]`
- `luminescence_layers: tuple[str, ...]`
- `camera_model: str`
- `camera_max_frame_size: tuple[int, int]`
- `hardware_features: dict[str, frozenset[str]]` **(pass-3 amendment per Eric locked decision D5; resolves pass-3 F-I-4)**: per-subsystem hardware-gated capability set (e.g. `{'trigger_board', 'led_board_trig_io', 'argolight_slide', 'temperature_sensor'}`). Independent of `firmware_features`: this surface captures USB-to-IO trigger device presence and other non-firmware-gated hardware capabilities. Ships with empty default; callers treat empty as "feature unknown" not "feature absent". Resets on reconnect.

**Runtime state** (`scope.runtime_state`, mutable, refreshed on driver
events):
- `firmware_versions: dict[str, str]` (`{'motor': '3.0.9', 'led': '3.0.7'}`) -- changes when boards are reflashed
- `firmware_features: dict[str, frozenset[str]]` (per subsystem; populated once FW4.0 ships; reflects current firmware, not declared support). **Pass-2 C-10 + pass-3 verification**: ships with empty default. Callers treat empty as "feature unknown" not "feature absent". The same empty-set semantic applies in Rule 8 capability-probe corollary text. FW4.0 promotion populates this surface; until then field firmware lacks `INFO.features` and the dict stays empty.
- (future: `connection_status`, `usb_link_state`, etc. as they accrete)

**Capability invalidation policy (pass-2 A-10 integrated, 2026-05-11)**: capabilities are immutable per Lumascope instance. Reconnect = new Lumascope = new capabilities. REST clients re-poll `/capabilities` on connection events. `add_capability_listener` is NOT provided -- capability changes are scope-lifecycle changes, not runtime mutations.

Sub-APIs read from BOTH surfaces as needed. `scope.runtime_state` is
refreshed at driver connect + at any post-FWUPDATE reconnect. Callers
that need a consistent snapshot use `scope.runtime_state.snapshot()`
which returns a frozen copy.

**Q11 amended**: kept unified-per-surface (one immutable surface +
one runtime surface), but the two surfaces are split so the immutable
contract is honest. Sub-APIs do NOT each own a capabilities surface.
Capability-probe gates use the immutable surface; recovery / version
gates use the runtime surface.

### 2.6 `Lumascope` composition root (what survives)

After the split, `Lumascope` is a thin facade with ~30 methods:

- Construction: `__init__` (constructs sub-APIs, wires them together,
  registers atexit emergency shutdown)
- Composition: holds `motion`, `illumination`, `imaging`, `diagnostics`,
  `capabilities`, **`io`** as public attributes (six sub-APIs per pass-3 D2)
- Lifecycle: `disconnect`, `_emergency_shutdown`, `acquire_exclusive`
- Hardware presence: `motor_connected`, `led_connected`,
  `_no_hardware`, `are_all_connected`, `_notify_partial_hardware`
- Executor wiring: `register_executors`, `register_executor_bundle`,
  `register_source_path`, `_require_executor` (until Wave 3 retires this
  pattern)
- Top-level info getters: `get_microscope_model`, `get_motor_info`,
  `get_led_info`, `get_camera_info`, `get_system_info`,
  `get_camera_profile_info` (these read from the relevant sub-APIs +
  capabilities; pure facades)
- Diagnostic factory: `create_diagnostic` (classmethod)
- Stop: `stop_motion` (delegates to `scope.motion.stop()`; kept on
  composition root for compat with the existing top-level emergency-stop
  call sites)

**Pass-3 amendment (2026-05-11) -- atexit pattern shape (resolves pass-2 M-4 per DJ2)**: the atexit ownership shape is pinned as **instance-removes-self-on-destruct**. Each Lumascope instance registers its own atexit hook at `__init__` (via `atexit.register(self._emergency_shutdown)`) and removes it at `disconnect` / `_emergency_shutdown` (via `atexit.unregister(self._emergency_shutdown)`). The lifecycle-inventory registry (new Rule 41) tracks both registrations and removals. This pairs cleanly with Rule 41's mechanical pre-commit gate that blocks new `atexit.register(...)` sites without a same-commit edit to the lifecycle-inventory doc.

**Pass-2 M-7 + pass-3 layer-audit §2.5 -- two top-level facades by design**: the rename question from pass-1 (`Lumascope` -> `Microscope` / `Scope`) stays answered as KEEP `Lumascope`. Pass-3 layer-audit's §2.1 Option 2 (Session as own layer per D4) makes the two-facade structure explicit: **`Lumascope` is the hardware-composition-root (what you compose); `ScopeSession` is the session-composition-root (what you call)**. L4/L3 mental-model continuity holds for both names; L2 callers target Session per §6.6 LumascopeSkills.md TOC restructure.

### 2.7 What leaves `Lumascope` entirely

**Image save / path / metadata** (cluster 17 + static duplicates).
**Q6 decision**: move out of the API entirely. Goes to `modules/image_save.py`
as plain functions. Static-method duplicates at the `lumascope_api.py`
file bottom retire (Rule 35 -- no parallel implementations). Callers
import directly:

```
from modules.image_save import (
    save_image,
    save_live_image,
    generate_image_save_path,
    generate_image_metadata,
    get_next_save_path,
    get_well_label,
    prepare_image_for_saving,
)
```

This retires both the instance methods AND the static duplicates in one
move.

**`compute_focus_score`** (pass-1 amendment to Q12). Pure function on a
frame array; no camera state needed. Same pattern as image-save helpers:
moves out of the API entirely to `modules/focus.py` as a free function.

```
from modules.focus import compute_focus_score, ...
```

Sets a cleaner precedent than the original Q12 resolution: **frame-analysis
functions are pure -> they live in `modules/`, not on sub-APIs.** Future
analysis functions (e.g. background subtraction, ROI metrics) follow the
same pattern.

**Pass-3 amendment (2026-05-11) -- Rule 33 decision-record comment (resolves pass-2 M-6)**: the Rule 33 decision comment lives at the new `modules/focus.py` module preamble, recording: "Considered `ImagingAPI.compute_focus_score` (frame analysis on a sub-API); rejected because the function is pure on a frame array, has no camera state, and analysis helpers are not API-layer concerns. Future frame-analysis functions (background subtraction, ROI metrics, PREFACE-style two-shot Z math per roadmap feature F2) follow the same pattern: pure helpers in `modules/`, not sub-API methods."

**`composite_capture` orchestration** (pass-1 amendment to §1 / M-1).
Without `scope.tasks` (Option A rejected per Eric 2026-05-09), composite
capture orchestration moves from `ui/composite_capture.py` to
`modules/composite_capture.py` as a free function that takes `scope` as an
argument. NOT a method on `Lumascope` or any sub-API. UI imports
`from modules.composite_capture import composite_capture` and calls
`composite_capture(scope, ...)`. Same pattern for protocol orchestration
(in `modules/protocol_runner.py` or equivalent). Multi-API workflow code
lives in `modules/`, calls into sub-APIs.

### 2.8 What stays in `Lumascope` for now (deferred)

**Operation flags** (`is_homing`, `is_capturing`, `is_focusing`,
`is_turreting`). **Q5 decision**: move to the relevant sub-API
(`scope.motion.is_homing`, `scope.imaging.is_capturing`, etc.). Deferred
to Phase 6 of the migration so we don't break listener-firing call sites
mid-split. During Phases 1-5 the flags stay on `Lumascope` as one-line
forwarders to the sub-API state; Phase 6 retires the forwarders.

**Objective / labware / turret-config / stage-offset** (cluster 21).
**Decision**: stays on `Lumascope` for now; this is microscope
configuration, not live hardware. Moves later (post-Phase 6) to
`modules/scope_session.py` as a separate refactor; not blocking.

### 2.9 Cross-process state and sub-API hosting environments (pass-2 A-5 integrated)

**Pass-3 amendment (2026-05-11) -- sub-API hosting environments (resolves pass-2 A-5)**: Kivy (GUI) + FastAPI (REST headless) + REST-server-co-resident-with-GUI all share the Lumascope instance. The model: **one Lumascope per process; one ScopeSession per client; locks process-scoped; no in-process state mirrors a remote service.** Sub-API listeners fire sync-from-calling-thread per existing Rule 33 decision. The bridge layer that schedules to UI is per-environment, not per-sub-API:

- GUI host (LVP.exe): `UIListenerBridge` (Kivy `Clock.schedule_once`)
- REST host (FastAPI): TBD per dedicated REST design session -- WebSocket subscriptions or polling; the bridge layer mirrors UIListenerBridge's shape but uses an asyncio-aware dispatcher
- REST-co-resident-with-GUI: both bridges instantiated; both consume the same Lumascope; both deliver via their respective schedulers

Sub-API listeners themselves do NOT know which host they're running under -- they fire from the calling thread (Pylon callback / IDS grab / motion-monitor / etc.); the bridge layer is what knows about Kivy Clock vs asyncio. This keeps sub-API code hosting-agnostic per Rule 15.

Implications for plugins (per §4.2 namespaces): a `ctx.plugins.ui` plugin assumes Kivy; a `ctx.plugins.rest` plugin assumes the REST app; a `ctx.plugins.live_processing` plugin's `frame_handler` may execute on the driver thread regardless of host, so handlers must be hosting-agnostic per Rule 15.

### 2.10 Session layer (ScopeSession) -- pass-3 amendment

**Pass-3 amendment (2026-05-11) -- Session as own layer (resolves pass-2 C-7 + pass-3 layer-audit §2.1 Option 2 per Eric locked decision D4)**: `ScopeSession` (`modules/scope_session.py`, 355 lines, 21 methods today) is promoted to a real layer between GUI/REST and the Lumascope API.

**Layer chain edit** (lands in the rule-change commit per CLAUDE.md Rule 1):

```
GUI / Engineering plugin / REST  ->  Session  ->  Lumascope API  ->
  Modules  ->  Drivers  ->  Hardware
```

**ScopeSession owns**:
- Session-scoped facade methods (`led_on`, `led_off`, `leds_off`, `led_on_sync`, `move_absolute`, `move_relative`, `move_home`, etc.) -- thin forwarders over the Lumascope sub-APIs
- L2 lifecycle (one Session per process for GUI, per REST client for REST, per test fixture for pytest)
- L2 entry points that GUI / REST / SDK callers all target
- Settings / executor / protocol-runner composition (`create_protocol_runner`, `start_executors`, `shutdown_executors`, `start_application_session`)
- Hosting-environment-specific scheduler injection: GUI Session takes `KivyClockScheduler`; REST Session takes `ThreadingTimerScheduler`; CLI Session takes synchronous scheduler; test Session takes per-test scheduler. Pairs with §2.9.
- **Focus history surface (pass-3 amendment per D12; F-I-1)**: `session.focus.get_well_focus(well, objective)` / `set_well_focus(well, objective, z)` / `narrow_search_range(well, objective)` -- backed by `data/focus_history.json` (per-scope persistent state). This is intentionally on Session, not Lumascope, because it's workflow state that spans protocol runs and lives at the session/workflow level, not the hardware level.

**Why Session, not Lumascope, owns focus history (D12)**: per-well focus is session-scoped state (which plate, which objective, current bench's focus history). Lumascope is hardware-scoped (which boards are connected, which axes can move, what state is the LED in right now). Coupling persistent workflow state to the hardware composition root would re-introduce the Lumascope-as-everything pattern Wave 7 is decomposing.

**L2 entry-point clarification** (pass-3 layer-audit §2.6): L2 callers (Matlab / micromanager / SDK script / REST client) target `ScopeSession`. The sub-API surface is L3/L4 visible (lives in source; appears in tests) but L2 consumers see Session methods. After Wave 7, LumascopeSkills.md restructures to lead with Session as the L2 entry point (see §6.6).

**ScopeSession survives Wave 7 intact**. The dedicated REST design session (deferred per pass-2 C-6/C-7) decides REST URL convention and whether REST-side adapters change shape; ScopeSession's existence at the Session layer is NOT deferred. The decision that's deferred is the URL routing on top of Session, not Session itself.

---

## 3. Resolution of the 13 open questions

| # | Question | Resolution |
|---|---|---|
| 1 | Naming collision (sub-API vs driver attr) | Driver handles private (`_motion_driver`, `_led_driver`, `_camera_driver`); sub-APIs take public names. Tests using driver-direct access migrate to sub-API per Rule 22, OR keep private-handle access with `# Driver-internal test: <reason>` comment. |
| 2 | Lumascope as composition root? | Yes, ~30-method facade. See §2.6. |
| 3 | Where do listeners live? | Per sub-API. `scope.motion.add_position_listener()`, `scope.illumination.add_led_listener()`, `scope.imaging.add_frame_listener()`. Sync-fire-from-calling-thread contract preserved (existing decision per Rule 33). |
| 4 | Where does lifecycle live? | Composition root constructs/disconnects sub-APIs in dependency order. Each sub-API exposes `_init(driver_handles, scope_root)` and `_disconnect()` private methods, called only by composition root. Mirrors new Rule 41 (lifecycle ownership explicit). |
| 5 | Where do operation flags live? | Per sub-API by ownership: `motion.is_homing`, `motion.is_turreting`, `motion.is_moving`; `imaging.is_capturing`, `imaging.is_focusing`. Deferred to Phase 6 of migration. |
| 6 | Image save helpers | Move out of Lumascope entirely to `modules/image_save.py` as plain functions. Static duplicates retire. See §2.7. |
| 7 | Plugin types hardcoded vs extensible? | Hardcoded for 4.x. Four namespaces (ui, post_processing, live_processing, rest) cover known extension axes. `live_processing` name locked but ships empty until Wave 7 Phase 4 per pass-3 D1. Adding a new category is a deliberate platform change. Rule 33 decision-record comment in `ctx/plugins/__init__.py`. |
| 8 | Plugin lifecycle hooks | `register(ctx)`, `unregister(ctx)`, `on_settings_changed(ctx, settings)`, optional `on_event(event_type, payload)`. See §4.3. |
| 9 | Plugin capability declaration | Yes, plugins declare `capabilities: tuple[str, ...]` in their `PluginSpec`. Used (not informational-only) by tech-support report + diagnostic probe per pass-1 M-3 / pass-3 DJ3. See §4.4. |
| 10 | REST URL mapping | **DEFERRED 2026-05-11** (pass-2 C-6) to dedicated REST design session. See §5 + §10 ScopeSession disposition row. |
| 11 | ScopeCapabilities fragmentation | **Pass-1 amended**: split into TWO unified surfaces. `scope.capabilities` = immutable scope identity (axes, led_colors, camera_model, etc.). `scope.runtime_state` = runtime-mutable values (firmware_versions, firmware_features). Both unified-per-surface; sub-APIs read from both, do not each own their own. See §2.5. |
| 12 | `compute_focus_score` placement | **Pass-1 amended**: moves to `modules/focus.py` as a free function (pure function on a frame array, same pattern as image-save helpers in Q6). Sets precedent: frame-analysis functions are pure -> live in `modules/`, not on sub-APIs. See §2.7. |
| 13 | Static helper duplicates | Retire in same pass as Q6 (image save move). Rule 35 -- no parallel implementations once the canonical path exists. |

---

## 4. Plugin platform specification

### 4.1 Goals

Single platform, namespace-scoped registries, lifecycle-aware. Reference
shape: `etaluma_engineering` (today's only plugin). Generalizes the
implicit contract at `etaluma-engineering/etaluma_engineering/__init__.py`
+ host hook at `lumaviewpro.py:716-741`.

### 4.2 Namespaces (hardcoded for 4.x)

| Namespace | Purpose | Today's example |
|---|---|---|
| `ctx.plugins.ui` | UI-extending plugins (Kivy widgets, accordion tabs, settings panels, toolbar buttons) | `etaluma_engineering` adds Engineering accordion tab |
| `ctx.plugins.post_processing` | Operate on saved files (offline batch processing). Inputs: per-step output directories, manifest. Outputs: new artifacts in the same or sibling directory. | (future intern project; pass-3 amendment per D9: today's five built-in `ProtocolPostProcessingExecutor` subclasses -- `composite_generation`, `zprojector`, `stitcher`, `stack_builder`, `video_builder` -- retire into this namespace during 4.1.5) |
| `ctx.plugins.live_processing` | Subscribe to live frames via `scope.imaging.add_frame_listener`; runs in capture thread (must be fast). **Pass-3 amendment per D1**: namespace name LOCKED in 4.x set; ships EMPTY until Wave 7 Phase 4 builds the per-frame listener registry + driver-side fire sites (Pylon / IDS / FX2 / Sim) + thread contract + drop policy. Roadmap feature F6 (live image processing -- summation, illumination flattening) is the first real consumer. | (future, post-Phase-4) |
| `ctx.plugins.rest` | Register REST endpoints under `/plugins/<plugin-name>/...`. Pairs with `REST_API_PLAN.md` Phase 1. | (future) |

Adding a fifth namespace is a deliberate platform-spec change, not a
runtime extension. Per Rule 33, the decision to hold the namespace set at
four lives as a comment in `modules/plugins/__init__.py` so future
contributors see the decision before proposing a fifth.

### 4.3 Plugin entry-point contract

A plugin package's top-level `__init__.py` exposes:

```
__version__ = "X.Y.Z"

def register(ctx):
    """Register plugin with LVP. Called once at app startup."""
    ...

def unregister(ctx):
    """Tear down plugin. Called at app shutdown.
    Release subscribed listeners, save plugin state, close file handles."""
    ...

def on_settings_changed(ctx, settings):
    """Called when one of the plugin's declared subscription keys changes.
    Plugin declares subscription keys via PluginSpec.subscribes_to.
    Default is no-op if subscribes_to is empty."""
    ...
```

Optional `on_event(event_type, payload)` for plugins that subscribe to
named LVP events (e.g. `protocol_started`, `protocol_completed`,
`hardware_disconnected`).

**Pass-1 amendment (I-3 — `on_settings_changed` granularity)**: plugins
declare subscription keys in `PluginSpec.subscribes_to` (e.g.
`subscribes_to=("manual_video.max_fps", "camera.gain")`). The host calls
`on_settings_changed` only when one of those keys changes. Without
subscription keys, the hook fires on every settings change and plugins
get spammed; with subscription keys, hook firing is targeted. Empty
`subscribes_to` = hook never fires (default).

Host loading flow (replaces today's ad-hoc try/except in
`lumaviewpro.py:716-741`):

1. App startup. Widget tree + AppContext + Lumascope all initialized.
2. Discover installed plugins. Today: `import etaluma_engineering`. Future:
   walk `entry_points` group `lvp.plugins`. (Decision: 4.x ships with
   discovery via `entry_points`; the import-by-name fallback is removed.)
3. For each plugin:
   - Read `__version__`, `requires_lvp_version`, `PluginSpec`
   - Version-check against host
   - Call `register(ctx)` wrapped in try/except
   - On success: log `INFO` + record in `ctx.plugins.<namespace>` registry
   - On failure: log `ERROR` + fire `notifications.error` (per Rule 14) +
     skip plugin (do not abort app)
4. On shutdown (per new Rule 41 -- lifecycle ownership): call
   `unregister(ctx)` for each registered plugin in reverse-registration
   order. Wrapped in try/except; failure logs `WARNING`, does not block
   shutdown.

### 4.4 PluginSpec object

Every plugin declares a spec at registration time:

```
@dataclass(frozen=True)
class PluginSpec:
    name: str                         # unique within namespace
    version: str                      # plugin version, semver
    requires_lvp_version: str         # semver constraint, e.g. ">=4.1.0"
    description: str                  # one-line, shown in About dialog
    capabilities: tuple[str, ...]     # API surfaces this plugin uses
    subscribes_to: tuple[str, ...] = ()  # settings keys for on_settings_changed
    author: str                       # for support contact
    url: str = ""                     # optional homepage / repo
```

**Pass-1 amendment (M-3 — PluginSpec.capabilities commitment)**:
`capabilities` is USED, not informational-only. Two concrete uses in 4.x:
- **Tech-support report** lists each loaded plugin + its declared
  capability set so a customer issue with "plugin X breaks Y" has the
  plugin's claimed reach at hand.
- **Diagnostic probe** (`run_pylon_diagnostic_probe` and similar) attaches
  active-plugin manifest so the probe output records which plugins were
  loaded when the data was collected.

`capabilities` strings are the dotted-path names plugins use (e.g.
`"scope.imaging"`, `"scope.imaging.add_frame_listener"`,
`"modules.image_save"`). Not enforced as a sandbox in 4.x; future strict
mode (4.2+) may add runtime gates.

**Pass-1 amendment (I-3 — subscribes_to keys)**: `subscribes_to` lists
settings-tree keys (dot-path notation, e.g.
`("manual_video.max_fps", "manual_video.max_duration")`). Host fires
`on_settings_changed(ctx, settings)` only when one of those keys
changes. Empty tuple = hook never fires.

### 4.5 Per-namespace registry interface

`ctx.plugins.<namespace>.register(spec, **handler_kwargs)`:

- **`ui.register(spec, mount_point, builder)`**:
  `mount_point` is a named widget-tree location (e.g.
  `'left_sidebar.accordion'`, `'right_sidebar.image_settings.bottom'`,
  `'main_toolbar'`). `builder` is a callable returning the Kivy widget.
  Replaces today's `accordion.add_widget(tab)` reach-through with a
  named mount surface.
- **`post_processing.register(spec, processor)`**:
  `processor` is a callable `processor(input_dir, manifest, output_dir)
  -> ProcessorResult`. Lifecycle: invoked by user from a "Run plugin"
  menu OR scheduled at protocol-completion time per processor's spec.
- **`live_processing.register(spec, frame_handler)`**:
  `frame_handler` is `frame_handler(frame, metadata) -> None`. Wired
  to `scope.imaging.add_frame_listener` at registration; auto-removed at
  `unregister`. Performance contract: handler must return in <16 ms or
  next-frame is dropped (logged WARNING per new Rule 42 budget entry).
- **`rest.register(spec, router)`**:
  `router` is a sub-router (FastAPI `APIRouter` or equivalent). Mounted
  at `/plugins/<name>/`. Plugin owns its endpoint paths under that
  prefix. **Pass-3 amendment (2026-05-11) -- REST middleware shared (resolves pass-2 A-12 per Eric locked decision D11)**: plugin REST routes mount UNDER the same FastAPI app + middleware, NOT around it. The `scope.is_command_safe_for_rest` dangerous-command gate applies to plugin endpoints same as core endpoints. Plugins cannot bypass the safety gate by being mounted before / parallel to the middleware. No `PluginSpec.requires_unsafe_commands` flag in 4.x -- the platform is pure-default-safe; 4.2+ may add opt-in if needed (deferred per default-safe principle).

**Pass-3 amendment (2026-05-11) -- per-namespace health surface (resolves pass-2 A-11)**: each namespace exposes `ctx.plugins.<namespace>.health() -> NamespaceHealth` returning a dataclass with `loaded: list[PluginStatus]`, `failed: list[PluginStatus]`, `last_runtime_errors: list[RuntimeError]`. The tech-support report (per PluginSpec.capabilities use case in §4.4) calls `health()` on every namespace to record which plugins were loaded + which failed at the time the support data was collected. Diagnostic probes (`run_pylon_diagnostic_probe` etc.) likewise attach the active-plugin manifest via these `health()` calls.

**Pass-1 amendment (M-2 — REST conflict resolution)**: if two plugins
attempt to register at the same `/plugins/<name>/` prefix, the second
registration RAISES `PluginRegistrationError` and the second plugin is
not loaded (per Rule 5 — fail visible). Same shape across all four
namespaces: name collisions abort the colliding registration, log
`ERROR`, fire `notifications.error` per Rule 14, do not abort app or
unload the first plugin.

**Pass-1 amendment (lifecycle on partial-load failure)**: if
`register(ctx)` fails partway through (e.g. registered against
`ctx.plugins.ui` successfully but failed at `ctx.plugins.live_processing`),
host calls `unregister(ctx)` on the partially-loaded plugin to give it
a chance to clean up. If `unregister` also fails, log `WARNING` and
continue. The plugin is not added to the loaded-plugin list either way.

### 4.6 Mount points for `ctx.plugins.ui`

**Pass-3 amendment (2026-05-11) -- one initial mount point (resolves pass-2 I-9 + M-9 per Eric locked decision D10)**: lock ONE name initially -- `left_sidebar.accordion` -- the engineering plugin's mount. Additional mount points get added when the first plugin needs each, per Sec 4.2 "deliberate platform change" pattern. This defers four of the five names from earlier drafts.

| Mount point | Where in widget tree | Use |
|---|---|---|
| `left_sidebar.accordion` | MotionSettings accordion | Engineering tab today |

**Planned (added when first real plugin needs each)**: `right_sidebar.image_settings.bottom`, `right_sidebar.protocol_settings.bottom`, `main_toolbar`, `settings_dialog.tabs`. Each addition is a deliberate platform-spec change per §4.2; the widget-shape contract per mount point (size, layout, lifecycle) is specified at the time the mount point is added, when the first consumer's needs make the contract concrete.

Plugins refer to mount points by name; the host maps name to widget
attachment. New mount points require a platform-spec change (deliberate),
not a plugin-side ad-hoc reach. M-9 mount-point naming convention is dropped per D10 (only one mount point survives initial lock); convention picked at next-mount-point addition time per §4.2.

### 4.7 Loading sequence (concrete)

Replaces `lumaviewpro.py:716-741`:

```
# Pseudo-code. Real implementation ships in Phase A of the plugin platform
# implementation (see migration §6 below).

def _load_plugins(self, ctx):
    discovered = importlib.metadata.entry_points(group='lvp.plugins')
    for ep in discovered:
        try:
            plugin = ep.load()
            spec = getattr(plugin, 'spec', None)
            if not spec:
                logger.warning(f'[Plugins ] {ep.name}: no PluginSpec, skipping')
                continue
            if not _version_compatible(spec.requires_lvp_version, LVP_VERSION):
                logger.warning(
                    f'[Plugins ] {spec.name} v{spec.version} requires LVP'
                    f' {spec.requires_lvp_version}; have {LVP_VERSION}; skipping'
                )
                continue
            plugin.register(ctx)
            ctx.plugins._registered.append(plugin)
            logger.info(f'[Plugins ] {spec.name} v{spec.version} loaded')
        except Exception as e:
            logger.error(f'[Plugins ] {ep.name} load failed: {e}', exc_info=True)
            notifications.error(
                'Plugin load failed',
                f'{ep.name} did not load. Other features unaffected.'
            )
```

Shutdown (per Rule 41, called from `LumaViewProApp.on_stop`):

```
def _unload_plugins(self, ctx):
    for plugin in reversed(ctx.plugins._registered):
        try:
            plugin.unregister(ctx)
        except Exception as e:
            logger.warning(f'[Plugins ] {plugin.__name__} unregister failed: {e}')
```

### 4.7.1 Plugin update flow on version mismatch (pass-2 A-8 integrated)

**Pass-3 amendment (2026-05-11)**: three load-failure classes with distinct L1-readable surfaces (per Rule 28 -- title is a short noun phrase, body says what happened + how to fix):

| Class | Title | Body shape |
|---|---|---|
| Version mismatch | "Plugin update required" | "Plugin `<name>` requires LVP >= `<required>`; current is `<current>`. Update plugin or LVP to continue using it." |
| Missing dependency | "Plugin dependency missing" | "Plugin `<name>` requires `<dep>`; install with `pip install <dep>` and restart." |
| Runtime error | "Plugin load failed" | "Plugin `<name>` failed to load: `<exception summary>`. See log for trace." |

The host detects which class by examining the exception type during `plugin.register(ctx)`: explicit `PluginVersionError` raised by version-check; `ModuleNotFoundError` / `ImportError` caught as missing-dependency; everything else as runtime error. Notification per Rule 28's non-fatal-warning shape: operation continues without this plugin; OK-dismissable; system runs with what it has.

### 4.7.2 Backcompat shim for older plugins lacking PluginSpec (pass-2 A-9 integrated)

**Pass-3 amendment (2026-05-11)**: for one minor LVP release post-platform-launch, the host detects plugins lacking `PluginSpec` and synthesizes a default Spec from the plugin's `__version__` attribute plus the assumed defaults: `namespace='ui'`, `mount_point='left_sidebar.accordion'` (today's engineering-plugin mount per D10), empty `subscribes_to`, empty `capabilities`. Bench operators with un-refactored engineering-plugin installs continue to see the Engineering tab unchanged across the platform-launch release. The next release retires the shim; plugins must declare `PluginSpec` from that point forward.

---

## 5. REST surface mapping

**Status (pass-2 C-6 DEFERRED 2026-05-11)**: this section is design-only-not-locked. REST URL convention conflicts with `REST_API_PLAN.md` flat `/api/v1/{move,led,camera,focus}` (lines 156-170) and is deferred to a dedicated REST design session. The session decides which convention wins and produces a V2 plan that supersedes both this §5 and the current REST_API_PLAN URL convention. The rule-change commit + Wave 0 + Wave 1 + the §10 namespace lock ship WITHOUT REST URL specifics.

For design-only reference, the hierarchical convention proposed by this doc:

| Sub-API | REST prefix | Example endpoints |
|---|---|---|
| `scope.motion` | `/motion` | `POST /motion/move`, `POST /motion/home`, `GET /motion/position`, `GET /motion/state` |
| `scope.illumination` | `/illumination` | `POST /illumination/set`, `POST /illumination/off`, `GET /illumination/state` |
| `scope.imaging` | `/imaging` | `POST /imaging/capture`, `GET /imaging/frame`, `POST /imaging/gain`, `POST /imaging/exposure` |
| `scope.diagnostics` | `/diagnostics` | `GET /diagnostics/probe`, `POST /diagnostics/bandwidth_test` |
| `scope.capabilities` | `/capabilities` | `GET /capabilities` |
| `scope.io` (pass-3 D2) | `/io` | (TBD; trigger I/O surface ships when scope.io methods land) |
| Plugin REST | `/plugins/<name>` | per-plugin |

Convention pinned per Track C Rule 30 edit (wire-contract symmetry):
URL + request shape + response shape are L2 contracts once Phase 1 ships.
Pre-Phase-1, structural changes are free.

---

## 6. Migration path

Backcompat-unconstrained per Rule 30 edit (no L2 consumers yet). Direct
rename + move; no forwarders, no deprecation cycles ACROSS PHASES.

**Pass-1 amendment (C-3 — intra-phase consistency)**: each phase ships as
**two commits**, not one. The original "single atomic commit per phase"
approach forced a single huge diff touching every file (262 internal call
sites + 124 external = often 50+ files per phase) which is unreviewable
in practice. Two-commit pattern:

1. **Commit A (additive)**: introduce new sub-API surface; add one-line
   forwarders on `Lumascope` that delegate to the new sub-API. Old API
   still works (forwarders); new API also works. Both call paths
   exercised by existing tests; no breakage. CI green.
2. **Commit B (deletion)**: remove forwarders. Update remaining external
   callers to the new API. Old API gone. Tests still green.

This bounds intra-phase risk: if Commit A fails CI, the forwarder approach
is wrong and we abort the phase. If Commit A passes and Commit B fails
CI, the forwarder removal can be incremental (commit B-i for each
remaining caller group) until clean. Either way, the API is never in
half-migrated state on the branch.

Each phase is two commits (A + B). Each commit updates
`LumascopeSkills.md` in the same commit per Rule 34. Commit messages
specify "Phase N Commit A: add new + forwarders" / "Phase N Commit B:
retire old + forwarders".

### Phase 1 -- Composition root + sub-API skeleton (5-7 days, was 1-2 days)

**Pass-4 amendment (2026-05-14) -- Phase 1 ships as delegating facades; bodies stay**: Phase 1 (LVP `009f1fe`) shipped the sub-API namespaces (`scope.motion`, etc.) as **delegating wrappers** that route calls back to `_lumascope.py`. The original pass-3 spec said bodies move INTO sub-API classes during Phase 1; that approach was inverted to lock the namespace + freeze the API shape WITHOUT taking the 6000-line body-relocation risk in a single commit. Body relocation happens incrementally during Phase 2-7 caller migrations: each phase migrates its cluster of callers AND moves the corresponding bodies in the same commit. The end-state architecture is identical to the pass-3 spec; the path is smaller, more reviewable, and lower-risk. **Namespace LOCKED**: `scope.motion / .illumination / .imaging / .diagnostics / .capabilities / .io` are the final names for 4.x. Test triage from the original plan still applies (259 sites across 14 files; see pass-3 amendment below) but the work spreads across Phases 2-7 instead of concentrating in Phase 1.

**Pass-3 cost re-band (2026-05-11, resolves pass-2 C-11 + pass-3 layer-audit Fact 3)**: pass-2 estimated 94 driver-direct test sites across 9 files; pass-3 layer-audit verified the actual count is **259 sites across 14 files** (~2.75x undercount). The five files pass-2 missed: `test_state_observer.py` (54 sites), `test_audit_fixes.py` (29), `test_microscope_settings.py` (9), `test_driver_registry.py` (5), `test_led_toggle_reliability.py` (4). Phase 1 cost re-bands to:
- Skeleton + driver-rename sweep: 1-2 days
- Test triage (per-site decision: migrate to sub-API per Rule 22 OR keep private-handle access with comment): 3-5 days
- conftest fixture updates: 1 day
- Cross-repo coordination (Firmware + Firmware/etaluma-engineering + LVP -- see pass-3 layer-audit Fact 2: engineering plugin lives in the Firmware repo, not in LVP): 1-2 days
- **Total Phase 1**: 5-7 days, not 1-2

Work items:
- Create `modules/lumascope_api/` package: `__init__.py`, `motion.py`,
  `illumination.py`, `imaging.py`, `diagnostics.py`, `capabilities.py`,
  `io.py` (six sub-APIs per D2; `io.py` ships as empty skeleton)
  (move existing capability code in)
- Each sub-API class is empty except `__init__(driver_handle, scope_root)`
  + `_init()` + `_disconnect()`
- `Lumascope.__init__` instantiates each sub-API; assigns to `self.motion`,
  `self.illumination`, `self.imaging`, `self.diagnostics`,
  `self.capabilities`, `self.io`
- Driver attribute renames: `self.motion -> self._motion_driver`,
  `self.led -> self._led_driver`, `self.camera -> self._camera_driver`.
  Sweep all 268 internal call sites
- Update `etaluma_engineering` plugin (in the Firmware repo, not LVP): any `scope.motion` driver-direct
  access becomes `scope._motion_driver` (same-commit edit; the engineering plugin lives at `Firmware/etaluma-engineering/etaluma_engineering/` and ships its own `pip install -e` cycle separately from LVP)
- Update tests: 259 driver-direct sites across 14 files get per-site triage; migrate to sub-API per Rule 22 OR keep private-handle access (`scope._motion_driver.<method>()`) with a comment justifying the driver-internal test. Update `scope_capabilities.py:9-10` docstring same-commit (per pass-2 M-5)
- Methods stay on `Lumascope` for this phase; nothing moves yet
- All existing public callers still work (sub-APIs exist but empty)

**Acceptance** (pass-2 A-7 integrated, 2026-05-11):
- Existing test suite compiles (no import errors) and passes
- Tests using `scope.motion.<driver_method>()` are migrated to `scope._motion_driver.<driver_method>()` or to the new sub-API; conftest fixtures updated
- Smoke assertion in conftest: `isinstance(scope.motion, MotionAPI)` (not a driver instance)
- `scope.motion`, `scope.illumination`, `scope.imaging`, `scope.diagnostics`, `scope.capabilities`, `scope.io` all accessible as sub-API instances; `scope._motion_driver`, `scope._led_driver`, `scope._camera_driver` accessible as driver instances
- `scope_capabilities.py:9-10` docstring updated to reference new private-handle pattern (pass-2 M-5)
- Engineering plugin builds + loads cleanly from the Firmware repo against the updated LVP

### Phase 2 -- Move motion methods to scope.motion (2-3 days)

- Move ~40 motion methods from `Lumascope` to `MotionAPI`
- Each moved method's body either (a) stays the same (was already
  driver-delegating, now delegates via `self._driver` instead of
  `self.scope_root._motion_driver`), or (b) updates to read from
  `self._owned_state` (was `self.scope_root._pos_cache` etc.)
- Move `_pos_cache`, `_axis_state`, `_arrival_events`, position listeners
  + lock to `MotionAPI`
- Move motion-monitor thread to `MotionAPI._init` / `_disconnect`
- Update internal callers: `self.move_absolute_position(...)` ->
  `self.motion.move_absolute_position(...)` if called from non-MotionAPI
  Lumascope code; intra-MotionAPI calls stay as `self.method(...)`
- Update external callers in LVP: `scope.move_absolute_position(...)` ->
  `scope.motion.move_absolute_position(...)`. Mass rename via grep + sed
  with manual review per file
- Update tests
- Update `LumascopeSkills.md` motion section
- Operation flags (`is_homing`, `is_turreting`) get one-line forwarders on
  Lumascope pointing at `self.motion.<flag>`

### Phase 3 -- Move illumination methods + collapse LED state (2-3 days)

- Move ~30 LED methods from `Lumascope` to `IlluminationAPI`
- Combine with Wave 2 STATE-LED-1: `_led_state` lives in `IlluminationAPI`
  as the single source of truth; `LEDBoard.led_ma` retires
- LED listeners + lock move
- Update internal + external callers (`scope.led_on(...)` ->
  `scope.illumination.led_on(...)`)
- Update tests
- Update `LumascopeSkills.md` illumination section
- `etaluma_engineering` plugin: any `scope.led_*` -> `scope.illumination.*`

### Phase 4 -- Move imaging methods + live_processing infrastructure (6-8 days, was 3-4 days)

**Pass-3 amendment (2026-05-11) -- live_processing infrastructure lands here (per D1; resolves pass-2 C-8 + pass-3 F-C-1)**: the `add_frame_listener` infrastructure that `ctx.plugins.live_processing` depends on is built during this phase, paired with the ImagingAPI relocation. Cost re-band adds ~3-4 days for the listener registry + driver-side fire sites + thread contract + drop policy.

- Move ~50 camera methods from `Lumascope` to `ImagingAPI`
- Move `_camera_cache`, `_frame_buffer`, camera listeners, frame_validity,
  scale_bar
- Move camera diagnostics (`get_camera_temps`, etc.) -- keep with imaging
  per audit recommendation
- Operation flags (`is_capturing`, `is_focusing`) get one-line forwarders
- `compute_focus_score` moves to `modules/focus.py` (per pass-1 C-4 amendment, not into ImagingAPI)
- **Per-frame listener registry on `ImagingAPI`**: `add_frame_listener(handler)` / `remove_frame_listener(handler)` / `_fire_frame_listeners(frame, metadata)`. Sync-fire-from-calling-thread per Rule 33; drop policy when handler exceeds 16ms budget (per Rule 42 budget entry `plugin_live_processing_handler_ms` in `docs/PERFORMANCE_BUDGETS.md`)
- **Driver-side fire sites**: each camera driver calls the listener registry from its frame handler -- `PylonCamera._store_frame` (Pylon ImageHandlerBase callback thread), IDS grab thread, FX2 USB-stream thread, SimulatedCamera tick thread. Each requires careful threading + lock discipline per Track B CONC-2
- **Drop policy + budget violation surface**: counter + WARNING log per Rule 42 + notification per Rule 28 when budget is exceeded
- **Test infrastructure**: live-processing listener test for `SimulatedCamera` with failure-injection per Rule 11 strengthening
- Update internal + external callers
- Update tests
- Update `LumascopeSkills.md` imaging section
- `ctx.plugins.live_processing` namespace now has working infrastructure (was empty in earlier phases per §4.2)

### Phase 5 -- Move diagnostics + extract image_save (1-2 days)

- Move ~10 diagnostic probes from `Lumascope` to `DiagnosticsAPI`
- Move ~15 image-save helpers (instance + static duplicates) from
  `Lumascope` to `modules/image_save.py` as plain functions
- Retire static-method duplicates in same commit (Rule 35)
- Update callers: `scope.save_image(...)` -> `from modules.image_save
  import save_image; save_image(...)`
- Update tests
- Update `LumascopeSkills.md`: new "Image Save" section pointing at
  `modules/image_save`; diagnostics section

### Phase 6 -- Cleanup composition root + retire forwarders (1-2 days)

- Verify `Lumascope` is now ~30 methods: lifecycle + executor + atexit +
  facade getters
- Retire operation-flag forwarders on `Lumascope`: callers update to
  `scope.motion.is_homing` etc.
- Final `LumascopeSkills.md` sync (per §6.6 TOC restructure: Session-led, six sub-APIs accessible as `scope.<name>`)
- Run full test suite + bench validation

**Total (pass-3 re-band)**: 3-5 weeks calendar / 15-25 days dev time (resolves pass-2 C-12 + pass-3 layer-audit Fact 3). Previous "10-16 days" estimate is RETIRED; previous META synthesis "multi-month" framing is also retired. Drivers of the wider band:
- 259 driver-direct test sites across 14 files (not 94 across 9 per pass-2; layer-audit verified count)
- Phase 4 includes live_processing infrastructure (+3-4 days per D1)
- Phase 3 includes IlluminationAPI channel-spec rename + ScopeCapabilities `illuminators` widening (+3-5 days per D3)
- LumascopeSkills.md TOC restructure (per §6.6, +1-2 days)
- Cross-repo coordination LVP + Firmware + Firmware/etaluma-engineering (3-repo per pass-3 layer-audit Fact 2)
- ScopeSession Session-layer integration (one-line Rule 1 edit but the conceptual shift carries through every Phase's L2 contract update)

### Plugin platform implementation phases (separate, post-design)

**Pass-3 amendment (2026-05-11) -- Phase B split (resolves pass-2 C-9 + C-13; THREE-repo coupling per pass-3 layer-audit Fact 2)**: Phase B splits into B1 + B2 to acknowledge that the engineering plugin lives in the Firmware repo (not LVP) and crosses THREE repos: LVP / Firmware / Firmware/etaluma-engineering (which is a sub-package of Firmware).

- **Phase A (1-2 days)**: build `modules/plugins/__init__.py` with
  registry-of-registries; implement `ctx.plugins.<namespace>`,
  PluginSpec, mount points (one initial name per D10), loading flow per §4
- **Phase B1 (1-2 days, was Phase B part 1)**: ship updated `Firmware/etaluma-engineering/` with `[project.entry-points]` block in `pyproject.toml` declaring the new entry-point group `lvp.plugins`. The plugin code itself still uses today's `register(ctx)` shape; BOTH discovery paths (entry-points + import-by-name fallback) work. Install on every dev machine via `pip install -e Firmware/etaluma-engineering/`. Verify no behavior change.
- **Phase B2 (1 day, was Phase B part 2)**: LVP-side switches discovery to `entry_points` group `lvp.plugins`; retire the import-by-name fallback. Engineering plugin (now refactored to register against `ctx.plugins.ui` with the locked `left_sidebar.accordion` mount point) loads via entry-points only. Requires every dev/bench/lab machine to have completed Phase B1's `pip install -e` first; the operational sequence is "B1 lands -> reinstall on every machine -> B2 lands."
- **Phase C (intern's work, weeks)**: first NEW post-processing plugin built
  against the platform. Hardens the contract; reveals gaps for fix-up.

**D9 -- Built-in post-processing retires into the plugin platform (4.1.5)**: today's five `ProtocolPostProcessingExecutor` subclasses (`composite_generation`, `zprojector`, `stitcher`, `stack_builder`, `video_builder`) migrate to `ctx.plugins.post_processing` during 4.1.5, validating the plugin contract on real workflows BEFORE the intern's project. This addresses pass-3 F-I-3 and avoids parallel post-processing surfaces.

Plugin platform Phase A + Phase B1 + Phase B2 can ship BEFORE the API sub-API
migration; the three-repo coupling means each phase is a coordinated release across LVP and Firmware-side plugin install.

### 6.6 LumascopeSkills.md TOC restructure (pass-2 A-6 integrated)

**Pass-3 amendment (2026-05-11)**: Rule 34 forces same-commit `LumascopeSkills.md` updates for each sub-API method move, but doesn't pre-decide the new TOC shape. Pinning the final TOC now so each Phase 2-5 commit slots its docstring updates into the right Phase 6 position:

**Phase 1 inserts skeleton sections** for each sub-API + Session layer; Phases 2-5 fill them per the migration; Phase 6 final TOC:

1. Lumascope composition root
2. ScopeSession session layer
3. scope.motion
4. scope.illumination
5. scope.imaging
6. scope.diagnostics
7. scope.capabilities
8. scope.io
9. modules (image_save, focus, composite_capture, protocol_runner, focus_history)
10. plugin platform reference
11. REST surface reference

L2 callers (Matlab / micromanager / SDK / REST clients) target Session per pass-3 layer-audit §2.6; sub-API sections are L3/L4-visible reference for OEMs and contributors. The Session lead positions ScopeSession as the L2 entry point per D4.

### 6.7 Plugin testing harness (pass-2 A-13 integrated)

**Pass-3 amendment (2026-05-11)**: Phase A deliverables include `tests/plugin_test_harness.py` -- a fixture-driven entry point that gives plugin authors "configured ctx with mocked sub-APIs." Reusable across `ctx.plugins.ui`, `ctx.plugins.post_processing`, `ctx.plugins.live_processing`, `ctx.plugins.rest` namespaces. Without this, every plugin author (intern's first plugin, future OEM contributions, retired-built-ins post-D9) re-invents the harness. Cost: ~1-2 days in Phase A.

---

## 7. Cost re-estimate

**Pass-3 cost re-band (2026-05-11, resolves pass-2 C-11 + C-12 + pass-3 layer-audit Fact 3)**: previous "10-16 days" estimate is RETIRED. Previous META synthesis "multi-month" framing is also RETIRED. Settle on a single band across both docs:

| Work | Cost |
|---|---|
| Sub-API migration (Phases 1-6, including 259-site test triage, IlluminationAPI channel-spec rename, live_processing infrastructure in Phase 4, ScopeCapabilities `illuminators`/`hardware_features` widening, LumascopeSkills.md TOC restructure, three-repo coordination) | 15-25 dev days / 3-5 weeks calendar |
| Plugin platform (Phases A + B1 + B2; intern's Phase C separate; built-in post-processing retire = 4.1.5 work per D9) | 4-6 days |
| Plugin testing harness (per §6.7, pass-2 A-13) | 1-2 days |
| Rule-change commit (referencing locked namespace, Session layer in Rule 1, six sub-APIs) | 2-3 hours |
| **Total before intern arrives** | **3-5 weeks calendar; 15-25 dev days; longer if D9 built-in post-processing retire lands in same wave** |

Plugin platform (Phases A + B1 + B2) lands BEFORE intern arrives so her work
hardens the canonical pattern. API sub-API migration (Phases 1-6) can
overlap or follow.

---

## 8. What this design unblocks

- **Rule 35a (API surface decomposition)** can name the six sub-API names
  concretely (`scope.motion`, `scope.illumination`, `scope.imaging`, `scope.diagnostics`, `scope.capabilities`, `scope.io`)
- **Rule 30 edit** (pre-L2-ship freedom) ships in same commit
- **Rule 43-optional** (one extension surface) points at this doc's §4
- **Wave 2 STATE-LED-1** combines into Phase 3 (LED state collapse + sub-API
  move = one set of changes)
- **Wave 5 R3 (capability expansion)** combines with Phase 1 + 5
  (capabilities sub-API and field expansion)
- **Wave 6 plugin platform** has its design ready; implementation Phase
  A-B1-B2 can ship in the 4.1 window
- **Intern's post-processing project** has a contract to build against; built-in post-processing retire (D9) into the platform first
- **REST Phase 1** is DEFERRED pending the dedicated REST design session that resolves pass-2 C-6 / C-7 (URL convention; ScopeSession disposition is resolved per D4 -- Session as own layer -- but the URL routing is the open question)

---

## 9. Followups outside this design's scope

- **Objective / labware / turret-config / stage-offset** (cluster 21 ~12
  methods): stays on `Lumascope` for now, moves later to
  `modules/scope_session.py` as a separate refactor
- **Executor topology cleanup** (Wave 3 R4 + new Rule 35c): independent
  of sub-API split; runs in Wave 3
- **Listener async-fire decision**: keep sync-fire-from-calling-thread
  per existing Rule 33 decision; revisit if a sub-API listener proves to
  block the calling thread
- **Plugin sandboxing / capability enforcement**: 4.x ships informational
  capability declaration; strict mode is a future platform-version
  decision
- **Plugin auto-discovery via `entry_points`** vs import-by-name: ships
  with discovery only; import-by-name fallback retired
- **REST plugin endpoint authentication**: `REST_API_PLAN.md` covers; not
  in scope here

---

## 10. Decisions list (for the rule-change commit reference)

The rule-change commit references the following locked names:

- **Sub-API attribute names (SIX, pass-3 amended per D2)**: `scope.motion`, `scope.illumination`,
  `scope.imaging`, `scope.diagnostics`, `scope.capabilities`, **`scope.io`** (reserved for USB-to-IO trigger device F9; ships as empty placeholder in Wave 7 Phase 1)
- Runtime state surface: `scope.runtime_state` (pass-1 amendment;
  firmware_versions + firmware_features + future runtime-mutable values)
- **Hardware features surface (pass-3 D5)**: `scope.capabilities.hardware_features: dict[str, frozenset[str]]` alongside `firmware_features` -- captures USB-to-IO trigger and other non-firmware-gated hardware capabilities
- **Illuminator capability shape (pass-3 D3)**: `scope.capabilities.illuminators: tuple[IlluminatorSpec, ...]` with each spec carrying `kind`/`channels_or_patterns`/`unit`/`max_intensity`; `led_colors`/`led_channels`/`led_max_ma` become derived properties for backcompat
- **IlluminationAPI canonical surface (pass-3 D3)**: `set_channel(channel_spec, intensity)` / `clear_channel(channel_spec)` / `get_illuminator_state(illuminator_id)` / `illuminator_states`. `led_on(channel, mA)` / `led_off(channel)` / `leds_off()` survive as Rule 30 backcompat forwarders for the current LED illuminator.
- Driver private handle names: `scope._motion_driver`,
  `scope._led_driver`, `scope._camera_driver`
- Sub-API class names: `MotionAPI`, `IlluminationAPI`, `ImagingAPI`,
  `DiagnosticsAPI`, `Capabilities`, `RuntimeState`, **`IOAPI`** (pass-3 D2; empty in 4.x)
- Sub-API module locations: `modules/lumascope_api/<name>.py`
- Plugin namespace set: `ctx.plugins.{ui, post_processing,
  live_processing, rest}` (hardcoded for 4.x; `live_processing` name LOCKED but namespace ships empty until Wave 7 Phase 4 builds the per-frame listener infrastructure per D1)
- Plugin entry point group: `lvp.plugins`
- Plugin lifecycle hooks: `register(ctx)`, `unregister(ctx)`,
  `on_settings_changed(ctx, settings)`, optional `on_event(...)`
- PluginSpec `subscribes_to` field for targeted `on_settings_changed`
  (pass-1 amendment)
- PluginSpec `capabilities` used by tech-support report + diagnostic
  probe (pass-1 amendment — not informational-only; pass-2 I-10 considered drop-as-redundant + REJECTED per DJ3 -- the field is consumed by support tooling)
- **Mount points (ONE initial, pass-3 amended per D10)**: `left_sidebar.accordion` only. Additional mount points add when first plugin needs each per §4.2 "deliberate platform change" pattern.
- REST URL conventions: **DEFERRED 2026-05-11**. Pass-2 C-6 surfaced a
  conflict with `REST_API_PLAN.md` flat `/api/v1/{move,led,camera}` already
  locked. Per Eric: defer to a dedicated REST design session; §5 of this
  doc is design-only-not-locked until that session ships a V2 plan.
- REST conflict resolution: duplicate name raises `PluginRegistrationError`,
  second plugin not loaded (pass-1 amendment)
- **REST middleware shared (pass-3 D11)**: plugin REST routes mount UNDER the same FastAPI app + middleware, NOT around it. `scope.is_command_safe_for_rest` dangerous-command gate applies to plugin endpoints same as core endpoints. No `PluginSpec.requires_unsafe_commands` flag in 4.x.
- **ScopeSession Session layer (pass-3 D4)**: ScopeSession promoted to its own **Session layer** in the canonical layer chain. Rule 1 edits to `GUI / REST -> Session -> Lumascope API -> Modules -> Drivers -> Hardware`. ScopeSession is the session-composition-root; Lumascope remains hardware-composition-root. Resolves pass-2 C-7. The REST design session decides REST URL convention on top of Session; Session's existence is NOT deferred.
- **Focus history surface (pass-3 D12)**: per-(scope_id, plate_id, well, objective) focus history stored in `data/focus_history.json`; API on Session, not Lumascope: `session.focus.get_well_focus(well, objective)` / `set_well_focus(...)` / `narrow_search_range(...)`. 4.1.5 work.
- **Atexit pattern shape (pass-3 DJ2)**: instance-removes-self-on-destruct -- each Lumascope instance registers its own atexit hook at init and removes it at disconnect/_emergency_shutdown. Pairs with Rule 41 lifecycle-inventory registry.
- **D9 built-in post-processing retirement**: `modules/stitcher.py`, `modules/zprojector.py`, `modules/composite_generation.py`, `modules/stack_builder.py`, `modules/video_builder.py` migrate to `ctx.plugins.post_processing` during 4.1.5 (validates contract on real workflows before intern's project).
- New module: `modules/focus.py` (pure functions; `compute_focus_score`
  moves here, pass-1 amendment to Q12)
- New module: `modules/composite_capture.py` (free function
  `composite_capture(scope, ...)`, pass-1 amendment to M-1)
- New doc: `docs/L2_CONSUMERS.md` (initially empty; named list of
  external L2 consumers that triggers Rule 30 freeze when populated)
- New doc: `docs/PERFORMANCE_BUDGETS.md` (per META Wave 0 R-4; skeleton with 5 initial budget rows including `plugin_live_processing_handler_ms`)

Rule changes referencing this doc:
- Rule 30 (L2 stability) edit: pre-L2-ship freedom + sharp freeze
  trigger (PyPI publish + named external consumer)
- Rule 35a (API surface decomposition cap): names the SIX sub-APIs as
  the canonical decomposition pattern; thresholds are qualitative + WARN
- Rule 43-optional (one extension surface): points at this doc's §4
- Rule 1 (layer architecture): adds Session row -- `GUI/REST -> Session -> Lumascope API -> Modules -> Drivers -> Hardware` (pass-3 D4)

---

## 11. Open questions for Eric (RESOLVED 2026-05-11)

1. **Mount point names**: ship the 5 initial names from §4.6; add more as needs surface. **Locked.**
2. **Plugin discovery mechanism**: pure `entry_points`; retire import-by-name. Requires smoke-test on Eric's etaluma-engineering install before locking. **Locked.**
3. **`Lumascope` class name**: keep `Lumascope` (L3/L4 mental-model continuity; class shrinks but name carries meaning). **Locked.**
4. **Sub-API initialization order**: parallel construct (constructors are pure-stash, no I/O); sequential `_init` in dependency order — capabilities, then motion/illumination/imaging (parallel-safe at `_init`), then diagnostics (last, depends on all three). **Locked.**

---

## 12. Doc lifecycle

This design doc stays live until all six API migration phases land.
Then moves to `docs/completed/PLUGIN_API_DESIGN_2026-05-09.md` with a
header noting "spec realized in commits SHA1-SHA6, see
`LumascopeSkills.md` for the canonical L2 surface."

The plugin platform spec (§4) stays referenced as long as plugins are
still being built against it. Updates to the spec require a doc edit +
minor LVP version bump (per Rule 30 once REST + plugin contract has
external L2 consumers).
