# LumaViewPro Changelog

## 4.0.0 (in development)

- **A move returns once it has arrived (SDK, breaking)**: `scope.motion.move_absolute` and
  `move_relative` wait for the axis and raise `MoveNotCompletedError` when it did not arrive;
  the `wait_until_complete` argument is removed. They used to return as soon as the board took
  the command, so a stall, a lost board or a stop never reached the caller. For work done while
  the axis travels, or axes moved together, `start_move_absolute` / `start_move_relative` return
  a started move whose `wait()` gives the same outcome. A waited move no longer holds the IO
  lane while it travels. `wait_until_finished_moving` waits for the axes moving when it is
  called and raises instead of returning `False` on a timeout or `True` for an axis left
  unknown. `session.go_to_step` returns once X, Y and Z have arrived;
  `session.start_go_to_step` is its started form. The engineering plugin needs 1.0.12 or
  later.
- **A run with a failed image save ends `incomplete` (SDK, breaking)**: `files_complete`'s
  `files` is now `'written'` or `'incomplete'` (was `'abandoned'`). A save that failed on
  disk used to count as written, so the composite merge, the hyperstack build and the
  post-processing auto-run read a folder missing an image; they now refuse it, with
  `RunFilesNotWrittenError` reason `write_batch_save_failed`.
- **One labware and objective catalogue per scope (SDK)**: `Lumascope(..., source_path=None)`
  reads `data/labware.json` and `data/objectives.json` once, from `source_path` (the
  installation's own folder when none is given), and exposes `scope.source_path`,
  `scope.wellplate_loader` and `scope.objective_helper`. The session's same-named members
  are now the scope's, read-only. Before, a session started on another folder read its own
  catalogues from that folder while the scope, autofocus and the run read the installation's,
  so they could disagree about which objectives and plates exist. `ScopeSession.create`'s
  `source_path` now defaults to the installation's folder (was `'.'`), and is refused beside
  `scope=`: pass it to `Lumascope(source_path=...)` instead. A caller-built scope's session
  saves settings and reads `data/tiling.json` under that scope's folder.
  `scope.protocols.register_source_path` is removed.
- **A missing or unusable catalogue file raises `InstallationFileError` (breaking)**:
  `labware.json` / `objectives.json` failures raise `modules.exceptions.InstallationFileError`
  (with `.file_path`) when the scope is built, and nothing is left running. It used to be a
  popup plus a later `ConfigError`, which the GUI read as bad stored settings.
- **Protocol takes the scope's catalogues (SDK, breaking)**: `Protocol.validate_steps`,
  `validate_for_run`, `apply_tiling`, `Protocol.from_config` and `Protocol.create_empty` take
  `objective_helper` (and `wellplate_loader` where a plate is read); a `Protocol` no longer
  builds its own. `scope.protocols.create_protocol` / `load_protocol` pass them for you.
  `ProtocolTimeEstimator` requires its loader.
- **Post-processed file names**: the objective token is derived from the id the run recorded,
  so an objective the current catalogue lacks is named rather than omitted. A `short_name`
  written into `objectives.json` is now ignored: capture and post-processing both derive
  the token from the id, so one objective cannot be named two ways (none ships with one).

- **One session factory: `ScopeSession.create_headless` is removed (SDK, breaking)**:
  build every session with `ScopeSession.create(settings, ..., simulate=...)`. What
  `create_headless()` did becomes `ScopeSession.create(ScopeSession.load_user_settings(root), simulate=True)`;
  `create_headless(settings=s)` becomes `ScopeSession.create(s, simulate=True)`.
  `load_user_settings(source_path)` is new: it reads the user's configuration the
  way the GUI does and raises `ConfigError` when `source_path` is not an LVP
  installation root. `settings` stays a required argument of `create`, so a
  script cannot be configured from disk by omission; `simulate=False` builds the
  same session on real hardware.

- **`ScopeSession.create` takes the host's injections as keyword arguments (SDK)**:
  the factory signature is now
  `create(settings, source_path='.', scope=None, io_executor=None, camera_executor=None, *, simulate=False, ui_dispatcher=None, af_ui_update_func=None, settings_saved_hook=None, engineering_mode=False, display_ctx_provider=None)`.
  `simulate=True` builds a simulated scope declared with `settings['microscope']`;
  `ui_dispatcher` is a `Clock.schedule_once(func, dt)`-shaped callable that marshals
  the four executor lanes' callbacks onto the host's UI thread (None runs them
  inline on the worker, which is the headless form); `af_ui_update_func(pos)` is
  the Z readout for the autofocus runner AND the capture engine (one callable,
  two consumers); `settings_saved_hook(settings_snapshot)` fires after a successful
  `save_settings` with the dict that was written; `engineering_mode` is stored on
  the session; `display_ctx_provider` is HOST-ONLY (the Kivy display thread) and is
  not an L2 parameter. `create_headless(settings=None, source_path='.', engineering_mode=False)`
  keeps its signature and is now exactly `create(simulate=True)` with the settings
  resolved from disk when none are passed.

  **Breaking for SDK callers**: (a) `session.shutdown()` now turns the LEDs off,
  stops motion and DISCONNECTS a scope the factory built (`create` with no `scope=`,
  or `create_headless`) -- afterwards `scope.motor_connected` is False,
  `scope.imaging.is_streaming()` is False and `scope.diagnostics.get_microscope_model()`
  returns None; a caller-passed scope (`create(..., scope=my_scope)`) is untouched and
  stays the caller's to disconnect, as do both scopes after `session.set_scope(...)`,
  and a second `shutdown()` logs one info line and does nothing. (b) `configure_scope()`
  asks the motor board for its model first and WRITES a catalogued reported model into
  the caller's `settings['microscope']` when it differs from the stored selection
  (hardware truth outranks the selection) -- this includes a simulated scope, so a
  caller's settings dict can come back changed; an uncatalogued or absent report leaves
  the stored model. (c) A simulated scope now reports its DECLARED model, so
  `Lumascope(simulate=True, configured_model='LS850')` reports `LS850` and therefore has
  NO turret axis -- `create_headless(settings=...)` against the shipped template's
  `LS850` loses the turret the old code reported by accident; a bare
  `Lumascope(simulate=True)` still reports the module-global default. (d)
  `start_executors` is internal (not part of the L2 API surface): the factories start
  the lanes they build, and calling it again silently spawns a SECOND worker thread on
  each lane -- drop it from every recipe that follows a factory. (e) With caller-passed
  lanes (`io_executor=` / `camera_executor=`) and a factory-built scope, the LED drain
  at shutdown runs on the caller's io lane only while that lane's worker thread is
  alive, then the caller's lanes are shut down with wait.
- **Session factories configure the scope they build (SDK)**: `ScopeSession.create`
  (when it builds the scope) and `create_headless` now run the settings-to-scope
  bring-up -- turret slot keys normalized, slot-1 objective adopted, labware
  selected, `scope.initialize(...)` applied -- and release the camera start
  gate before returning, so a headless session can save an image without a
  further `initialize`. New public member `session.configure_scope()` for a
  caller-passed scope or a directly constructed session.

  **Breaking for SDK callers**: a hand-built settings dict must carry `frame`
  and `objective_id` (`ConfigError` names the missing key; the old `'4x'`
  default, which named no shipped objective, is gone); `create_headless()`
  raises `ConfigError` instead of returning a session on empty settings when
  `source_path` (default: the CWD) holds no `data/settings.json`; and
  `start_application_session(disable_homing=True)` performs no startup motion
  at all -- it no longer positions the turret, and no longer raises on an
  unhomed one.
- **The Session owns the objective question (SDK)**: `session.objective_question()`
  reports whether the objective is unknowable (never confirmed on this install,
  or the turret on an unassigned slot) and `session.confirm_objective(...)`
  answers it; `select_objective`, `assign_turret_objective`,
  `clear_turret_objective` and `set_turret_position` are the writers. Every
  member refuses an id that is not exactly a catalogue key. The resolved-optics
  record (`[Optics   ]`) now fires from the Session at bring-up and on every
  objective change, so a headless session logs a real scale; a headless turret
  move onto an unassigned slot logs a warning.
- **API stability**: Lumascope SDK + REST API are PRE-RELEASE. Subject to
  breaking changes in 4.1 / 4.1.5 / 4.2. See `docs/LumascopeSkills.md`
  preface for the migration plan. Internal LumaViewPro use is not
  affected by the freeze trigger.
- **Saved-image bit depth (file-format change)**: full-pixel-depth TIFFs now
  store raw, right-aligned sensor values (a 12-bit frame is `0..4095`) and
  record the true depth in the OME-TIFF `SignificantBits` tag, instead of
  left-justifying the data into the 16-bit container (`value * 16`). This is
  the standard OME-TIFF representation and fixes dim/grainy rendering of 10-bit
  captures and a 16-bit overflow on summed full-depth frames. Read-back
  (Post-Processing video, hyperstacks) scales by the file's `SignificantBits`,
  so existing left-justified files (tagged 16-bit) still read correctly -- no
  migration needed.

- **Protocol video frame counts (behavior change)**: video steps now record
  at the configured rate instead of an uncapped hardware-paced loop. With
  the shipped default step rate of 5 fps -- against the ~40 fps the old
  loop delivered on a fast camera -- a default-config video step records
  ~87.5% fewer frames than before. This is correct behavior replacing
  defect behavior: the old loop's extra frames were never the configured
  rate, and its delivery silently truncated under load. Each recording now
  writes a `recording_manifest.json` with the measured frame rate, frame
  count, and per-frame timestamps; raise the step's fps to record more
  frames.
- **Saved channel identity (metadata fix + SDK signature break)**: a saved
  image now records the channel it was acquired on, independently of how it is
  displayed. Manual captures, composites and composite exports previously
  stamped every file `Channel.Name = "BF"` regardless of the LED that lit them,
  because the metadata argument carrying that fact defaulted to brightfield and
  only the protocol path passed it -- so Quick Enhance read a green frame back
  as brightfield and declined to color it. `Channel.Modality` had the same
  defect one field over: it was derived from the false-color toggle, so one file
  could carry `Channel.Name = "Green"` next to `Channel.Modality = "BF"`. A
  16-bit non-OME fluorescence or luminescence capture saved with false color OFF
  now records `Modality = "MIF"` where it previously recorded `"BF"`; nothing
  else on disk changes, and no existing file is rewritten. Manual captures also
  begin recording their real LED drive current instead of `0`.

  **Breaking for SDK callers**: `save_image`, `save_live_image` and
  `prepare_image_for_saving` now require keyword-only `channel` and
  `false_color_on`; the `color` and `true_color` parameters are gone, and
  `write_video_frame`'s `layer_color` is now `channel`. A save can no longer be
  constructed without stating what it imaged. Post-processing outputs whose
  source channel cannot be determined record `"Unknown"` rather than asserting
  brightfield.

- **Per-recording OME-TIFF hyperstacks**: protocol video runs produce one
  hyperstack per well per scan (T = frame order, per-plane timing),
  including headless / REST runs. In Fiji, open hyperstacks via
  `Plugins > Bio-Formats > Importer` with Color mode = Composite to see
  channel colors; a plain File > Open shows ImageJ default LUTs.

Detailed release notes for 4.0.0-betaN tags live in
`LVP_4.0.0_CHANGELOG.md` (release-engineering log).
