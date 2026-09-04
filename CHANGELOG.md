# LumaViewPro Changelog

## 4.0.0 (in development)

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
