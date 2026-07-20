# Stitching Safety TDD Evidence

Source plan: user journeys and acceptance criteria derived in this TDD run.

## User journeys

- As a microscope user, I can run Fast Preview without an unconstrained panorama algorithm so I can rapidly inspect tile geometry.
- As a microscope user, I can run Quality Stitch without changing source pixels, channel identity, or brightfield values.
- As a microscope user, a 0% overlap scan is placed from recorded stage geometry without invented registration.
- As a support user, I see a plain-language popup while exact diagnostics remain in the application log.

## RED and GREEN evidence

The initial RED run was:

```text
.venv/bin/python -m pytest tests/test_stitcher.py -q
ImportError: cannot import name 'infer_stage_overlap'
```

The final GREEN run was:

```text
.venv/bin/python -m pytest tests/test_stitcher.py tests/test_stitcher_plugin.py tests/test_issue_166_kv_lock_bindings.py -q
70 passed
```

`git diff --check` and `py_compile` also passed for the production Python files.

## Guarantees

| Guarantee | Test target | Result |
|---|---|---|
| Recorded stage spacing that equals tile size is treated as 0% overlap. | `TestStageConstrainedStitchModes.test_infers_zero_overlap_from_stage_spacing` | PASS |
| Quality at 0% overlap never invokes feature matching or overlap registration. | `test_quality_at_zero_overlap_uses_geometry_without_feature_matching` | PASS |
| Source-preserving composition does not average overlapping source pixels. | `test_source_preserving_mode_never_averages_overlap_pixels` | PASS |
| Quality with real overlap uses bounded local registration. | `test_quality_overlap_route_uses_bounded_local_registration` | PASS |
| The user popup does not expose raw worker messages and directs support to logs. | `test_stitcher_popup_never_surfaces_raw_worker_message` | PASS |
| The UI declares Fast Preview, Quality, and estimate behavior. | `test_stitch_ui_explains_modes_and_time_estimation` | PASS |
| Prebuilt composites are excluded from raw-channel stitching. | `test_stitcher_ignores_prebuilt_composites` | PASS |

## Known validation gap

The active virtual environment has neither `ruff` nor `coverage` installed, so lint and coverage could not be run. No claim of 80% coverage is made here. The focused pytest suite and Python compile checks are the completed local validation.
