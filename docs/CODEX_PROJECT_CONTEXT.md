# Codex Project Context

## Project Overview

LumaViewPro is a Python/Kivy microscope control application. It coordinates
hardware control, image acquisition, protocol execution, and offline
post-processing such as stitching, compositing, Z-projection, video creation,
and stack building.

This repository has a broad regression suite and several hardware-adjacent
paths. Keep changes narrow, preserve existing operator workflows, and avoid
refactors unless they are required for the task being implemented.

## Current Branch Progress

The current tiling/stitching branch has merged the current `origin/main`. The
tiling/stitching work has extracted GUI-free stitching selection into
`modules/stitching_core.py`. That extraction is intentional and should be kept.

Current stitching route behavior:

- BF uses feature stitching first, then overlap registration, then
  stage-position placement, then simple grid placement.
- Fluorescence and other non-BF channels use overlap registration, then
  stage-position placement, then simple grid placement.
- Per-group fallback metadata is preserved in `metadata["fallback_reason"]`
  and related fields.

The current task adds caller/operator visibility when a successful stitch used
a fallback algorithm. This should remain a successful post-processing result,
not a failure, because output was generated.

## Constraints

- Do not change default stitching route behavior unless explicitly requested.
- Do not change stitching algorithm behavior while documenting or surfacing
  degraded mode.
- Keep `scripts/compare_tiling_stitch_sample.py` unchanged unless a clear,
  task-blocking issue is found.
- Preserve `metadata["fallback_reason"]`.
- Avoid broad rewrites in post-processing. Prefer result metadata and existing
  popup/status surfaces.
- Be careful in dirty worktrees. Do not revert changes made by other agents or
  users.
- Use focused tests that do not require microscope hardware.

## Useful Files

- `modules/stitching_core.py`: per-group stitching algorithm selection,
  fallback chain, and GUI-free stitch helpers.
- `modules/stitcher.py`: production stitch post-processor integration,
  grouping, output filenames, and protocol record writes.
- `modules/protocol_post_processor.py`: shared post-processing folder runner
  and result summary surface.
- `ui/post_processing.py`: operator-facing post-processing popups and
  callbacks.
- `tests/test_stitcher.py`: focused stitching algorithm, route, fallback, and
  output contract coverage.

## Next Steps

- If reviewers ask for more visibility, prefer extending the existing
  post-processing result summary or notification center rather than changing
  algorithm selection.
- If adding more stitching tests, keep them synthetic and hardware-free.
- If validating captured data, use scripts only as read-only comparison tools
  unless the task explicitly requests script changes.
