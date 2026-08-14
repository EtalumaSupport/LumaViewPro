# Enhance color and sharpening TDD evidence

## Scope

Derived from the reported `0green_s.tiff` failure: preserve the Green visual
channel for metadata-less legacy false-color TIFFs and make close fluorescent
bead saddles clearer without modifying raw inputs.

## User journeys

1. As a fluorescence user, I can Enhance a legacy one-plane Green RGB TIFF
   whose filename starts with a numeric acquisition prefix, so its derived TIFF
   remains Green instead of being labeled and rendered as BF.
2. As a microscopy user, I can see a clearer saddle between close beads without
   lifting the dark background or treating the result as quantitative data.

## RED evidence

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/python -m pytest tests/test_quick_enhance.py -q`

The added tests failed before implementation:

- `0green_s.tiff` exported as `(32, 32)` gray instead of `(32, 32, 3)` Green RGB.
- The close-pair saddle ratio stayed `0.32407406`, failing the required 10 percent reduction.

The recipe generation test also failed when it required pipeline version `3`:
the pre-change code returned `2`.

## GREEN evidence

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/python -m pytest tests/test_quick_enhance.py tests/test_enhance_file_or_folder.py -q`

Exit status `0`: all selected Quick Enhance and picker-flow tests passed.

`./.venv/bin/python -m py_compile modules/quick_enhance.py`

Exit status `0`.

`git diff --check`

Exit status `0`.

## Guarantees

| Guarantee | Test | Result |
| --- | --- | --- |
| A metadata-less one-plane Green TIFF named `0green_s.tiff` exports as Green RGB with Green metadata and leaves the source bytes unchanged. | `test_metadata_less_legacy_green_rgb_with_numeric_prefix_exports_green` | PASS |
| Fixed Enhance deepens a synthetic close-bead saddle by at least 10 percent without lifting its dark baseline. | `test_fixed_recipe_deepens_a_close_bead_pair_saddle_without_lifting_background` | PASS |
| The derived recipe records the new generation and signal-gated unsharp operation. | `test_recipe_has_required_provenance_and_quantitative_warning` | PASS |

## Original-image replay

The original raw image was copied to a temporary directory and run through
`QuickEnhancer.export_file`; the supplied raw file was not modified. The output
was Green-only RGB, carried `channel: Green`, and wrote a recipe sidecar. For
the reviewed upper-center pair, valley-to-peak ratio improved from `0.5077` to
`0.4320`; foreground edge energy rose from `342.177` to `357.814`.

## Coverage limitation

Coverage could not be measured in this checkout because
`./.venv/bin/python -m coverage --version` returned `No module named coverage`.
No dependency was installed or changed during this task.
