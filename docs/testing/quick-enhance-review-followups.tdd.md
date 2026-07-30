# Quick Enhance Review Follow-ups TDD Evidence

Source plan: the remaining Quick Enhance review findings and Eric's panel-copy feedback.

## User journeys

- As a microscope user, I see the Quick Enhance preview before the export action so I know what will be saved.
- As a microscope user, I can process a large image without coordinate-grid allocations that dominate memory usage.
- As a future supported color-input user, non-finite values cannot leak into the derived RGB output.

## RED and GREEN evidence

RED, before production changes:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_quick_enhance.py::test_illumination_plane_avoids_full_resolution_coordinate_grids \
  tests/test_quick_enhance.py::test_color_algorithm_sanitizes_non_finite_channels_before_reconstruction \
  tests/test_quick_enhance_kv.py::test_quick_enhance_kv_offers_one_fixed_quick_enhance_action -q
3 failed
```

GREEN, after the minimal changes:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tests/test_quick_enhance.py tests/test_quick_enhance_kv.py tests/test_image_utils.py -q
55 passed
```

`git diff --check` also passed.

## Guarantees

| Guarantee | Test target | Result |
|---|---|---|
| Preview appears before the save action; status is last; redundant copy stays removed. | `test_quick_enhance_kv_offers_one_fixed_quick_enhance_action` | PASS |
| The illumination plane cannot create a full-resolution `np.mgrid` coordinate pair. | `test_illumination_plane_avoids_full_resolution_coordinate_grids` | PASS |
| Sanitized RGB channels, rather than unsanitized source channels, drive reconstruction. | `test_color_algorithm_sanitizes_non_finite_channels_before_reconstruction` | PASS |

## Known validation gap

The active virtual environment does not have `pytest-cov`; its `--cov` options are unrecognized. No coverage percentage is claimed. The Kivy parser test and focused Quick Enhance/image-utils suite completed locally.
