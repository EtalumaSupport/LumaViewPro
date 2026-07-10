# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""data/settings.json defaults match what the code actually reads.

settings.json is the defaults file merged into current.json by
_deep_merge_defaults. Keys the code reads-with-a-default should be seeded
here so the schema -- not a scattered .get() default -- is the source of
truth, and keys the code deletes should not be reintroduced.

- image_mode: the consolidated capture/save selector; seeded to the 8-bit
  default, and the two retired toggles it replaced must not be reintroduced.
- per-layer video_config.fps: microscope_settings self-heals a missing fps
  to DEFAULT_VIDEO_FPS (30); seeded to match so the default lives in schema.
- disable_protocol_accordions: popped (permanently disabled, no longer a
  setting) -- must not be carried in defaults.
"""

from __future__ import annotations

import json
import pathlib

# pin-justified: the shipped defaults in data/settings.json are the
# contract a fresh install receives.
SETTINGS = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / 'data' / 'settings.json').read_text()
)


def test_image_mode_seeded_and_legacy_toggles_removed():
    assert SETTINGS['image_mode'] == '8bit'
    assert 'use_full_pixel_depth' not in SETTINGS
    assert 'false_color_16bit' not in SETTINGS


def test_dead_disable_protocol_accordions_removed():
    assert 'disable_protocol_accordions' not in SETTINGS


def test_every_layer_video_config_seeds_fps():
    layers = [k for k, v in SETTINGS.items() if isinstance(v, dict) and 'video_config' in v]
    assert layers, 'expected per-layer video_config blocks'
    for layer in layers:
        vc = SETTINGS[layer]['video_config']
        # 30 == DEFAULT_VIDEO_FPS (microscope_settings); seeding matches the
        # code's self-heal so behavior is unchanged.
        assert vc.get('fps') == 30, f'{layer} video_config missing fps=30: {vc}'
