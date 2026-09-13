# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refusal's remedy has to fit the refusal.

When a post-processing function turns a folder away it says what it excluded.
Saying so is not enough on its own: a user who pointed the stitcher at a folder
of composites learns that composites are derived outputs, and still does not
learn that the route they wanted works the other way round -- stitch the
channels, then composite the result.

The remedy is keyed on the PAIR (function that refused, what it excluded)
rather than on the function, because the stitcher also turns away videos and
hyperstacks, and "stitch the channels first" is wrong for those. Wrong advice
in the refusal popup is the defect this exists to remove, so an unknown pair
says nothing at all.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.common_utils import PostFunction
from modules.protocol_post_processor import _remedy_for


class TestTheRemedyFitsTheRefusal:
    def test_stitching_refusing_composites_names_the_supported_route(self):
        remedy = _remedy_for(PostFunction.STITCHED, {PostFunction.COMPOSITE})
        assert 'Stitch the source channel images first' in remedy
        assert 'Stitched folder' in remedy

    def test_stitching_refusing_videos_offers_nothing(self):
        """The guard against a per-function remedy: videos are not answered by
        stitching channels first, so the refusal says only what it excluded."""
        assert _remedy_for(PostFunction.STITCHED, {PostFunction.VIDEO}) == ''

    def test_a_mixed_folder_names_only_the_remedy_that_applies(self):
        remedy = _remedy_for(PostFunction.STITCHED, {PostFunction.COMPOSITE, PostFunction.VIDEO})
        assert 'Stitch the source channel images first' in remedy
        assert remedy.count('Stitch the source channel images first') == 1

    def test_another_function_refusing_composites_offers_nothing(self):
        assert _remedy_for(PostFunction.ZPROJECT, {PostFunction.COMPOSITE}) == ''

    def test_nothing_excluded_adds_nothing(self):
        assert _remedy_for(PostFunction.STITCHED, set()) == ''
