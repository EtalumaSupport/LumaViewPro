# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""One unresolvable objective id must not take down a whole post-processing run.

The objective lookup answers an id it cannot resolve with None -- a documented
return, and the only answer available for a protocol that names an objective
the catalogue no longer holds (the catalogue was edited, or the install was
downgraded, between the run and the post-processing of its files).

The post-processing filename builder subscripted that return directly, so the
None became `TypeError: 'NoneType' object is not subscriptable`. It is raised
inside the loop that plans output names across every group, so a single bad id
in a single well aborted the post-processing of the entire run -- stitches,
composites, projections and stacks for every other well included.

The capture lane, which asks the same question of the same loader for the same
purpose, has always handled the None: it logs a warning naming the id and
carries None as the short name. Two lanes, one loader, one return value, two
different beliefs about what it can be. This pins them to the same one.
"""

import pandas as pd
import pytest

from modules.protocol_post_processor import ProtocolPostProcessor
from modules.common_utils import PostFunction

UNRESOLVABLE = 'ZZZ_not_a_prefix'


class _NameOnlyPostProcessor(ProtocolPostProcessor):
    """The smallest concrete subclass: the filename helper is what is under
    test, and every algorithm reaches it through the same base method."""

    @staticmethod
    def _get_groups(df: pd.DataFrame) -> pd.DataFrame:
        return df

    @staticmethod
    def _filter_ignored_types(df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _generate_filename(self, df: pd.DataFrame, **kwargs) -> str:
        raise NotImplementedError

    def _group_algorithm(self, path, df: pd.DataFrame):
        raise NotImplementedError

    @staticmethod
    def _add_record(protocol_post_record, alg_metadata: dict, root_path):
        raise NotImplementedError


@pytest.fixture
def turret_post_processor():
    """A turret scope: the only configuration that asks the loader at all."""
    return _NameOnlyPostProcessor(post_function=PostFunction.COMPOSITE, has_turret=True)


def test_an_unresolvable_objective_id_does_not_raise(turret_post_processor):
    """The defect: one bad id anywhere aborted every group in the run."""
    assert turret_post_processor._get_objective_short_name_if_has_turret(UNRESOLVABLE) is None


def test_a_resolvable_objective_id_still_names_the_lens(turret_post_processor):
    """The fix must not cost the feature it guards: a real id still resolves to
    its short name, which is what puts the objective in the filename."""
    assert turret_post_processor._get_objective_short_name_if_has_turret('4x Oly') == '4xOly'


def test_a_scope_with_no_turret_never_asks(turret_post_processor):
    """Unchanged, and pinned because it is the other source of None: without a
    turret the lookup is skipped entirely, so None already had to be a legal
    return here -- which is why no caller needed changing."""
    no_turret = _NameOnlyPostProcessor(post_function=PostFunction.COMPOSITE, has_turret=False)
    assert no_turret._get_objective_short_name_if_has_turret(UNRESOLVABLE) is None
    assert no_turret._get_objective_short_name_if_has_turret('4x Oly') is None
