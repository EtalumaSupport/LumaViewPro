"""A plugin reads what this LumaViewPro does through its plugin API level.

No version could say it: ``version.txt``'s line is the promoted release name,
and hundreds of trunk commits share one beta number. The host carries one
integer, raised by any commit that changes behaviour a plugin relies on; a
plugin that needs a behaviour reads ``session.plugin_api_level`` at its own
entry and refuses below it. Every level has its row in LumascopeSkills.md
saying what it added, so a plugin author can find which level to ask for.
"""

import re
from pathlib import Path

from modules.plugins import PLUGIN_API_LEVEL
from modules.scope_session import ScopeSession
from tests.settings_fixtures import complete_settings

_SKILLS = Path(__file__).resolve().parents[1] / 'docs' / 'LumascopeSkills.md'


def test_the_session_answers_the_hosts_level():
    session = ScopeSession.create(complete_settings(), simulate=True, warn_pre_release=False)
    try:
        assert session.plugin_api_level == PLUGIN_API_LEVEL
    finally:
        session.shutdown()


def test_every_level_says_what_it_added():
    rows = re.findall(r'^\| level (\d+) \|', _SKILLS.read_text(), flags=re.MULTILINE)
    assert [int(r) for r in rows] == list(range(1, PLUGIN_API_LEVEL + 1)), (
        'LumascopeSkills.md "Plugin API level" needs one row per level, '
        f'1..{PLUGIN_API_LEVEL}, in order; found {rows}'
    )
