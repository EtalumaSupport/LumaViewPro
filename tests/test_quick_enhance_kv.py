"""Quick Enhance KV must parse before the application constructs a window."""

import re
from pathlib import Path
import subprocess
import sys


KV_PATH = Path(__file__).resolve().parents[1] / 'ui' / 'lumaviewpro.kv'

# The five named post-processing panel rules (Object Analysis is an inline
# BoxLayout inside the accordion and is covered by the wrap guard below).
PANEL_RULE_HEADERS = [
    '<VideoCreationControls>:',
    '<StitchControls>:',
    '<ZProjectionControls>:',
    '<CompositeGenControls>:',
    '<QuickEnhanceControls>:',
]


def _indent(line: str) -> int:
    """Indentation width, mirroring Kivy's parser (tab expands to 4)."""
    prefix = line[: len(line) - len(line.lstrip(' \t'))]
    return len(prefix.replace('\t', '    '))


def _rule_block(content: str, rule_header: str) -> str:
    """The text of one top-level kv rule, up to the next top-level rule."""
    start = content.index(rule_header)
    nxt = re.search(r'\n<', content[start + 1 :])
    return content[start : start + 1 + nxt.start()] if nxt else content[start:]


def test_lumaviewpro_kv_parses_with_quick_enhance_controls():
    kv_path = Path(__file__).resolve().parents[1] / 'ui' / 'lumaviewpro.kv'
    script = (
        'from pathlib import Path; '
        'from kivy.lang.parser import Parser; '
        f'path = Path({str(kv_path)!r}); '
        'Parser(content=path.read_text(encoding="utf-8"), filename=str(path))'
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_quick_enhance_kv_offers_one_click_enhance_without_panel_prose():
    content = KV_PATH.read_text(encoding='utf-8')
    quick_enhance_rule = content[
        content.index('<QuickEnhanceControls>:') : content.index('<QuickEnhanceUtilityButton@')
    ]

    assert quick_enhance_rule.count('FileOrFolderChooseBTN:') == 1
    assert "text: 'Enhance'" in quick_enhance_rule
    assert 'text: root.status_text' in quick_enhance_rule
    for removed in (
        "text: 'Choose Image'",
        "text: 'Choose Folder'",
        "text: 'Quick Enhance'",
        "text: 'Save Enhanced Image'",
        'text: root.warning_text',
        'quick_enhance_preview_image',
    ):
        assert removed not in quick_enhance_rule
    assert 'height: self.texture_size[1] + dp(12)' in quick_enhance_rule


def test_every_post_processing_panel_is_scroll_wrapped():
    """Guard against controls clipping at the minimum application window size."""
    content = KV_PATH.read_text(encoding='utf-8')
    lines = _rule_block(content, '<PostProcessingAccordion>:').splitlines()
    item_indices = [i for i, ln in enumerate(lines) if ln.strip() == 'AccordionItem:']
    assert len(item_indices) >= 6, 'expected the six post-processing accordion items'

    for idx in item_indices:
        item_indent = _indent(lines[idx])
        first_child_widget = None
        for ln in lines[idx + 1 :]:
            stripped = ln.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if _indent(ln) <= item_indent:
                break
            if re.match(r'^[A-Z]\w*:$', stripped):
                first_child_widget = stripped[:-1]
                break
        assert first_child_widget == 'PostProcessingPanelScroll'


def test_panel_rules_declare_their_own_minimum_height():
    content = KV_PATH.read_text(encoding='utf-8')
    for header in PANEL_RULE_HEADERS:
        block = _rule_block(content, header)
        assert re.search(r'size_hint_y:\s*None', block), f'{header} must declare size_hint_y: None'
        assert re.search(r'height:\s*self\.minimum_height', block), (
            f'{header} must declare height: self.minimum_height'
        )


def test_quick_enhance_disclaimer_is_not_rendered_in_the_panel():
    kv_rule = _rule_block(KV_PATH.read_text(encoding='utf-8'), '<QuickEnhanceControls>:')
    assert 'warning_text' not in kv_rule

    py_src = (Path(__file__).resolve().parents[1] / 'ui' / 'post_processing.py').read_text(
        encoding='utf-8'
    )
    assert 'QUANTITATIVE_USE_WARNING' not in py_src
