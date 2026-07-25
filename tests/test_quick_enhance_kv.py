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


def test_quick_enhance_kv_uses_short_labels_without_custom_controls():
    kv_path = Path(__file__).resolve().parents[1] / 'ui' / 'lumaviewpro.kv'
    content = kv_path.read_text(encoding='utf-8')
    quick_enhance_rule = content[
        content.index('<QuickEnhanceControls>:') : content.index('<QuickEnhanceUtilityButton@')
    ]

    assert (
        "values: ('Auto (Recommended)', 'Brightfield / Phase', 'Contrast Only')"
        in quick_enhance_rule
    )
    assert "text: 'Choose Image'" in quick_enhance_rule
    assert "text: 'Choose Folder'" in quick_enhance_rule
    assert "text: 'Show Original' if root.show_after else 'Show Enhanced'" in quick_enhance_rule
    assert "text: 'Update Preview'" in quick_enhance_rule
    assert "text: 'Save Folder' if root.source_folder else 'Save Image'" in quick_enhance_rule
    assert "text: 'Run'" in quick_enhance_rule
    assert 'disabled: not root.input_ready' in quick_enhance_rule
    assert 'Custom' not in quick_enhance_rule


def test_every_post_processing_panel_is_scroll_wrapped():
    """Structural guard for panel clipping: the accordion viewport is
    shorter than several panels at the 1024x600 minimum window, so every
    AccordionItem's first child widget must be the shared scroll wrapper.
    A future panel added without it fails here instead of silently
    clipping its lower controls."""
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
        assert first_child_widget == 'PostProcessingPanelScroll', (
            f'AccordionItem at kv line offset {idx} hosts {first_child_widget!r} '
            f'directly; every panel must sit inside PostProcessingPanelScroll '
            f'or its lower controls clip at the minimum window size'
        )


def test_panel_rules_declare_their_own_minimum_height():
    """The wrapper only scrolls a panel that reports its content height:
    each named panel rule must be self-describing (size_hint_y None +
    minimum_height), so any future instantiation scrolls correctly."""
    content = KV_PATH.read_text(encoding='utf-8')
    for header in PANEL_RULE_HEADERS:
        block = _rule_block(content, header)
        assert re.search(r'size_hint_y:\s*None', block), f'{header} must declare size_hint_y: None'
        assert re.search(r'height:\s*self\.minimum_height', block), (
            f'{header} must declare height: self.minimum_height'
        )


def test_quick_enhance_disclaimer_is_single_homed():
    """The disclaimer's string store is QUANTITATIVE_USE_WARNING; the kv
    caption REFLECTS it (root.warning_text) rather than duplicating the
    words, and no status-line append re-introduces a second rendering.
    Also guards the literal backslash-n regression: the old duplicated
    caption text carried an escaped newline that rendered as characters."""
    kv_rule = _rule_block(KV_PATH.read_text(encoding='utf-8'), '<QuickEnhanceControls>:')
    assert 'text: root.warning_text' in kv_rule, 'caption must reflect the warning store'
    assert '\\n' not in kv_rule, 'no escaped-newline literals in the Quick Enhance rule'
    assert 'visual inspection' not in kv_rule, 'disclaimer text must not be duplicated in kv'

    py_src = (Path(__file__).resolve().parents[1] / 'ui' / 'post_processing.py').read_text(
        encoding='utf-8'
    )
    occurrences = py_src.count('QUANTITATIVE_USE_WARNING')
    assert occurrences == 2, (
        f'QUANTITATIVE_USE_WARNING must appear exactly twice in ui/post_processing.py '
        f'(import + warning_text attribute); found {occurrences} -- a status-text '
        f'append duplicating the caption has likely been re-introduced'
    )


def test_quick_enhance_documentation_explains_the_short_mode_names():
    documentation = (Path(__file__).resolve().parents[1] / 'docs' / 'QUICK_ENHANCE.md').read_text(
        encoding='utf-8'
    )

    assert '## Modes' in documentation
    assert 'Auto (Recommended)' in documentation
    assert 'Brightfield / Phase' in documentation
    assert 'Contrast Only' in documentation
