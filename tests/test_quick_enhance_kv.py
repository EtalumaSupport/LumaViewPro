"""Quick Enhance KV must parse before the application constructs a window."""

from pathlib import Path
import subprocess
import sys


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


def test_quick_enhance_documentation_explains_the_short_mode_names():
    documentation = (Path(__file__).resolve().parents[1] / 'docs' / 'QUICK_ENHANCE.md').read_text(
        encoding='utf-8'
    )

    assert '## Modes' in documentation
    assert 'Auto (Recommended)' in documentation
    assert 'Brightfield / Phase' in documentation
    assert 'Contrast Only' in documentation
