# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""How a post-processed image was made: each build in order, with its settings.

A derived image's row in the post-processing record carries its derivation
as text: the builds that made it, oldest first, each with every setting
that decided its pixels -- ``ZProject(method=max)>Stitched(mode=quality)``.
Its kinds (stitched, z-projected, ...) are read from that text, and its
file name's post tokens are rendered from it. Two images of the same
source made two ways therefore never share a build's group or a name, and
a rebuild with new settings is never taken for one already made.

A raw capture's derivation is empty. A row recorded before derivations
were kept knows its kinds but not their settings or order; each of its
steps is unknown (``ZProject(?)``) and renders the kind's bare token.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping

from modules.common_utils import POST_TOKEN_VIDEO, PostFunction
from modules.image_mode import OutputFormat
from modules.stitch_mode import StitchMode

_STEP_SEPARATOR = '>'
_SETTING_SEPARATOR = ','
_UNKNOWN = '?'
_RESERVED = frozenset('>,=()')

# Composite and video settings run to several values (a threshold and a
# colour per channel), too many to spell out in a file name; the name
# carries a short code of them, and the record keeps them whole.
_CODE_LENGTH = 6

STITCH_FAST_PREVIEW_TOKEN = 'FastPreview'


@dataclasses.dataclass(frozen=True)
class Step:
    """One build in a derivation and the settings that decided its pixels.

    Attributes:
        kind: The build.
        settings: Each setting as (name, value), in the build's own order;
            None when the step was recorded before settings were kept.
    """

    kind: PostFunction
    settings: tuple[tuple[str, str], ...] | None = ()

    def __post_init__(self):
        for name, value in self.settings or ():
            if _RESERVED & set(name + value):
                raise ValueError(
                    f'A {self.kind.value} setting cannot hold any of {"".join(sorted(_RESERVED))}: '
                    f'{name}={value}'
                )

    def text(self) -> str:
        if self.settings is None:
            return f'{self.kind.value}({_UNKNOWN})'
        inner = _SETTING_SEPARATOR.join(f'{name}={value}' for name, value in self.settings)
        return f'{self.kind.value}({inner})'

    def setting(self, name: str) -> str | None:
        """The value of setting ``name``, or None when it is unknown."""
        return None if self.settings is None else dict(self.settings)[name]

    def code(self) -> str:
        """A short code that changes whenever any of this step's settings do."""
        return hashlib.sha256(self.text().encode()).hexdigest()[:_CODE_LENGTH]


def extended(derivation: str, step: Step) -> str:
    """The derivation of an output ``step`` made from images derived as ``derivation``."""
    return step.text() if derivation == '' else f'{derivation}{_STEP_SEPARATOR}{step.text()}'


def steps(derivation: str) -> tuple[Step, ...]:
    """The steps of ``derivation``, oldest first.

    Raises:
        ValueError: the text is not a derivation.
    """
    if derivation == '':
        return ()
    parsed = []
    for text in derivation.split(_STEP_SEPARATOR):
        kind, opened, rest = text.partition('(')
        if not opened or not rest.endswith(')'):
            raise ValueError(f'{text!r} is not a derivation step')
        inner = rest[:-1]
        if inner == _UNKNOWN:
            parsed.append(Step(PostFunction(kind), None))
            continue
        settings = tuple(
            tuple(pair.split('=', 1)) for pair in inner.split(_SETTING_SEPARATOR) if pair != ''
        )
        parsed.append(Step(PostFunction(kind), settings))
    return tuple(parsed)


def kinds(derivation: str) -> dict[str, bool]:
    """Each kind's column value for an image derived as ``derivation``."""
    made = {step.kind for step in steps(derivation)}
    return {kind.value: kind in made for kind in PostFunction}


def made_unrecorded(derivation: str) -> bool:
    """Whether a step of ``derivation`` was recorded before settings were kept."""
    return any(step.settings is None for step in steps(derivation))


def raw_columns() -> dict[str, object]:
    """A raw capture's row columns for how it was made: no derivation, no kind."""
    return {'Derivation': '', **kinds('')}


def from_kinds(flags: Mapping[str, bool]) -> str:
    """The derivation of a row recorded before derivations were kept.

    Its kinds are known; their settings and order are not, so each step is
    unknown, in the kinds' declared order, and two rows of the same kinds
    read as the same derivation.
    """
    return _STEP_SEPARATOR.join(
        Step(kind, None).text() for kind in PostFunction if bool(flags[kind.value])
    )


def name_tokens(derivation: str) -> tuple[str, ...]:
    """The post tokens of a file name for an image derived as ``derivation``."""
    tokens: list[str] = []
    for step in steps(derivation):
        known = step.settings is not None
        if step.kind is PostFunction.ZPROJECT:
            tokens.append(f'zproj_{step.setting("method")}' if known else 'zproj')
        elif step.kind is PostFunction.STITCHED:
            tokens.append('stitched')
            if known and step.setting('mode') == StitchMode.FAST_PREVIEW:
                tokens.append(STITCH_FAST_PREVIEW_TOKEN)
        elif step.kind is PostFunction.COMPOSITE:
            # The name's channel is 'Composite'; only its settings need a token.
            if known:
                tokens.append(step.code())
        elif step.kind is PostFunction.VIDEO:
            tokens.append(POST_TOKEN_VIDEO)
            if known:
                tokens.append(step.code())
        elif step.kind is PostFunction.HYPERSTACK:
            tokens.append('hyperstack')
    return tuple(tokens)


def color_text(color: tuple[float, float, float]) -> str:
    """A colour as a setting's value: ``#rrggbb``."""
    return '#' + ''.join(f'{round(channel * 255):02x}' for channel in color)


def format_settings(output_format: OutputFormat) -> tuple[tuple[str, str], ...]:
    """A stitch's or a z-projection's format setting: stated for OME-TIFF only.

    A plain build states none, as every one built before OME builds were
    made, so a folder built then is never built again; an OME build states
    its format, so it never shares a group or a name with a plain one of
    the same images.
    """
    return (('format', output_format),) if output_format == OutputFormat.OME_TIFF else ()
