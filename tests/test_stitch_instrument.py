"""The stitch-registration instrument scores what it is given and labels only what it can.

The instrument (``tests/capability/stitch_instrument.py``) is the measure
#838 and #860's design is planned from, so its own two jobs are pinned
here: the scorer counts each outcome and names every seam it could not
score, and the label proposers never propose an offset the content does
not ask for.
"""

from tests.capability import stitch_instrument as si


def _seam(i):
    return si.Seam(i, i + 1, 900, 0)


def test_the_scorer_counts_every_outcome_and_names_every_seam_it_cannot_score():
    rule = si.Rule('accepts a score of 0.5 or more', 'quality', lambda a: a.score >= 0.5)
    answers = {
        _seam(0): si.Answer(5, -2, 0.9),  # accepted, at the label: right
        _seam(1): si.Answer(9, -2, 0.9),  # accepted, 4 px off: wrong
        _seam(2): si.Answer(5, -2, 0.1),  # declined where there is an offset: missed
        _seam(3): si.Answer(3, 3, 0.1),  # declined on nothing to match: declined right
        _seam(4): si.Answer(3, 3, 0.9),  # accepted on nothing to match: wrong
        _seam(5): si.Answer(5, -2, 0.9, border=True),  # unknown: never scored
    }
    labels = {
        _seam(0): si.Label('offset', 5.4, -2.3),
        _seam(1): si.Label('offset', 5.0, -2.0),
        _seam(2): si.Label('offset', 5.0, -2.0),
        _seam(3): si.Label('no_content'),
        _seam(4): si.Label('no_content'),
        _seam(5): si.Label('unknown', why='proposers disagree'),
    }

    score = si.score_seams(rule, answers, labels)

    assert score.counts == {'right': 1, 'wrong': 2, 'missed': 1, 'declined right': 1}
    assert score.unknown == ['5->6: proposers disagree']
    assert score.border == 0, 'an unknown seam is counted nowhere, its border included'


def test_the_proposers_never_label_an_offset_the_content_does_not_ask_for():
    """On every synthetic scene, each tile with its own noise: a proposed label is the construction's, within 1 px.

    The periodic ruling, the straight edge and the row pattern are where a
    wide search finds a confident wrong peak; a proposer there must decline
    (``unknown``), never propose. A featureless scene is never proposed. The
    textured control is proposed on every seam.
    """
    wrong = []
    for scene in si.SCENES:
        for overlap in (0.1, 0.2):
            tiles, truth = si.synthetic_set(scene, overlap)
            proposed = si.propose_labels(tiles)
            for seam, label in proposed.items():
                if label.kind != 'offset':
                    continue
                true = truth[seam]
                if true.kind != 'offset':
                    wrong.append(f'{scene} {overlap}: {seam.key} proposed on {true.kind}')
                elif max(abs(label.dx - true.dx), abs(label.dy - true.dy)) > si.RIGHT_WITHIN_PX:
                    wrong.append(
                        f'{scene} {overlap}: {seam.key} proposed ({label.dx:.2f}, {label.dy:.2f}), '
                        f'built ({true.dx:.2f}, {true.dy:.2f})'
                    )
            if scene == 'texture':
                assert all(lab.kind == 'offset' for lab in proposed.values()), (
                    f'texture {overlap}: {[(s.key, lab.why) for s, lab in proposed.items() if lab.kind != "offset"]}'
                )
    assert not wrong, '\n'.join(wrong)
