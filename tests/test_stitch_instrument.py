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
    textured control is proposed on every seam. The scenes at one overlap
    are cut at one stage pattern, so they are each other's other specimens
    for check (c), which a label from phase alone must pass.
    """
    wrong = []
    for overlap in (0.1, 0.2):
        built = {scene: si.synthetic_set(scene, overlap) for scene in si.SCENES}
        checked = si.across_specimens(
            {
                scene: {seam.key: label for seam, label in si.propose_labels(tiles).items()}
                for scene, (tiles, _truth) in built.items()
            }
        )
        for scene, (_tiles, truth) in built.items():
            proposed = {seam: checked[scene][seam.key] for seam in truth}
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


def test_the_fit_keeps_only_seams_that_fit_however_many_it_must_set_aside():
    """Check (b): every relation the fit keeps is within its bound, three set aside if it takes three.

    Three bad seams on a 3x3, on both axes and of different sizes. Which
    seams are set aside is the fit's greedy choice and is not pinned: a set
    of bad seams can have another explanation as small, which least squares
    cannot tell apart.
    """
    stage = [(gx * 900.0, gy * 900.0) for gy in range(3) for gx in range(3)]
    tiles = [{'x_px': x, 'y_px': y} for x, y in stage]
    errors = {'0->1': (6.0, 0.0), '4->5': (0.0, 9.0), '3->6': (-7.0, 0.0)}
    relations = {
        seam: (
            5.0 + errors.get(seam.key, (0.0, 0.0))[0],
            -4.0 + errors.get(seam.key, (0.0, 0.0))[1],
        )
        for seam in si.seams(tiles)
    }

    kept, set_aside = si.fit_check(len(stage), 0, stage, relations)

    _, residuals = si.fit_positions(len(stage), 0, stage, kept)
    assert max(residuals.values()) <= si.FIT_RESIDUAL_PX, residuals
    assert len(set_aside) == 3, sorted(s.key for s in set_aside)
    assert set(kept) | set(set_aside) == set(relations)


def test_a_label_from_phase_alone_stands_only_where_the_other_specimens_agree():
    """Check (c) is all a phase-alone label stands on: unchecked, or off the others, it is unknown.

    A label both proposers agreed on is left as it is when there are too
    few other specimens to check it against; one from phase alone is not.
    """
    move = '0->1'

    def phase_alone(dx):
        return si.Label('offset', dx, -4.0, why='phase alone, NCC (14.0, -4.0)', phase_alone=True)

    def agreed(dx):
        return si.Label('offset', dx, -4.0)

    one_other = si.across_specimens(
        {'defocused': {move: phase_alone(9.0)}, 'plate': {move: agreed(9.2)}}
    )
    assert one_other['defocused'][move].kind == 'unknown'
    assert 'fewer than two other specimens' in one_other['defocused'][move].why
    assert one_other['plate'][move] == agreed(9.2), 'an agreed label with too few others stands'

    no_witness = si.across_specimens(
        {
            'defocused': {move: phase_alone(9.0)},
            'plate': {move: agreed(9.2)},
            'ruling': {move: phase_alone(9.1)},
        }
    )
    assert no_witness['defocused'][move].kind == 'unknown', 'a phase-alone label is no witness'

    others = {'plate': {move: agreed(9.2)}, 'usaf': {move: agreed(8.9)}}
    assert si.across_specimens({'defocused': {move: phase_alone(9.4)}, **others})['defocused'][
        move
    ] == phase_alone(9.4)
    off = si.across_specimens({'defocused': {move: phase_alone(13.5)}, **others})['defocused'][move]
    assert off.kind == 'unknown'
    assert "against the other specimens'" in off.why

    # 1.5 px from the others' median (9.05): inside the 2 px an agreed label
    # is held to, outside the 1 px a phase-alone label is.
    assert si.across_specimens({'defocused': {move: agreed(10.55)}, **others})['defocused'][
        move
    ] == agreed(10.55)
    assert (
        si.across_specimens({'defocused': {move: phase_alone(10.55)}, **others})['defocused'][
            move
        ].kind
        == 'unknown'
    )
