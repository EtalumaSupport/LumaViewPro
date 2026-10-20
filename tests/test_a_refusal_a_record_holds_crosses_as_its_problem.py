# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A refusal a record holds crosses a wire as a refused call's problem carries it.

The status record holds the refusal a run would get now, and a client reads
it as it reads a refused call: ``kind``, ``reason``, ``title``, ``detail``
and ``remedy``, beside the fields the refusal's type publishes. Those are
the problem's keys less the three only an answer has (``type``, ``status``,
``instance``), composed once (``wire_encoding.outcome_fields``) for both. In
the same process the record keeps the exception itself, and the OpenAPI
description names a refusal's model where a type says one is sent.
"""

from __future__ import annotations

import pathlib

from modules import wire_encoding
from modules.exceptions import ArgumentRefusedError, ProtocolRunRefusedError, Remedy
from rest.problems import answered_by_member
from rest.routes import RefusalAnswer, answer_type


def _encoded(value: object) -> object:
    def refuse(v: object) -> object:
        raise AssertionError(f'{v!r} is no refusal')

    return wire_encoding.encode(value, live_folder=pathlib.Path(), handle=refuse, job=refuse)


def _problem_less_answer_keys(exception: Exception) -> dict:
    body = dict(answered_by_member(exception, '00000000-0000-0000-0000-000000000000').body)
    for key in ('type', 'status', 'instance'):
        del body[key]
    return body


def test_a_held_refusal_carries_its_problems_keys():
    refusal = ProtocolRunRefusedError(
        'files_writing_stalled',
        "A protocol's files have stopped writing.",
        remedy=Remedy('recover_stalled_files', 'Recover', 'Wait'),
    )

    sent = _encoded(refusal)

    assert sent == _problem_less_answer_keys(refusal)
    assert sent['kind'] == 'refusal'
    assert sent['reason'] == 'files_writing_stalled'
    assert sent['detail'] == "A protocol's files have stopped writing."
    assert sent['remedy'] == {
        'member': 'recover_stalled_files',
        'confirm_text': 'Recover',
        'cancel_text': 'Wait',
    }


def test_a_held_refusal_carries_the_fields_its_type_publishes():
    refusal = ArgumentRefusedError('not_a_number', argument='illumination_ma', value='bright')

    sent = _encoded(refusal)

    assert sent == _problem_less_answer_keys(refusal)
    assert sent['argument'] == 'illumination_ma'


def test_in_the_same_process_the_record_keeps_the_exception():
    refusal = ProtocolRunRefusedError('already_running', 'A run is in progress.')

    assert wire_encoding.plain({'run_refused': refusal}) == {'run_refused': refusal}


def test_the_description_names_a_refusals_model():
    alternatives = wire_encoding.outbound(
        'ProtocolRunRefusedError | None',
        wire_encoding.project_classes(),
        wire_encoding.project_aliases(),
    )

    assert [a.form for a in alternatives] == [wire_encoding.REFUSAL, wire_encoding.SCALAR]
    assert answer_type(alternatives[:1]) is RefusalAnswer
