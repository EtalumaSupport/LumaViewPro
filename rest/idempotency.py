# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A retried request runs once: the first answers of requests sent with ``Idempotency-Key``.

A client that lost the answer to a POST -- a dropped connection, a timeout
-- cannot tell whether the call ran. Sent again with the same
``Idempotency-Key``, it runs nothing and is answered what the first request
was answered. A key is held from the moment its first request is admitted,
so a retry that arrives while the first still runs waits for that answer.

The answers are held apart from jobs: a request's answer is what a retry
is owed, and a 200 is no job. They are held for ``KEY_AGE``, at most
``KEY_LIMIT`` of them, the oldest let go first, so a client that keys every
request lets go only its own oldest keys, never another client's job.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import datetime
import hashlib
import json

import fastapi

from rest import problems

KEY_LIMIT = 10_000
KEY_AGE = datetime.timedelta(hours=24)
# Headers each answer makes for itself, which a replay does not repeat.
_OWN_HEADERS = frozenset({'content-length', 'x-request-id'})


@dataclasses.dataclass(frozen=True)
class Recorded:
    """An answer as its request was given it, to be given again.

    Attributes:
        status: The HTTP status.
        body: The body, byte for byte.
        headers: The headers the answer carried, but those each answer makes.
    """

    status: int
    body: bytes
    headers: dict[str, str]

    @classmethod
    def of(cls, response: fastapi.Response) -> Recorded:
        """The answer *response* is."""
        headers = {k: v for k, v in response.headers.items() if k.lower() not in _OWN_HEADERS}
        return cls(response.status_code, bytes(response.body), headers)

    def replayed(self) -> fastapi.Response:
        """The answer again, marked as a replay."""
        return fastapi.Response(
            self.body,
            status_code=self.status,
            headers={**self.headers, 'Idempotent-Replayed': 'true'},
        )


def digest(member: str, body: dict[str, object]) -> str:
    """What a request asks, whatever order its body's keys came in."""
    asked = json.dumps({'member': member, 'body': body}, sort_keys=True, default=str)
    return hashlib.sha256(asked.encode()).hexdigest()


@dataclasses.dataclass
class _Held:
    digest: str
    # The first request's answer, or what it raised instead of answering.
    answer: asyncio.Future[Recorded | BaseException]
    claimed: datetime.datetime


class IdempotencyKeys:
    """The keys clients have sent, each with its first request's answer.

    Used on the server's loop only, where every request is admitted.
    """

    def __init__(self) -> None:
        self._held: collections.OrderedDict[str, _Held] = collections.OrderedDict()

    def claim(self, key: str, asked: str) -> tuple[bool, asyncio.Future[Recorded | BaseException]]:
        """Claim *key* for a request asking *asked* (its ``digest``).

        Returns:
            Whether the key is this request's -- it runs, and settles the
            future with its answer -- and the future the first request's
            answer settles.

        Raises:
            ServerRefusedError: ``idempotency_key_reused``, the key was sent
                with another member or body.
        """
        self._prune()
        held = self._held.get(key)
        if held is not None:
            if held.digest != asked:
                raise problems.ServerRefusedError(
                    'idempotency_key_reused',
                    f'Idempotency-Key {key!r} was sent with another request: '
                    'a key names one request, sent again only to retry it.',
                )
            return False, held.answer
        held = _Held(asked, asyncio.get_running_loop().create_future(), problems.now())
        self._held[key] = held
        return True, held.answer

    def forget(self, key: str, answer: asyncio.Future[Recorded | BaseException]) -> None:
        """Let *key* go: its first request ended without an answer, so a retry runs it."""
        if self._held.get(key) is not None and self._held[key].answer is answer:
            del self._held[key]

    def _prune(self) -> None:
        oldest = problems.now() - KEY_AGE
        while self._held:
            key, held = next(iter(self._held.items()))
            if len(self._held) <= KEY_LIMIT and held.claimed >= oldest:
                return
            del self._held[key]
