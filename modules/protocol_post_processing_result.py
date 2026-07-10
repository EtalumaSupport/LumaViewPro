"""Typed result for one post-processing group.

The base-class ``load_folder`` loop needs the output depth of each generated
artifact so its completion line can report whether the input depth round-tripped
through the operation. That value used to travel in a free-form ``metadata`` dict
under a string key; every post-processor built the dict by hand and none placed
the key where the consumer read it, so the consumer raised ``KeyError`` on the
first successful group of every type. This type removes that whole failure mode:
a successful result cannot be constructed without its output depth, so the
consumer reads a typed attribute that is guaranteed present.
"""

import pathlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PostProcResult:
    """The outcome of one post-processing group.

    A success is built only through :meth:`ok`, which requires
    ``significant_bits``; a failure through :meth:`failed`, which carries none
    (no artifact was produced). The pixel array is deliberately absent: the base
    class never reads the produced image back (each subclass writes its own file),
    so a result that dropped the buffer after the write is fully sufficient and
    does not retain multi-gigabyte stacks.
    """

    status: bool
    significant_bits: int | None = None
    error: str | None = None
    record_metadata: dict = field(default_factory=dict)
    actual_output_file_loc: pathlib.Path | None = None

    @classmethod
    def ok(
        cls,
        *,
        significant_bits: int,
        record_metadata: dict | None = None,
        actual_output_file_loc=None,
    ) -> 'PostProcResult':
        """Build a successful result. ``significant_bits`` is required so a
        depth-less success is unconstructible."""
        if significant_bits is None:
            raise ValueError('PostProcResult.ok requires significant_bits')
        return cls(
            status=True,
            significant_bits=int(significant_bits),
            record_metadata=record_metadata or {},
            actual_output_file_loc=actual_output_file_loc,
        )

    @classmethod
    def failed(cls, error: str, *, record_metadata: dict | None = None) -> 'PostProcResult':
        """Build a failed result. No artifact was produced, so no depth."""
        return cls(status=False, error=error, record_metadata=record_metadata or {})

    @classmethod
    def from_group_result(cls, result: dict) -> 'PostProcResult':
        """Adapt a subclass ``_group_algorithm`` inner dict at the load_folder
        boundary. Each subclass still assembles an internal dict (its inner
        writer is shared with standalone save paths that hand back the image
        array a result has no field for); this converts that dict into the typed
        result the base class consumes. A success dict missing ``significant_bits``
        raises here at the one boundary, not deep in the base-class loop."""
        if not result['status']:
            return cls.failed(
                result.get('error') or 'post-processing failed',
                record_metadata=result.get('metadata', {}),
            )
        return cls.ok(
            significant_bits=result['significant_bits'],
            record_metadata=result.get('metadata', {}),
            actual_output_file_loc=result.get('actual_output_file_loc'),
        )
