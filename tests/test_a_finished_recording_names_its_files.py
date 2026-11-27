# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""A finished manual recording names every file it wrote (``RecordingFinished.files``).

An MP4 is written into the shared ``Manual/`` folder beside every other
recording's, so a caller that cannot read the names from the answer cannot
tell which file is its own. Each format's recording, on the simulated
scope, ends on its own at its 1 s limit; its answer names each file it
wrote under the live folder, once, and no other.
"""

import pytest

from _lumascope.modules.image_mode import RecordingFormat


def _recording_in(recording_format):
    return {'video_recording_format': recording_format.value, 'video': {'max_duration_s': 1}}


# parametrize-justified: each format is its own finish path (one MP4; frames
# and a manifest; frames, a manifest and a hyperstack), each its own session.
@pytest.mark.parametrize(
    'sim_session',
    [_recording_in(f) for f in RecordingFormat],
    ids=[f.value for f in RecordingFormat],
    indirect=True,
)
def test_a_finished_recording_names_each_file_it_wrote(sim_session, tmp_path):
    finished = sim_session.manual_recording.start().result(timeout=60)

    on_disk = {path for path in tmp_path.rglob('*') if path.is_file()}
    assert on_disk, 'the recording wrote nothing'
    assert sorted(finished.files) == sorted(on_disk)
