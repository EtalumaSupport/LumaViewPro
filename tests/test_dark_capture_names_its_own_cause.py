"""A dark capture must say it was dark, not that the camera is gone.

Bench, 2026-09-15: a bright-field capture at 0.01 ms was correctly rejected by
the dark-floor guard -- 14 us of integration IS black -- and the user was told

    'camera inactive or not grabbing. Check the live view, then try again.'

while the camera took gain and exposure writes in that same millisecond and the
run held no disconnect of any kind. The operator went looking for a reconnect
fault that did not exist.

The guard is right and stays: issues #671 and #721 are the record of a near-black
frame being silently saved, and `tests/test_dark_floor_capture_guard.py` locks the
rejection itself. What was wrong is the CAUSE reported afterwards.

The failure ladder that reported it is gone: a capture that gets no frame now
raises its own reason, and a dark frame is not a failure at all -- it is
delivered, saved and marked. These lock that it stays delivered and marked.
"""


def test_the_dark_flag_is_cleared_per_capture():
    """A stale flag would misattribute the NEXT capture.

    The measurement sets the flag deep in the grab loop and the caller reads
    it after; without a reset at the top of each capture, one dark capture
    would mark every later frame dark.
    """
    import inspect

    from modules.lumascope_api.imaging import ImagingAPI

    src = inspect.getsource(ImagingAPI._capture_and_wait_impl)
    assert 'self._dark_saved = False' in src, (
        'capture_and_wait must clear the dark flag before each capture'
    )


def test_a_dark_frame_is_saved_and_never_refused():
    """The replacement tripwire.

    Its predecessor asserted the opposite -- that the dark-floor block
    still contained a ``return None`` -- on the reasoning that a silently
    saved black frame re-opened issues #671 and #721. That reasoning was
    retired deliberately: the stale pre-LED frame #671-B reported is
    owned by tracked illumination state and the retry, not by inspecting
    pixels, and refusing the frame destroyed real dark observations
    instead. The block must now carry no refusal at all, and must file
    the fact that lets callers tell a dark capture from a lit one.
    """
    import inspect

    from modules.lumascope_api.imaging import ImagingAPI

    src = inspect.getsource(ImagingAPI._get_image_impl)
    block = src.split('if dark_floor_check:', 1)[1].split('if verify_chunk_targets:', 1)[0]
    assert 'return None' not in block, (
        'a dark frame must be delivered; a refusal here destroys a real '
        'observation the operator can see on screen'
    )
    assert 'self._dark_saved = True' in block, (
        'the darkness must still be recorded, or no caller can tell a dark '
        'capture from a lit one without re-measuring pixels'
    )
