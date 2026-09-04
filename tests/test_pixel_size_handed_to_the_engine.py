"""Pixel size is a value the engine is handed, never one it looks up.

Every image scale the application writes -- the PhysicalSizeX in an OME-TIFF,
the scale bar burned into a live frame, the tile spacing a protocol is laid out
with -- derives from two numbers the scope's capabilities hold: the sensor's
pixel pitch and the tube lens focal length. The resolver that combined them
with an objective's focal length used to find those two numbers by reading the
GUI's application context. Headless, the context is unset, so the resolver
answered None: every headless image was written with no scale, and a tiled
protocol could not be built at all.

Now the resolver takes the capabilities as a required argument. The producers
that hold a scope pass its capabilities; the tiling chain is handed the same
record by the caller that owns the scope; the GUI's readouts go through a
GUI-side getter that reads the live scope. Nothing below the GUI reaches for a
process-wide store to learn a scale.
"""

import ast
import pathlib
from types import SimpleNamespace

import pytest

import modules.app_context as _app_ctx
import modules.common_utils as common_utils
import modules.image_save as image_save
from modules.tiling_config import TilingConfig
from tests.ast_seams import parse_module

REPO = pathlib.Path(__file__).resolve().parents[1]
TILING_CONFIGS = REPO / 'data' / 'tiling.json'


@pytest.fixture
def no_context(monkeypatch):
    """The headless condition: nothing has published an application context."""
    monkeypatch.setattr(_app_ctx, 'ctx', None)


class TestAHeadlessImageCarriesItsScale:
    def test_metadata_from_a_scope_that_knows_its_optics(self, sim_scope, no_context):
        """The defect itself: a scope that can report its optics wrote images
        with no scale whenever no GUI context existed in the process."""
        from modules.labware_loader import WellPlateLoader

        sim_scope.runtime_state.set_objective('20x Oly')
        sim_scope.runtime_state.set_labware(WellPlateLoader().get_plate('96 well microplate'))
        sim_scope.runtime_state.set_stage_offset({'x': 0.0, 'y': 0.0})
        objective = sim_scope.runtime_state.get_current_objective()
        caps = sim_scope.capabilities
        expected = (
            caps.pixel_size_um
            / (caps.lens_focal_length_mm / objective['focal_length'])
            * sim_scope.imaging._binning_size
        )

        metadata = image_save.generate_image_metadata(sim_scope, channel='BF', x=0, y=0, z=0)

        assert metadata['pixel_size_um'] == pytest.approx(expected, abs=1e-4), metadata.get(
            'pixel_size_um'
        )


class TestTheResolverIsHandedTheOptics:
    def test_no_capabilities_no_answer(self):
        """The argument is required: a resolver that could answer without it
        would answer from a store the caller never saw."""
        with pytest.raises(TypeError, match='capabilities'):
            common_utils.get_pixel_size(focal_length=9.0, binning_size=1)

    def test_handed_optics_it_computes_with_no_context(self, scale_capabilities, no_context):
        assert common_utils.get_pixel_size(
            focal_length=9.0, binning_size=1, capabilities=scale_capabilities
        ) == pytest.approx(2.0 / (47.8 / 9.0))

    def test_a_scope_that_cannot_report_its_optics_yields_none(self, no_context):
        """Preserved: no invented scale. A scope with unknown optics still
        answers None, and its callers still degrade honestly."""
        caps = SimpleNamespace(pixel_size_um=None, lens_focal_length_mm=None)

        assert (
            common_utils.get_pixel_size(focal_length=9.0, binning_size=1, capabilities=caps) is None
        )
        assert (
            common_utils.get_field_of_view(
                focal_length=9.0,
                frame_size={'width': 100, 'height': 100},
                binning_size=1,
                capabilities=caps,
            )
            is None
        )


class TestTheTilingChainIsHandedItsScale:
    def test_a_grid_is_laid_out_with_no_context_in_the_process(
        self, scale_capabilities, no_context
    ):
        tiles = TilingConfig(tiling_configs_file_loc=TILING_CONFIGS).get_tile_centers(
            config_label='2x2',
            focal_length=9.0,
            frame_size={'width': 1900, 'height': 1900},
            fill_factor=1.0,
            binning_size=1,
            capabilities=scale_capabilities,
        )

        assert len(tiles) == 4
        assert tiles['A2']['x'] != tiles['A1']['x'], 'a real grid must have real spacing'

    def test_a_tiled_protocol_is_built_with_no_context_in_the_process(
        self, scale_capabilities, no_context
    ):
        """The chain above the layout: a protocol is a data object handed a
        scale, and building one headless needs nothing in the process."""
        from tests.test_tiling_overlap import _make_protocol_from_config

        protocol = _make_protocol_from_config(overlap_percent=0, capabilities=scale_capabilities)

        assert len(protocol.steps()) == 4


class TestTheHelperFamilyReachesNoContext:
    def test_common_utils_imports_no_context(self):
        """Structural, at every scope: the two function-local imports inside
        the resolver family were the module's only reach for the context."""
        tree = parse_module('modules/common_utils.py')
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Import)
                and any(a.name == 'modules.app_context' for a in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and (
                    node.module == 'modules.app_context'
                    or (
                        node.module == 'modules'
                        and any(a.name == 'app_context' for a in node.names)
                    )
                )
            )
        ]
        assert not offenders, (
            f'modules/common_utils.py imports the application context at {offenders}'
        )
