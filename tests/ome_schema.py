# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The OME 2016-06 schema gate for the tests: a real XSD validator.

``ome_types.from_xml`` builds a model from the description and checks no
element order and no ID reference, so a misordered Instrument or an
ObjectiveSettings naming an absent Objective passes it. The schema itself
(ome-types bundles ``ome-2016-06.xsd``) checks both.
"""

import functools
import pathlib
import xml.etree.ElementTree as ET

import ome_types
import tifffile as tf
import xmlschema

OME = 'http://www.openmicroscopy.org/Schemas/OME/2016-06'


@functools.cache
def _schema() -> xmlschema.XMLSchema:
    return xmlschema.XMLSchema(str(pathlib.Path(ome_types.__file__).parent / 'ome-2016-06.xsd'))


def schema_errors(description: str) -> list[str]:
    """Every way ``description`` breaks the OME 2016-06 schema, empty when valid."""
    return [str(error).splitlines()[0] for error in _schema().iter_errors(description)]


def valid_description(path: pathlib.Path) -> ET.Element:
    """The OME description of the file at ``path``, parsed, after asserting
    it is valid against the schema."""
    description = tf.tiffcomment(path)
    assert schema_errors(description) == []
    return ET.fromstring(description)


def q(tag: str) -> str:
    """``tag`` in the OME namespace, for ElementTree's find."""
    return f'{{{OME}}}{tag}'
