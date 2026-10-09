# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""The REST server: the declared API over HTTP, for a client on the same computer.

A host beside ``ui/``: it reaches the scope only through the Session and
the members marked for the wire (``modules.api_surface``), in the forms
``modules.wire_encoding`` declares. Nothing imports this package, or its
dependencies (FastAPI, uvicorn, pydantic), unless the server starts.
"""
