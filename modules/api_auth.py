# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
API authentication for the LumaViewPro REST API.

Provides API key generation, validation, and a middleware-style check
function. Keys are stored in the settings JSON under rest_api.api_key.

Security model:
  - When api_key is null/empty: no authentication required (localhost-only use)
  - When api_key is set: all API requests must include the key via
    X-API-Key header or ?api_key= query parameter
  - When binding to non-localhost: api_key MUST be set in the installation's
    settings file; until it is, every request is rejected
"""

import hmac
import logging
import secrets

logger = logging.getLogger('LVP.modules.api_auth')

# Key length in bytes (32 bytes = 256 bits, rendered as 64 hex chars)
_KEY_LENGTH_BYTES = 32


def generate_api_key() -> str:
    """Generate a cryptographically random API key.

    Returns:
        A 64-character hex string suitable for use as an API key.
    """
    return secrets.token_hex(_KEY_LENGTH_BYTES)


def validate_api_key(provided_key: str, stored_key: str) -> bool:
    """Constant-time comparison of provided key against stored key.

    Uses hmac.compare_digest to prevent timing attacks.

    Args:
        provided_key: The key from the API request.
        stored_key: The key from settings.

    Returns:
        True if the keys match, False otherwise.
    """
    if not provided_key or not stored_key:
        return False
    return hmac.compare_digest(provided_key, stored_key)


def is_auth_required(settings: dict) -> bool:
    """Check if API authentication is required based on settings.

    Auth is required when:
      - rest_api.api_key is set to a non-empty string, OR
      - rest_api.host is not localhost (127.0.0.1 or ::1)
    """
    rest_api = settings.get('rest_api', {})
    api_key = rest_api.get('api_key')
    host = rest_api.get('host', '127.0.0.1')

    # Non-localhost binding always requires auth
    if host not in ('127.0.0.1', '::1', 'localhost'):
        return True

    # Auth required if key is configured
    return bool(api_key)


def check_auth(settings: dict, provided_key: str | None) -> tuple[bool, str]:
    """Check if a request is authorized.

    Args:
        settings: The application settings dict.
        provided_key: The API key from the request (header or query param).

    Returns:
        (authorized, message) tuple. authorized is True if the request
        should proceed, False if it should be rejected.
    """
    rest_api = settings.get('rest_api', {})
    stored_key = rest_api.get('api_key')
    host = rest_api.get('host', '127.0.0.1')

    # No key configured and localhost-only: allow all requests
    if not stored_key and host in ('127.0.0.1', '::1', 'localhost'):
        return True, 'ok'

    # Non-localhost with no key configured: reject (misconfiguration)
    if not stored_key:
        logger.error(
            'REST API bound to %s without api_key configured -- rejecting request for safety', host
        )
        return False, 'API key required when binding to non-localhost'

    # Key configured: validate
    if not provided_key:
        return False, 'API key required (X-API-Key header or api_key query param)'

    if validate_api_key(provided_key, stored_key):
        return True, 'ok'

    logger.warning('Invalid API key attempt')
    return False, 'Invalid API key'
