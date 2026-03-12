# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""
Tests for API authentication module (modules/api_auth.py).
"""

import pytest
from modules.api_auth import (
    generate_api_key,
    validate_api_key,
    is_auth_required,
    check_auth,
    ensure_api_key_for_non_localhost,
)


class TestGenerateApiKey:
    """Verify API key generation."""

    def test_returns_hex_string(self):
        key = generate_api_key()
        assert isinstance(key, str)
        int(key, 16)  # Should not raise — valid hex

    def test_correct_length(self):
        key = generate_api_key()
        assert len(key) == 64  # 32 bytes = 64 hex chars

    def test_unique_each_call(self):
        keys = {generate_api_key() for _ in range(10)}
        assert len(keys) == 10  # All unique


class TestValidateApiKey:
    """Verify constant-time key comparison."""

    def test_matching_keys(self):
        key = generate_api_key()
        assert validate_api_key(key, key) is True

    def test_mismatched_keys(self):
        assert validate_api_key('abc', 'def') is False

    def test_empty_provided_key(self):
        assert validate_api_key('', 'stored_key') is False

    def test_empty_stored_key(self):
        assert validate_api_key('provided_key', '') is False

    def test_none_keys(self):
        assert validate_api_key(None, 'stored') is False
        assert validate_api_key('provided', None) is False


class TestIsAuthRequired:
    """Verify auth requirement logic."""

    def test_localhost_no_key(self):
        settings = {'rest_api': {'host': '127.0.0.1', 'api_key': None}}
        assert is_auth_required(settings) is False

    def test_localhost_with_key(self):
        settings = {'rest_api': {'host': '127.0.0.1', 'api_key': 'abc123'}}
        assert is_auth_required(settings) is True

    def test_non_localhost_no_key(self):
        settings = {'rest_api': {'host': '0.0.0.0', 'api_key': None}}
        assert is_auth_required(settings) is True

    def test_non_localhost_with_key(self):
        settings = {'rest_api': {'host': '0.0.0.0', 'api_key': 'abc123'}}
        assert is_auth_required(settings) is True

    def test_missing_rest_api_section(self):
        assert is_auth_required({}) is False

    def test_ipv6_localhost(self):
        settings = {'rest_api': {'host': '::1', 'api_key': None}}
        assert is_auth_required(settings) is False


class TestCheckAuth:
    """Verify request authorization."""

    def test_localhost_no_key_allows_all(self):
        settings = {'rest_api': {'host': '127.0.0.1', 'api_key': None}}
        ok, msg = check_auth(settings, None)
        assert ok is True

    def test_valid_key_accepted(self):
        key = generate_api_key()
        settings = {'rest_api': {'host': '127.0.0.1', 'api_key': key}}
        ok, msg = check_auth(settings, key)
        assert ok is True

    def test_invalid_key_rejected(self):
        settings = {'rest_api': {'host': '127.0.0.1', 'api_key': 'correct'}}
        ok, msg = check_auth(settings, 'wrong')
        assert ok is False
        assert 'Invalid' in msg

    def test_missing_key_rejected(self):
        settings = {'rest_api': {'host': '127.0.0.1', 'api_key': 'set'}}
        ok, msg = check_auth(settings, None)
        assert ok is False
        assert 'required' in msg

    def test_non_localhost_no_key_rejects(self):
        settings = {'rest_api': {'host': '0.0.0.0', 'api_key': None}}
        ok, msg = check_auth(settings, None)
        assert ok is False
        assert 'non-localhost' in msg


class TestEnsureApiKeyForNonLocalhost:
    """Verify auto-generation of API keys."""

    def test_generates_key_for_non_localhost(self):
        settings = {'rest_api': {'host': '0.0.0.0', 'api_key': None}}
        key = ensure_api_key_for_non_localhost(settings)
        assert key is not None
        assert len(key) == 64
        assert settings['rest_api']['api_key'] == key

    def test_no_key_for_localhost(self):
        settings = {'rest_api': {'host': '127.0.0.1', 'api_key': None}}
        key = ensure_api_key_for_non_localhost(settings)
        assert key is None

    def test_no_overwrite_existing_key(self):
        settings = {'rest_api': {'host': '0.0.0.0', 'api_key': 'existing'}}
        key = ensure_api_key_for_non_localhost(settings)
        assert key is None
        assert settings['rest_api']['api_key'] == 'existing'
