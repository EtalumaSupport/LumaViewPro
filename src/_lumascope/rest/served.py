# Copyright (c) 2023-2026 Etaluma, Inc. MIT License. See LICENSE file.
"""What a host serves, as the one judge of the ``rest_api`` settings reads it.

The load leaves ``rest_api`` as the user wrote it, since the shipped value
is not always the narrower one (an empty client list admits every client),
and only a host that serves can answer a value it cannot use by not
serving. The server's builder and the application both read the section
here, so a host that builds the application itself gets the same reading.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import socket

from _lumascope.modules.exceptions import RestServerNotStartedError

_PORT_RANGE = (1024, 65535)
"""The REST server's port, inclusive: above the ones the system reserves."""

_Address = ipaddress.IPv4Address | ipaddress.IPv6Address
_Network = ipaddress.IPv4Network | ipaddress.IPv6Network


@dataclasses.dataclass(frozen=True)
class Served:
    """Where a host serves and whom it admits, as the judge read it from ``rest_api``."""

    address: _Address
    port: int
    # Empty admits every client that can reach the address.
    allowed_clients: tuple[_Network, ...]

    @property
    def family(self) -> socket.AddressFamily:
        """The socket family the address is bound with."""
        return socket.AF_INET6 if self.address.version == 6 else socket.AF_INET

    def admits(self, peer: str | None) -> bool:
        """Whether a client at *peer* is served: every client without an allowlist, else one in it.

        A peer that is no address -- none, or a name -- is in no entry.
        """
        if not self.allowed_clients:
            return True
        try:
            client = ipaddress.ip_address(peer)
        except ValueError:
            return False
        return any(client in entry for entry in self.allowed_clients)


@dataclasses.dataclass(frozen=True)
class Reach:
    """How a client reaches a server: what the scope PC shows while it serves."""

    address: str
    port: int
    # 'http'.
    scheme: str
    # How many entries the allowlist holds; 0 admits every client.
    allowed_clients: int

    @property
    def url(self) -> str:
        """The API's address for a client, under ``/api/v1``."""
        return f'{self.scheme}://{self._host}:{self.port}/api/v1'

    @property
    def _host(self) -> str:
        # An IPv6 address is bracketed beside its port.
        return f'[{self.address}]' if ':' in self.address else self.address

    def __str__(self) -> str:
        if self.allowed_clients:
            plural = '' if self.allowed_clients == 1 else 's'
            guard = f'{self.allowed_clients} allowed client{plural}'
        elif ipaddress.ip_address(self.address).is_loopback:
            guard = 'this computer only'
        else:
            guard = 'any client on the network'
        return f'{self._host}:{self.port}, {self.scheme.upper()}, {guard}'


def judged(section: object) -> Served:
    """What to serve, read from the ``rest_api`` settings, or the refusal of every value that cannot be used.

    A host serves no section it cannot read in full: ``enabled`` is judged
    though only the GUI acts on it. The address is one address of this
    computer, never a wildcard, which would serve every network it is on;
    an address that is not this computer's is refused by the bind. Each
    allowlist entry is an address or a network of the address's family,
    and a loopback address takes none, since no other computer reaches it.

    Raises:
        RestServerNotStartedError: Naming each value that cannot be used.
    """
    if type(section) is not dict:
        raise RestServerNotStartedError(
            None, None, (('rest_api', f'it must be a block of settings, not {section!r}'),)
        )
    refused = []
    enabled = section.get('enabled')
    if type(enabled) is not bool:
        refused.append(('rest_api.enabled', f'it must be true or false, not {enabled!r}'))
    port = section.get('port')
    low, high = _PORT_RANGE
    # A socket takes a whole port; bool is excluded by name: True is an int.
    if type(port) is not int or not low <= port <= high:
        refused.append(
            ('rest_api.port', f'it must be a whole number from {low} to {high}, not {port!r}')
        )
    address, why = _address(section.get('address'))
    if why:
        refused.append(('rest_api.address', why))
    clients, why = _clients(section.get('allowed_clients'), address)
    refused.extend(('rest_api.allowed_clients', w) for w in why)
    if refused:
        raise RestServerNotStartedError(section.get('address'), port, tuple(refused))
    return Served(address, port, clients)


def _address(written: object) -> tuple[_Address | None, str | None]:
    """The address to serve on, or why *written* cannot be one."""
    try:
        if type(written) is not str:
            raise ValueError
        address = ipaddress.ip_address(written)
    except ValueError:
        # A name such as localhost is refused too: it can resolve to more
        # than one address, and the user names the one served.
        return None, (
            'it must be one IP address of this computer, such as 127.0.0.1 or 192.168.10.2, '
            f'not {written!r}'
        )
    if address.is_unspecified:
        return None, (
            f'{written!r} would serve every network this computer is on; name the one '
            "address of this computer's that clients reach it at"
        )
    if getattr(address, 'scope_id', None):
        # A zoned link-local address is not bound by its spelling alone.
        return None, (
            f'{written!r} is a link-local address with a zone, which is not served; '
            "name this computer's IPv4 address on the link"
        )
    return address, None


def _clients(written: list, address: _Address | None) -> tuple[tuple[_Network, ...], list[str]]:
    """The allowlist's entries, or why each one written cannot be used.

    A list: the load refuses a settings file whose ``allowed_clients`` is
    not one, as it does any setting of the wrong shape.
    """
    entries = []
    why = []
    for entry in written:
        try:
            if type(entry) is not str:
                raise TypeError
            # strict: an address with host bits under its prefix is refused,
            # never widened to the network it is in.
            network = ipaddress.ip_network(entry, strict=True)
        except (TypeError, ValueError):
            why.append(
                f'{entry!r} is not an address or a network, such as 192.168.10.5 or 192.168.10.0/24'
            )
            continue
        if address is not None and network.version != address.version:
            why.append(
                f'{entry!r} is IPv{network.version}, and the server is reached at the '
                f'IPv{address.version} address {str(address)!r}, so it could never be served'
            )
            continue
        entries.append(network)
    if entries and address is not None and address.is_loopback:
        why.append(
            f'the server is reached at the loopback address {str(address)!r}, which no other '
            'computer reaches; name the address clients reach it at, or leave the list empty'
        )
    return tuple(entries), why
