# Serving REST

Starting the REST server and its settings, and serving a client on another computer: the address, the clients allowed, and the scope PC's network and firewall.

## Serving it

Two hosts serve the API, the same API from either:

- **LumaViewPro**, beside its window, when `rest_api.enabled` is `true` in its
  settings (shipped `false`). An installed LumaViewPro serves this way.
- **`python -m lumascope.rest`**, with no window, run from LumaViewPro's
  source; an installed copy has no such command. It serves whatever `enabled`
  says, though `enabled` must still be `true` or `false`.

```bash
python -m lumascope.rest --simulate      # a simulated scope; leave out --simulate for the hardware
```

It brings the scope up as LumaViewPro does -- the user's settings, the startup
home, the plugins -- and serves where `rest_api` says (shipped
`127.0.0.1:8000`, this computer only). It takes the installation's
single-instance lock, so it will not start while LumaViewPro or another server
runs from this installation, simulated or not. A failed startup home is
reported and the server still serves, so a client can home again. Ctrl-C or
SIGTERM closes it: new calls are refused (`'server_closing'`), the session
shuts down with the event stream open, and the stream's last event is
`closing`. It exits `0`, or `1`, printing why, when it could not take the lock
or serve, stopped before it served, or the session did not close cleanly.

To serve a session your own program has brought up, build its server, the one
every host serves through:

```python norun
from lumascope import build_server

server = build_server(session, shut_down=session.shutdown)
server.run()    # until Ctrl-C, whose close shuts the session down
if not server.close_begun:    # it stopped before it served
    session.shutdown()
elif server.close_failed is not None:    # what the session's shutdown raised
    print(server.close_failed)
```

A program that mounts the API in an ASGI application of its own takes it from
`build_app(session)`; its `allowed_clients` hold whoever serves it.

**The settings.** The server reads `rest_api` from the session's settings. It
is installation-only: no call changes it, from any caller
(`'installation_only'`), so it is edited in the installation's
`data/current.json` ([Settings](python.md#bringing-a-session-up-in-python)),
with LumaViewPro closed, since LumaViewPro writes its whole settings file back
as it runs. The port is 1024 to 65535. A `rest_api` the server cannot use, or
an address and port it cannot take, is refused `RestServerNotStartedError`
before anything is served, from every host: LumaViewPro shows it and runs on
without a server, and `python -m lumascope.rest` prints it and exits `1`.
While it serves, `server.reach` says where and to whom (`python -m
lumascope.rest` prints it, and LumaViewPro shows it in Advanced Settings).

## Serving another computer

Shipped, the server answers this computer only (`127.0.0.1`). To drive the
scope from another computer over a dedicated cable, or a private network the
lab trusts, name the scope PC's address on that link:

```json
"rest_api": {
    "enabled": true,
    "port": 8000,
    "address": "192.168.10.2",
    "allowed_clients": ["192.168.10.3"]
}
```

- `address` is one address of the scope PC, IPv4 or IPv6, never `0.0.0.0` or
  `::` (every network the PC is on), never a name such as `localhost` (write
  `127.0.0.1`), and not a link-local IPv6 address with a zone (`fe80::1%eth0`).
- `allowed_clients`, optional, lists the client addresses or networks served
  (`"192.168.10.3"`, `"192.168.10.0/24"`); a request from any other address is
  refused `403` (`client_not_allowed`, [Problems](rest.md#problems)). Empty, every client that can reach the
  address is served. A network is written as its network address: `192.168.10.0/24`,
  not `192.168.10.5/24`. Each entry is the address's family, and a list beside
  `127.0.0.1` is refused, since no other computer reaches it.

This is plain HTTP with no credential: anyone who can reach the address drives
the scope, and the control session's id travels unencrypted. The allowlist
guards against a client on the wrong network, not an attacker on the same one;
use it on a cable or a network you trust.

On the scope PC:

- **Connect the network before LumaViewPro starts.** An unplugged cable takes
  its address with it, so a server started before the cable is in cannot take
  the address: LumaViewPro says so and runs with REST off until it restarts,
  and `python -m lumascope.rest` prints it and exits `1`.
- **The firewall.** The installer adds no firewall rule: only REST on a
  network address needs one. Windows asks, the first time LumaViewPro serves
  on a network, whether to allow it, and answering needs an administrator.
  Allow it on Public networks too: a direct cable with fixed addresses is an
  unidentified network, which Windows counts as Public. Cancelling the prompt
  makes Windows block LumaViewPro, and a block outranks any allow. Where no
  one can answer the prompt, or after a cancel, an administrator opens it in
  a Command Prompt run as administrator, the first line removing any block:

  ```
  netsh advfirewall firewall delete rule name=all dir=in program="C:\Program Files\LumaViewPro\lumaviewpro.exe"
  netsh advfirewall firewall add rule name="LumaViewPro REST" dir=in action=allow program="C:\Program Files\LumaViewPro\lumaviewpro.exe" enable=yes
  ```

  `netsh advfirewall firewall delete rule name="LumaViewPro REST"` removes
  it. A site whose policy ignores local rules needs its IT to add the rule.
  A copy run from source serves through its `python.exe`, and Windows asks
  for that program instead. On Linux, a firewall that is on (firewalld on Fedora and RHEL, ufw when enabled
  on Ubuntu) drops the connection with no prompt and no message: open the port,
  for example `sudo firewall-cmd --permanent --add-port=8000/tcp && sudo
  firewall-cmd --reload`, or `sudo ufw allow from 192.168.10.3 to any port 8000
  proto tcp`.
