# REST from another computer

Serving the REST API to a client on another computer: the address, the clients allowed, and the scope PC's network and firewall.

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
  refused `403` (`client_not_allowed`). Empty, every client that can reach the
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
  the address; it says so, and REST is off until LumaViewPro restarts.
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
