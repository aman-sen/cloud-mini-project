# Task 5: Subnet allocation for 10.20.0.0/16

Method: allocate largest first, each block directly after the previous one, smallest prefix that fits.
Usable hosts = 2^(32-prefix) - 2.

| Subnet     | Hosts needed | Prefix | Network     | Usable range            | Broadcast    | Usable hosts |
|------------|-------------:|:------:|-------------|-------------------------|--------------|-------------:|
| app        | 1,000        | /22    | 10.20.0.0   | 10.20.0.1 - 10.20.3.254 | 10.20.3.255  | 1,022        |
| public     | 250          | /24    | 10.20.4.0   | 10.20.4.1 - 10.20.4.254 | 10.20.4.255  | 254          |
| data       | 100          | /25    | 10.20.5.0   | 10.20.5.1 - 10.20.5.126 | 10.20.5.127  | 126          |
| management | 20           | /27    | 10.20.5.128 | 10.20.5.129 - 10.20.5.158 | 10.20.5.159 | 30           |

## Working

* **app (1,000 hosts)**: /23 gives 2^9-2 = 510 (too small); /22 gives 2^10-2 = 1,022 >= 1,000. Start 10.20.0.0; block size 1,024 = 4 x 256, so it spans 10.20.0.0 - 10.20.3.255. Next free address: 10.20.4.0.
* **public (250 hosts)**: /25 gives 126 (too small); /24 gives 2^8-2 = 254 >= 250. Start 10.20.4.0 (a multiple of 256, aligned). Spans 10.20.4.0 - 10.20.4.255. Next free: 10.20.5.0.
* **data (100 hosts)**: /26 gives 62 (too small); /25 gives 2^7-2 = 126 >= 100. Start 10.20.5.0 (multiple of 128, aligned). Spans 10.20.5.0 - 10.20.5.127. Next free: 10.20.5.128.
* **management (20 hosts)**: /28 gives 14 (too small); /27 gives 2^5-2 = 30 >= 20. Start 10.20.5.128 (multiple of 32, aligned). Spans 10.20.5.128 - 10.20.5.159.

Verified programmatically in `tests/test_network.py` (Python `ipaddress`): no overlaps, all inside 10.20.0.0/16.

## Why a stateless NACL needs separate reply rules, and why rule order matters (< 0.5 page)

A network ACL is *stateless*: it judges every packet on its own and keeps no table of connections. A firewall/security group that is *stateful* would see that an inbound request to TCP 443 was allowed and automatically permit the reply. A NACL does not remember the request, so the reply (which travels in the opposite direction, from port 443/8000 *to* the client's ephemeral port 1024-65535) is just another packet that must match an allow rule. That is why the file has an outbound rule for ephemeral ports back to clients (rule 110 outbound), an inbound ephemeral rule for the answers coming back from the app subnet (rule 130 inbound), and an outbound rule for the request towards the app subnet on 8000. Without these, connections would open but the responses would be dropped.

Rules are evaluated from the lowest number upward and the first match wins. The block for 198.51.100.0/24 must override the "allow TCP 80/443 from anywhere" rules, because 198.51.100.x addresses are also inside 0.0.0.0/0. If the deny had a higher number (e.g. 200) than the allow rules (100, 110), a packet from 198.51.100.9 to port 443 would match rule 110 first and be allowed; the deny would never be reached. Giving the deny the lowest number (90) makes it match first, so the abusive range is dropped even on web ports.
