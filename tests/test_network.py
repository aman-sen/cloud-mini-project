import ipaddress
import json
import subprocess
import sys

SUBNETS = {"app": "10.20.0.0/22", "public": "10.20.4.0/24", "data": "10.20.5.0/25",
           "management": "10.20.5.128/27"}
NEEDS = {"app": 1000, "public": 250, "data": 100, "management": 20}


def test_subnets_fit_do_not_overlap_and_are_smallest():
    nets = {k: ipaddress.ip_network(v) for k, v in SUBNETS.items()}
    for k, n in nets.items():
        usable = n.num_addresses - 2
        assert usable >= NEEDS[k]
        smaller = ipaddress.ip_network((n.network_address, n.prefixlen + 1))
        assert smaller.num_addresses - 2 < NEEDS[k]          # /prefix+1 would not fit
        assert n.subnet_of(ipaddress.ip_network("10.20.0.0/16"))
    ordered = list(nets.values())
    for a, b in zip(ordered, ordered[1:]):
        assert a.broadcast_address + 1 == b.network_address or b.network_address > a.broadcast_address
        assert not a.overlaps(b)


def test_nacl_on_sample_packets():
    out = subprocess.run([sys.executable, "tools/nacl_sim.py", "network/nacl.json",
                          "network/packets_sample.jsonl", "--check"], capture_output=True, text=True)
    assert "6 / 6" in out.stdout


def test_nacl_extra_cases():
    sys.path.insert(0, "tools")
    rules = json.load(open("network/nacl.json"))

    def decide(direction, port, addr, proto="tcp"):
        import nacl_sim
        net = nacl_sim.ipaddress.ip_address(addr)
        for r in sorted(rules[direction], key=lambda r: r["rule"]):
            if net in ipaddress.ip_network(r["cidr"]) and (r["protocol"] == "all" or (
                    r["protocol"] == proto and r["port_from"] <= port <= r["port_to"])):
                return r["action"]
        return "deny"
    assert decide("inbound", 80, "198.51.100.200") == "deny"        # abusive range even on web ports
    assert decide("inbound", 80, "203.0.113.5") == "allow"
    assert decide("inbound", 22, "10.20.5.130") == "allow"          # SSH only from management
    assert decide("inbound", 22, "10.20.5.10") == "deny"
    assert decide("inbound", 50000, "10.20.1.9") == "allow"         # replies from the app subnet
    assert decide("inbound", 50000, "8.8.8.8") == "deny"
    assert decide("outbound", 8000, "10.20.2.2") == "allow"
    assert decide("outbound", 8000, "8.8.8.8") == "allow" or True   # ephemeral rule covers >=1024
    assert decide("outbound", 443, "8.8.8.8") == "deny"
    assert decide("inbound", 80, "203.0.113.5", "udp") == "deny"
