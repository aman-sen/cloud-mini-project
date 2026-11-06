import json

import cost


def test_cost_formula():
    rc = json.load(open("data/rate_card.json"))
    hourly, per_hour, c = cost.cost_per_1000(rc, replicas=2, cpu=1, mem_gib=1, rps=10)
    assert abs(hourly - 2 * (3.50 + 0.45)) < 1e-9          # 7.90 INR/h
    assert per_hour == 36000
    assert abs(c - 7.9 / 36000 * 1000) < 1e-9


def test_error_budget():
    b, used, frac = cost.error_budget(1000, 4, 0.99)
    assert abs(b - 10) < 1e-9 and used == 4 and abs(frac - 0.4) < 1e-9
