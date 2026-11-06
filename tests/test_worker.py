import json

import worker


def lines(n):
    return [json.dumps({"id": f"C{i:04d}", "text": f"t{i}"}) for i in range(n)]


def test_sharding_partitions_every_line_exactly_once():
    for count in (1, 2, 3, 7):
        seen = []
        for idx in range(count):
            seen += [r["id"] for r in worker.select_shard(lines(50), idx, count)]
        assert sorted(seen) == [f"C{i:04d}" for i in range(50)]


def test_shard_rule_is_line_number_mod_count():
    ids = [r["id"] for r in worker.select_shard(lines(10), 1, 3)]
    assert ids == ["C0001", "C0004", "C0007"]


def test_prompt_keeps_json_braces():
    tpl = open("prompts/teacher_prompt.txt", encoding="utf-8").read()
    p = worker.build_prompt(tpl, "set an alarm for six")
    assert p.rstrip().endswith("Command: set an alarm for six") and '{"intent"' in p
    assert "{command}" not in p
