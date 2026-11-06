import json

import clean


def rec(i, text, raw):
    return {"id": i, "text": text, "teacher_raw": raw}


def good(intent="weather_query", date=None, time=None, place=None):
    return json.dumps({"intent": intent, "date": date, "time": time, "place": place})


def test_each_rejection_reason_and_identity():
    records = [
        rec("C1", "weather in london", good(place="london")),                      # kept
        rec("C2", "weather in paris", "{not json"),                                # a
        rec("C3", "weather in rome", json.dumps({"intent": "weather_query"})),     # b
        rec("C4", "weather in oslo", good(intent="play_music", place="oslo")),     # c
        rec("C5", "weather in kyiv", good(place="Kyiv, Ukraine")),                 # d (not in text)
        rec("C6", "weather in london", good(place="london")),                      # duplicate of C1
        rec("C0", "alarm at seven am", good("alarm_set", time="seven am")),        # kept, sorts first
    ]
    kept, c = clean.clean(records)
    assert [r["id"] for r in kept] == ["C0", "C1"]
    assert (c["a_invalid_json"], c["b_bad_keys"], c["c_bad_intent"], c["d_bad_slot"],
            c["duplicates"], c["kept"], c["input"]) == (1, 1, 1, 1, 1, 2, 7)
    assert c["input"] == sum(v for k, v in c.items() if k != "input")


def test_duplicate_keeps_smallest_id():
    kept, c = clean.clean([rec("C9", "same", good()), rec("C2", "same", good())])
    assert [r["id"] for r in kept] == ["C2"] and c["duplicates"] == 1


def test_extra_key_and_non_string_slot_rejected():
    extra = json.dumps({"intent": "alarm_set", "date": None, "time": None, "place": None, "x": 1})
    assert clean.classify(rec("1", "a", extra)) == "b"
    assert clean.classify(rec("1", "a", good(date="7"))) == "d"
    bad = '{"intent": "alarm_set", "date": 7, "time": null, "place": null}'
    assert clean.classify(rec("1", "7", bad)) == "d"


def test_output_is_deterministic_and_checksum_matches():
    import hashlib
    kept, _ = clean.clean([rec("C2", "b", good()), rec("C1", "a", good())])
    t1, t2 = clean.serialise(kept), clean.serialise(list(kept))
    assert t1 == t2
    assert clean.sha256_line(t1).split()[0] == hashlib.sha256(t1.encode()).hexdigest()


def test_shard_coverage_check():
    missing, dup = clean.check_shards([{"id": "C1"}, {"id": "C1"}], ["C1", "C2"])
    assert missing == ["C2"] and dup == ["C1"]
