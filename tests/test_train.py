import hashlib
import json
import tarfile

import train


class FakeTok:
    eos_token_id = 0

    def __call__(self, s):
        return {"input_ids": [ord(c) % 250 + 1 for c in s]}


def test_example_masks_prompt_and_ends_with_eos():
    ids, labels = train.build_example(FakeTok(), "hi", '{"intent": "alarm_set", "date": null, "time": null, "place": null}', 500)
    n_prompt = len(train.prompt_text("hi"))
    assert labels[:n_prompt] == [-100] * n_prompt           # no loss on the command
    assert labels[n_prompt:] == ids[n_prompt:] and ids[-1] == 0
    assert len(ids) == len(labels)


def test_truncation():
    ids, labels = train.build_example(FakeTok(), "x" * 100, '{"intent": "alarm_set", "date": null, "time": null, "place": null}', 40)
    assert len(ids) == len(labels) == 40


def test_target_is_canonical_json_with_key_order():
    t = train.target_text('{"place": null, "intent": "alarm_set", "time": null,  "date": null}')
    assert json.loads(t)["intent"] == "alarm_set" and t.startswith(" {")


def test_package_checksum_matches_release_asset(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"abc" * 1000)
    (tmp_path / "config.json").write_text("{}")
    tar = train.package(str(tmp_path), "v9")
    expect = hashlib.sha256(open(tar, "rb").read()).hexdigest()
    assert (tmp_path / "model.sha256").read_text().split()[0] == expect
    with tarfile.open(tar) as t:
        assert sorted(t.getnames()) == ["config.json", "model.safetensors"]
    first = open(tar, "rb").read()
    train.package(str(tmp_path), "v9")          # packaging again must give byte-identical output
    assert open(tar, "rb").read() == first
