import split_suites as s
import pytest

def test_partition_is_disjoint_and_nonempty():
    buckets = {suite: s.split_of(suite) for suite in s.rsi_common.GOLD_SUITES}
    for pool in ("train", "holdout", "canary"):
        assert any(v == pool for v in buckets.values()), f"{pool} empty"
    # each suite in exactly one pool (disjoint by construction)
    assert set(buckets.values()) == {"train", "holdout", "canary"}

def test_no_answer_leakage(tmp_path):
    rfc_root = "/home/tyler/AI/rfc/wt-rfc"
    result = s.write_split(rfc_root, tmp_path / "split.json")
    train = set(result["_hashes"]["train"])
    assert not (set(result["_hashes"]["holdout"]) & train)
    assert not (set(result["_hashes"]["canary"]) & train)
    assert (tmp_path / "split.json").exists()

def test_leakage_raises_on_planted_duplicate(tmp_path, monkeypatch):
    # a fingerprint shared between train and holdout must trip the fatal check
    monkeypatch.setattr(s, "answer_hashes",
                        lambda root: {"train": {"DUP"}, "holdout": {"DUP"}, "canary": {"C"}})
    with pytest.raises(s.LeakageError):
        s.write_split("ignored", tmp_path / "split.json")

def test_empty_pool_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "answer_hashes",
                        lambda root: {"train": set(), "holdout": {"A"}, "canary": {"B"}})
    with pytest.raises(s.LeakageError):
        s.write_split("ignored", tmp_path / "split.json")
