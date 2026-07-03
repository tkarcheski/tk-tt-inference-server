import split_suites as s

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
