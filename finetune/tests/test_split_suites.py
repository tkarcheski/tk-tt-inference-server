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

def test_exec_suites_land_in_an_evaluated_pool():
    # run_loop._eval_arm only evaluates holdout/canary; an exec suite mapping to
    # train would silently never run (the trap the split_of pin closes).
    for suite in s.rsi_common.EXEC_SUITES:
        pool = s.split_of(suite)
        assert pool in ("holdout", "canary"), \
            f"exec suite {suite} maps to {pool}: silently never evaluated"

def test_exec_pin_survives_a_train_bucketing_suite(monkeypatch):
    # Regression trap for the pin itself. If split_of's EXEC_SUITES pin is ever
    # removed, the committed-split guard (test_ci_leakage) only fails until
    # split.json is regenerated — a regenerated file is self-consistent, so an
    # exec suite that buckets to train would then vanish from the eval path
    # with every test green. docker/c is a real planned sibling suite that
    # buckets to train (1 < 7), so it is exactly the future suite that would
    # silently never run without the pin.
    fake = "docker/c"
    assert s.bucket(fake) < 7, "fixture must bucket to train for the trap to bite"
    monkeypatch.setattr(s.rsi_common, "EXEC_SUITES", [fake])
    assert s.split_of(fake) in ("holdout", "canary")

def test_exec_and_gold_suites_are_disjoint():
    # A suite in both lists would be fingerprinted for training AND pinned to an
    # eval pool — contradictory: its gold answers would enter _hashes while the
    # suite itself is asserted to carry no trainable target.
    overlap = set(s.rsi_common.EXEC_SUITES) & set(s.rsi_common.GOLD_SUITES)
    assert not overlap, f"suites listed as both gold and exec: {overlap}"

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
