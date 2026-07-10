import json
import pathlib

import split_suites as s


def test_ci_no_leakage():
    d = s.answer_hashes("/home/tyler/AI/rfc/wt-rfc")
    assert not ((set(d["holdout"]) | set(d["canary"])) & set(d["train"]))


def test_committed_split_matches_deterministic_bucketing():
    # guards against anyone hand-editing split.json to move a suite into the
    # wrong pool (which could sneak a test suite into the training pool).
    split = json.load(open(pathlib.Path(__file__).parent.parent / "split.json"))
    for pool in ("train", "holdout", "canary"):
        for suite in split["splits"][pool]:
            assert s.split_of(suite) == pool, f"{suite} in {pool} but buckets to {s.split_of(suite)}"
    assigned = [x for p in ("train", "holdout", "canary") for x in split["splits"][p]]
    # The committed pool listing spans every evaluated suite: gold-answer suites
    # plus execution-graded suites (e.g. docker/python), which are pinned to an
    # eval pool and never enter the train fingerprint set.
    assert sorted(assigned) == sorted(s.rsi_common.EVAL_SUITES)
