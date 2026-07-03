import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).parent.parent


def test_dataset_contains_only_train_suites(tmp_path):
    out = tmp_path / "out"
    subprocess.run([
        "/home/tyler/.rsi-loop-venv/bin/python", str(ROOT / "build_dataset.py"),
        "--rfc-root", "/home/tyler/AI/rfc/wt-rfc",
        "--split", str(ROOT / "split.json"),
        "--out-dir", str(out),
    ], check=True)
    train_suites = set(json.load(open(ROOT / "split.json"))["splits"]["train"])
    rows = [json.loads(l) for l in open(out / "train.jsonl")]
    assert rows, "no rows emitted"
    assert {r["suite"] for r in rows} <= train_suites
    assert not (out / "val.jsonl").exists()  # random val split removed


def test_no_train_row_overlaps_test_fingerprints(tmp_path):
    out = tmp_path / "out"
    subprocess.run([
        "/home/tyler/.rsi-loop-venv/bin/python", str(ROOT / "build_dataset.py"),
        "--rfc-root", "/home/tyler/AI/rfc/wt-rfc",
        "--split", str(ROOT / "split.json"), "--out-dir", str(out),
    ], check=True)
    split = json.load(open(ROOT / "split.json"))
    test_fps = set(split["_hashes"]["holdout"]) | set(split["_hashes"]["canary"])
    rows = [json.loads(line) for line in open(out / "train.jsonl")]
    assert rows, "no train rows emitted"
    for rec in rows:
        user, target = rec["messages"][1]["content"], rec["messages"][2]["content"]
        fp = hashlib.sha256((user + "\x00" + target).encode()).hexdigest()
        assert fp not in test_fps, f"train row leaked a test fingerprint: {user[:60]!r}"
