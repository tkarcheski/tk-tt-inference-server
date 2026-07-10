import hashlib, json, sys
import rsi_common
from build_dataset import iter_scenarios, special_pair, resolve_prompt, resolve_target

SALT = rsi_common.SALT

class LeakageError(Exception): ...

def bucket(suite_id: str) -> int:
    return int(hashlib.sha256((SALT + suite_id).encode()).hexdigest(), 16) % 10

def split_of(suite_id: str) -> str:
    # Execution-graded suites (rsi_common.EXEC_SUITES) are eval-only: they carry no
    # gold answer, so they contribute zero training fingerprints and are pinned to
    # an eval pool. Pin to holdout — the sanity read — rather than the canary
    # promotion gate, so a first-of-its-kind execution suite is exercised
    # end-to-end without immediately driving promotion decisions. They can never
    # land in train (run_loop only evaluates holdout/canary), so they never leak.
    if suite_id in rsi_common.EXEC_SUITES:
        return "holdout"
    b = bucket(suite_id)
    return "train" if b < 7 else ("holdout" if b < 9 else "canary")

# --- reuse build_dataset.build()'s exact (user, target) extraction so the
# --- fingerprint matches the real training key (single source of truth). ---
def _pair(item):
    pair = special_pair(item) or (resolve_prompt(item), resolve_target(item))
    user, target = pair
    return (user, target) if (user and target) else None

def answer_hashes(rfc_root):
    out = {"train": set(), "holdout": set(), "canary": set()}
    for suite, item in iter_scenarios(rfc_root):
        if suite not in rsi_common.GOLD_SUITES:
            continue
        pair = _pair(item)
        if not pair:
            continue
        user, target = pair
        fp = hashlib.sha256((user + "\x00" + target).encode()).hexdigest()
        out[split_of(suite)].add(fp)
    return out

def write_split(rfc_root, out_path):
    hashes = answer_hashes(rfc_root)
    empty = [p for p in ("train", "holdout", "canary") if not hashes[p]]
    if empty:
        raise LeakageError(f"empty fingerprint pool(s) {empty}: refusing to write a vacuous split (is rfc_root correct?)")
    leaked = (hashes["holdout"] | hashes["canary"]) & hashes["train"]
    if leaked:
        raise LeakageError(f"{len(leaked)} answer(s) shared between train and holdout/canary")
    doc = {
        "salt": SALT,
        # Pool listing spans gold-answer + execution-graded suites; fingerprints
        # (_hashes, the actual leakage firewall) remain gold-only, so exec suites
        # are evaluated without ever entering the train fingerprint set.
        "splits": {p: sorted(s for s in rsi_common.EVAL_SUITES if split_of(s) == p)
                   for p in ("train", "holdout", "canary")},
        "_hashes": {k: sorted(v) for k, v in hashes.items()},
    }
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return doc

if __name__ == "__main__":
    d = write_split(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "finetune/split.json")
    print(json.dumps(d["splits"], indent=2))
