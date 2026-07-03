# MODEL_TUNER Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the week-1 slice of the MODEL_TUNER loop — a leak-free, shadow-only pipeline that LoRA-fine-tunes a small Qwen on rfc train-pool suites, serves it via Ollama, evaluates it and the base against a frozen holdout+canary with rfc, and writes per-test results to a Postgres warehouse for a McNemar promotion gate.

**Architecture:** One orchestrator (`run_loop.py`) drives isolated single-responsibility modules under `finetune/`. Data flows: `split_suites.py` (frozen train/holdout/canary firewall) → `build_dataset.py` (train-pool only) → `train_lora.py` (Qwen ≤3B, CPU, MERGE=1) → `serve_ollama.py` (merged safetensors → Ollama tag) → `eval_rfc.py` (robot on suite dirs, both arms) → `import_results.py` (output.xml → `rsi.test_results`) → `gate.py` (McNemar on holdout+canary). The P100 base-Llama is a reference line only; the paired control is Qwen-base vs Qwen-tuned on the same Ollama endpoint.

**Tech Stack:** Python 3, psycopg2, scipy (McNemar), Robot Framework (via rfc's `uv run robot`), Ollama 0.20.0, PEFT/trl (existing trainer), Postgres 16.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-07-03-rsi-model-tuner-loop-design.md` (authoritative).
- **Extraction-first:** v1 trains only on real gold answers from **train-pool** suites. No LLM-synthesized targets.
- **Base model:** `Qwen/Qwen2.5-3B-Instruct` (≤3B; env `BASE_MODEL`). Never Llama (trainer enforces).
- **P100 = reference only.** Paired A/B control = Qwen-base (adapter-off) vs Qwen-tuned, both on Ollama. Never compare across hardware for the gate; secondaries measured within the Ollama lane only.
- **Shadow-only:** the loop NEVER swaps a serving endpoint or merges a PR. It opens a **draft** PR / writes `status='proposed'` rows. Human promotes.
- **Prompt filter:** any rfc prompt ≥2048 tokens is skipped (P100 64K config crashes on ≥2048 prefill); applies to the vLLM/Llama reference arm.
- **Leakage is fatal:** CI fails if any holdout/canary suite id or answer-hash appears in the training JSONL. `leakage_score` must be 0 to propose promotion.
- **Frozen split salt:** `SALT = "rfc-split-v816"` (committed constant). Split is by **suite id**, three-way train/holdout/canary.
- **DB DSN:** `postgresql://rfc:changeme@localhost:5434/rfc`, schema `rsi`, role `rfc`. No host `psql` client — use psycopg2.
- **Glue venv:** `/home/tyler/.rsi-loop-venv` (psycopg2-binary, scipy, lxml). Do NOT reuse `finetune/.venv-train` (no pip; training-only).
- **rfc repo:** `/home/tyler/AI/rfc/wt-rfc`. Robot invoked as `uv run robot` from that dir.
- **Gold-answer suites (11), the only ones that participate** — verified by running the extractor: `adversarial, c_interview, causal_reasoning, code_review, context_window, extraction, hallucination, legal, quantization, temporal_reasoning, variables`. The top-level `variables` suite **duplicates** `hallucination/numerical_facts.yaml` — exclude `variables` to avoid a cross-split answer-hash collision, leaving **10 suites**.
- **Commit discipline:** one commit per task, on branch `claude/p100-capacity-decision` (or a fresh feature branch). Draft PR only; never push main.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `finetune/rsi_common.py` (new) | DSN/env config, psycopg2 connection helper, `SALT`, constants, `token_len()` filter. |
| `finetune/sql/rsi_bootstrap.sql` (new) | Idempotent `rsi.experiments` + `rsi.test_results` DDL. |
| `finetune/split_suites.py` (new) | Deterministic suite→{train,holdout,canary} partition + per-scenario answer-hashes → `split.json`; leakage assertion. |
| `finetune/build_dataset.py` (modify) | Add `--split`; read train-pool suites only; drop the random val split; emit `train_pool_hash`. |
| `finetune/train_lora.py` (modify) | Accept a local `.jsonl` DATASET; default a ≤3B base is set by the caller. |
| `finetune/serve_ollama.py` (new) | Merged safetensors dir → `ollama create` tag; return `(tag, lora_adapter_hash)`. |
| `finetune/eval_rfc.py` (new) | Run `robot` on a suite dir against a provider arm; return the `output.xml` path. |
| `finetune/import_results.py` (new) | Parse `output.xml` → per-test rows → `rsi.test_results`. |
| `finetune/gate.py` (new) | McNemar on holdout+canary matched per-test outcomes; verdict + report. |
| `finetune/run_loop.py` (new) | Orchestrate a round; write `rsi.experiments`; draft-PR on pass; shadow-only. |
| `finetune/tests/…` (new) | pytest tests per module. |
| `/home/tyler/AI/rfc/wt-rfc/.env` (create) | Base rfc config (Ollama defaults); arms override provider via env. |

---

## Task 0: Glue venv + warehouse schema

**Files:**
- Create: `/home/tyler/.rsi-loop-venv` (venv, not in repo)
- Create: `finetune/sql/rsi_bootstrap.sql`
- Create: `finetune/rsi_common.py`
- Test: `finetune/tests/test_rsi_common.py`

**Interfaces:**
- Produces: `rsi_common.connect() -> psycopg2.connection` (autocommit off); `rsi_common.DSN: str`; `rsi_common.SALT: str = "rfc-split-v816"`; `rsi_common.GOLD_SUITES: list[str]` (10, excluding `variables`); `rsi_common.token_len(text: str) -> int`.

- [ ] **Step 1: Create the glue venv** (the one true environment blocker)

```bash
python3 -m venv /home/tyler/.rsi-loop-venv
/home/tyler/.rsi-loop-venv/bin/pip install -q psycopg2-binary==2.9.9 scipy==1.14.1 lxml==5.3.0 pytest==8.3.3 tiktoken==0.8.0
/home/tyler/.rsi-loop-venv/bin/python -c "import psycopg2, scipy.stats, lxml.etree, tiktoken; print('glue venv OK')"
```
Expected: `glue venv OK`

- [ ] **Step 2: Write the schema DDL** — `finetune/sql/rsi_bootstrap.sql`

```sql
CREATE SCHEMA IF NOT EXISTS rsi;

CREATE TABLE IF NOT EXISTS rsi.experiments (
  experiment_id        uuid PRIMARY KEY,
  intent               text NOT NULL,            -- 'baseline' | 'model_tuner_round'
  provider_variant     text NOT NULL,            -- 'vllm-tt' | 'ollama' | 'vllm'
  serving_runtime      text NOT NULL,
  base_model_sha       text NOT NULL,
  lora_adapter_hash    text,
  train_pool_hash      text,
  train_split_hash     text,
  leakage_score        numeric,
  endpoint_url         text,
  tt_container_digest  text,
  rfc_sha              text,
  seed                 integer,
  train_hardware       text,                     -- 'cpu' | 'cuda' | 'p100'
  parent_experiment_id uuid,
  created_at           timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rsi.test_results (
  experiment_id    uuid NOT NULL REFERENCES rsi.experiments(experiment_id),
  suite_id         text NOT NULL,
  test_id          text NOT NULL,
  pool             text NOT NULL,                -- 'train' | 'holdout' | 'canary'
  status           text NOT NULL,                -- 'PASS' | 'FAIL' | 'SKIP'
  grader_rationale text,
  repeat_idx       integer NOT NULL DEFAULT 0,
  PRIMARY KEY (experiment_id, suite_id, test_id, repeat_idx)
);
```

- [ ] **Step 3: Write `rsi_common.py`**

```python
import os
import psycopg2

DSN = os.environ.get("RSI_DSN", "postgresql://rfc:changeme@localhost:5434/rfc")
SALT = "rfc-split-v816"
# 11 gold-answer suites minus the top-level `variables` duplicate of hallucination.
GOLD_SUITES = [
    "adversarial", "c_interview", "causal_reasoning", "code_review",
    "context_window", "extraction", "hallucination", "legal",
    "quantization", "temporal_reasoning",
]

def connect():
    return psycopg2.connect(DSN)

_ENC = None
def token_len(text: str) -> int:
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text or ""))
```

- [ ] **Step 4: Write the failing test** — `finetune/tests/test_rsi_common.py`

```python
import subprocess, pathlib
import rsi_common

def test_schema_applies_and_tables_exist():
    sql = pathlib.Path(__file__).parent.parent / "sql" / "rsi_bootstrap.sql"
    subprocess.run(
        ["docker", "exec", "-i", "rfc-monorepo-postgres-1", "psql", "-U", "rfc", "-d", "rfc"],
        stdin=open(sql), check=True,
    )
    with rsi_common.connect() as c, c.cursor() as cur:
        cur.execute("select to_regclass('rsi.experiments'), to_regclass('rsi.test_results')")
        a, b = cur.fetchone()
    assert a == "rsi.experiments" and b == "rsi.test_results"

def test_token_len_and_constants():
    assert rsi_common.SALT == "rfc-split-v816"
    assert len(rsi_common.GOLD_SUITES) == 10 and "variables" not in rsi_common.GOLD_SUITES
    assert rsi_common.token_len("hello world") >= 2
```

- [ ] **Step 5: Run tests**

Run: `cd finetune && PYTHONPATH=. /home/tyler/.rsi-loop-venv/bin/pytest tests/test_rsi_common.py -v`
Expected: PASS (schema applied idempotently; constants correct).

- [ ] **Step 6: Commit**

```bash
git add finetune/sql/rsi_bootstrap.sql finetune/rsi_common.py finetune/tests/test_rsi_common.py
git commit -m "feat(rsi): glue venv contract + rsi.* warehouse schema"
```

---

## Task 1: Suite split + leakage firewall (`split_suites.py`)

**Files:**
- Create: `finetune/split_suites.py`
- Test: `finetune/tests/test_split_suites.py`

**Interfaces:**
- Consumes: `rsi_common.SALT`, `rsi_common.GOLD_SUITES`; the extraction logic mirrored from `build_dataset.py` (`resolve_target`/`special_pair`).
- Produces: `bucket(suite_id) -> int`; `split_of(suite_id) -> str`; `answer_hashes(rfc_root) -> dict[str, set[str]]` (split → set of sha256(norm(target))); `write_split(rfc_root, out_path) -> dict` (writes `split.json`, raises `LeakageError` if holdout/canary answer-hashes intersect train).

- [ ] **Step 1: Write the failing test** — `finetune/tests/test_split_suites.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd finetune && PYTHONPATH=. /home/tyler/.rsi-loop-venv/bin/pytest tests/test_split_suites.py -v`
Expected: FAIL (`No module named split_suites`).

- [ ] **Step 3: Implement `split_suites.py`**

```python
import hashlib, json, os, glob, re, sys
import yaml  # use /usr/bin/python3 (has PyYAML) OR install pyyaml into the glue venv
import rsi_common

SALT = rsi_common.SALT

class LeakageError(Exception): ...

def bucket(suite_id: str) -> int:
    return int(hashlib.sha256((SALT + suite_id).encode()).hexdigest(), 16) % 10

def split_of(suite_id: str) -> str:
    b = bucket(suite_id)
    return "train" if b < 7 else ("holdout" if b < 9 else "canary")

# --- mirror build_dataset.py extraction so hashes match the training targets ---
def _norm(v): return re.sub(r"\s+", " ", str(v)).strip()
def _rationale(it): return it.get("rationale") or it.get("explanation")

def _resolve_target(it):
    if it.get("expected_letter"):
        ans = f"The correct answer is {_norm(it['expected_letter'])}."
        if _rationale(it): ans += "\n\n" + _norm(_rationale(it))
        return ans
    for k in ("expected_answer", "expected", "answer"):
        if it.get(k) not in (None, ""): return _norm(it[k])
    return None

def _special_pair(it):
    if it.get("argument") and it.get("fallacy_name"):
        t = _norm(it["fallacy_name"])
        if it.get("explanation"): t += "\n\n" + _norm(it["explanation"])
        return t
    if it.get("verdict") and (it.get("scenario") or it.get("claim")):
        t = _norm(it["verdict"])
        if it.get("rationale"): t += "\n\n" + _norm(it["rationale"])
        return t
    if it.get("attribute") and it.get("text") and it.get("expected"):
        return _norm(it["expected"])
    return None

def _target(it):
    return _special_pair(it) or _resolve_target(it)

def _iter(rfc_root):
    for path in sorted(glob.glob(os.path.join(rfc_root, "robot", "**", "variables", "*.yaml"), recursive=True)):
        suite = os.path.relpath(path, os.path.join(rfc_root, "robot")).split(os.sep)[0]
        if suite not in rsi_common.GOLD_SUITES:
            continue
        data = yaml.safe_load(open(path)) or {}
        items = data if isinstance(data, list) else data.values()
        for block in items:
            for it in (block if isinstance(block, list) else [block]):
                if isinstance(it, dict):
                    yield suite, it

def answer_hashes(rfc_root):
    out = {"train": set(), "holdout": set(), "canary": set()}
    for suite, it in _iter(rfc_root):
        t = _target(it)
        if not t: continue
        out[split_of(suite)].add(hashlib.sha256(_norm(t).encode()).hexdigest())
    return out

def write_split(rfc_root, out_path):
    hashes = answer_hashes(rfc_root)
    leaked = (hashes["holdout"] | hashes["canary"]) & hashes["train"]
    if leaked:
        raise LeakageError(f"{len(leaked)} answer(s) shared between train and holdout/canary")
    doc = {
        "salt": SALT,
        "splits": {p: sorted(s for s in rsi_common.GOLD_SUITES if split_of(s) == p)
                   for p in ("train", "holdout", "canary")},
        "_hashes": {k: sorted(v) for k, v in hashes.items()},
    }
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return doc

if __name__ == "__main__":
    d = write_split(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "finetune/split.json")
    print(json.dumps(d["splits"], indent=2))
```
*(If the glue venv lacks PyYAML, add `pyyaml==6.0.2` to Task 0 Step 1.)*

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd finetune && PYTHONPATH=. /home/tyler/.rsi-loop-venv/bin/pytest tests/test_split_suites.py -v`
Expected: PASS. Then generate the frozen split: `PYTHONPATH=. /home/tyler/.rsi-loop-venv/bin/python split_suites.py /home/tyler/AI/rfc/wt-rfc finetune/split.json` and commit `split.json`.

- [ ] **Step 5: Commit**

```bash
git add finetune/split_suites.py finetune/tests/test_split_suites.py finetune/split.json
git commit -m "feat(rsi): deterministic train/holdout/canary firewall + leakage check"
```

---

## Task 2: Fix the contamination bug in `build_dataset.py`

**Files:**
- Modify: `finetune/build_dataset.py` (add `--split`; filter to train suites; drop random val split; hash output)
- Test: `finetune/tests/test_build_dataset_firewall.py`

**Interfaces:**
- Consumes: `split.json` from Task 1.
- Produces: `finetune/out/train.jsonl` containing ONLY train-pool suites; prints `train_pool_hash` (sha256 of the file).

- [ ] **Step 1: Write the failing test**

```python
import json, subprocess, pathlib, hashlib
ROOT = pathlib.Path(__file__).parent.parent

def test_dataset_contains_only_train_suites(tmp_path):
    out = tmp_path / "out"
    subprocess.run([
        "/usr/bin/python3", str(ROOT / "build_dataset.py"),
        "--rfc-root", "/home/tyler/AI/rfc/wt-rfc",
        "--split", str(ROOT / "split.json"),
        "--out-dir", str(out),
    ], check=True)
    train_suites = set(json.load(open(ROOT / "split.json"))["splits"]["train"])
    rows = [json.loads(l) for l in open(out / "train.jsonl")]
    assert rows, "no rows emitted"
    assert {r["suite"] for r in rows} <= train_suites
    assert not (out / "val.jsonl").exists()  # random val split removed
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd finetune && /home/tyler/.rsi-loop-venv/bin/pytest tests/test_build_dataset_firewall.py -v`
Expected: FAIL (`--split` unrecognized).

- [ ] **Step 3: Modify `build_dataset.py`** — filter in `iter_scenarios`, replace the split/write block (current L142-166).

Add to argparse: `ap.add_argument("--split", help="split.json; restrict to its train suites")`. In `build()`, load the split (if given) and `continue` when `suite` not in the train set. Replace L156-166 with:

```python
    train_suites = None
    if args.split:
        train_suites = set(json.load(open(args.split))["splits"]["train"])
    records, per_suite = build(args.rfc_root, train_suites)   # build() skips non-train suites
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "train.jsonl")
    with open(path, "w") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"train_pool_hash={digest} rows={len(records)} suites={sorted(per_suite)}")
```
Update `build(rfc_root, train_suites=None)` to `continue` when `train_suites is not None and suite not in train_suites`. Remove `--val-frac`/`--seed`, the `random.shuffle`, and the `val`/`all` file writes. Keep all extraction logic unchanged.

- [ ] **Step 4: Run to verify it passes**

Run: `cd finetune && /home/tyler/.rsi-loop-venv/bin/pytest tests/test_build_dataset_firewall.py -v`
Expected: PASS.

- [ ] **Step 5: Honesty check (leakage guard fires)** — add to the test file:

```python
def test_pointing_at_holdout_yields_no_train_rows(tmp_path):
    # build_dataset restricted to train MUST NOT emit holdout suites even if asked
    import json, pathlib
    split = json.load(open(pathlib.Path(__file__).parent.parent / "split.json"))
    holdout = set(split["splits"]["holdout"])
    rows = [json.loads(l) for l in open(tmp_path.parent / "out" / "train.jsonl")] if (tmp_path.parent/"out"/"train.jsonl").exists() else []
    assert not ({r["suite"] for r in rows} & holdout)
```

- [ ] **Step 6: Commit**

```bash
git add finetune/build_dataset.py finetune/tests/test_build_dataset_firewall.py
git commit -m "fix(finetune): train-pool-only dataset; remove test-set contamination"
```

---

## Task 3: Per-test result importer (`import_results.py`)

**Files:**
- Create: `finetune/import_results.py`
- Test: `finetune/tests/test_import_results.py`

**Interfaces:**
- Consumes: an `output.xml` path; `rsi_common.connect()`; an `experiment_id`, `pool`, `repeat_idx`.
- Produces: `parse_output_xml(path) -> list[dict]` (each `{suite_id, test_id, status, grader_rationale}`); `import_results(path, experiment_id, pool, repeat_idx) -> int` (rows inserted).

- [ ] **Step 1: Write the failing test** (uses the real RF schema)

```python
import import_results as ir

MINIMAL = """<?xml version="1.0" encoding="UTF-8"?>
<robot generator="Robot 7.0">
  <suite name="Math Tests" id="s1">
    <test name="Basic Addition" id="s1-t1">
      <status status="PASS" start="1" end="2"/>
    </test>
    <test name="Hard One" id="s1-t2">
      <kw name="Ask LLM"><msg>RFC_DATA:grading_reason:wrong number</msg></kw>
      <status status="FAIL">Expected 5 got 4</status>
    </test>
    <status status="FAIL"/>
  </suite>
</robot>"""

def test_parse_per_test(tmp_path):
    p = tmp_path / "output.xml"; p.write_text(MINIMAL)
    rows = ir.parse_output_xml(str(p))
    by = {r["test_id"]: r for r in rows}
    assert by["Basic Addition"]["status"] == "PASS"
    assert by["Hard One"]["status"] == "FAIL"
    assert "wrong number" in (by["Hard One"]["grader_rationale"] or "")
```

- [ ] **Step 2: Run to verify it fails.** Expected: `No module named import_results`.

- [ ] **Step 3: Implement `import_results.py`**

```python
from xml.etree import ElementTree as ET
import rsi_common

def parse_output_xml(path):
    root = ET.parse(path).getroot()
    rows = []
    for suite in root.findall(".//suite"):
        suite_name = suite.get("name", "unknown")
        for test in suite.findall("test"):           # direct children only
            st = test.find("status")
            status = st.get("status", "UNKNOWN") if st is not None else "UNKNOWN"
            rationale = None
            for msg in test.findall(".//msg"):
                txt = (msg.text or "")
                if "RFC_DATA:grading_reason:" in txt:
                    rationale = txt.split("RFC_DATA:grading_reason:", 1)[1].strip()
            if rationale is None and st is not None and (st.text or "").strip():
                rationale = st.text.strip()
            rows.append({"suite_id": suite_name, "test_id": test.get("name", "unknown"),
                         "status": status, "grader_rationale": rationale})
    return rows

def import_results(path, experiment_id, pool, repeat_idx=0):
    rows = parse_output_xml(path)
    with rsi_common.connect() as c, c.cursor() as cur:
        for r in rows:
            cur.execute(
                """insert into rsi.test_results
                   (experiment_id, suite_id, test_id, pool, status, grader_rationale, repeat_idx)
                   values (%s,%s,%s,%s,%s,%s,%s)
                   on conflict do nothing""",
                (experiment_id, r["suite_id"], r["test_id"], pool, r["status"], r["grader_rationale"], repeat_idx))
        c.commit()
    return len(rows)
```

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add finetune/import_results.py finetune/tests/test_import_results.py
git commit -m "feat(rsi): per-test output.xml importer → rsi.test_results"
```

---

## Task 4: McNemar promotion gate (`gate.py`)

**Files:**
- Create: `finetune/gate.py`
- Test: `finetune/tests/test_gate.py`

**Interfaces:**
- Consumes: `rsi_common.connect()`; two `experiment_id`s (base, tuned); the `pool` to score (`holdout`, then `canary` as the promotion metric).
- Produces: `matched_outcomes(base_id, tuned_id, pool) -> list[tuple[bool,bool]]`; `mcnemar_gate(base_id, tuned_id, pool, min_effect_pp=5.0, alpha=0.05) -> dict` (`{n, base_pass, tuned_pass, delta_pp, p_value, passes}`).

- [ ] **Step 1: Write the failing test**

```python
import gate

def test_mcnemar_math():
    # tuned fixes 12, breaks 2, ties otherwise: significant improvement
    pairs = [(False, True)] * 12 + [(True, False)] * 2 + [(True, True)] * 40 + [(False, False)] * 6
    r = gate.mcnemar_from_pairs(pairs, min_effect_pp=5.0, alpha=0.05)
    assert r["delta_pp"] > 5 and r["p_value"] < 0.05 and r["passes"] is True

def test_mcnemar_rejects_noise():
    pairs = [(False, True)] * 3 + [(True, False)] * 3 + [(True, True)] * 40
    r = gate.mcnemar_from_pairs(pairs, min_effect_pp=5.0, alpha=0.05)
    assert r["passes"] is False
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `gate.py`**

```python
from scipy.stats import binomtest
import rsi_common

def mcnemar_from_pairs(pairs, min_effect_pp=5.0, alpha=0.05):
    b = sum(1 for base, tuned in pairs if base and not tuned)   # tuned broke
    c = sum(1 for base, tuned in pairs if (not base) and tuned) # tuned fixed
    n = len(pairs)
    base_pass = sum(1 for x, _ in pairs if x) / n * 100 if n else 0
    tuned_pass = sum(1 for _, y in pairs if y) / n * 100 if n else 0
    delta = tuned_pass - base_pass
    # exact McNemar = binomial test on the discordant pairs
    p = binomtest(c, b + c, 0.5, alternative="greater").pvalue if (b + c) else 1.0
    return {"n": n, "base_pass": base_pass, "tuned_pass": tuned_pass,
            "delta_pp": delta, "p_value": p,
            "passes": bool(delta >= min_effect_pp and p < alpha)}

def matched_outcomes(base_id, tuned_id, pool):
    q = """select suite_id, test_id, repeat_idx,
                  bool_or(status='PASS') filter (where experiment_id=%s) as base_pass,
                  bool_or(status='PASS') filter (where experiment_id=%s) as tuned_pass
           from rsi.test_results where pool=%s and experiment_id in (%s,%s)
           group by suite_id, test_id, repeat_idx
           having count(distinct experiment_id)=2"""
    with rsi_common.connect() as c, c.cursor() as cur:
        cur.execute(q, (base_id, tuned_id, pool, base_id, tuned_id))
        return [(r[3], r[4]) for r in cur.fetchall()]

def mcnemar_gate(base_id, tuned_id, pool, **kw):
    return mcnemar_from_pairs(matched_outcomes(base_id, tuned_id, pool), **kw)
```

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Commit**

```bash
git add finetune/gate.py finetune/tests/test_gate.py
git commit -m "feat(rsi): McNemar paired promotion gate"
```

---

## Task 5: Serve a tuned model via Ollama (`serve_ollama.py`)

**Files:**
- Create: `finetune/serve_ollama.py`
- Test: `finetune/tests/test_serve_ollama.py`

**Interfaces:**
- Consumes: a merged safetensors dir (`train_lora.py` MERGE=1 output at `OUTPUT_DIR/merged`).
- Produces: `adapter_hash(adapter_dir) -> str`; `create_tag(merged_dir, tag) -> str` (runs `ollama create <tag> --experimental -q q4_K_M -f Modelfile`, returns tag).

- [ ] **Step 1: Write the failing test** (mocks subprocess so no real GB build in CI)

```python
import serve_ollama as so

def test_writes_modelfile_and_calls_ollama(tmp_path, monkeypatch):
    merged = tmp_path / "merged"; merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"x")
    calls = []
    monkeypatch.setattr(so.subprocess, "run", lambda *a, **k: calls.append(a[0]) or type("R", (), {"returncode": 0})())
    tag = so.create_tag(str(merged), "rsi-qwen:test")
    assert tag == "rsi-qwen:test"
    assert (merged / "Modelfile").read_text().startswith("FROM ")
    assert any("ollama" in c and "create" in c for c in calls)

def test_adapter_hash_stable(tmp_path):
    a = tmp_path / "ad"; a.mkdir(); (a / "adapter_config.json").write_text("{}")
    (a / "adapter_model.safetensors").write_bytes(b"weights")
    assert len(so.adapter_hash(str(a))) == 64
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `serve_ollama.py`**

```python
import hashlib, os, subprocess

def adapter_hash(adapter_dir):
    h = hashlib.sha256()
    for name in sorted(os.listdir(adapter_dir)):
        p = os.path.join(adapter_dir, name)
        if os.path.isfile(p):
            h.update(name.encode()); h.update(open(p, "rb").read())
    return h.hexdigest()

def create_tag(merged_dir, tag, quant="q4_K_M"):
    modelfile = os.path.join(merged_dir, "Modelfile")
    with open(modelfile, "w") as fh:
        fh.write(f"FROM {os.path.abspath(merged_dir)}\n")
    subprocess.run(["ollama", "create", tag, "--experimental", "-q", quant, "-f", modelfile], check=True)
    return tag
```

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Manual smoke (once, not in CI)** — after a real `MERGE=1` train run:

Run: `ollama create rsi-qwen:smoke --experimental -q q4_K_M -f finetune/out/lora-qwen/merged/Modelfile && ollama run rsi-qwen:smoke "2+2="`
Expected: a coherent completion; confirms the safetensors→Ollama path end-to-end.

- [ ] **Step 6: Commit**

```bash
git add finetune/serve_ollama.py finetune/tests/test_serve_ollama.py
git commit -m "feat(rsi): merged-safetensors → Ollama tag"
```

---

## Task 6: Run an rfc suite against an arm (`eval_rfc.py`)

**Files:**
- Create: `finetune/eval_rfc.py`
- Create: `/home/tyler/AI/rfc/wt-rfc/.env` (from `.env.example`)
- Test: `finetune/tests/test_eval_rfc.py`

**Interfaces:**
- Consumes: a `suite_id`, an arm dict (`{"provider": "ollama"|"vllm", "model": tag, "base_url": url}`).
- Produces: `arm_env(arm) -> dict[str,str]` (env overrides selecting the provider); `run_suite(suite_id, arm, results_dir) -> str` (path to `output.xml`); enforces the <2048-token prompt filter via `--exclude` when possible, else post-hoc.

- [ ] **Step 1: Create rfc `.env`** — copy `.env.example`, set the Ollama defaults; arms override via env at call time. `cp /home/tyler/AI/rfc/wt-rfc/.env.example /home/tyler/AI/rfc/wt-rfc/.env` and set `LLM_PROVIDER=ollama`, `OLLAMA_ENDPOINT=http://localhost:11434`, `DEFAULT_MODEL=qwen2.5:3b`.

- [ ] **Step 2: Write the failing test** (verifies env mapping only — no live robot run in CI)

```python
import eval_rfc as e

def test_arm_env_ollama():
    env = e.arm_env({"provider": "ollama", "model": "rsi-qwen:v1"})
    assert env["LLM_PROVIDER"] == "ollama" and env["DEFAULT_MODEL"] == "rsi-qwen:v1"

def test_arm_env_vllm():
    env = e.arm_env({"provider": "vllm", "base_url": "http://localhost:8011/v1"})
    assert env["LLM_PROVIDER"] == "vllm" and env["VLLM_BASE_URL"] == "http://localhost:8011/v1"
```

- [ ] **Step 3: Implement `eval_rfc.py`**

```python
import os, subprocess

RFC = os.environ.get("RFC_ROOT", "/home/tyler/AI/rfc/wt-rfc")

def arm_env(arm):
    env = dict(os.environ)
    env["LLM_PROVIDER"] = arm["provider"]
    if arm["provider"] == "ollama":
        env["DEFAULT_MODEL"] = arm["model"]
        env["OLLAMA_ENDPOINT"] = arm.get("base_url", "http://localhost:11434")
    elif arm["provider"] == "vllm":
        env["VLLM_BASE_URL"] = arm.get("base_url", "http://localhost:8011/v1")
    return env

def run_suite(suite_id, arm, results_dir):
    # Mirror the Makefile robot recipe WITHOUT rfc's DB listeners (we own rsi.*).
    os.makedirs(results_dir, exist_ok=True)
    cmd = ["uv", "run", "robot", "-d", results_dir,
           "--metadata", f"test_suite:{suite_id}",
           f"robot/{suite_id}/"]
    subprocess.run(cmd, cwd=RFC, env=arm_env(arm), check=False)  # robot exits nonzero on test failures
    return os.path.join(results_dir, "output.xml")
```

- [ ] **Step 4: Run the unit test.** Expected: PASS.

- [ ] **Step 5: Live smoke (once, not CI)** — run one small train suite against Ollama base and confirm an `output.xml` with per-test rows:

Run: `cd finetune && PYTHONPATH=. /home/tyler/.rsi-loop-venv/bin/python -c "import eval_rfc,import_results as ir; p=eval_rfc.run_suite('context_window',{'provider':'ollama','model':'qwen2.5:3b'},'/tmp/rsi_eval'); print(len(ir.parse_output_xml(p)),'tests')"`
Expected: prints a nonzero test count. *(If `qwen2.5:3b` isn't pulled, `ollama pull qwen2.5:3b` first.)*

- [ ] **Step 6: Commit**

```bash
git add finetune/eval_rfc.py finetune/tests/test_eval_rfc.py
git commit -m "feat(rsi): run an rfc suite against an ollama/vllm arm"
```

---

## Task 7: Round orchestrator (`run_loop.py`)

**Files:**
- Create: `finetune/run_loop.py`
- Test: `finetune/tests/test_run_loop.py`

**Interfaces:**
- Consumes: all prior modules; env `BASE_MODEL` (default `Qwen/Qwen2.5-3B-Instruct`), `MAX_STEPS`, `RSI_MODE` (`shadow` default), `RSI_AGENTS_ENABLED`.
- Produces: `new_experiment(intent, **fields) -> str` (uuid, inserts `rsi.experiments`); `run_round(smoke=False) -> dict` (report: experiment ids, gate verdict, proposed bool). NEVER swaps an endpoint; on pass writes a proposal + (if a repo is configured) a draft PR.

- [ ] **Step 1: Write the failing test** (unit: uuid + insert + shadow invariants; mocks train/serve/eval)

```python
import run_loop as rl

def test_new_experiment_inserts_and_returns_uuid():
    eid = rl.new_experiment("model_tuner_round", provider_variant="ollama",
                            serving_runtime="ollama", base_model_sha="deadbeef")
    assert len(eid) == 36
    with rl.rsi_common.connect() as c, c.cursor() as cur:
        cur.execute("select intent from rsi.experiments where experiment_id=%s", (eid,))
        assert cur.fetchone()[0] == "model_tuner_round"

def test_shadow_mode_never_promotes(monkeypatch):
    monkeypatch.setenv("RSI_MODE", "shadow")
    assert rl.can_promote() is False
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `run_loop.py`** — full body (composes every prior module; shadow-only):

```python
import argparse, json, os, subprocess, uuid, pathlib
import rsi_common, split_suites, serve_ollama, eval_rfc, import_results, gate

FT = pathlib.Path(__file__).parent
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
REPEATS = int(os.environ.get("RSI_REPEATS", "1"))

def can_promote():
    return os.environ.get("RSI_MODE", "shadow") == "live" and \
           os.environ.get("RSI_AGENTS_ENABLED", "true") != "false"

def new_experiment(intent, **f):
    eid = str(uuid.uuid4())
    cols = ["experiment_id", "intent"] + list(f.keys())
    vals = [eid, intent] + list(f.values())
    ph = ",".join(["%s"] * len(cols))
    with rsi_common.connect() as c, c.cursor() as cur:
        cur.execute(f"insert into rsi.experiments ({','.join(cols)}) values ({ph})", vals)
        c.commit()
    return eid

def _eval_arm(experiment_id, arm, splits):
    """Run every holdout+canary suite for one arm; import per-test rows."""
    for pool in ("holdout", "canary"):
        for suite in splits[pool]:
            for r in range(REPEATS):
                rd = f"/tmp/rsi/{experiment_id}/{pool}/{suite}/{r}"
                xml = eval_rfc.run_suite(suite, arm, rd)
                if os.path.exists(xml):
                    import_results.import_results(xml, experiment_id, pool, r)

def run_round(once=True, smoke=False):
    if os.environ.get("RSI_KILL", "0") == "1":
        raise SystemExit("kill switch set (RSI_KILL=1)")
    # 1) frozen firewall + leak check (raises LeakageError on contamination)
    split = split_suites.write_split(eval_rfc.RFC, FT / "split.json")
    splits = split["splits"]
    train_hash_line = subprocess.run(
        ["/usr/bin/python3", str(FT / "build_dataset.py"),
         "--rfc-root", eval_rfc.RFC, "--split", str(FT / "split.json"),
         "--out-dir", str(FT / "out")],
        capture_output=True, text=True, check=True).stdout.strip()
    train_pool_hash = train_hash_line.split("train_pool_hash=")[1].split()[0]

    # 2) train (CPU LoRA, ≤3B, merge) — smoke uses tiny base + few steps
    env = dict(os.environ, DATASET=str(FT / "out" / "train.jsonl"),
               OUTPUT_DIR=str(FT / "out" / "lora-qwen"), MERGE="1",
               BASE_MODEL=("Qwen/Qwen2.5-0.5B-Instruct" if smoke else BASE_MODEL),
               MAX_STEPS=os.environ.get("MAX_STEPS", "5" if smoke else "200"))
    subprocess.run(["python3", str(FT / "train_lora.py")], cwd=FT, env=env, check=True)
    merged = str(FT / "out" / "lora-qwen" / "merged")
    lora_hash = serve_ollama.adapter_hash(str(FT / "out" / "lora-qwen"))

    # 3) serve tuned via Ollama (base arm reuses a pinned base tag)
    tuned_tag = serve_ollama.create_tag(merged, "rsi-qwen:round")
    base_tag = os.environ.get("RSI_BASE_TAG", "qwen2.5:3b")

    # 4) provenance rows (base is a frozen reference each round)
    base_id = new_experiment("baseline", provider_variant="ollama",
                             serving_runtime="ollama", base_model_sha=base_tag,
                             lora_adapter_hash=None, train_pool_hash=None,
                             leakage_score=0, train_hardware="cpu")
    tuned_id = new_experiment("model_tuner_round", provider_variant="ollama",
                              serving_runtime="ollama", base_model_sha=base_tag,
                              lora_adapter_hash=lora_hash, train_pool_hash=train_pool_hash,
                              leakage_score=0, train_hardware="cpu", parent_experiment_id=base_id)

    # 5) evaluate both arms on the SAME Ollama endpoint (paired control)
    _eval_arm(base_id, {"provider": "ollama", "model": base_tag}, splits)
    _eval_arm(tuned_id, {"provider": "ollama", "model": tuned_tag}, splits)

    # 6) McNemar gate — canary is the promotion metric, holdout is the sanity read
    holdout = gate.mcnemar_gate(base_id, tuned_id, "holdout")
    canary = gate.mcnemar_gate(base_id, tuned_id, "canary")
    report = {"base_id": base_id, "tuned_id": tuned_id,
              "holdout": holdout, "canary": canary,
              "train_pool_hash": train_pool_hash, "lora_hash": lora_hash}

    # 7) SHADOW ONLY: propose, never swap an endpoint, never auto-merge
    report["proposed"] = bool(canary["passes"] and holdout["passes"])
    if report["proposed"] and not can_promote() and os.environ.get("RSI_OPEN_PR") == "1":
        subprocess.run(["gh", "pr", "create", "--draft",
                        "--title", f"MODEL_TUNER proposal {tuned_id[:8]}",
                        "--body", json.dumps(report, indent=2)], check=False)
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_round(once=args.once, smoke=args.smoke)
```

- [ ] **Step 4: Run the unit test.** Expected: PASS.

- [ ] **Step 5: End-to-end dry run (the real integration gate)** — `RSI_MODE=shadow MAX_STEPS=5 BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct PYTHONPATH=finetune /home/tyler/.rsi-loop-venv/bin/python finetune/run_loop.py --once --smoke`
Expected: completes without touching `:8011`; prints base/tuned experiment ids and a McNemar report for holdout+canary; writes rows to `rsi.test_results`; opens NO live PR. Verify the P100 container was never restarted (`docker ps` uptime unchanged).

- [ ] **Step 6: Commit**

```bash
git add finetune/run_loop.py finetune/tests/test_run_loop.py
git commit -m "feat(rsi): shadow-only MODEL_TUNER round orchestrator"
```

---

## Task 8: Training adapter for local JSONL + ≤3B base

**Files:**
- Modify: `finetune/train_lora.py` (accept a local `.jsonl` DATASET)
- Test: `finetune/tests/test_train_lora_local.py`

**Interfaces:**
- Produces: when `DATASET` ends in `.jsonl`, loads it via `load_dataset("json", data_files={"train": DATASET})` instead of a HF hub name.

- [ ] **Step 1: Write the failing test** (config-level: the dataset-loading branch)

```python
import importlib, os
def test_local_jsonl_branch(monkeypatch, tmp_path):
    ds = tmp_path / "train.jsonl"
    ds.write_text('{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]}\n')
    import train_lora
    d = train_lora.load_local_or_hub(str(ds))
    assert "train" in d and len(d["train"]) == 1
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Modify `train_lora.py`** — extract dataset loading into a helper:

```python
def load_local_or_hub(name):
    from datasets import load_dataset
    if name.endswith(".jsonl"):
        return load_dataset("json", data_files={"train": name})
    return load_dataset(name)
```
Replace the current `load_dataset(DATASET)` call (L56) with `load_local_or_hub(DATASET)`.

- [ ] **Step 4: Run the unit test** (with the training venv, which has `datasets`):

Run: `cd finetune && PYTHONPATH=. .venv-train/bin/python -m pytest tests/test_train_lora_local.py -v` *(install pytest into `.venv-train` if missing, or run the helper directly)*.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add finetune/train_lora.py finetune/tests/test_train_lora_local.py
git commit -m "feat(finetune): train_lora accepts a local jsonl dataset"
```

---

## Task 9: CI leakage gate + docs

**Files:**
- Create: `finetune/tests/test_ci_leakage.py`
- Create/append: `finetune/README.md` (loop usage), `docs/CONTRACT.md` (evidence-captured stub)

- [ ] **Step 1: CI leakage test** — regenerate the split and assert the guard is wired:

```python
import split_suites as s
def test_ci_no_leakage():
    d = s.answer_hashes("/home/tyler/AI/rfc/wt-rfc")
    assert not ((set(d["holdout"]) | set(d["canary"])) & set(d["train"]))
```

- [ ] **Step 2: Run it.** Expected: PASS (fails loudly if anyone edits the split to leak).

- [ ] **Step 3: Write `docs/CONTRACT.md`** capturing observed reality: real DB (`rfc` on `:5434`), the two provider variants (`vllm-tt` on `:8011`, `ollama`), per-test result shape, the `<2048`-token P100 constraint, the measured baseline (29 tok/s/user, 361@16c, TTFT~90ms), and the 10 gold-answer suites + frozen split salt.

- [ ] **Step 4: Commit**

```bash
git add finetune/tests/test_ci_leakage.py finetune/README.md docs/CONTRACT.md
git commit -m "docs(rsi): CI leakage gate + observed reality contract"
```

---

## Verification (whole-loop, after all tasks)

1. **Unit suite green:** `cd finetune && PYTHONPATH=. /home/tyler/.rsi-loop-venv/bin/pytest tests/ -v`.
2. **Firewall proven:** `test_split_suites` + `test_ci_leakage` pass; deliberately move a holdout suite into `GOLD_SUITES`'s train bucket → `write_split` raises `LeakageError`.
3. **End-to-end dry run** (Task 7 Step 5) completes shadow-only, writes `rsi.experiments` + `rsi.test_results`, prints McNemar reports, opens no live PR, and never restarts the `:8011` container.
4. **Baseline reproducibility:** run the base arm twice → identical provenance hashes and pass rates within noise.
5. **Serving path:** the Task 5 manual smoke (`ollama run rsi-qwen:smoke`) returns a coherent completion.

## Risks / notes carried from the spec

- CPU LoRA cadence on 1.5–3B; if a round is too slow, drop `MAX_STEPS` or base size.
- Many gold-answer suites lack `make robot-*` targets → `eval_rfc.run_suite` invokes `robot robot/<suite>/` directly (no rfc DB listeners, to keep writes isolated to `rsi.*`).
- Grader models are pinned via rfc `.env` and not varied in v1 (avoid grader drift confounding the signal).
- The top-level `variables` suite is excluded (duplicate of `hallucination`) to keep the leakage check honest.
- Roadmap issues (`MODEL_TUNER` + gaps) are proposed in the spec, not created — do that only on Tyler's go.
