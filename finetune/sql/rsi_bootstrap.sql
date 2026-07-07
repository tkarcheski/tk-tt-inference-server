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
  PRIMARY KEY (experiment_id, suite_id, test_id, pool, repeat_idx)
);

-- One row per auto-published model (a round that cleared the promotion gate).
-- UNIQUE(tuned_id) gives the loop idempotency: a tuned model publishes at most once.
CREATE TABLE IF NOT EXISTS rsi.publications (
  version           text PRIMARY KEY,            -- v{seed}-{adapter_hash[:8]}
  tuned_id          uuid NOT NULL REFERENCES rsi.experiments(experiment_id),
  submodule_commit  text,                        -- git-LFS registry commit SHA
  release_url       text,                        -- GitHub release on the fork
  holdout_delta_pp  numeric,
  canary_delta_pp   numeric,
  created_at        timestamptz NOT NULL DEFAULT now(),
  UNIQUE (tuned_id)
);

-- The rolling "champion" baseline pushed to the Ollama registry
-- (tkarcheski/rsi-qwen:3b-latest). A round lands here only when it clears the
-- promotion gate AND the ollama push succeeds. The is_current row is the baseline
-- the loop's next round must beat (self-improving ratchet). cluster_status tracks
-- large-cluster (RFC-chat) validation feedback.
CREATE TABLE IF NOT EXISTS rsi.baselines (
  version           text PRIMARY KEY,            -- v{seed}-{adapter_hash[:8]}
  tuned_id          uuid NOT NULL REFERENCES rsi.experiments(experiment_id),
  ollama_ref        text NOT NULL,               -- tkarcheski/rsi-qwen:3b-latest
  holdout_delta_pp  numeric,
  canary_delta_pp   numeric,
  pushed_at         timestamptz NOT NULL DEFAULT now(),
  is_current        boolean NOT NULL DEFAULT true,
  cluster_status    text NOT NULL DEFAULT 'pending',  -- pending | validated | rejected
  UNIQUE (tuned_id)
);
-- At most one current champion at a time.
CREATE UNIQUE INDEX IF NOT EXISTS rsi_baselines_one_current
  ON rsi.baselines (is_current) WHERE is_current;
