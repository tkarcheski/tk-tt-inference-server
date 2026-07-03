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
