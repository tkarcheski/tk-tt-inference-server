import dataset_loader


def test_local_jsonl_branch(tmp_path):
    ds = tmp_path / "train.jsonl"
    ds.write_text('{"messages":[{"role":"user","content":"hi"},'
                  '{"role":"assistant","content":"yo"}]}\n')
    d = dataset_loader.load_local_or_hub(str(ds))
    assert "train" in d and len(d["train"]) == 1
