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
