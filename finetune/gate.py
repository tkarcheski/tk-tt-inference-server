import contextlib
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
    with contextlib.closing(rsi_common.connect()) as c, c.cursor() as cur:
        cur.execute(q, (base_id, tuned_id, pool, base_id, tuned_id))
        return [(r[3], r[4]) for r in cur.fetchall()]

def mcnemar_gate(base_id, tuned_id, pool, **kw):
    return mcnemar_from_pairs(matched_outcomes(base_id, tuned_id, pool), **kw)
