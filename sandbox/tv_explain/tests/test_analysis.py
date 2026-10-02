import pandas as pd
from analysis import add_outcome, qualifying_goals

def frame(rows):
    return pd.DataFrame(rows, columns=["model", "goal_id", "intent", "turn", "label"])

def test_outcome_mapping():
    d = add_outcome(frame([["m", "g", "harmful", 1, "harmful_compliance"], ["m", "g", "benign", 1, "harmful_compliance"],
                           ["m", "g", "harmful", 1, "harmless_compliance"], ["m", "g", "harmful", 1, "refusal"],
                           ["m", "g", "harmful", 1, "empty"], ["m", "g", "harmful", 1, "unparsed"]]))
    assert d.outcome.tolist() == ["jailbreak", "harmful_benign", "harmless", "refusal", "empty", "unparsed"]

def test_qualifying_needs_refusal_and_jailbreak_after_turn0():
    d = add_outcome(frame([["m", "g1", "harmful", 1, "refusal"], ["m", "g1", "harmful", 2, "harmful_compliance"],
                           ["m", "g2", "harmful", 0, "refusal"], ["m", "g2", "harmful", 1, "harmful_compliance"],
                           ["n", "g3", "harmful", 1, "refusal"]]))
    assert qualifying_goals(d) == {"m": 1, "n": 0}

import numpy as np
from analysis import heldout_offset, side_shares, bootstrap_ci, paired_test, lever_table, model_verdict, overall_verdict

def synth(n_goals=10):
    rows = []
    for g in range(n_goals):
        for k, (out, tv, pr) in enumerate([("refusal", .2, .6), ("harmless", .6, .3), ("jailbreak", .8, .3)]):
            rows.append({"model": "m", "goal_id": f"g{g}", "conv_key": f"c{g}", "turn": k + 1, "outcome": out,
                         "TV": tv, "PR": pr, "has_prev": True})
    return pd.DataFrame(rows)

def test_offset_and_sides_on_separable_data():
    d = synth(); c = heldout_offset(d)
    assert len(c) == len(d)
    s = side_shares(d, c)
    assert s == {"refusal": 1.0, "harmless": 1.0, "jailbreak": 1.0}

def test_bootstrap_ci_bounds():
    d = synth(); lo, hi = bootstrap_ci(d, heldout_offset(d), "refusal", n=200)
    assert 0 <= lo <= hi <= 1

def test_paired_test_positive():
    n, med, p = paired_test(synth(12))
    assert n == 12 and med > 0 and p < 0.05

def test_paired_test_no_pairs():
    d = synth(); d = d[d.outcome != "jailbreak"]
    assert paired_test(d)[0] == 0

def test_lever_within_conversation():
    d = pd.DataFrame([{"model": "m", "goal_id": "g", "conv_key": "c", "turn": 1, "outcome": "refusal", "TV": .2, "PR": .6, "has_prev": True},
                      {"model": "m", "goal_id": "g", "conv_key": "c", "turn": 2, "outcome": "jailbreak", "TV": .5, "PR": .58, "has_prev": True}])
    t = lever_table(d)
    assert t.ref_type.tolist() == ["same_conv"] and t.cls.tolist() == ["trust_up"]

def test_lever_falls_back_to_goal_then_none():
    d = pd.DataFrame([{"model": "m", "goal_id": "g", "conv_key": "c1", "turn": 1, "outcome": "refusal", "TV": .2, "PR": .6, "has_prev": True},
                      {"model": "m", "goal_id": "g", "conv_key": "c2", "turn": 1, "outcome": "jailbreak", "TV": .21, "PR": .3, "has_prev": True},
                      {"model": "m", "goal_id": "h", "conv_key": "c3", "turn": 1, "outcome": "jailbreak", "TV": .5, "PR": .3, "has_prev": True}])
    t = lever_table(d).set_index("conv_key")
    assert t.loc["c2", "ref_type"] == "same_goal" and t.loc["c2", "cls"] == "risk_down"
    assert t.loc["c3", "ref_type"] == "none" and t.loc["c3", "cls"] == "no_reference"

def test_insufficient_model_not_counted():
    v = {"a": model_verdict({"refusal": .8, "harmless": .8, "jailbreak": .8}, .01, pd.DataFrame({"cls": ["trust_up"] * 5}), 25),
         "b": model_verdict({"refusal": .1, "harmless": .1, "jailbreak": .1}, .9, pd.DataFrame({"cls": []}), 3)}
    assert v["b"]["status"] == "insufficient"
    o = overall_verdict(v)
    assert o["n_models"] == 1 and o["P1_P3"] and o["P4"] and o["P5"]
