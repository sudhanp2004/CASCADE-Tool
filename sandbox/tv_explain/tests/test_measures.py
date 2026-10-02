import numpy as np, pandas as pd
from measures import add_tv_pr

def test_tv_is_previous_nonempty_reply_and_has_prev():
    d = pd.DataFrame({"conv_key": ["a", "a", "a", "a", "b"], "turn": [0, 1, 2, 3, 0],
                      "user": ["u0", "u1", "u2", "u3", "v0"], "reply": ["r0", "", "r2", "r3", "s0"]})
    trust = {"r0": .9, "r2": .2, "r3": .5, "s0": .7}
    out = add_tv_pr(d, trust_fn=lambda xs: np.array([trust[x] for x in xs]), pr_fn=lambda xs: np.array([len(x) / 10 for x in xs]))
    assert out.turn.tolist() == [0, 2, 3, 0]                     # empty reply dropped
    assert out.TV.tolist() == [0.0, .9, .2, 0.0]                 # turn 2 gets r0, not the empty turn-1 reply
    assert out.has_prev.tolist() == [False, True, True, False]
    assert np.allclose(out.PR, [.2, .2, .2, .2])

def test_first_reply_empty_means_no_prev():
    d = pd.DataFrame({"conv_key": ["a", "a"], "turn": [0, 1], "user": ["u0", "u1"], "reply": ["", "r1"]})
    out = add_tv_pr(d, trust_fn=lambda xs: np.ones(len(xs)), pr_fn=lambda xs: np.zeros(len(xs)))
    assert out.turn.tolist() == [1] and out.has_prev.tolist() == [False]
