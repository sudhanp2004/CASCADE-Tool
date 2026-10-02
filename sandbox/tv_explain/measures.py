"""Frozen TV and PR (exactly as in sandbox/tv_test): TV_t = instantaneous trust of the previous non-empty reply,
gate*(0.6*S_E + 0.4*I_D); PR_t = request_probe on the user message. Empty replies are dropped."""
import os, sys
import numpy as np, pandas as pd

TT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tv_test")

def _default_fns():
    sys.path.insert(0, TT)
    from risk_signal import enc
    from components import s_e, i_d
    ns = {"__file__": os.path.join(TT, "eval_baseline.py")}
    exec(open(os.path.join(TT, "eval_baseline.py")).read().split("cas = pd.read_parquet")[0], ns)
    gate, FIX = ns["gate"], ns["FIX"]
    QP = np.load(os.path.join(TT, "request_probe.npz"))
    def trust_fn(xs):
        E = enc(xs); return np.array([gate(x, FIX) for x in xs]) * (.6 * s_e(E) + .4 * np.array([i_d(x) for x in xs]))
    def pr_fn(xs):
        return 1 / (1 + np.exp(-(enc(xs) @ QP["coef"] + float(QP["intercept"]))))
    return trust_fn, pr_fn

def add_tv_pr(df, trust_fn=None, pr_fn=None):
    if trust_fn is None or pr_fn is None:
        trust_fn, pr_fn = _default_fns()
    d = df[df.reply.str.strip() != ""].copy()
    reps, reqs = sorted(set(d.reply)), sorted(set(d.user))
    T = dict(zip(reps, trust_fn(reps))); P = dict(zip(reqs, pr_fn(reqs)))
    tv, hp, last_key, prev = [], [], None, None
    for r in d.itertuples():
        if r.conv_key != last_key: last_key, prev = r.conv_key, None
        tv.append(0.0 if prev is None else float(T[prev])); hp.append(prev is not None); prev = r.reply
    d["TV"], d["has_prev"], d["PR"] = tv, hp, [float(P[u]) for u in d.user]
    return d.reset_index(drop=True)
