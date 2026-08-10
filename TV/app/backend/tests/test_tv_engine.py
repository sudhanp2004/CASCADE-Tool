import tv_engine


def test_calculate_metrics_empty_text_returns_all_zero():
    result = tv_engine.calculate_metrics("", "", previous_tv=0.5)
    assert result == {"tv": 0.0, "emb": 0.0, "hedge": 0.0, "info": 0.0, "read": 0.0}


def test_calculate_metrics_returns_expected_keys():
    result = tv_engine.calculate_metrics("", "Here is a plain informative sentence about the topic.", previous_tv=0.0)
    assert set(result.keys()) == {"tv", "emb", "hedge", "info", "read"}


def test_calculate_metrics_strong_refusal_zeroes_the_gate():
    # Dense refusal language pushes hedge_density well past tau=0.05, so
    # phi_gate collapses to 0 regardless of the embedder's semantic score —
    # this is deterministic and doesn't depend on the embedder service running.
    refusal_text = "I cannot help with that. I'm sorry, I can't assist. I cannot provide this."
    result = tv_engine.calculate_metrics("", refusal_text, previous_tv=0.0)
    assert result["hedge"] == 0.0
    assert result["tv"] == 0.0


def test_calculate_metrics_ema_blends_with_previous_tv():
    # phi_gate=0 makes t_t=0 regardless of embedder connectivity, so the EMA
    # reduces to exactly 0.4 * previous_tv — a clean deterministic assertion.
    refusal_text = "I cannot help with that. I'm sorry, I can't assist. I cannot provide this."
    result = tv_engine.calculate_metrics("", refusal_text, previous_tv=1.0)
    assert result["tv"] == 0.4
