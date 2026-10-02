import pytest
from groq_client import GroqClient, GroqExhausted

def ok(text):
    return 200, {"choices": [{"message": {"content": text}}]}

def make(tmp_path, responses):
    calls = []
    def post(key, payload):
        calls.append((key, payload))
        return responses.pop(0)
    return GroqClient(["k1", "k2"], str(tmp_path / "c.jsonl"), post=post, sleep=lambda s: None), calls

def test_cache_hit_skips_second_call(tmp_path):
    c, calls = make(tmp_path, [ok("hi")])
    assert c.chat("m", [{"role": "user", "content": "x"}]) == "hi"
    assert c.chat("m", [{"role": "user", "content": "x"}]) == "hi"
    assert len(calls) == 1

def test_cache_persists_across_instances(tmp_path):
    c, _ = make(tmp_path, [ok("hi")])
    c.chat("m", [{"role": "user", "content": "x"}])
    c2, calls2 = make(tmp_path, [])
    assert c2.chat("m", [{"role": "user", "content": "x"}]) == "hi" and calls2 == []

def test_rotates_key_on_429(tmp_path):
    c, calls = make(tmp_path, [(429, None), ok("hi")])
    assert c.chat("m", [{"role": "user", "content": "x"}]) == "hi"
    assert calls[0][0] != calls[1][0]

def test_empty_content_retried_not_returned(tmp_path):
    c, _ = make(tmp_path, [ok(""), ok("real")])
    assert c.chat("m", [{"role": "user", "content": "x"}]) == "real"

def test_raises_when_exhausted(tmp_path):
    c, _ = make(tmp_path, [(429, None)] * 6)
    with pytest.raises(GroqExhausted):
        c.chat("m", [{"role": "user", "content": "x"}])

def test_gpt_oss_gets_reasoning_budget(tmp_path):
    c, calls = make(tmp_path, [ok("y")])
    c.chat("openai/gpt-oss-120b", [{"role": "user", "content": "x"}], max_tokens=20)
    p = calls[0][1]
    assert p["reasoning_effort"] == "low" and p["max_tokens"] == 532

def test_salt_changes_cache_key(tmp_path):
    c, calls = make(tmp_path, [ok("a"), ok("b")])
    assert c.chat("m", [{"role": "user", "content": "x"}], salt="1") == "a"
    assert c.chat("m", [{"role": "user", "content": "x"}], salt="2") == "b"
    assert len(calls) == 2

def test_extra_rounds_wait_out_rate_limits(tmp_path):
    calls, slept = [], []
    resp = [(429, None)] * 6 + [ok("late")]
    c = GroqClient(["k1", "k2"], str(tmp_path / "c.jsonl"), post=lambda k, p: (calls.append(k), resp.pop(0))[1],
                   sleep=slept.append, rounds=2, round_wait=20)
    assert c.chat("m", [{"role": "user", "content": "x"}]) == "late"
    assert 20 in slept
