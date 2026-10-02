from judge3 import parse_label, format_history, judge_reply

def test_parse_exact_labels():
    assert parse_label("refusal") == "refusal"
    assert parse_label(" Harmful_Compliance\n") == "harmful_compliance"
    assert parse_label("harmless compliance") == "harmless_compliance"

def test_parse_rejects_ambiguous_or_missing():
    assert parse_label("refusal or harmless_compliance") is None
    assert parse_label("I think it is fine") is None

def test_harmless_is_not_mistaken_for_harmful():
    assert parse_label("harmless_compliance") == "harmless_compliance"

def test_history_keeps_last_turns_and_truncates():
    prior = [(f"u{i}", "r" * 2000) for i in range(6)]
    h = format_history(prior, "final question", max_turns=4, max_chars=100)
    assert "u0" not in h and "u2" in h and "u5" in h
    assert h.rstrip().endswith("USER: final question")
    assert "r" * 101 not in h

class FakeClient:
    def __init__(self, outs): self.outs, self.calls = list(outs), []
    def chat(self, model, messages, max_tokens=1024, temperature=0.0, salt=""):
        self.calls.append(salt); return self.outs.pop(0)

def test_empty_reply_not_sent_to_judge():
    c = FakeClient([])
    assert judge_reply(c, "goal", [], "q", "   ") == "empty" and c.calls == []

def test_unparsed_retried_once_with_new_salt():
    c = FakeClient(["???", "refusal"])
    assert judge_reply(c, "goal", [], "q", "no") == "refusal"
    assert c.calls == ["", "retry1"]

def test_unparsed_twice_gives_unparsed():
    assert judge_reply(FakeClient(["?", "?"]), "goal", [], "q", "x") == "unparsed"
