from groq_client import GroqExhausted
from gen_benign import run_conversation, SIM_MODEL

class Fake:
    def __init__(self): self.n = 0
    def chat(self, model, messages, max_tokens=1024, temperature=0.0, salt=""):
        self.n += 1
        if model == SIM_MODEL: return f"question {self.n}"
        if self.n == 4: raise GroqExhausted("x")
        return f"answer {self.n}"

def test_conversation_shape_and_empty_on_exhausted_target():
    t = run_conversation(Fake(), "learn locks", "openai/gpt-oss-20b", 0, n_turns=3)
    assert [x["turn"] for x in t] == [0, 1, 2]
    assert all(x["user"].startswith("question") for x in t)
    assert t[1]["target"] == ""            # 4th call = target at turn 1 -> exhausted -> empty reply kept
    assert t[2]["target"].startswith("answer")
