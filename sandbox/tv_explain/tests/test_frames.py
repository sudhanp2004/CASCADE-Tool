import json
import pandas as pd
from frames import from_messages, load_attack, load_benign, prior_pairs, COLS

def conv(key, msgs, jbb, intent="harmful"):
    return {"conv_key": key, "source": "t", "model": "m", "goal": "g", "goal_id": "g", "intent": intent,
            "messages": msgs, "jbb": jbb}

def test_one_row_per_assistant_message_with_turn_index():
    m = [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"},
         {"role": "user", "content": "u1"}, {"role": "assistant", "content": ""}]
    d = from_messages([conv("c", m, [0, 1])])
    assert list(d.columns) == COLS
    assert d.turn.tolist() == [0, 1] and d.user.tolist() == ["u0", "u1"] and d.reply.tolist() == ["a0", ""]
    assert d.jbb.tolist() == [0, 1]

def test_label_length_mismatch_raises():
    m = [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}]
    try:
        from_messages([conv("c", m, [0, 1])]); assert False
    except ValueError:
        pass

def test_prior_pairs_within_conversation_only():
    m = [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"},
         {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    d = from_messages([conv("c1", m, [0, 0]), conv("c2", m, [0, 0])])
    p = prior_pairs(d)
    assert p[0] == [] and p[1] == [("u0", "a0")] and p[2] == []

def test_load_attack_real_files():
    d = load_attack()
    assert set(d.source) == {"cascade", "tv_gen"} and (d.intent == "harmful").all()
    assert d.groupby("source").conv_key.nunique().to_dict() == {"cascade": 250, "tv_gen": 400}
    assert d.model.nunique() == 7

def test_load_benign(tmp_path):
    rec = {"goal_index": 3, "harmful_goal": "H", "benign_goal": "B", "target": "openai/gpt-oss-20b", "attempt": 1,
           "turns": [{"turn": 0, "user": "hi", "target": "hello"}, {"turn": 1, "user": "more", "target": ""}]}
    p = tmp_path / "b.jsonl"; p.write_text(json.dumps(rec) + "\n")
    d = load_benign(str(p))
    assert d.goal.tolist() == ["B", "B"] and d.goal_id.tolist() == ["H", "H"]
    assert (d.intent == "benign").all() and (d.jbb == -1).all() and d.reply.tolist() == ["hello", ""]
