import json

import app as app_module


def _parse_sse_events(body):
    events = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if chunk.startswith("data: "):
            try:
                events.append(json.loads(chunk[len("data: "):]))
            except json.JSONDecodeError:
                pass
    return events


def test_index_route_returns_200():
    client = app_module.app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200


def test_get_datasets_returns_all_four_with_expected_fields():
    client = app_module.app.test_client()
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    data = resp.get_json()
    keys = {d["key"] for d in data}
    assert keys == {"anthropic", "tomgibbs", "safemtdata", "cosafe"}
    for d in data:
        assert d["threshold"] > 0
        assert d["test_total"] == d["test_jailbreak"] + d["test_safe"]


def test_embedder_status_returns_boolean_ok_field():
    client = app_module.app.test_client()
    resp = client.get("/api/embedder_status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "ok" in data
    assert isinstance(data["ok"], bool)
    # The embedder isn't running in the test environment, so this should be
    # False here — but the important thing this test guards is the shape.


def test_demo_stream_unknown_dataset_returns_single_error_event():
    client = app_module.app.test_client()
    resp = client.get("/api/demo/stream?dataset=doesnotexist&label=any")
    events = _parse_sse_events(resp.get_data(as_text=True))
    assert len(events) == 1
    assert "error" in events[0]


def test_demo_stream_known_dataset_emits_turns_and_verdict():
    client = app_module.app.test_client()
    resp = client.get("/api/demo/stream?dataset=anthropic&label=any")
    events = _parse_sse_events(resp.get_data(as_text=True))
    statuses = [e.get("status") for e in events]
    assert "turn_start" in statuses
    assert "turn_text" in statuses
    assert "verdict" in statuses

    verdict = next(e for e in events if e.get("status") == "verdict")
    assert verdict["predicted"] in ("jailbreak", "safe")
    assert verdict["ground_truth"] in ("jailbreak", "safe")
    assert isinstance(verdict["correct"], bool)


def test_demo_stream_label_filter_forces_ground_truth():
    client = app_module.app.test_client()
    resp = client.get("/api/demo/stream?dataset=anthropic&label=jailbreak")
    events = _parse_sse_events(resp.get_data(as_text=True))
    verdict = next(e for e in events if e.get("status") == "verdict")
    assert verdict["ground_truth"] == "jailbreak"
