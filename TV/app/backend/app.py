import json
import os
import subprocess
import sys
import time

import requests
from flask import Flask, Response, jsonify, render_template, request

import loaders
from tv_engine import calculate_metrics

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(BACKEND_DIR)
TV_DIR = os.path.dirname(APP_DIR)

app = Flask(
    __name__,
    template_folder=os.path.join(APP_DIR, "frontend", "templates"),
    static_folder=os.path.join(APP_DIR, "frontend", "static"),
)

DATASET_INDEX = loaders.get_dataset_index()

EMBEDDER_URL = "http://127.0.0.1:5001/similarity"


def is_embedder_healthy():
    """Pings the embedder microservice. Returns True/False — never raises.

    Used both by the startup auto-launcher and by /api/embedder_status so the
    frontend can detect a down embedder instead of silently getting a biased
    s_e_hat=0.5 out of tv_engine.calculate_metrics()'s fallback."""
    try:
        requests.post(EMBEDDER_URL, json={"text": "test"}, timeout=1)
        return True
    except requests.exceptions.RequestException:
        return False


def ensure_embedder_service():
    if is_embedder_healthy():
        print("[*] Embedder microservice is already running.")
        return

    print("[*] Embedder microservice not detected. Starting it now...")
    embedder_path = os.path.join(TV_DIR, "embedder_service.py")
    log_path = os.path.join(BACKEND_DIR, "embedder.log")
    log_file = open(log_path, "a")
    subprocess.Popen([sys.executable, embedder_path], stdout=log_file, stderr=log_file)
    for _ in range(30):
        if is_embedder_healthy():
            print("[*] Embedder microservice successfully started on port 5001.")
            return
        time.sleep(1)
    print(f"[!] Warning: Embedder microservice failed to start within 30 seconds. See {log_path} for details.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    result = []
    for key, data in DATASET_INDEX.items():
        test = data["test"]
        n_jb = sum(1 for c in test if c["ground_truth"] == "jailbreak")
        n_safe = sum(1 for c in test if c["ground_truth"] == "safe")
        result.append({
            "key": key,
            "name": data["name"],
            "threshold": data["threshold"],
            "test_total": len(test),
            "test_jailbreak": n_jb,
            "test_safe": n_safe,
        })
    return jsonify(result)


@app.route("/api/embedder_status", methods=["GET"])
def embedder_status():
    return jsonify({"ok": is_embedder_healthy()})


@app.route("/api/demo/stream", methods=["GET"])
def demo_stream():
    dataset_key = request.args.get("dataset", "").strip()
    label_filter = request.args.get("label", "any").strip()

    if dataset_key not in DATASET_INDEX:
        error_payload = json.dumps({"error": f"Unknown dataset: {dataset_key}"})
        return Response(f"data: {error_payload}\n\n", mimetype="text/event-stream")

    conversation = loaders.pick_conversation(DATASET_INDEX, dataset_key, label_filter)
    if conversation is None:
        error_payload = json.dumps({"error": f"No '{label_filter}' examples in {dataset_key} test split"})
        return Response(f"data: {error_payload}\n\n", mimetype="text/event-stream")

    threshold = DATASET_INDEX[dataset_key]["threshold"]

    def generate():
        prev_tv = 0.0
        crossed = False
        for turn_index, turn in enumerate(conversation["turns"]):
            yield f"data: {json.dumps({'status': 'turn_start', 'turn_index': turn_index, 'role': turn['role']})}\n\n"
            yield f"data: {json.dumps({'status': 'turn_text', 'turn_index': turn_index, 'text': turn['text']})}\n\n"

            if turn["role"] == "assistant":
                metrics = calculate_metrics("", turn["text"], previous_tv=prev_tv)
                prev_tv = metrics["tv"]
                update_payload = {"status": "tv_update", "turn_index": turn_index}
                update_payload.update(metrics)
                yield f"data: {json.dumps(update_payload)}\n\n"

                if not crossed and metrics["tv"] >= threshold:
                    crossed = True
                    crossed_payload = {
                        "status": "threshold_crossed",
                        "turn_index": turn_index,
                        "tv": metrics["tv"],
                        "threshold": threshold,
                    }
                    yield f"data: {json.dumps(crossed_payload)}\n\n"

        predicted = "jailbreak" if crossed else "safe"
        ground_truth = conversation["ground_truth"]
        verdict_payload = {
            "status": "verdict",
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": predicted == ground_truth,
        }
        yield f"data: {json.dumps(verdict_payload)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    ensure_embedder_service()
    # 127.0.0.1 (not 0.0.0.0) + debug=False: this is a live-demo tool, not
    # meant to expose the Werkzeug interactive debugger on the network, and
    # debug's reloader would otherwise double-load the dataset index (~230MB)
    # into memory on startup.
    app.run(host="127.0.0.1", port=5050, debug=False)
