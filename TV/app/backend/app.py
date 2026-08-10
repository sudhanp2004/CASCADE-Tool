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


def ensure_embedder_service():
    embedder_url = "http://127.0.0.1:5001/similarity"
    try:
        requests.post(embedder_url, json={"text": "test"}, timeout=1)
        print("[*] Embedder microservice is already running.")
        return
    except requests.exceptions.RequestException:
        print("[*] Embedder microservice not detected. Starting it now...")
        embedder_path = os.path.join(TV_DIR, "embedder_service.py")
        subprocess.Popen([sys.executable, embedder_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            try:
                requests.post(embedder_url, json={"text": "test"}, timeout=1)
                print("[*] Embedder microservice successfully started on port 5001.")
                return
            except requests.exceptions.RequestException:
                time.sleep(1)
        print("[!] Warning: Embedder microservice failed to start within 30 seconds.")


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
    app.run(host="0.0.0.0", port=5050, debug=True)
