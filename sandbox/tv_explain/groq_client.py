"""Groq chat client: rotates over all GROQ_API_KEY* in keys.json, gives gpt-oss a reasoning budget, caches every
non-empty reply on disk (resumable, thread-safe). Raises GroqExhausted instead of returning "" when every try fails."""
import hashlib, json, os, threading, time
import requests

URL = "https://api.groq.com/openai/v1/chat/completions"
KEYS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tv_gen", "keys_ds", "keys.json")

class GroqExhausted(RuntimeError):
    pass

def load_keys(path=KEYS_JSON):
    d = json.load(open(path))
    return [v for k, v in sorted(d.items()) if k.startswith("GROQ_API_KEY") and str(v).startswith("gsk_")]

def _http_post(key, payload):
    r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=120)
    return r.status_code, (r.json() if r.status_code == 200 else None)

class GroqClient:
    def __init__(self, keys, cache_path, post=_http_post, sleep=time.sleep, rounds=1, round_wait=30):
        assert keys, "no Groq keys"
        self.keys, self.post, self.sleep, self.i = list(keys), post, sleep, 0
        self.rounds, self.round_wait = rounds, round_wait
        self.cache_path, self.cache, self.lock = cache_path, {}, threading.Lock()
        if os.path.exists(cache_path):
            for line in open(cache_path):
                r = json.loads(line); self.cache[r["k"]] = r["v"]

    def chat(self, model, messages, max_tokens=1024, temperature=0.0, salt=""):
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if "gpt-oss" in model:
            payload["reasoning_effort"] = "low"; payload["max_tokens"] = max_tokens + 512
        k = hashlib.sha256(json.dumps([payload, salt], sort_keys=True).encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        for n in range(len(self.keys) * 3 * self.rounds):
            if n and n % (len(self.keys) * 3) == 0:
                self.sleep(self.round_wait)              # every key failed this round: wait out the per-minute limit
            with self.lock:
                key = self.keys[self.i % len(self.keys)]
            try:
                code, body = self.post(key, payload)
            except Exception:
                with self.lock: self.i += 1
                self.sleep(2); continue
            if code == 200:
                text = (body["choices"][0]["message"].get("content") or "").strip()
                if text:
                    with self.lock:
                        self.cache[k] = text
                        with open(self.cache_path, "a") as f:
                            f.write(json.dumps({"k": k, "v": text}) + "\n")
                    return text
                with self.lock: self.i += 1
                continue
            if code in (429, 413, 498, 401, 403):
                with self.lock: self.i += 1
                self.sleep(1); continue
            self.sleep(2)
        raise GroqExhausted(f"{model}: no non-empty reply after {len(self.keys) * 3 * self.rounds} tries")
