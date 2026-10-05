"""Routes the rewriter/judge model (gpt-oss-120b in the shared code) to Claude Haiku 4.5 for this spike only;
target models still go to Groq. Haiku replies are cached in their own file (resumable)."""
import hashlib, json, os, threading
import anthropic

HAIKU = "claude-haiku-4-5"
ROUTED = {"openai/gpt-oss-120b"}
ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env.example")

def _key():
    for line in open(ENV):
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("ANTHROPIC_API_KEY not found")

class Router:
    def __init__(self, groq, cache_path):
        self.groq, self.cache_path, self.lock = groq, cache_path, threading.Lock()
        self.client = anthropic.Anthropic(api_key=_key(), max_retries=6)
        self.cache = {}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                r = json.loads(line); self.cache[r["k"]] = r["v"]

    def chat(self, model, messages, max_tokens=1024, temperature=0.0, salt=""):
        if model not in ROUTED:
            return self.groq.chat(model, messages, max_tokens=max_tokens, temperature=temperature, salt=salt)
        k = hashlib.sha256(json.dumps([HAIKU, messages, max_tokens, temperature, salt], sort_keys=True).encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        r = self.client.messages.create(model=HAIKU, max_tokens=max_tokens, temperature=temperature, messages=messages)
        text = "".join(b.text for b in r.content if b.type == "text").strip()
        with self.lock:
            self.cache[k] = text
            with open(self.cache_path, "a") as f: f.write(json.dumps({"k": k, "v": text}) + "\n")
        return text
