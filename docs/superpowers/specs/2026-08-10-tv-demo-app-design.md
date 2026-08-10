# Trust Vector Demo App — Design

**Date:** 2026-08-10
**Status:** Approved
**Location:** `TV/app/`

## Purpose

A standalone application (backend + UI) that demonstrates the Trust Vector (TV)
metric live, for a thesis/defense showcase. It replays real, held-out
conversations from the four multi-turn datasets already validated in
`sandbox_data/TV_Threshold_Test_Report.md`, computing TV turn-by-turn exactly
as the live CASCADE attack pipeline does, and shows whether TV's live verdict
matches the dataset's own ground-truth safe/jailbreak label.

This is a demo/showcase tool, not part of the production CASCADE-Tool attack
pipeline. It must run independently, without any of the Groq/Portkey/target-model
API keys the main `backend/app.py` requires.

## Non-goals

- Not a replacement for `sandbox_data/evaluate_*.py` (offline threshold-fitting
  scripts) or `threshold_sweep.py`. Thresholds are precomputed and hardcoded
  here, not refit.
- No support for the 3 single-turn datasets (WildGuardMix, BeaverTails,
  SorryBench-Human-Judgement) — excluded per design discussion, since there's
  no multi-turn TV climb to animate for them.
- No live target-model calls, no attacker LLM, no Groq/Portkey — this app only
  replays pre-recorded dataset conversations.

## Architecture

Standalone Flask app under `TV/app/`, independent of `backend/app.py`.

**Why standalone instead of extending the main backend:** the main app imports
`key_manager` at module load time (reads Groq/Portkey/API keys from `.env`)
and is oriented around live attack orchestration. Bolting a dataset-replay
demo onto that file would couple an unrelated concern to code that shouldn't
need any of those keys to run, and would add more surface area to an
already-1300-line file. A standalone app needs only: Flask, `requests`,
`spacy` (`en_core_web_sm`), `textstat`, `numpy`, `scikit-learn` (only for
rebuilding the train/test split at startup, not fit — thresholds are
hardcoded).

```
TV/app/
  backend/
    app.py          # Flask routes, SSE stream endpoint, embedder auto-start
    loaders.py       # 4 per-dataset parsers + train/test split reconstruction
    tv_engine.py      # calculate_metrics() — copied from TV/calculate_metrics.py
  frontend/
    templates/
      index.html
    static/
      css/style.css   # adapted from frontend/static/css/style.css (dark theme, gauge, chat bubbles)
      js/demo.js
```

The embedder microservice (`TV/embedder_service.py`) is auto-started as a
background subprocess on port 5001 by `backend/app.py`, identical to how the
main app's `ensure_embedder_service()` works — same embedding pipeline, same
port, so TV values are computed exactly the same way as in the validated
offline runs and the live attack tool.

## Data loading & the train/test split

At startup, `loaders.py` rebuilds the same train/test split each
`evaluate_*.py` script used, and only samples demo conversations from the
**test** split — the genuinely held-out data the report's accuracy numbers
describe.

| Dataset | Source file(s) in `TV/datasets/` | Format | Ground truth | Split |
|---|---|---|---|---|
| Anthropic | `anthropic/{train,test}.json` | `transcript` string, `Human:`/`Assistant:` blocks | Heuristic: refusal-string match on final assistant turn | Pre-split files, use `test.json` |
| Tom-Gibbs | `tomgibbs/{train,test}.json` | `Multi-turn conversation` (stringified `ast.literal_eval`-able list of `{role,content}`) | **Ground truth**: `Jailbroken.Multi-turn` field (stringified dict) | Pre-split files, use `test.json` |
| SafeMTData_1K | `safemtdata/SafeMTData_1K.json` | `conversations` (list of `{role,content}`) | Heuristic: refusal-string match on final assistant turn | Reconstruct via `train_test_split(test_size=0.2, random_state=42, stratify=labels)`, matching `evaluate_safemtdata.py` |
| CoSafe | `cosafe/cosafe_all_rollouts_guardgen.json` | `conversation` (history) + 4 `model_rollouts` + `rewards` per row | **Ground truth**: guard-model `reward` (1=safe, 0=unsafe) per rollout | Reconstruct via `train_test_split(test_size=0.2, random_state=42, stratify=labels_per_row)`, matching `evaluate_cosafe.py`; each row expands into up to 4 (history + one rollout) examples |

Each loader normalizes its dataset into a common in-memory record:

```python
{
  "id": str,                                # stable index into the source list
  "turns": [{"role": "human"|"assistant", "text": str}, ...],  # full conversation, in order
  "ground_truth": "jailbreak" | "safe",
}
```

Malformed rows (e.g. Tom-Gibbs entries where `ast.literal_eval` fails) are
skipped at load time; the count of skipped rows per dataset is logged to
stdout at startup, matching the existing `evaluate_*.py` behavior.

CoSafe note: the `conversation` history is replayed as-is (human/assistant
turns), then the selected `model_rollouts[i]` is appended as the final
assistant turn, with `ground_truth` set from `rewards[i]` (0 → jailbreak, 1 →
safe).

## Threshold table (hardcoded, from `TV_Threshold_Test_Report.md`)

```python
DATASET_THRESHOLDS = {
    "anthropic":   0.3358,
    "tomgibbs":    0.5128,
    "safemtdata":  0.4845,
    "cosafe":      0.4066,
}
```

## The core mechanic — live, blind threshold crossing

`TV_Threshold_Test_Report.md`'s accuracy numbers use an **asymmetric**
measurement: jailbreak conversations are checked at turn N-1, safe
conversations at the final turn N — chosen knowing the label in advance, for
validation purposes.

This app instead does the **blind, live** version matching how TV would
actually gate a real conversation: as each assistant turn's TV is computed in
order, the app checks `TV >= threshold` with no knowledge of the label. The
first turn where this is true is flagged as the crossing point. This is
stricter than the report's headline numbers (see report §3.2, the symmetric
N-1/N-1 case) — a small info tooltip near the verdict panel explains this
distinction, linking the discrepancy back to the report rather than hiding it.

**Verdict rule:** `predicted = "jailbreak" if TV crossed the threshold at any
turn, else "safe"`. Compared against `ground_truth` for the ✓/✗ badge.

## Backend API

- `GET /` — renders `index.html`.
- `GET /api/datasets` — returns the 4 datasets with their threshold and
  test-split example counts (total / jailbreak / safe), for populating the
  selector panel.
- `GET /api/demo/stream?dataset=<key>&label=any|jailbreak|safe` — SSE stream.
  Picks one random matching conversation from that dataset's test split (or
  errors if no example matches the requested label) and streams:
  - `turn_start` (`role`, `turn_index`)
  - `turn_text` (full text for that turn — no token-level streaming needed
    since this is replaying static text, not a live model call)
  - `tv_update` (after each assistant turn: `tv`, `emb`, `hedge`, `info`,
    `read`, `turn_index`) — reuses `calculate_metrics()` unchanged
  - `threshold_crossed` (emitted once, on the first turn where `tv >=
    threshold`)
  - `verdict` (`predicted`, `ground_truth`, `correct: bool`) — final event

This mirrors the SSE pattern already used by `/api/attack/stream` in
`backend/app.py`, so `demo.js` can reuse the same `fetch` + `ReadableStream`
parsing approach as `stream.js`.

## Frontend

Single page, dark theme reused from `frontend/static/css/style.css`:

- **Dataset panel** (left): select one of 4 datasets (shows its threshold +
  test-split counts), a label filter (Any / Force jailbreak / Force safe),
  and a **Start** button.
- **Chat window** (center): human/assistant bubbles appended as `turn_start`
  + `turn_text` events arrive; an inline "⚠ THRESHOLD CROSSED" divider drops
  into the chat when that event fires. Conversation keeps playing to the end
  regardless of when the threshold crosses, for full context.
- **Telemetry panel** (right): live TV gauge (reusing the existing
  `.gauge-block`/`.fill` CSS classes), current turn counter, and — once the
  `verdict` event arrives — a result card: predicted verdict, ground-truth
  label, ✓/✗ correct badge, and the final turn's metric breakdown (semantic
  alignment / info density / refusal gate).

## Error handling

- Embedder service unreachable: `calculate_metrics()` already degrades to
  `sim_refusal=sim_compliance=0.0` (existing behavior, unchanged) — TV still
  computes, just without the semantic-alignment term contributing signal.
  Surfaced as a small warning badge in the telemetry panel if the embedder
  ping fails at startup.
- Dataset file missing from `TV/datasets/`: that dataset is excluded from
  `/api/datasets` (not shown in the selector) rather than erroring the whole
  app, with a startup log line.
- No test-split example matches the requested `label` filter (shouldn't
  happen given known class balances, but defensively handled): SSE stream
  emits an `error` event instead of hanging.

## Testing

This is a demo tool, not production infrastructure — testing is scoped
accordingly (YAGNI: no full pytest suite).

- **Loader sanity check** (manual script or `if __name__ == "__main__"` block
  in `loaders.py`): each of the 4 loaders parses a non-trivial number of
  test-split examples successfully, printing counts and skip counts — same
  shape as the `print()` diagnostics already in `evaluate_*.py`.
- **Manual verification**: start the app, run through all 4 datasets with
  each label filter (Any/Force jailbreak/Force safe), confirm the gauge
  updates, the threshold-crossed marker appears at the right point, and the
  verdict/ground-truth/correct badge renders sensibly.
