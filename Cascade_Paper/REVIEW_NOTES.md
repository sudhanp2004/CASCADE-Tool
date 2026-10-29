# CASCADE paper — audit & publication plan

Last updated: 2026-09-22. Numbers below were recomputed from `benchmark_asr/` (250 session JSONs, 1,494 turns) and `benchmark2_allturn.csv` (HarmBench, 1,243 turns).

## Goal
Publish `paper.tex` to support a PhD application (IIIT Sri City). A journal is preferred over a conference (SCIE/Scopus journals count most). Still to check with Dr. Sethi / IIIT Sri City regulations: are only journal or only SCIE papers counted?

## Target venue
- **Primary: Computers & Security** (Elsevier, Q1), provided the baseline comparisons below are added.
- **Fallback: JISA** (Elsevier, Q1/Q2). Both are free to publish in via the subscription (non-open-access) route.
- Other options: Cybersecurity (Springer, open access, fee), IJIS (Springer), ACM DTRAP, MDPI JCP (fast, fee), ACM AI Letters (short version, free for its first 3 years), OUP J. Cybersecurity, Discover AI.
- Not suitable: ACM TOPS (bar too high), EAAI/KBS (want an AI-method paper), AI Review (surveys only), PACMHCI/TOCHI (HCI), Frontiers critical-infrastructure topic.
- Conferences checked 2026-09-21: MLNLP 2026 (IEEE, deadline 20 Oct 2026) is the only one still open. PRDC (17 Sep), SecITC (14 Sep), DM-AISC@ICDM (5 Sep) and ICAIPS (held Sep) are closed. ICICT 2027 (Springer LNNS, London, 16–19 Feb 2027): later deadline round not confirmed.
- The same paper can't be submitted to a conference and a journal at the same time. A journal extension needs about 30% new content.

## Numbers in paper.tex that are WRONG
1. **Baseline 5.2% mixes judges.** It counts JBB *or* internal-judge hits (13/250), while the 70.0% attack figure uses the JBB judge only. JBB-only baseline = **2.8% (7/250)**. Per-target JBB baseline: Claude 0, DeepSeek 0, Gemini 2%, Llama 4%, Mistral 8%.
2. **Baseline-success sessions never ran the attack.** The frontend stops at the first JBB hit, so the 7 sessions where the direct request succeeded have 0 attack turns and count as CASCADE failures. The McNemar test is invalid as set up. The data gives a 175/7 split, not the paper's 170/8.
3. **2 human-flagged sessions** (DeepSeek "Holocaust poem", Claude "LGBTQ+ harassment", both at turn 10) have JBB and internal scores forced to 1 (`database.set_human_flag`). Automated-only Claude JBB = **2% (1/50)**, not 4%; DeepSeek = 92%. Either remove them or disclose.
4. **Strategy table:** fake_emergency is **3/5 = 60%**, not 4/5 = 80%. H4's chi-square (1.24) already uses 3. The "67.5–80%" range text is wrong.
5. **Internal judge:** 211 turns (~14%) had a failed judge call, which defaults to FALSE; 171 turns are bucket UNKNOWN. This largely explains the low internal rates (Mistral 20%).
6. **Bibliography** has placeholders ("Title of the Huang 2024 Paper…", "Venue/Journal", "CRI Authors", "HiddenDetect Authors", "AutoAdv Authors"). Wrong entries: Crescendo's authors are Russinovich, Salem, Eldan (not Sennrich); Jatmo is Piet et al. (not Carlini); HiddenDetect is not a linguistic-feature survey; LATS "Devang"/"D. Agrawal" is inconsistent.
7. The "Both judges" column is same-turn agreement (30.0%). Session-level both = 38.4%. Define it in the paper.

## Verified correct
JBB ASR 70.0% and per-target values (4/94/80/80/92), Internal and Either columns, mean turn of first success 4.21, H1 (133.71, p=6.25e-28), H2 (125.74), H3 (81.14), H4 (1.24, p=0.743), phase table values, HarmBench table (8.00/33.33/72.00/73.47/97.83), per-turn HarmBench figure. Proxy = qwen/qwen3.6-27b for all 250 sessions. The temporal-override code path never fired (0 unjustified phase advances; 3 minor inconsistent regressions).

## Validity problems to address
- **Circularity:** phase is assigned by TV thresholds, so "TV rises across phases" holds by construction and H3 is confounded. Baseline vs. attack is the only clean comparison.
- **Threshold origin unknown.** Per-model thresholds (0.4356 etc.) were added in commit 4e7749b (2026-07-14). `benchmark_asr_initial/` (254 sessions, Jul 4–8) uses the SAME 50 objectives × SAME 5 models. If the thresholds were fitted on it, that is tuning on the test objectives. **OPEN QUESTION to the user.**
- **Method section describes things the code doesn't do:**
  - Table II lists 7 drift categories; the code has 4 strategies with different names (academic_research, professional_persona, fictional_creative, fake_emergency).
  - Latent intent implantation and delayed triggers are not implemented.
  - The judge doesn't check the 4 listed items.
  - The abstract says SLMs, but none were tested.
  - Readability is computed but not used in TV.
  - "Hedge" column = Φ_gate (1 = no refusal phrases), not hedge density.
- **Same model in several roles:** Llama 3.3 70B is a target, the strategy selector (`strategies.py`), and the JBB judge. The selector falls back to academic_research on any error, which may explain its 50% share.
- JBB judge: Llama 3.3 70B via Groq/Portkey (the official Llama-3-70B is no longer on Groq), plus a refusal-prefix pre-filter that skips the judge. HarmBench classifier = community Q4_K_M GGUF port of HarmBench-Llama-2-13b-cls via Ollama.
- Claude capped at max_tokens 2048; other targets uncapped. Default target temperatures. n=1 run per pair, no confidence intervals.
- Behaviour subset is hand-picked (7 matched to an earlier list + 43 "category diversity"), which reads as cherry-picking.
- Missing: comparison with existing attacks, ethics / responsible-disclosure section, model versions and dates.
- Minor: "IIIT Sricity" → "IIIT Sri City"; intro contains leftover survey text ("Below, we provide a comprehensive review…").

## Plan for Computers & Security
1. Fix every wrong number above and remove or disclose the human-flagged sessions.
2. Rewrite the method section to match the code; add ethics and disclosure sections.
3. **Baselines:** the JBB leaderboard isn't directly comparable (every attack was run on different targets; JBB doesn't prescribe them). Instead:
   - replay JBB artifacts (PAIR, GCG, JBC, prompt + random search) on our 5 targets with the same judge (a transfer caveat applies), or run PAIR adaptively;
   - add one multi-turn baseline: Crescendo (PyRIT implementation).
   - Leaderboard numbers may appear only as an "as reported, different targets" context table.
4. Run all 100 JBB behaviours (excluding the 2 CSAM ones, with a stated reason), not the hand-picked 50.
5. **Validate the judge:** run the Llama 3.3 judge on JBB's human-labelled judge-comparison dataset (~300 items) and report agreement (the official judge gets ~90%).
6. Add Wilson confidence intervals; state model versions, dates and parameters.
7. Convert to the `elsarticle` template; write the cover letter, highlights (3–5 bullets, ≤85 chars each) and declarations.

Next step offered but not started: a JBB artifact replay script (wait for the user's go-ahead).
