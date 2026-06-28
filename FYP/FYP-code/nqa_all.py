# nq_hallucination_pipeline.py
# ============================================================
# Paper-grade Natural Questions (NQ) Hallucination Detector + Strong Mitigation
# FEVER-style closed-loop pipeline adapted to NQ
#
# Goal:
#   "尽量改正确(少幻觉)" ——允许更多 abstain
#
# Key features:
# 1) GOLD uses different NLI model (DeBERTa) to reduce leakage
# 2) Detector uses RoBERTa-large-MNLI
# 3) Threshold tuned on VAL, fixed on TEST
# 4) Paired Monte-Carlo on TEST
# 5) Mitigation with:
#       - always_mild
#       - always_strict_K
#       - detector-conditioned policy
# 6) Stronger factual score with contradiction penalty
# 7) Strict rewrite can use span extraction from reference
#
# Outputs:
# - nq_paper_overall_summary.json / .csv
# - nq_paper_error_cases.csv
# - nq_mitigation_cases.csv
# - nq_answers_cache.csv
#
# Requirements:
#   pip install datasets transformers torch scikit-learn tqdm selfcheckgpt ollama
#   ollama pull llama2:7b
#   ollama pull qwen2.5:7b-instruct
# ============================================================

import os
import ssl
import csv
import json
import re
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
import ollama

# ============================================================
# CONFIG
# ============================================================

# ---------- Files ----------
ANSWER_CACHE_FILE = "nq_answers_cach_all.csv"

OUT_JSON_FILE = "nq_paper_overall_summary_all.json"
OUT_CSV_FILE  = "nq_paper_overall_summary_all.csv"

ERROR_CASES_FILE = "nq_paper_error_cases_all.csv"
MAX_ERROR_LOGS_PER_MODE = 200

MITI_CASES_FILE = "nq_mitigation_cases_all.csv"
MAX_CASES_PER_RUN = 500

# ---------- Sampling ----------
SEED = 42
N_SAMPLES = 5

SUBSET_SIZE = 800
POOL_SIZE = 800
VAL_RATIO = 0.2

NUM_RUNS = 3
SAMPLE_SIZE = 200

# ---------- Ollama ----------
GEN_MODEL   = "llama2:7b"
JUDGE_MODEL = "qwen2.5:7b-instruct"

# ---------- NLI models ----------
DETECTOR_NLI_MODEL = "roberta-large-mnli"

GOLD_NLI_CANDIDATES = [
    "microsoft/deberta-v3-base-mnli",
    "microsoft/deberta-v3-large-mnli",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
]

# ---------- Scoring weights ----------
W_JUDGE = 0.35
W_FACT  = 0.45
W_SC    = 0.20

FACT_CONTRA_ALPHA = 1.8
GOLD_CONTRA_THRESH = 0.50
GOLD_THR = 0.50

# ---------- Mitigation ----------
MITI_K = 10
MILD_TEMP = 0.3
STRICT_TEMP = 0.15

COUNT_EMPTY_AS_ABSTAIN = True
ABSTAIN_IF_STILL_RISKY = True
ABSTAIN_MARGIN = 0.08
MID_KIND = "strict"

USE_SPAN_EXTRACTION_IN_STRICT = True
SPAN_MAX_WORDS = 28

TUNE_T_HIGH = True
VAL_TUNE_MAX_Q = 150
TUNE_W_ABSTAIN = 0.10
TUNE_W_REGRESS = 0.50

# ---------- HF cache ----------
os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface/models"
ssl._create_default_https_context = ssl._create_unverified_context

# ---------- Repro ----------
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ============================================================
# Utilities
# ============================================================

def ensure_csv_header(path, fieldnames):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def safe_mean(xs):
    return float(np.mean(xs)) if xs else 0.0


def safe_std(xs, ddof=0):
    return float(np.std(xs, ddof=ddof)) if xs else 0.0


def load_cache_answers(path):
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = r.get("question", "")
            a = r.get("answer", "")
            if q and a:
                cache.setdefault(q, []).append(a)
    return cache


def append_cache_answers(path, rows):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "answer"])
        for row in rows:
            w.writerow(row)


def normalize_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list) and len(x) > 0:
        return str(x[0]).strip()
    return str(x).strip()


def normalize_question(q):
    if isinstance(q, str):
        return q.strip()
    if isinstance(q, dict):
        t = q.get("text", None)
        if isinstance(t, str):
            return t.strip()
        if isinstance(t, list) and len(t) > 0:
            return str(t[0]).strip()
    if isinstance(q, list) and len(q) > 0:
        return normalize_question(q[0])
    return str(q).strip()


def parse_final_answer_text(answer_text: str):
    """
    Parse NQ final answer from strict/mild rewrite output:
    FINAL_ANSWER: ...
    Fallback: raw text
    """
    if not answer_text:
        return ""
    m = re.search(r"^\s*FINAL_ANSWER\s*:\s*(.+?)\s*$", answer_text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    return answer_text.strip()


# ============================================================
# Load dataset + subset
# ============================================================

print("\nLoading Natural Questions (validation split)...")
raw_dataset = load_dataset(
    "natural_questions",
    split="validation",
    cache_dir=os.environ.get("HF_DATASETS_CACHE", None)
)
print("Dataset loaded:", len(raw_dataset))

# ============================================================
# Extract usable references from Natural Questions
# ============================================================

def extract_reference_from_nq_item(item):

    ann_raw = item.get("annotations", None)
    if not ann_raw:
        return ""

    if isinstance(ann_raw, list):
        if len(ann_raw) == 0:
            return ""
        ann = ann_raw[0]
    else:
        ann = ann_raw

    short_answers = ann.get("short_answers", [])
    yes_no = ann.get("yes_no_answer", "NONE")

    doc = item.get("document_text", "")
    if not doc:
        return ""

    tokens = doc.split(" ")

    # short answer
    if isinstance(short_answers, list):

        for sa in short_answers:

            start = sa.get("start_token")
            end = sa.get("end_token")

            if start is None or end is None:
                continue

            if 0 <= start < len(tokens) and start < end <= len(tokens):

                ref = " ".join(tokens[start:end]).strip()

                if ref:
                    return ref

    # yes/no
    if yes_no in ["YES", "NO"]:
        return yes_no

    return ""
# ============================================================
# Filter dataset for usable references
# ============================================================

valid_items = []

for item in tqdm(raw_dataset, desc="Filtering NQ for valid refs"):

    ref = extract_reference_from_nq_item(item)

    if ref:

        valid_items.append({
            "question": normalize_question(item.get("question", "")),
            "reference": ref
        })


print("Valid items with usable references:", len(valid_items))
if len(valid_items) == 0:
    raise RuntimeError(
        "No valid NQ items were extracted. "
        "Please check extract_reference_from_nq_item() and the dataset structure."
    )

# ============================================================
# Debug check (VERY IMPORTANT)
# ============================================================

print("\nExample extracted references:\n")

for i in range(min(5, len(valid_items))):
    print("Q:", valid_items[i]["question"])
    print("REF:", valid_items[i]["reference"])
    print()


# ============================================================
# Create subset for experiments
# ============================================================

rng = np.random.RandomState(SEED)

subset_size = min(SUBSET_SIZE, len(valid_items))

subset = rng.choice(
    valid_items,
    size=subset_size,
    replace=False
).tolist()

print("Final subset size:", len(subset))
if len(subset) == 0:
    raise RuntimeError("Subset is empty after sampling.")

# ============================================================
# Load NLI models
# ============================================================

def load_nli_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    mdl.eval()
    return tok, mdl


print("\nLoading DETECTOR NLI model:", DETECTOR_NLI_MODEL)
det_tok, det_nli = load_nli_model(DETECTOR_NLI_MODEL)

gold_tok, gold_nli = None, None
print("\nLoading GOLD NLI model with fallback...")
for cand in GOLD_NLI_CANDIDATES:
    try:
        print("  trying:", cand)
        gold_tok, gold_nli = load_nli_model(cand)
        print("  -> loaded GOLD:", cand)
        break
    except Exception as e:
        print("  failed:", cand, "|", str(e)[:140])

if gold_tok is None:
    raise RuntimeError("Failed to load any GOLD NLI model candidates.")


@torch.no_grad()
def nli_ent_con(tok, mdl, premise, hypothesis):
    if not premise or not hypothesis:
        return 0.0, 0.0

    inputs = tok(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    probs = F.softmax(mdl(**inputs).logits, dim=-1)[0].detach().cpu().numpy()
    # contradiction=0, neutral=1, entailment=2
    return float(probs[2]), float(probs[0])


def detector_mnli_signal(reference: str, answer: str):
    if not reference or not answer:
        return 0.0, 0.0
    return nli_ent_con(det_tok, det_nli, reference, answer)


def gold_contradiction_prob(reference: str, answer: str):
    _, con = nli_ent_con(gold_tok, gold_nli, reference, answer)
    return float(con)


def nq_gold_ok(answer: str, reference: str, thresh=GOLD_THR):
    """
    Gold=True if:
      - answer entails reference
      - answer does NOT contradict reference
    """
    if not answer or not reference:
        return False
    ent, con = nli_ent_con(gold_tok, gold_nli, reference, answer)
    return (ent > thresh) and (con < thresh)


# ============================================================
# SelfCheckGPT-NLI
# ============================================================

print("\nLoading SelfCheckGPT-NLI...")
selfcheck = SelfCheckNLI(device=device)


def _extract_reasonish_text(ans: str) -> str:
    if not ans:
        return ""
    m = re.search(r"^\s*FINAL_ANSWER\s*:\s*(.+?)\s*$", ans, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    parts = [p.strip() for p in re.split(r"[.\n]", ans) if p.strip()]
    return parts[0] if parts else ans.strip()


def selfcheck_group(answers):
    if len(answers) < 2:
        return 0.0

    snippets = []
    for a in answers:
        s = _extract_reasonish_text(a)
        if s:
            snippets.append(s)

    if len(snippets) < 2:
        return 0.0

    scores = selfcheck.predict(sentences=snippets, sampled_passages=snippets)
    return float(np.mean(scores)) if len(scores) else 0.0


# ============================================================
# Ollama helpers
# ============================================================

def ollama_chat_retry(model, messages, options, max_tries=3, sleep_s=0.8):
    last_err = None
    for _ in range(max_tries):
        try:
            return ollama.chat(model=model, messages=messages, options=options)
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise last_err


# ============================================================
# Judge
# ============================================================

def judge_truth(question: str, answer: str, reference: str) -> bool:
    prompt = f"""
You are a strict factuality evaluator for Natural Questions.

Mark True if:
- The answer contains the Reference Answer (or its exact meaning),
- AND it contains NO false or contradictory statements.

Extra correct information is allowed and should NOT cause False.

Mark False ONLY if the answer misses the reference meaning OR contains factual errors/contradictions.

Question: {question}
Reference Answer: {reference}
Model Answer: {answer}

Respond with exactly one token: True or False.
"""
    try:
        resp = ollama_chat_retry(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
            max_tries=2
        )
        return resp["message"]["content"].strip().lower().startswith("true")
    except Exception:
        return False


# ============================================================
# Answer generation + cache
# ============================================================

ensure_csv_header(ANSWER_CACHE_FILE, ["question", "answer"])
answer_cache = load_cache_answers(ANSWER_CACHE_FILE)


def generate_answer(question: str):
    try:
        resp = ollama_chat_retry(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": question}],
            options={"temperature": 0.8, "top_p": 0.9},
            max_tries=2
        )
        return resp["message"]["content"]
    except Exception:
        return ""


def ensure_answers(question: str, n_samples: int):
    cur = answer_cache.get(question, [])
    need = n_samples - len(cur)
    if need <= 0:
        return cur[:n_samples]

    new_rows = []
    for _ in range(need):
        ans = generate_answer(question)
        if ans:
            cur.append(ans)
            new_rows.append({"question": question, "answer": ans})

    if new_rows:
        append_cache_answers(ANSWER_CACHE_FILE, new_rows)

    answer_cache[question] = cur
    return cur[:n_samples]


# ============================================================
# Scoring
# ============================================================

def fact_score_from_ent_con(ent: float, con: float) -> float:
    return float(max(ent - FACT_CONTRA_ALPHA * con, 0.0))


def mode_score(sc, judge_ok, ent, con, mode):
    judge_score = 1.0 if judge_ok else 0.0
    sc_score = 1.0 - float(sc)
    fact_score = fact_score_from_ent_con(float(ent), float(con))

    wj, wf, ws = W_JUDGE, W_FACT, W_SC

    if mode == "full":
        score = wj * judge_score + wf * fact_score + ws * sc_score
    elif mode == "no_judge":
        score = (wf * fact_score + ws * sc_score) / (wf + ws)
    elif mode == "no_mnli":
        score = (wj * judge_score + ws * sc_score) / (wj + ws)
    elif mode == "no_selfcheck":
        score = (wj * judge_score + wf * fact_score) / (wj + wf)
    elif mode == "judge_only":
        score = judge_score
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return float(score)


def score_single_answer_full(question, reference, answer_text, sc_override=None):
    answer_for_eval = parse_final_answer_text(answer_text)
    judge_ok = judge_truth(question, answer_for_eval, reference)
    ent, con = detector_mnli_signal(reference, answer_for_eval)
    sc = float(sc_override) if sc_override is not None else 0.0
    ok_score = mode_score(sc=sc, judge_ok=judge_ok, ent=ent, con=con, mode="full")
    return ok_score, {
        "judge_ok": judge_ok,
        "ent": float(ent),
        "con": float(con),
        "sc": float(sc)
    }


# ============================================================
# Mitigation
# ============================================================

def build_span_extraction_prompt(question: str, reference: str):
    return f"""
You are extracting supporting spans from a reference answer for Natural Questions.

Question:
{question}

Reference answer:
{reference}

Task:
Copy 1-2 short spans (exact substrings) from the Reference answer that are most relevant.

Rules:
- Copy verbatim from the Reference answer
- Do NOT paraphrase
- If insufficient, output: NONE

Output format exactly:
SPAN1: "<...>"
SPAN2: "<...>"   (optional)
"""


def extract_spans(question: str, reference: str) -> str:
    if not reference.strip():
        return "NONE"
    try:
        resp = ollama_chat_retry(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": build_span_extraction_prompt(question, reference)}],
            options={"temperature": 0.0},
            max_tries=2
        )["message"]["content"].strip()
    except Exception:
        return "NONE"

    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    spans = []
    for ln in lines:
        if ln.upper().startswith("SPAN"):
            m = re.search(r"\"(.+?)\"", ln)
            if m:
                s = m.group(1).strip()
                if s:
                    ws = s.split()
                    s = " ".join(ws[:SPAN_MAX_WORDS])
                    spans.append(s)

    if not spans:
        return "NONE"

    out = []
    for i, s in enumerate(spans[:2], 1):
        out.append(f'SPAN{i}: "{s}"')
    return "\n".join(out)


def build_mild_rewrite_prompt(question, reference, prev_answer):
    return f"""
Question: {question}

Reference answer:
{reference}

Task:
- Rewrite the answer so it preserves only the reference meaning.
- Remove unsupported or incorrect information.
- Do NOT add any new facts.

Output format exactly:
FINAL_ANSWER: <one short answer>

Previous Answer:
{prev_answer}
"""


def build_strict_rewrite_prompt(question, reference, prev_answer, spans_block: str):
    spans_part = f"\nRelevant spans (verbatim):\n{spans_block}\n" if spans_block else ""
    return f"""
You are a strict factual correction system for Natural Questions.
Your goal is to minimize hallucination.

Question:
{question}

Reference answer:
{reference}
{spans_part}

Rules:
- Use ONLY the reference answer meaning.
- DO NOT add any new facts.
- If the reference is too limited, output an abstaining answer.

Output format exactly:
FINAL_ANSWER: <one short answer grounded only in the reference>

Previous answer:
{prev_answer}
"""


def rewrite_once(question, reference, prev_answer, kind="mild"):
    if kind == "mild":
        prompt = build_mild_rewrite_prompt(question, reference, prev_answer)
        temp = MILD_TEMP
    else:
        spans_block = ""
        if USE_SPAN_EXTRACTION_IN_STRICT:
            spans_block = extract_spans(question, reference)
        prompt = build_strict_rewrite_prompt(question, reference, prev_answer, spans_block=spans_block)
        temp = STRICT_TEMP

    try:
        resp = ollama_chat_retry(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temp, "top_p": 0.9},
            max_tries=2
        )
        return resp["message"]["content"]
    except Exception:
        return ""


def compute_nei_and_abstain(answer_text: str):
    if not answer_text or not answer_text.strip():
        nei = False
        abstain = True if COUNT_EMPTY_AS_ABSTAIN else False
        return nei, abstain

    ans = parse_final_answer_text(answer_text).lower()
    abstain = (
        "cannot determine" in ans
        or "insufficient" in ans
        or "do not know" in ans
        or "don't know" in ans
    )
    return abstain, abstain


# ============================================================
# Candidate filter + rerank
# ============================================================

def _filter_candidates(question, reference, candidates):
    kept = []
    for c in candidates:
        if not c or not c.strip():
            continue

        ans = parse_final_answer_text(c)
        if not ans:
            continue

        ent, con = detector_mnli_signal(reference, ans)
        if ent < con:
            continue

        gcon = gold_contradiction_prob(reference, ans) if reference.strip() else 0.0
        if gcon >= GOLD_CONTRA_THRESH:
            continue

        kept.append(c)

    return kept


def _group_by_label(candidates):
    groups = {
        "ANSWER": [],
        "ABSTAIN": []
    }
    for c in candidates:
        ans = parse_final_answer_text(c).lower()
        if (
            "cannot determine" in ans
            or "insufficient" in ans
            or "do not know" in ans
            or "don't know" in ans
        ):
            groups["ABSTAIN"].append(c)
        else:
            groups["ANSWER"].append(c)
    return groups


def detector_conditioned_correct(
    question, reference, base_answer,
    t_low_ok, t_high_ok,
    K=10,
    mid_kind="strict",
    abstain_if_still_risky=True,
    abstain_threshold=None
):
    if abstain_threshold is None:
        abstain_threshold = float(t_low_ok)

    base_ok, _ = score_single_answer_full(question, reference, base_answer, sc_override=None)
    if base_ok >= t_high_ok:
        return base_answer, base_ok, "keep", base_ok

    def gen_candidates(kind, K_local):
        cands = []
        for _ in range(int(K_local)):
            ai = rewrite_once(question, reference, base_answer, kind=kind)
            if ai and ai.strip():
                cands.append(ai)
        return cands

    if base_ok >= t_low_ok:
        kind = "strict" if mid_kind == "strict" else "mild"
        raw_cands = gen_candidates(kind=kind, K_local=K)
        tag = f"mid_{kind}_K{K}"
    else:
        raw_cands = gen_candidates(kind="strict", K_local=K)
        tag = f"strict_K{K}"

    if not raw_cands:
        return base_answer, base_ok, f"{tag}_all_fail_keep", base_ok

    filtered = _filter_candidates(question, reference, raw_cands)
    if not filtered:
        filtered = raw_cands

    groups = _group_by_label(filtered)

    best_group_label = None
    best_group_best_ok = -1.0
    best_group_best_ans = None
    best_group_sc = 0.0

    for lbl, cands in groups.items():
        if not cands:
            continue

        sc_grp = selfcheck_group(cands)
        base_ok_sc, _ = score_single_answer_full(question, reference, base_answer, sc_override=sc_grp)

        local_best_ans = base_answer
        local_best_ok = base_ok_sc

        for ai in cands:
            oki, _ = score_single_answer_full(question, reference, ai, sc_override=sc_grp)
            if oki > local_best_ok:
                local_best_ok = oki
                local_best_ans = ai

        if local_best_ok > best_group_best_ok:
            best_group_best_ok = local_best_ok
            best_group_best_ans = local_best_ans
            best_group_label = lbl
            best_group_sc = sc_grp

    if best_group_best_ans is None:
        return base_answer, base_ok, f"{tag}_no_group_keep", base_ok

    best_ans = best_group_best_ans
    best_ok = best_group_best_ok
    action = f"{tag}_group={best_group_label}_sc_rerank"

    if abstain_if_still_risky and (best_ok < abstain_threshold):
        abst_ans = "FINAL_ANSWER: I cannot determine the answer from the provided reference."
        abst_ok, _ = score_single_answer_full(question, reference, abst_ans, sc_override=best_group_sc)
        if abst_ok >= best_ok:
            return abst_ans, abst_ok, f"{action}_abstain", base_ok

    return best_ans, best_ok, action, base_ok


def always_rewrite_correct(question, reference, base_answer, kind="mild", K=1):
    if kind == "mild":
        ai = rewrite_once(question, reference, base_answer, kind="mild")
        if not ai:
            ok0, _ = score_single_answer_full(question, reference, base_answer, sc_override=None)
            return base_answer, ok0
        ok1, _ = score_single_answer_full(question, reference, ai, sc_override=None)
        return ai, ok1

    raw = []
    for _ in range(max(1, K)):
        ai = rewrite_once(question, reference, base_answer, kind="strict")
        if ai and ai.strip():
            raw.append(ai)

    if not raw:
        ok0, _ = score_single_answer_full(question, reference, base_answer, sc_override=None)
        return base_answer, ok0

    filtered = _filter_candidates(question, reference, raw)
    if not filtered:
        filtered = raw

    groups = _group_by_label(filtered)

    best_ok = -1.0
    best_ans = base_answer

    for lbl, cands in groups.items():
        if not cands:
            continue

        sc_grp = selfcheck_group(cands)
        base_ok_sc, _ = score_single_answer_full(question, reference, base_answer, sc_override=sc_grp)

        local_best_ok = base_ok_sc
        local_best_ans = base_answer

        for ai in cands:
            oki, _ = score_single_answer_full(question, reference, ai, sc_override=sc_grp)
            if oki > local_best_ok:
                local_best_ok = oki
                local_best_ans = ai

        if local_best_ok > best_ok:
            best_ok = local_best_ok
            best_ans = local_best_ans

    if best_ok < 0:
        ok0, _ = score_single_answer_full(question, reference, base_answer, sc_override=None)
        return base_answer, ok0

    return best_ans, float(best_ok)


# ============================================================
# Build pool
# ============================================================

print("\nBuilding question pool...")
rng = np.random.RandomState(SEED)
pool_q_idxs = rng.choice(len(subset), size=min(POOL_SIZE, len(subset)), replace=False).tolist()
rng.shuffle(pool_q_idxs)

val_n = int(len(pool_q_idxs) * VAL_RATIO)
val_q_idxs = pool_q_idxs[:val_n]
test_q_idxs = pool_q_idxs[val_n:]
print(f"Pool size: {len(pool_q_idxs)} | val questions: {len(val_q_idxs)} | test questions: {len(test_q_idxs)}")
if len(val_q_idxs) == 0 or len(test_q_idxs) == 0:
    raise RuntimeError(
        f"Invalid split: val={len(val_q_idxs)}, test={len(test_q_idxs)}. "
        "Please increase valid subset size or check reference extraction."
    )

def build_question_pack(q_idx: int):
    item = subset[int(q_idx)]
    q = item["question"]
    ref = item["reference"]

    answers = ensure_answers(q, N_SAMPLES)
    sc = selfcheck_group(answers)

    per_answer = []
    for ans in answers:
        ans_eval = parse_final_answer_text(ans)

        gold_ok = nq_gold_ok(ans_eval, ref)
        judge_ok = judge_truth(q, ans_eval, ref)
        ent, con = detector_mnli_signal(ref, ans_eval)

        per_answer.append({
            "answer": ans,
            "gold_ok": bool(gold_ok),
            "judge_ok": bool(judge_ok),
            "ent": float(ent),
            "con": float(con),
        })

    return {
        "q_idx": int(q_idx),
        "q": q,
        "ref": ref,
        "answers": answers,
        "sc": float(sc),
        "per_answer": per_answer
    }


print("\nPrecomputing VAL + TEST packs (slow due to Ollama)...")
val_packs = [build_question_pack(i) for i in tqdm(val_q_idxs, desc="VAL packs")]
test_packs = [build_question_pack(i) for i in tqdm(test_q_idxs, desc="TEST packs")]


# ============================================================
# Detection eval + threshold tuning
# ============================================================

def eval_on_packs(packs, mode, threshold, run_id=0, log_errors_path=None, max_log=0):
    tp = fp = tn = fn = 0
    all_scores = []
    q_stds = []

    error_logged = 0
    if log_errors_path:
        ensure_csv_header(log_errors_path, [
            "mode","run","q_idx","question","reference",
            "answer","gold_ok","pred_ok","score","threshold",
            "judge_ok","ent","con","selfcheck"
        ])

    for pack in packs:
        sc = pack["sc"]
        scores_this_q = []

        for a in pack["per_answer"]:
            score = mode_score(
                sc=sc,
                judge_ok=a["judge_ok"],
                ent=a["ent"],
                con=a["con"],
                mode=mode
            )
            pred_ok = (score >= threshold)
            gold_ok = a["gold_ok"]

            pred_halluc = not pred_ok
            gold_halluc = not gold_ok

            all_scores.append(score)
            scores_this_q.append(score)

            if pred_halluc and gold_halluc:
                tp += 1
            elif pred_halluc and not gold_halluc:
                fp += 1
            elif (not pred_halluc) and (not gold_halluc):
                tn += 1
            else:
                fn += 1

            if log_errors_path and error_logged < max_log and (pred_ok != gold_ok):
                with open(log_errors_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=[
                        "mode","run","q_idx","question","reference",
                        "answer","gold_ok","pred_ok","score","threshold",
                        "judge_ok","ent","con","selfcheck"
                    ])
                    w.writerow({
                        "mode": mode,
                        "run": run_id,
                        "q_idx": pack["q_idx"],
                        "question": pack["q"],
                        "reference": pack["ref"],
                        "answer": a["answer"],
                        "gold_ok": gold_ok,
                        "pred_ok": pred_ok,
                        "score": score,
                        "threshold": threshold,
                        "judge_ok": a["judge_ok"],
                        "ent": a["ent"],
                        "con": a["con"],
                        "selfcheck": sc
                    })
                error_logged += 1

        q_stds.append(float(np.std(scores_this_q)) if len(scores_this_q) > 1 else 0.0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "answer_score_std": float(np.std(all_scores)) if all_scores else 0.0,
        "question_score_std": float(np.mean(q_stds)) if q_stds else 0.0
    }


def tune_threshold_on_val(val_packs, mode, grid=None):
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)

    scores = []
    golds = []
    for pack in val_packs:
        sc = pack["sc"]
        for a in pack["per_answer"]:
            scores.append(mode_score(sc, a["judge_ok"], a["ent"], a["con"], mode))
            golds.append(a["gold_ok"])

    scores = np.array(scores, dtype=np.float32)
    golds = np.array(golds, dtype=np.bool_)
    gold_halluc = ~golds

    best_t = 0.5
    best_f1 = -1.0

    for t in grid:
        pred_ok = scores >= t
        pred_halluc = ~pred_ok

        tp = np.sum(pred_halluc & gold_halluc)
        fp = np.sum(pred_halluc & (~gold_halluc))
        fn = np.sum((~pred_halluc) & gold_halluc)

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0

        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    return best_t, best_f1


# ============================================================
# Mitigation eval
# ============================================================

def eval_mitigation_on_packs(
    packs, policy_name, t_low_ok, t_high_ok, K=10,
    run_id=0, log_cases_path=None, max_log=0,
    abstain_threshold=None
):
    total = 0
    gold_ok_cnt = 0
    nei_cnt = 0
    abstain_cnt = 0

    improve_cnt = 0
    regress_cnt = 0
    abstain_from_wrong_cnt = 0

    base_ok_scores = []
    final_ok_scores = []
    actions = {}

    if log_cases_path:
        ensure_csv_header(log_cases_path, [
            "run","policy","q_idx","question","reference",
            "base_answer","final_answer",
            "base_ok_score","final_ok_score","delta_ok_score",
            "base_gold_ok","final_gold_ok",
            "final_NEI","action"
        ])

    logged = 0

    for pack in packs:
        q = pack["q"]
        ref = pack["ref"]
        q_idx = pack.get("q_idx", -1)

        base_answer = pack["per_answer"][0]["answer"] if pack["per_answer"] else ""

        base_ok, _ = score_single_answer_full(q, ref, base_answer, sc_override=None)
        base_gold_ok = nq_gold_ok(parse_final_answer_text(base_answer), ref)

        if policy_name == "base":
            final_answer = base_answer
            final_ok = base_ok
            action = "base_keep"
        elif policy_name == "always_mild":
            final_answer, final_ok = always_rewrite_correct(q, ref, base_answer, kind="mild", K=1)
            action = "always_mild"
        elif policy_name == "always_strict_K":
            final_answer, final_ok = always_rewrite_correct(q, ref, base_answer, kind="strict", K=K)
            action = f"always_strict_K{K}_group_sc"
        elif policy_name == "policy":
            final_answer, final_ok, action, _ = detector_conditioned_correct(
                q, ref, base_answer,
                t_low_ok=t_low_ok,
                t_high_ok=t_high_ok,
                K=K,
                mid_kind=MID_KIND,
                abstain_if_still_risky=ABSTAIN_IF_STILL_RISKY,
                abstain_threshold=abstain_threshold
            )
        else:
            raise ValueError(f"Unknown mitigation policy: {policy_name}")

        final_gold_ok = nq_gold_ok(parse_final_answer_text(final_answer), ref)
        nei, abstain = compute_nei_and_abstain(final_answer)

        total += 1
        gold_ok_cnt += (1 if final_gold_ok else 0)
        nei_cnt += (1 if nei else 0)
        abstain_cnt += (1 if abstain else 0)

        if (not base_gold_ok) and final_gold_ok:
            improve_cnt += 1
        if base_gold_ok and (not final_gold_ok):
            regress_cnt += 1
        if (not base_gold_ok) and nei:
            abstain_from_wrong_cnt += 1

        base_ok_scores.append(base_ok)
        final_ok_scores.append(final_ok)
        actions[action] = actions.get(action, 0) + 1

        if log_cases_path and logged < max_log:
            with open(log_cases_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "run","policy","q_idx","question","reference",
                    "base_answer","final_answer",
                    "base_ok_score","final_ok_score","delta_ok_score",
                    "base_gold_ok","final_gold_ok",
                    "final_NEI","action"
                ])
                w.writerow({
                    "run": run_id,
                    "policy": policy_name,
                    "q_idx": q_idx,
                    "question": q,
                    "reference": ref,
                    "base_answer": base_answer,
                    "final_answer": final_answer,
                    "base_ok_score": float(base_ok),
                    "final_ok_score": float(final_ok),
                    "delta_ok_score": float(final_ok - base_ok),
                    "base_gold_ok": bool(base_gold_ok),
                    "final_gold_ok": bool(final_gold_ok),
                    "final_NEI": bool(nei),
                    "action": action
                })
            logged += 1

    gold_ok_rate = gold_ok_cnt / total if total else 0.0
    halluc_rate = 1.0 - gold_ok_rate

    base_ok_mean = float(np.mean(base_ok_scores)) if base_ok_scores else 0.0
    final_ok_mean = float(np.mean(final_ok_scores)) if final_ok_scores else 0.0
    delta_ok_mean = float(np.mean(np.array(final_ok_scores) - np.array(base_ok_scores))) if base_ok_scores else 0.0

    return {
        "policy": policy_name,
        "n_questions": int(total),
        "final_gold_ok_rate": float(gold_ok_rate),
        "final_halluc_rate": float(halluc_rate),
        "nei_rate": float(nei_cnt / total) if total else 0.0,
        "abstain_rate": float(abstain_cnt / total) if total else 0.0,
        "base_ok_score_mean": base_ok_mean,
        "final_ok_score_mean": final_ok_mean,
        "delta_ok_score_mean": delta_ok_mean,
        "action_counts": actions,
        "paired": {
            "improve_cnt": int(improve_cnt),
            "regress_cnt": int(regress_cnt),
            "abstain_from_wrong_cnt": int(abstain_from_wrong_cnt),
        }
    }


# ============================================================
# Tune t_high
# ============================================================

def _select_val_for_tune(val_packs, max_q=None):
    if max_q is None or max_q <= 0:
        return val_packs
    return val_packs[: min(max_q, len(val_packs))]


def tune_t_high_on_val(val_packs_for_tune, t_low_ok, K, abstain_threshold):
    hi_max = float(min(0.95, t_low_ok + 0.25))
    grid = np.linspace(t_low_ok, hi_max, 11)

    best_hi = float(min(0.95, t_low_ok + 0.05))
    best_obj = 1e9
    best_stats = None

    for hi in grid:
        total = 0
        final_ok = 0
        abstain_cnt = 0
        regress_cnt = 0

        for pack in val_packs_for_tune:
            q = pack["q"]
            ref = pack["ref"]
            base_answer = pack["per_answer"][0]["answer"] if pack["per_answer"] else ""

            base_gold_ok = nq_gold_ok(parse_final_answer_text(base_answer), ref)

            final_answer, _, _, _ = detector_conditioned_correct(
                q, ref, base_answer,
                t_low_ok=t_low_ok,
                t_high_ok=float(hi),
                K=K,
                mid_kind=MID_KIND,
                abstain_if_still_risky=ABSTAIN_IF_STILL_RISKY,
                abstain_threshold=abstain_threshold
            )

            final_gold_ok = nq_gold_ok(parse_final_answer_text(final_answer), ref)
            _, abstain = compute_nei_and_abstain(final_answer)

            total += 1
            final_ok += (1 if final_gold_ok else 0)
            abstain_cnt += (1 if abstain else 0)

            if base_gold_ok and (not final_gold_ok):
                regress_cnt += 1

        if total == 0:
            continue

        final_ok_rate = final_ok / total
        halluc_rate = 1.0 - final_ok_rate
        abstain_rate = abstain_cnt / total
        regress_rate = regress_cnt / total

        obj = halluc_rate + TUNE_W_ABSTAIN * abstain_rate + TUNE_W_REGRESS * regress_rate

        if obj < best_obj:
            best_obj = obj
            best_hi = float(hi)
            best_stats = {
                "halluc_rate": float(halluc_rate),
                "abstain_rate": float(abstain_rate),
                "regress_rate": float(regress_rate),
                "final_ok_rate": float(final_ok_rate),
            }

    return best_hi, best_obj, best_stats


# ============================================================
# MAIN
# ============================================================

modes = ["full", "no_judge", "no_mnli", "no_selfcheck", "judge_only"]

print("\nTuning threshold on VAL for each mode...")
mode_thresholds = {}
for m in modes:
    t_star, f1_star = tune_threshold_on_val(val_packs, m)
    mode_thresholds[m] = float(t_star)
    print(f"  mode={m:12s} | best_thr={t_star:.2f} | val_F1={f1_star:.4f}")

t_low_ok = float(mode_thresholds["full"])
t_high_ok_default = float(min(0.95, t_low_ok + 0.05))
abstain_threshold = float(min(0.95, t_low_ok + ABSTAIN_MARGIN))

print(
    f"\nMitigation thresholds (default): "
    f"t_low_ok={t_low_ok:.3f}, t_high_ok={t_high_ok_default:.3f}, abstain_thr={abstain_threshold:.3f} | "
    f"K={MITI_K} | MID_KIND={MID_KIND}"
)

t_high_ok = t_high_ok_default
if TUNE_T_HIGH:
    val_for_tune = _select_val_for_tune(val_packs, max_q=VAL_TUNE_MAX_Q)
    print(f"\nTuning t_high on VAL (n={len(val_for_tune)} questions) to reduce hallucination...")

    if len(val_for_tune) == 0:
        print("  skipped: val_for_tune is empty, keep default t_high_ok.")
    else:
        t_high_ok, best_obj, best_stats = tune_t_high_on_val(
            val_packs_for_tune=val_for_tune,
            t_low_ok=t_low_ok,
            K=MITI_K,
            abstain_threshold=abstain_threshold
        )

        if best_stats is None:
            print("  skipped: no valid tuning stats produced, keep default t_high_ok.")
            t_high_ok = t_high_ok_default
        else:
            print(
                f"  tuned t_high_ok={t_high_ok:.3f} | obj={best_obj:.4f} | "
                f"halluc={best_stats['halluc_rate']:.4f} "
                f"abstain={best_stats['abstain_rate']:.4f} "
                f"regress={best_stats['regress_rate']:.4f}"
            )

print(
    f"\nMitigation thresholds (FINAL): "
    f"t_low_ok={t_low_ok:.3f}, t_high_ok={t_high_ok:.3f}, abstain_thr={abstain_threshold:.3f} | "
    f"K={MITI_K} | MID_KIND={MID_KIND}"
)

print("\nRunning paired Monte-Carlo on TEST...")
rng = np.random.RandomState(SEED)

summary = {m: [] for m in modes}
miti_policies = ["base", "always_mild", "always_strict_K", "policy"]
miti_summary = {p: [] for p in miti_policies}

test_q_count = len(test_packs)
if SAMPLE_SIZE > test_q_count:
    raise ValueError(f"SAMPLE_SIZE={SAMPLE_SIZE} > number of test questions={test_q_count}")

for run_id in range(NUM_RUNS):
    q_sel = rng.choice(test_q_count, size=SAMPLE_SIZE, replace=False)
    packs_run = [test_packs[i] for i in q_sel]

    for m in modes:
        thr = mode_thresholds[m]
        metrics = eval_on_packs(
            packs_run,
            mode=m,
            threshold=thr,
            run_id=run_id,
            log_errors_path=ERROR_CASES_FILE,
            max_log=MAX_ERROR_LOGS_PER_MODE
        )
        summary[m].append(metrics)

    for p in miti_policies:
        metrics = eval_mitigation_on_packs(
            packs_run,
            policy_name=p,
            t_low_ok=t_low_ok,
            t_high_ok=t_high_ok,
            K=MITI_K,
            run_id=run_id,
            log_cases_path=MITI_CASES_FILE,
            max_log=MAX_CASES_PER_RUN,
            abstain_threshold=abstain_threshold
        )
        miti_summary[p].append(metrics)


# ============================================================
# Final summaries + save
# ============================================================

print("\n=== FINAL SUMMARY (Detection) ===")
print(f"{'mode':12s} | {'F1_mean':>8s} {'F1_std':>8s} | {'Q-std':>10s} | {'A-std':>10s} | {'thr*':>6s}")
print("-" * 100)

rows = []
summary_json = {
    "config": {
        "SEED": SEED,
        "N_SAMPLES": N_SAMPLES,
        "SUBSET_SIZE": SUBSET_SIZE,
        "POOL_SIZE": POOL_SIZE,
        "VAL_RATIO": VAL_RATIO,
        "NUM_RUNS": NUM_RUNS,
        "SAMPLE_SIZE": SAMPLE_SIZE,
        "detector_nli": DETECTOR_NLI_MODEL,
        "gold_nli_candidates": GOLD_NLI_CANDIDATES,
        "judge_model": JUDGE_MODEL,
        "gen_model": GEN_MODEL,
        "weights": {"W_JUDGE": W_JUDGE, "W_FACT": W_FACT, "W_SC": W_SC},
        "fact_contra_alpha": FACT_CONTRA_ALPHA,
        "gold_contra_thresh": GOLD_CONTRA_THRESH,
        "gold_thr": GOLD_THR,
        "mitigation": {
            "MITI_K": MITI_K,
            "MID_KIND": MID_KIND,
            "MILD_TEMP": MILD_TEMP,
            "STRICT_TEMP": STRICT_TEMP,
            "USE_SPAN_EXTRACTION_IN_STRICT": USE_SPAN_EXTRACTION_IN_STRICT,
            "SPAN_MAX_WORDS": SPAN_MAX_WORDS,
            "COUNT_EMPTY_AS_ABSTAIN": COUNT_EMPTY_AS_ABSTAIN,
            "ABSTAIN_IF_STILL_RISKY": ABSTAIN_IF_STILL_RISKY,
            "ABSTAIN_MARGIN": ABSTAIN_MARGIN,
            "t_low_ok": t_low_ok,
            "t_high_ok": t_high_ok,
            "abstain_threshold": abstain_threshold,
            "tune_t_high": TUNE_T_HIGH,
            "val_tune_max_q": VAL_TUNE_MAX_Q,
            "tune_obj_weights": {"w_abstain": TUNE_W_ABSTAIN, "w_regress": TUNE_W_REGRESS},
        },
        "files": {
            "ANSWER_CACHE_FILE": ANSWER_CACHE_FILE,
            "OUT_JSON_FILE": OUT_JSON_FILE,
            "OUT_CSV_FILE": OUT_CSV_FILE,
            "ERROR_CASES_FILE": ERROR_CASES_FILE,
            "MITI_CASES_FILE": MITI_CASES_FILE,
        }
    },
    "thresholds": mode_thresholds,
    "modes": {},
    "mitigation": {}
}

full_f1s = [r["f1"] for r in summary["full"]]
full_mean = safe_mean(full_f1s)

for m in modes:
    f1s = [r["f1"] for r in summary[m]]
    qstds = [r["question_score_std"] for r in summary[m]]
    astds = [r["answer_score_std"] for r in summary[m]]

    summary_json["modes"][m] = {
        "f1": {"values": f1s, "mean": safe_mean(f1s), "std": safe_std(f1s)},
        "question_score_std": {"values": qstds, "mean": safe_mean(qstds), "std": safe_std(qstds)},
        "answer_score_std": {"values": astds, "mean": safe_mean(astds), "std": safe_std(astds)},
    }

    delta_mean = full_mean - safe_mean(f1s) if m != "full" else 0.0

    rows.append({
        "mode": m,
        "thr_star": float(mode_thresholds[m]),
        "f1_mean": safe_mean(f1s),
        "f1_std": safe_std(f1s),
        "question_score_std_mean": safe_mean(qstds),
        "question_score_std_std": safe_std(qstds),
        "answer_score_std_mean": safe_mean(astds),
        "answer_score_std_std": safe_std(astds),
        "delta_f1_vs_full_mean": float(delta_mean)
    })

    print(
        f"{m:12s} | "
        f"{safe_mean(f1s):8.4f} {safe_std(f1s):8.4f} | "
        f"Q-std={safe_mean(qstds):.4f} | "
        f"A-std={safe_mean(astds):.4f} | "
        f"{mode_thresholds[m]:6.2f}"
    )

print("\n=== MITIGATION SUMMARY (Question-level final answer) ===")
print(f"{'policy':16s} | {'gold_ok_mean':>10s} {'halluc_mean':>10s} | {'NEI_mean':>8s} {'abstain_mean':>12s} | {'Δok_score':>10s} | {'impr/reg/abst':>14s}")
print("-" * 120)

for p in miti_policies:
    ok_rates = [x["final_gold_ok_rate"] for x in miti_summary[p]]
    halluc_rates = [x["final_halluc_rate"] for x in miti_summary[p]]
    nei_rates = [x["nei_rate"] for x in miti_summary[p]]
    abstain_rates = [x["abstain_rate"] for x in miti_summary[p]]
    deltas = [x["delta_ok_score_mean"] for x in miti_summary[p]]

    impr = [x["paired"]["improve_cnt"] for x in miti_summary[p]]
    reg  = [x["paired"]["regress_cnt"] for x in miti_summary[p]]
    abst = [x["paired"]["abstain_from_wrong_cnt"] for x in miti_summary[p]]

    summary_json["mitigation"][p] = {
        "runs": miti_summary[p],
        "final_gold_ok_rate_mean": safe_mean(ok_rates),
        "final_gold_ok_rate_std": safe_std(ok_rates),
        "final_halluc_rate_mean": safe_mean(halluc_rates),
        "final_halluc_rate_std": safe_std(halluc_rates),
        "nei_rate_mean": safe_mean(nei_rates),
        "nei_rate_std": safe_std(nei_rates),
        "abstain_rate_mean": safe_mean(abstain_rates),
        "abstain_rate_std": safe_std(abstain_rates),
        "delta_ok_score_mean": safe_mean(deltas),
        "delta_ok_score_std": safe_std(deltas),
        "paired_mean": {
            "improve_cnt": safe_mean(impr),
            "regress_cnt": safe_mean(reg),
            "abstain_from_wrong_cnt": safe_mean(abst),
        }
    }

    print(
        f"{p:16s} | "
        f"{safe_mean(ok_rates):10.4f} {safe_mean(halluc_rates):10.4f} | "
        f"{safe_mean(nei_rates):8.4f} {safe_mean(abstain_rates):12.4f} | "
        f"{safe_mean(deltas):10.4f} | "
        f"{int(round(safe_mean(impr))):4d}/{int(round(safe_mean(reg))):3d}/{int(round(safe_mean(abst))):4d}"
    )

print("\n=== Paired ΔF1 per run (full - ablation) ===")
for m in modes:
    if m == "full":
        continue
    diffs = []
    for r in range(NUM_RUNS):
        diffs.append(summary["full"][r]["f1"] - summary[m][r]["f1"])
    print(f"{m:12s}: ΔF1 mean={safe_mean(diffs):.4f} std={safe_std(diffs):.4f}  (positive => full better)")

with open(OUT_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2, ensure_ascii=False)

ensure_csv_header(OUT_CSV_FILE, list(rows[0].keys()) if rows else ["mode"])
with open(OUT_CSV_FILE, "w", newline="", encoding="utf-8") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    else:
        f.write("mode\n")

print(f"\nSaved: {OUT_JSON_FILE}, {OUT_CSV_FILE}")
print(f"Mitigation cases saved: {MITI_CASES_FILE} (up to {MAX_CASES_PER_RUN} rows per policy per run)")
print(f"Optional detection error cases: {ERROR_CASES_FILE} (up to {MAX_ERROR_LOGS_PER_MODE} mismatches per mode per run)")