import os
import ssl
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from sklearn.model_selection import train_test_split
import ollama

# ============================================================
# Paper-grade FEVER Hallucination Detector Evaluation (TQA-style)
# + Detector-Conditioned Mitigation (closed-loop)
# + Multi-layer rerank: SelfCheck computed from K candidates and reused for scoring
# + Case logging for qualitative evidence (teacher-required)
# + Extra mitigation counters: improve/regress/abstain (paired evidence)
# ============================================================

# -----------------------------
# Config
# -----------------------------
ANSWER_CACHE_FILE = "fever_answers_cache_paper-f.csv"

OUT_JSON_FILE = "fever_paper_overall_summary-f.json"
OUT_CSV_FILE  = "fever_paper_overall_summary-f.csv"

ERROR_CASES_FILE = "fever_paper_error_cases-f.csv"
MAX_ERROR_LOGS_PER_MODE = 200

# NEW: mitigation case logs (teacher wants cases)
MITI_CASES_FILE = "fever_mitigation_cases.csv"
MAX_CASES_PER_RUN = 500

# Sampling / evaluation
N_SAMPLES = 5
SEED = 42

# FEVER stratified subset size
SUBSET_SIZE = 1000

# Build a pool from the subset first (so you can tune threshold on val)
POOL_SIZE = 800          # number of questions used in total (<= SUBSET_SIZE)
VAL_RATIO = 0.2          # val split inside POOL

# Paired Monte-Carlo runs on TEST split
NUM_RUNS = 3
SAMPLE_SIZE = 200        # per run, number of questions (question-level)

# Ollama models
JUDGE_MODEL = "qwen2.5:7b-instruct"
GEN_MODEL   = "llama2:7b"

# Detector NLI (feature model)
DETECTOR_NLI_MODEL = "roberta-large-mnli"

# Gold NLI (for factual contradiction check) -- different model to avoid leakage
GOLD_NLI_CANDIDATES = [
    "microsoft/deberta-v3-base-mnli",
    "microsoft/deberta-v3-large-mnli",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
]

# Aggregation weights
W_JUDGE = 0.35
W_FACT  = 0.45
W_SC    = 0.20

# Gold factual check thresholds
GOLD_CONTRA_THRESH = 0.50

# -----------------------------
# Mitigation config
# -----------------------------
MITI_K = 5
MILD_TEMP = 0.3
STRICT_TEMP = 0.2
COUNT_EMPTY_AS_ABSTAIN = True

# Safety valve (abstain to NEI if still risky after rerank)
ABSTAIN_IF_STILL_RISKY = True
ABSTAIN_MARGIN = 0.05   # abstain_threshold = t_low_ok + ABSTAIN_MARGIN

# In mid-risk bucket, use K-rerank (strict by default -> usually stronger than mild-once)
MID_KIND = "strict"     # "strict" or "mild"

# HF caches
os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface/models"
ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------------
# Utilities
# -----------------------------
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
            if q:
                cache.setdefault(q, []).append(a)
    return cache

def append_cache_answers(path, rows):
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "answer"])
        for row in rows:
            w.writerow(row)

def normalize_fever_label(x: str):
    if x is None:
        return None
    x = str(x).strip().upper()
    if x in ["SUPPORTED", "SUPPORTS", "SUPPORT"]:
        return "SUPPORTED"
    if x in ["REFUTED", "REFUTES", "REFUTE"]:
        return "REFUTED"
    if x in ["NOT ENOUGH INFO", "NEI", "NOT_ENOUGH_INFO"]:
        return "NOT ENOUGH INFO"
    return None

# -----------------------------
# FEVER evidence extraction
# -----------------------------
def extract_evidence_text(item):
    evid = item.get("evidence", None)
    if evid is None:
        return ""

    if isinstance(evid, list) and len(evid) > 0:
        first = evid[0]
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], (list, tuple)):
            first = first[0]

        if isinstance(first, (list, tuple)) and len(first) >= 3:
            txt = first[2]
            return str(txt).strip() if txt is not None else ""
        if isinstance(first, str):
            return first.strip()

    return ""

# -----------------------------
# Load dataset + build stratified subset
# -----------------------------
print("Loading FEVER dataset...")
raw_dataset = load_dataset(
    "fever/fever",
    "v1.0",
    cache_dir="D:/huggingface/datasets"
)["labelled_dev"]

labels = [s["label"] for s in raw_dataset]
all_idx = list(range(len(raw_dataset)))

_, selected_idx = train_test_split(
    all_idx,
    test_size=min(SUBSET_SIZE, len(raw_dataset)),
    stratify=labels,
    random_state=SEED
)

sampled = raw_dataset.select(selected_idx)

subset = []
for item in sampled:
    q = item["claim"]
    ev = extract_evidence_text(item)
    gold_label = normalize_fever_label(item["label"])
    subset.append({"question": q, "evidence": ev, "gold_label": gold_label})

print(f"FEVER stratified subset size: {len(subset)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -----------------------------
# Load NLI models: detector + gold
# -----------------------------
def load_nli_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    mdl.eval()
    return tok, mdl

print("\nLoading DETECTOR NLI model (feature model):", DETECTOR_NLI_MODEL)
det_tok, det_nli = load_nli_model(DETECTOR_NLI_MODEL)

gold_tok, gold_nli = None, None
print("\nLoading GOLD NLI model (for gold factual check) with fallback...")
for cand in GOLD_NLI_CANDIDATES:
    try:
        print("  trying:", cand)
        gold_tok, gold_nli = load_nli_model(cand)
        print("  -> loaded GOLD model:", cand)
        break
    except Exception as e:
        print("  failed:", cand, "|", str(e)[:140])

if gold_tok is None:
    raise RuntimeError("Failed to load any GOLD NLI model candidates.")

@torch.no_grad()
def nli_ent_con(tok, mdl, premise, hypothesis):
    inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(device)
    probs = F.softmax(mdl(**inputs).logits, dim=-1)[0].detach().cpu().numpy()
    # contradiction(0), neutral(1), entailment(2)
    return float(probs[2]), float(probs[0])

# -----------------------------
# Gold definition (FEVER)
# -----------------------------
def parse_fever_label_from_answer(answer_text: str):
    if not answer_text:
        return None
    t = answer_text.upper()
    if "NOT ENOUGH INFO" in t or "\nNEI" in t or " NEI" in t:
        return "NOT ENOUGH INFO"
    if "REFUTED" in t:
        return "REFUTED"
    if "SUPPORTED" in t:
        return "SUPPORTED"
    return None

def gold_factual_ok(answer_text: str, evidence: str):
    if not evidence or not answer_text:
        return True
    _, con = nli_ent_con(gold_tok, gold_nli, evidence, answer_text)
    return (con < GOLD_CONTRA_THRESH)

def fever_gold_ok(answer_text: str, gold_label: str, evidence: str):
    pred_label = parse_fever_label_from_answer(answer_text)
    if pred_label is None or gold_label is None:
        return False
    if pred_label != gold_label:
        return False
    if not gold_factual_ok(answer_text, evidence):
        return False
    return True

# -----------------------------
# Judge feature
# -----------------------------
def judge_truth(question: str, model_answer: str, gold_label: str, evidence: str) -> bool:
    prompt = f"""
You are a strict FEVER evaluator.

Your job:
1) Identify the FEVER label in the Model Answer (SUPPORTED / REFUTED / NOT ENOUGH INFO).
2) Check if it matches the Reference Label.
3) The answer may include extra explanation, but MUST NOT contain factual statements that contradict the Evidence.

Claim: {question}
Evidence: {evidence}
Reference Label: {gold_label}
Model Answer: {model_answer}

Return exactly one token: True or False.
"""
    try:
        resp = ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        return resp["message"]["content"].strip().lower().startswith("true")
    except Exception:
        return False

# -----------------------------
# Detector MNLI signal
# -----------------------------
def detector_mnli_signal(evidence: str, answer_text: str):
    if not evidence or not answer_text:
        return 0.0, 0.0
    ent, con = nli_ent_con(det_tok, det_nli, evidence, answer_text)
    return float(ent), float(con)

# -----------------------------
# SelfCheckGPT-NLI
# -----------------------------
print("\nLoading SelfCheckGPT-NLI...")
selfcheck = SelfCheckNLI(device=device)

def selfcheck_group(answers):
    """
    Average contradiction score across sampled answers.
    Higher => worse.
    """
    if len(answers) < 2:
        return 0.0
    sents = []
    for a in answers:
        sents.extend([x.strip() for x in a.split(".") if x.strip()])
    scores = selfcheck.predict(sentences=sents, sampled_passages=answers)
    return float(np.mean(scores))

# -----------------------------
# Answer generation + cache
# -----------------------------
ensure_csv_header(ANSWER_CACHE_FILE, ["question", "answer"])
answer_cache = load_cache_answers(ANSWER_CACHE_FILE)

def generate_answer(question: str, evidence: str):
    prompt = f"""
Claim: {question}
Evidence: {evidence}

Classify the claim into exactly one of:
SUPPORTED / REFUTED / NOT ENOUGH INFO

Then give a brief explanation.
"""
    try:
        resp = ollama.chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.8, "top_p": 0.9}
        )
        return resp["message"]["content"]
    except Exception:
        return ""

def ensure_answers(question: str, evidence: str, n_samples: int):
    cur = answer_cache.get(question, [])
    need = n_samples - len(cur)
    if need <= 0:
        return cur[:n_samples]

    new_rows = []
    for _ in range(need):
        ans = generate_answer(question, evidence)
        if ans:
            cur.append(ans)
            new_rows.append({"question": question, "answer": ans})

    if new_rows:
        append_cache_answers(ANSWER_CACHE_FILE, new_rows)

    answer_cache[question] = cur
    return cur[:n_samples]

# -----------------------------
# Score aggregation
# -----------------------------
def mode_score(sc, judge_ok, ent, con, mode):
    judge_score = 1.0 if judge_ok else 0.0
    sc_score = 1.0 - float(sc)  # higher better
    fact_score = max(float(ent) - float(con), 0.0)

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

# -----------------------------
# Mitigation helpers
# -----------------------------
def build_mild_rewrite_prompt(question, evidence, prev_answer):
    return f"""
Claim: {question}
Evidence: {evidence}

The previous answer may contain inaccuracies.
Task:
1) Output exactly one label: SUPPORTED / REFUTED / NOT ENOUGH INFO
2) Provide a brief explanation ONLY based on the evidence.
3) If the evidence is insufficient, choose NOT ENOUGH INFO.

Previous Answer:
{prev_answer}
"""

def build_strict_rewrite_prompt(question, evidence, prev_answer):
    # More structured => less hallucination in explanation
    return f"""
Claim: {question}
Evidence: {evidence}

IMPORTANT RULES:
- You MUST follow the output format exactly.
- You MUST NOT add any facts not present in the Evidence.
- If the evidence is insufficient, choose NOT ENOUGH INFO.
- In the explanation, include a short phrase copied from the Evidence.

Output format (must follow exactly):
LABEL: <SUPPORTED/REFUTED/NOT ENOUGH INFO>
EVIDENCE_QUOTE: "<copy a short phrase (5-15 words) from Evidence>"
REASON: <one sentence reasoning based only on the quoted evidence>

Previous Answer:
{prev_answer}
"""

def rewrite_once(question, evidence, prev_answer, kind="mild"):
    if kind == "mild":
        prompt = build_mild_rewrite_prompt(question, evidence, prev_answer)
        temp = MILD_TEMP
    else:
        prompt = build_strict_rewrite_prompt(question, evidence, prev_answer)
        temp = STRICT_TEMP

    try:
        resp = ollama.chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temp, "top_p": 0.9}
        )
        return resp["message"]["content"]
    except Exception:
        return ""

def score_single_answer_full(question, evidence, gold_label, answer_text, sc_override=None):
    """
    Single-answer ok_score.
    If sc_override is provided, use it; else sc=0 (single answer can't selfcheck).
    """
    judge_ok = judge_truth(question, answer_text, gold_label, evidence)
    ent, con = detector_mnli_signal(evidence, answer_text)
    sc = float(sc_override) if sc_override is not None else 0.0
    ok_score = mode_score(sc=sc, judge_ok=judge_ok, ent=ent, con=con, mode="full")
    return ok_score, {"judge_ok": judge_ok, "ent": ent, "con": con, "sc": sc}

def detector_conditioned_correct(question, evidence, gold_label, base_answer,
                                 t_low_ok, t_high_ok, K=5,
                                 mid_kind="strict",
                                 abstain_if_still_risky=True,
                                 abstain_threshold=None):
    """
    Stronger closed-loop mitigation:

      - ok >= t_high_ok: keep
      - t_low_ok <= ok < t_high_ok: generate K candidates (mid_kind), compute sc on candidates,
        rerank using full score with shared sc, pick best
      - ok < t_low_ok: strict K candidates, compute sc, rerank, pick best
      - Safety valve: if best_ok < abstain_threshold -> output NEI (if NEI scores >= best)

    """
    if abstain_threshold is None:
        abstain_threshold = float(t_low_ok)

    base_ok, _ = score_single_answer_full(question, evidence, gold_label, base_answer, sc_override=None)

    if base_ok >= t_high_ok:
        return base_answer, base_ok, "keep", base_ok

    def gen_candidates(kind, K_local):
        cands = []
        for _ in range(int(K_local)):
            ai = rewrite_once(question, evidence, base_answer, kind=kind)
            if ai and ai.strip():
                cands.append(ai)
        return cands

    def rerank_with_shared_sc(candidates, tag):
        sc_cands = selfcheck_group(candidates)

        # compare base under same sc (fair)
        best_ans = base_answer
        best_ok, _ = score_single_answer_full(question, evidence, gold_label, base_answer, sc_override=sc_cands)

        for ai in candidates:
            oki, _ = score_single_answer_full(question, evidence, gold_label, ai, sc_override=sc_cands)
            if oki > best_ok:
                best_ok = oki
                best_ans = ai

        if abstain_if_still_risky and (best_ok < abstain_threshold):
            nei_ans = "NOT ENOUGH INFO\nThe evidence is insufficient to determine the claim."
            nei_ok, _ = score_single_answer_full(question, evidence, gold_label, nei_ans, sc_override=sc_cands)
            if nei_ok >= best_ok:
                return nei_ans, nei_ok, f"{tag}_abstain_nei"

        return best_ans, best_ok, tag

    # mid risk: K-rerank
    if base_ok >= t_low_ok:
        kind = "strict" if mid_kind == "strict" else "mild"
        cands = gen_candidates(kind=kind, K_local=K)
        if len(cands) == 0:
            return base_answer, base_ok, f"mid_{kind}_K{K}_all_fail_keep", base_ok
        best_ans, best_ok, action = rerank_with_shared_sc(cands, tag=f"mid_{kind}_K{K}_sc_rerank")
        return best_ans, best_ok, action, base_ok

    # high risk: strict K-rerank
    cands = gen_candidates(kind="strict", K_local=K)
    if len(cands) == 0:
        return base_answer, base_ok, f"strict_K{K}_all_fail_keep", base_ok
    best_ans, best_ok, action = rerank_with_shared_sc(cands, tag=f"strict_K{K}_sc_rerank")
    return best_ans, best_ok, action, base_ok

def always_rewrite_correct(question, evidence, gold_label, base_answer, kind="mild", K=1):
    """
    Baselines:
      - mild: single rewrite
      - strict_K: strict K candidates + selfcheck rerank
    """
    if kind == "mild":
        ai = rewrite_once(question, evidence, base_answer, kind="mild")
        if not ai:
            ok0, _ = score_single_answer_full(question, evidence, gold_label, base_answer, sc_override=None)
            return base_answer, ok0
        ok1, _ = score_single_answer_full(question, evidence, gold_label, ai, sc_override=None)
        return ai, ok1

    candidates = []
    for _ in range(max(1, K)):
        ai = rewrite_once(question, evidence, base_answer, kind="strict")
        if ai and ai.strip():
            candidates.append(ai)

    if len(candidates) == 0:
        ok0, _ = score_single_answer_full(question, evidence, gold_label, base_answer, sc_override=None)
        return base_answer, ok0

    sc_cands = selfcheck_group(candidates)
    best_ans = base_answer
    best_ok, _ = score_single_answer_full(question, evidence, gold_label, base_answer, sc_override=sc_cands)

    for ai in candidates:
        oki, _ = score_single_answer_full(question, evidence, gold_label, ai, sc_override=sc_cands)
        if oki > best_ok:
            best_ok = oki
            best_ans = ai

    return best_ans, best_ok

def compute_nei_and_abstain(answer_text: str):
    if not answer_text or not answer_text.strip():
        nei = False
        abstain = True if COUNT_EMPTY_AS_ABSTAIN else False
        return nei, abstain
    lbl = parse_fever_label_from_answer(answer_text)
    nei = (lbl == "NOT ENOUGH INFO")
    abstain = nei
    return nei, abstain

# -----------------------------
# Build pool
# -----------------------------
print("\nBuilding question pool...")
rng = np.random.RandomState(SEED)
pool_q_idxs = rng.choice(len(subset), size=min(POOL_SIZE, len(subset)), replace=False).tolist()
rng.shuffle(pool_q_idxs)

val_n = int(len(pool_q_idxs) * VAL_RATIO)
val_q_idxs = pool_q_idxs[:val_n]
test_q_idxs = pool_q_idxs[val_n:]
print(f"Pool size: {len(pool_q_idxs)} | val questions: {len(val_q_idxs)} | test questions: {len(test_q_idxs)}")

def build_question_pack(q_idx: int):
    item = subset[int(q_idx)]
    q = item["question"]
    evidence = item["evidence"]
    gold_label = item["gold_label"]

    answers = ensure_answers(q, evidence, N_SAMPLES)
    sc = selfcheck_group(answers)

    per_answer = []
    for ans in answers:
        gold_ok = fever_gold_ok(ans, gold_label, evidence)
        judge_ok = judge_truth(q, ans, gold_label, evidence)
        ent, con = detector_mnli_signal(evidence, ans)
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
        "evidence": evidence,
        "gold_label": gold_label,
        "answers": answers,
        "sc": float(sc),
        "per_answer": per_answer
    }

print("\nPrecomputing features for VAL + TEST pool (this can be slow due to Ollama calls)...")
val_packs = [build_question_pack(i) for i in tqdm(val_q_idxs, desc="VAL packs")]
test_packs = [build_question_pack(i) for i in tqdm(test_q_idxs, desc="TEST packs")]

# -----------------------------
# Detection evaluation helpers
# -----------------------------
def eval_on_packs(packs, mode, threshold, run_id=0, log_errors_path=None, max_log=0):
    tp = fp = tn = fn = 0
    all_scores = []
    q_stds = []

    error_logged = 0
    if log_errors_path:
        ensure_csv_header(log_errors_path, [
            "mode","run","q_idx","question","gold_label","evidence",
            "answer","gold_ok","pred_ok","score","threshold",
            "judge_ok","ent","con","selfcheck"
        ])

    for pack in packs:
        sc = pack["sc"]
        scores_this_q = []

        for a in pack["per_answer"]:
            score = mode_score(sc=sc, judge_ok=a["judge_ok"], ent=a["ent"], con=a["con"], mode=mode)
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
                        "mode","run","q_idx","question","gold_label","evidence",
                        "answer","gold_ok","pred_ok","score","threshold",
                        "judge_ok","ent","con","selfcheck"
                    ])
                    w.writerow({
                        "mode": mode,
                        "run": run_id,
                        "q_idx": pack["q_idx"],
                        "question": pack["q"],
                        "gold_label": pack["gold_label"],
                        "evidence": pack["evidence"],
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
        f1 = (2*p*r/(p+r)) if (p+r) else 0.0

        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    return best_t, best_f1

# -----------------------------
# Mitigation evaluation (question-level) + CASE LOGGING + paired counters
# -----------------------------
def eval_mitigation_on_packs(packs, policy_name, t_low_ok, t_high_ok, K=5,
                             run_id=0, log_cases_path=None, max_log=0,
                             abstain_threshold=None):
    total = 0
    gold_ok_cnt = 0
    nei_cnt = 0
    abstain_cnt = 0

    # paired evidence counters (teacher-friendly)
    improve_cnt = 0   # base_wrong -> final_correct
    regress_cnt = 0   # base_correct -> final_wrong
    abstain_from_wrong_cnt = 0  # base_wrong -> final_NEI

    base_ok_scores = []
    final_ok_scores = []
    actions = {}

    if log_cases_path:
        ensure_csv_header(log_cases_path, [
            "run","policy","q_idx","question","evidence","gold_label",
            "base_answer","final_answer",
            "base_ok_score","final_ok_score","delta_ok_score",
            "base_pred_label","final_pred_label",
            "base_gold_ok","final_gold_ok",
            "final_NEI","action"
        ])

    logged = 0

    for pack in packs:
        q = pack["q"]
        ev = pack["evidence"]
        gl = pack["gold_label"]
        q_idx = pack.get("q_idx", -1)
        base_answer = pack["per_answer"][0]["answer"] if pack["per_answer"] else ""

        base_ok, _ = score_single_answer_full(q, ev, gl, base_answer, sc_override=None)
        base_gold_ok = fever_gold_ok(base_answer, gl, ev)
        base_pred = parse_fever_label_from_answer(base_answer)

        if policy_name == "base":
            final_answer = base_answer
            final_ok = base_ok
            action = "base_keep"
        elif policy_name == "always_mild":
            final_answer, final_ok = always_rewrite_correct(q, ev, gl, base_answer, kind="mild", K=1)
            action = "always_mild"
        elif policy_name == "always_strict_K":
            final_answer, final_ok = always_rewrite_correct(q, ev, gl, base_answer, kind="strict", K=K)
            action = f"always_strict_K{K}_sc"
        elif policy_name == "policy":
            final_answer, final_ok, action, _ = detector_conditioned_correct(
                q, ev, gl, base_answer,
                t_low_ok=t_low_ok,
                t_high_ok=t_high_ok,
                K=K,
                mid_kind=MID_KIND,
                abstain_if_still_risky=ABSTAIN_IF_STILL_RISKY,
                abstain_threshold=abstain_threshold
            )
        else:
            raise ValueError(f"Unknown mitigation policy: {policy_name}")

        final_gold_ok = fever_gold_ok(final_answer, gl, ev)
        final_pred = parse_fever_label_from_answer(final_answer)
        nei, abstain = compute_nei_and_abstain(final_answer)

        total += 1
        gold_ok_cnt += (1 if final_gold_ok else 0)
        nei_cnt += (1 if nei else 0)
        abstain_cnt += (1 if abstain else 0)

        # paired counters
        if (not base_gold_ok) and final_gold_ok:
            improve_cnt += 1
        if base_gold_ok and (not final_gold_ok):
            regress_cnt += 1
        if (not base_gold_ok) and nei:
            abstain_from_wrong_cnt += 1

        base_ok_scores.append(base_ok)
        final_ok_scores.append(final_ok)
        actions[action] = actions.get(action, 0) + 1

        # case log row
        if log_cases_path and logged < max_log:
            with open(log_cases_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "run","policy","q_idx","question","evidence","gold_label",
                    "base_answer","final_answer",
                    "base_ok_score","final_ok_score","delta_ok_score",
                    "base_pred_label","final_pred_label",
                    "base_gold_ok","final_gold_ok",
                    "final_NEI","action"
                ])
                w.writerow({
                    "run": run_id,
                    "policy": policy_name,
                    "q_idx": q_idx,
                    "question": q,
                    "evidence": ev,
                    "gold_label": gl,
                    "base_answer": base_answer,
                    "final_answer": final_answer,
                    "base_ok_score": float(base_ok),
                    "final_ok_score": float(final_ok),
                    "delta_ok_score": float(final_ok - base_ok),
                    "base_pred_label": base_pred,
                    "final_pred_label": final_pred,
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

# -----------------------------
# Main experiment
# -----------------------------
modes = ["full", "no_judge", "no_mnli", "no_selfcheck", "judge_only"]

print("\nTuning threshold on VAL for each mode...")
mode_thresholds = {}
for m in modes:
    t_star, f1_star = tune_threshold_on_val(val_packs, m)
    mode_thresholds[m] = t_star
    print(f"  mode={m:12s} | best_thr={t_star:.2f} | val_F1={f1_star:.4f}")

# Mitigation thresholds (ok_score scale)
t_low_ok = float(mode_thresholds["full"])
t_high_ok = float(min(0.95, t_low_ok + 0.05))  # aggressive => more rerank
abstain_threshold = float(min(0.95, t_low_ok + ABSTAIN_MARGIN))
print(f"\nMitigation thresholds (ok_score): t_low_ok={t_low_ok:.3f}, t_high_ok={t_high_ok:.3f} | "
      f"abstain_thr={abstain_threshold:.3f} | K={MITI_K} | MID_KIND={MID_KIND}")

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

# -----------------------------
# Final summaries
# -----------------------------
print("\n=== FINAL SUMMARY (Detection) ===")
print(f"{'mode':12s} | {'F1_mean':>8s} {'F1_std':>8s} | {'Q-std':>10s} | {'A-std':>10s} | {'thr*':>6s}")
print("-" * 100)

rows = []
summary_json = {
    "config": {
        "N_SAMPLES": N_SAMPLES,
        "SUBSET_SIZE": SUBSET_SIZE,
        "POOL_SIZE": POOL_SIZE,
        "VAL_RATIO": VAL_RATIO,
        "NUM_RUNS": NUM_RUNS,
        "SAMPLE_SIZE": SAMPLE_SIZE,
        "SEED": SEED,
        "detector_nli": DETECTOR_NLI_MODEL,
        "gold_nli_candidates": GOLD_NLI_CANDIDATES,
        "judge_model": JUDGE_MODEL,
        "gen_model": GEN_MODEL,
        "weights": {"W_JUDGE": W_JUDGE, "W_FACT": W_FACT, "W_SC": W_SC},
        "gold_contra_thresh": GOLD_CONTRA_THRESH,
        "mitigation": {
            "t_low_ok": t_low_ok,
            "t_high_ok": t_high_ok,
            "abstain_threshold": abstain_threshold,
            "K": MITI_K,
            "mid_kind": MID_KIND,
            "mild_temp": MILD_TEMP,
            "strict_temp": STRICT_TEMP,
            "count_empty_as_abstain": COUNT_EMPTY_AS_ABSTAIN,
            "rerank_selfcheck_from_candidates": True,
            "abstain_if_still_risky": ABSTAIN_IF_STILL_RISKY,
            "abstain_margin": ABSTAIN_MARGIN,
            "miti_cases_file": MITI_CASES_FILE
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
        "thr_star": mode_thresholds[m],
        "f1_mean": safe_mean(f1s),
        "f1_std": safe_std(f1s),
        "question_score_std_mean": safe_mean(qstds),
        "question_score_std_std": safe_std(qstds),
        "answer_score_std_mean": safe_mean(astds),
        "answer_score_std_std": safe_std(astds),
        "delta_f1_vs_full_mean": delta_mean
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

for p in ["base", "always_mild", "always_strict_K", "policy"]:
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
        "nei_rate_mean": safe_mean(nei_rates),
        "abstain_rate_mean": safe_mean(abstain_rates),
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

with open(OUT_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2, ensure_ascii=False)

ensure_csv_header(OUT_CSV_FILE, list(rows[0].keys()))
with open(OUT_CSV_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\nSaved: {OUT_JSON_FILE}, {OUT_CSV_FILE}")
print(f"Mitigation cases saved: {MITI_CASES_FILE} (up to {MAX_CASES_PER_RUN} rows per policy per run)")
print(f"Optional detection error cases: {ERROR_CASES_FILE} (up to {MAX_ERROR_LOGS_PER_MODE} mismatches per mode per run)")