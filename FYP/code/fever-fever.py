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
# Key upgrades:
#   1) Gold factual check uses DIFFERENT NLI model (DeBERTa) -> avoid leakage
#   2) Threshold tuned on VAL per mode, then fixed on TEST
#   3) Paired Monte-Carlo runs on TEST for fair ablation comparison
#   4) Outputs F1 mean/std, Q-std, A-std, thr*, ΔF1 vs full
#   5) Optional error cases CSV
# ============================================================

# -----------------------------
# Config
# -----------------------------
ANSWER_CACHE_FILE = "fever_answers_cache_paper-f.csv"

OUT_JSON_FILE = "fever_paper_overall_summary-f.json"
OUT_CSV_FILE  = "fever_paper_overall_summary-f.csv"

ERROR_CASES_FILE = "fever_paper_error_cases-f.csv"
MAX_ERROR_LOGS_PER_MODE = 200

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
SAMPLE_SIZE = 200        # per run, number of questions (not answers!)

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

# Aggregation weights (same spirit as your TQA paper-grade version)
W_JUDGE = 0.35
W_FACT  = 0.45
W_SC    = 0.20

# NLI thresholds used inside gold factual check
GOLD_CONTRA_THRESH = 0.50   # if answer contradicts evidence above this => gold False
GOLD_ENTAIL_THRESH = 0.00   # optional: not used strongly; kept for completeness

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
    """
    FEVER evidence field formats can vary depending on dataset variant.
    We try best-effort extraction of the first evidence sentence-like string.
    """
    evid = item.get("evidence", None)
    if evid is None:
        return ""

    # often: list of evidence entries, each is list/tuple of length>=3 where index 2 is text
    if isinstance(evid, list) and len(evid) > 0:
        first = evid[0]
        # Sometimes nested deeper
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], (list, tuple)):
            first = first[0]

        if isinstance(first, (list, tuple)) and len(first) >= 3:
            txt = first[2]
            return str(txt).strip() if txt is not None else ""
        # fallback: if it's already a string
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

# stratified sampling to SUBSET_SIZE
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
    subset.append({
        "question": q,
        "evidence": ev,
        "gold_label": gold_label
    })

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
    raise RuntimeError("Failed to load any GOLD NLI model candidates. Please check HF connectivity/cache/model names.")

@torch.no_grad()
def nli_ent_con(tok, mdl, premise, hypothesis):
    """
    Return (P_entail, P_contra)
    label order: contradiction(0), neutral(1), entailment(2)
    """
    inputs = tok(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)
    probs = F.softmax(mdl(**inputs).logits, dim=-1)[0].detach().cpu().numpy()
    return float(probs[2]), float(probs[0])

# -----------------------------
# Gold definition (FEVER):
# gold_ok = (predicted label matches dataset label) AND (NOT contradict evidence)
# Fact contradiction check uses GOLD NLI (DeBERTa), not detector NLI
# -----------------------------
def parse_fever_label_from_answer(answer_text: str):
    """
    Extract FEVER label token from LLM output.
    """
    if not answer_text:
        return None
    t = answer_text.upper()
    # order matters: "NOT ENOUGH INFO" should be checked before "SUPPORTED" etc
    if "NOT ENOUGH INFO" in t or "\nNEI" in t or " NEI" in t:
        return "NOT ENOUGH INFO"
    if "REFUTED" in t:
        return "REFUTED"
    if "SUPPORTED" in t:
        return "SUPPORTED"
    return None

def gold_factual_ok(answer_text: str, evidence: str):
    """
    True if answer does NOT contradict evidence.
    (We treat high contradiction probability as factual error.)
    Uses GOLD NLI model.
    """
    if not evidence or not answer_text:
        # if no evidence text, do not penalize by contradiction
        return True
    ent, con = nli_ent_con(gold_tok, gold_nli, evidence, answer_text)
    # if contradiction is high -> factual conflict with evidence
    return (con < GOLD_CONTRA_THRESH)

def fever_gold_ok(answer_text: str, gold_label: str, evidence: str):
    pred_label = parse_fever_label_from_answer(answer_text)
    if pred_label is None or gold_label is None:
        return False
    if pred_label != gold_label:
        return False
    # allow extra info, but must not contradict evidence
    if not gold_factual_ok(answer_text, evidence):
        return False
    return True

# -----------------------------
# Judge (Detector, NOT oracle)
# -----------------------------
def judge_truth(question: str, model_answer: str, gold_label: str, evidence: str) -> bool:
    """
    Judge is part of the detector features (NOT used as gold).
    It should allow extra correct explanation but penalize factual errors vs evidence.
    """
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
# Detector MNLI signal (RoBERTa) from evidence -> answer
# -----------------------------
def detector_mnli_signal(evidence: str, answer_text: str):
    """
    Returns:
      ent: P(entailment) evidence -> answer
      con: P(contradiction) evidence -> answer
    Uses DETECTOR NLI model (RoBERTa-MNLI).
    """
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
    Higher => less consistent => worse.
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
    """
    Generate FEVER-style classification answer using LLaMA2-7B.
    """
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
# Score aggregation (TQA-style)
# -----------------------------
def mode_score(sc, judge_ok, ent, con, mode):
    """
    Continuous score in [0, 1]-ish. Threshold applied externally.
    Signals:
      - judge_ok: bool
      - ent, con: detector MNLI signals evidence->answer
      - sc: SelfCheck contradiction score (higher worse)
    """
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
# Build a question pool (like TQA code style)
# -----------------------------
print("\nBuilding question pool...")
rng = np.random.RandomState(SEED)
pool_q_idxs = rng.choice(len(subset), size=min(POOL_SIZE, len(subset)), replace=False).tolist()
rng.shuffle(pool_q_idxs)

val_n = int(len(pool_q_idxs) * VAL_RATIO)
val_q_idxs = pool_q_idxs[:val_n]
test_q_idxs = pool_q_idxs[val_n:]

print(f"Pool size: {len(pool_q_idxs)} | val questions: {len(val_q_idxs)} | test questions: {len(test_q_idxs)}")

# Each question pack:
# {
#   "q_idx": int,
#   "q": str,
#   "evidence": str,
#   "gold_label": str,
#   "answers": list[str],
#   "sc": float,
#   "per_answer": [ {answer, gold_ok, judge_ok, ent, con} ... ]
# }
def build_question_pack(q_idx: int):
    item = subset[int(q_idx)]
    q = item["question"]
    evidence = item["evidence"]
    gold_label = item["gold_label"]

    answers = ensure_answers(q, evidence, N_SAMPLES)
    sc = selfcheck_group(answers)

    per_answer = []
    for ans in answers:
        # GOLD: dataset label match + DeBERTa contradiction check
        gold_ok = fever_gold_ok(ans, gold_label, evidence)

        # Detector features:
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
# Evaluation helpers
# -----------------------------
def eval_on_packs(packs, mode, threshold, run_id=0, log_errors_path=None, max_log=0):
    """
    Answer-level hallucination detection:
      gold_halluc = not gold_ok
      pred_halluc = (score < threshold)

    Also computes:
      A-std: std over all answer scores
      Q-std: mean over questions of std(scores within question)
    """
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

            # optional mismatch logging
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
    """
    Choose threshold that maximizes F1 on validation.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)

    # Precompute scores + gold labels once for speed
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
# Main experiment
# -----------------------------
modes = ["full", "no_judge", "no_mnli", "no_selfcheck", "judge_only"]

print("\nTuning threshold on VAL for each mode...")
mode_thresholds = {}
for m in modes:
    t_star, f1_star = tune_threshold_on_val(val_packs, m)
    mode_thresholds[m] = t_star
    print(f"  mode={m:12s} | best_thr={t_star:.2f} | val_F1={f1_star:.4f}")

# Paired Monte-Carlo on TEST
print("\nRunning paired Monte-Carlo on TEST...")
rng = np.random.RandomState(SEED)
summary = {m: [] for m in modes}

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

# -----------------------------
# Final summary + ΔF1 vs full
# -----------------------------
print("\n=== FINAL SUMMARY (Paper-grade FEVER: tuned threshold, decoupled gold) ===")
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
    },
    "thresholds": mode_thresholds,
    "modes": {}
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

ensure_csv_header(OUT_CSV_FILE, list(rows[0].keys()))
with open(OUT_CSV_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\nSaved: {OUT_JSON_FILE}, {OUT_CSV_FILE}")
print(f"Optional error cases: {ERROR_CASES_FILE} (up to {MAX_ERROR_LOGS_PER_MODE} mismatches per mode per run)")