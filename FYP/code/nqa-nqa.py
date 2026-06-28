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
import ollama

# ============================================================
# Paper-grade Natural Questions (NQ) Hallucination Detector Evaluation (TQA/FEVER-style)
#
# Upgrades vs your old NQ script:
#   1) GOLD uses DIFFERENT NLI model (DeBERTa) to avoid leakage
#   2) Detector MNLI uses RoBERTa-large-MNLI (feature model)
#   3) Threshold tuned on VAL per mode, then fixed on TEST
#   4) Paired Monte-Carlo runs on TEST for fair ablation comparison
#   5) Outputs F1 mean/std, Q-std, A-std, thr*, ΔF1 vs full + optional error cases
#
# Hallucination label:
#   gold_ok := answer entails reference AND does NOT contradict reference (via GOLD NLI)
#   gold_halluc := not gold_ok
# ============================================================

# -----------------------------
# Config
# -----------------------------
ANSWER_CACHE_FILE = "nq_answers_cache_paper-f.csv"

OUT_JSON_FILE = "nq_paper_overall_summary-f.json"
OUT_CSV_FILE  = "nq_paper_overall_summary-f.csv"

ERROR_CASES_FILE = "nq_paper_error_cases-f.csv"
MAX_ERROR_LOGS_PER_MODE = 200

# Sampling / evaluation
N_SAMPLES = 5
SEED = 42

# Build a pool first (for val/test split inside the pool)
SUBSET_MAX = 500      # how many NQ items you use total (from NQ validation)
POOL_SIZE  = 500      # pool size (<= SUBSET_MAX)
VAL_RATIO  = 0.2

# Paired Monte-Carlo on TEST split
NUM_RUNS = 3
SAMPLE_SIZE = 200      # per run, number of questions (not answers)

# Ollama models
JUDGE_MODEL = "qwen2.5:7b-instruct"
GEN_MODEL   = "llama2:7b"

# Detector NLI (feature model)
DETECTOR_NLI_MODEL = "roberta-large-mnli"

# Gold NLI (for gold labels) — different model to avoid leakage
GOLD_NLI_CANDIDATES = [
    "microsoft/deberta-v3-base-mnli",
    "microsoft/deberta-v3-large-mnli",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
]

# Aggregation weights (same spirit as your TQA/FEVER paper-grade)
W_JUDGE = 0.35
W_FACT  = 0.45
W_SC    = 0.20

# GOLD entail/contra threshold
GOLD_THR = 0.50  # typical 0.5

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
    """
    Cache schema: question, answer
    """
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

# -----------------------------
# Load NQ validation and build subset
# -----------------------------
print("Loading Natural Questions (validation split)...")
raw_dataset = load_dataset(
    "natural_questions",
    split="validation",
    cache_dir="D:/huggingface/datasets"
)
print("Dataset loaded:", len(raw_dataset))

def extract_reference_from_nq_item(item):
    """
    Best-effort extraction for a usable reference answer string.
    Priority:
      1) short answer text (if present)
      2) yes/no answer (YES/NO)
    """
    ann_raw = item.get("annotations", None)
    if ann_raw is None:
        return ""

    if isinstance(ann_raw, list):
        if len(ann_raw) == 0:
            return ""
        ann = ann_raw[0]
    elif isinstance(ann_raw, dict):
        ann = ann_raw
    else:
        return ""

    short_answers = ann.get("short_answers", [])
    yes_no = ann.get("yes_no_answer", "NONE")

    # short answer
    if isinstance(short_answers, list) and len(short_answers) > 0:
        first = short_answers[0]
        # Depending on dataset config, "text" can be string or list or missing
        if isinstance(first, dict) and "text" in first:
            ref = normalize_text(first["text"])
            if ref:
                return ref

    # yes/no
    if yes_no in ["YES", "NO"]:
        return yes_no

    return ""

valid_items = []
for item in tqdm(raw_dataset, desc="Filtering NQ for valid refs"):
    ref = extract_reference_from_nq_item(item)
    if ref:
        valid_items.append({
            "question": normalize_question(item.get("question", "")),
            "reference": ref
        })

print("Valid items with usable references:", len(valid_items))

rng = np.random.RandomState(SEED)
subset_size = min(SUBSET_MAX, len(valid_items))
subset = rng.choice(valid_items, size=subset_size, replace=False).tolist()
print("Final subset size:", len(subset))

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
print("\nLoading GOLD NLI model (for gold labels) with fallback...")
for cand in GOLD_NLI_CANDIDATES:
    try:
        print("  trying:", cand)
        gold_tok, gold_nli = load_nli_model(cand)
        print("  -> loaded GOLD model:", cand)
        break
    except Exception as e:
        print("  failed:", cand, "|", str(e)[:140])

if gold_tok is None:
    raise RuntimeError("Failed to load any GOLD NLI model candidates. Check HF connectivity/cache/model names.")

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
# GOLD label definition for NQ (DeBERTa-based)
# -----------------------------
def nq_gold_ok(answer: str, reference: str, thresh=GOLD_THR):
    """
    Gold=True if:
      - answer entails reference
      - answer does NOT contradict reference
    Using GOLD NLI model (DeBERTa), not detector NLI.
    """
    if not answer or not reference:
        return False
    ent, con = nli_ent_con(gold_tok, gold_nli, reference, answer)
    return (ent > thresh) and (con < thresh)

# -----------------------------
# Detector MNLI signal (RoBERTa) reference -> answer
# -----------------------------
def detector_mnli_signal(reference: str, answer: str):
    """
    Returns (entail, contra) using DETECTOR NLI model.
    """
    if not reference or not answer:
        return 0.0, 0.0
    ent, con = nli_ent_con(det_tok, det_nli, reference, answer)
    return float(ent), float(con)

# -----------------------------
# Judge (Detector feature, NOT gold)
# -----------------------------
def judge_truth(question: str, answer: str, reference: str) -> bool:
    """
    Judge should be '宽容型'：允许额外正确信息，只在出现事实错误/矛盾时判 False，
    并要求回答至少覆盖 reference 的含义（否则就 True/False? 这里按 NQ 任务：应包含 reference 含义才 True）
    """
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
        resp = ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        return resp["message"]["content"].strip().lower().startswith("true")
    except Exception:
        return False

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

def generate_answer(question: str):
    """
    Stochastic generation (LLaMA2).
    """
    try:
        resp = ollama.chat(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": question}],
            options={"temperature": 0.8, "top_p": 0.9}
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

# -----------------------------
# Score aggregation (TQA/FEVER-style)
# -----------------------------
def mode_score(sc, judge_ok, ent, con, mode):
    """
    Continuous score (roughly [0,1]). Threshold applied externally.
    Signals:
      - judge_ok: bool
      - ent, con: detector MNLI signals reference->answer
      - sc: SelfCheck contradiction score (higher worse)
    """
    judge_score = 1.0 if judge_ok else 0.0
    sc_score = 1.0 - float(sc)         # higher better
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
# Build question pool: split into VAL/TEST (inside subset)
# -----------------------------
print("\nBuilding question pool...")
pool_size = min(POOL_SIZE, len(subset))
pool_q_idxs = rng.choice(len(subset), size=pool_size, replace=False).tolist()
rng.shuffle(pool_q_idxs)

val_n = int(len(pool_q_idxs) * VAL_RATIO)
val_q_idxs = pool_q_idxs[:val_n]
test_q_idxs = pool_q_idxs[val_n:]

print(f"Pool size: {len(pool_q_idxs)} | val questions: {len(val_q_idxs)} | test questions: {len(test_q_idxs)}")

# Each question pack:
# {
#   "q_idx": int,
#   "q": str,
#   "ref": str,
#   "answers": list[str],
#   "sc": float,
#   "per_answer": [ {answer, gold_ok, judge_ok, ent, con} ... ]
# }
def build_question_pack(q_idx: int):
    item = subset[int(q_idx)]
    q = item["question"]
    ref = item["reference"]

    answers = ensure_answers(q, N_SAMPLES)
    sc = selfcheck_group(answers)

    per_answer = []
    for ans in answers:
        # GOLD: DeBERTa-based entail/contra vs reference
        gold_ok = nq_gold_ok(ans, ref)

        # Detector features:
        judge_ok = judge_truth(q, ans, ref)
        ent, con = detector_mnli_signal(ref, ans)

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

print("\nPrecomputing features for VAL + TEST pool (can be slow due to Ollama calls)...")
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

            # optional mismatch logging
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
    # paired: same question subset for every mode in this run
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
print("\n=== FINAL SUMMARY (Paper-grade NQ: tuned threshold, decoupled gold) ===")
print(f"{'mode':12s} | {'F1_mean':>8s} {'F1_std':>8s} | {'Q-std':>10s} | {'A-std':>10s} | {'thr*':>6s}")
print("-" * 100)

rows = []
summary_json = {
    "config": {
        "N_SAMPLES": N_SAMPLES,
        "SUBSET_MAX": SUBSET_MAX,
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
        "gold_thr": GOLD_THR,
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