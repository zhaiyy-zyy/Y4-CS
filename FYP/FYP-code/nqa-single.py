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
# Step 0: Basic Configuration
# ============================================================

ANSWER_CACHE_FILE = "nq_full_model_exp_answer_cache.csv"
OUT_SUMMARY_JSON = "nq_overall_summary.json"
OUT_SUMMARY_CSV  = "nq_overall_summary.csv"

N_SAMPLES = 5
NUM_RUNS = 3
SAMPLE_SIZE = 200
SEED = 42

os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface/models"
ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================
# Step 1: Load Natural Questions (validation split) and build subset
# ============================================================

print("Loading Natural Questions (validation split)...")
raw_dataset = load_dataset(
    "natural_questions",
    split="validation",
    cache_dir="D:/huggingface/datasets"
)
print("Dataset loaded:", len(raw_dataset))


def normalize_text(x):
    """Ensure answer text is a clean string."""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list) and len(x) > 0:
        return str(x[0]).strip()
    return ""


def normalize_question(q):
    """Ensure question field is a clean string."""
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
    return str(q)


valid_items = []
for item in raw_dataset:
    ann_raw = item.get("annotations", None)
    if ann_raw is None:
        continue

    if isinstance(ann_raw, list):
        if len(ann_raw) == 0:
            continue
        ann = ann_raw[0]
    elif isinstance(ann_raw, dict):
        ann = ann_raw
    else:
        continue

    short_answers = ann.get("short_answers", [])
    yes_no = ann.get("yes_no_answer", "NONE")

    # short answers
    if isinstance(short_answers, list) and len(short_answers) > 0:
        first = short_answers[0]
        if isinstance(first, dict) and "text" in first:
            ref = normalize_text(first["text"])
            if ref != "":
                valid_items.append({
                    "question": normalize_question(item["question"]),
                    "best_answer": ref
                })
                continue

    # yes/no answers
    if yes_no in ["YES", "NO"]:
        valid_items.append({
            "question": normalize_question(item["question"]),
            "best_answer": yes_no
        })
        continue

print("Valid items with real answers:", len(valid_items))

np.random.seed(SEED)
subset = list(np.random.choice(
    valid_items,
    size=min(500, len(valid_items)),
    replace=False
))
print("Final subset size:", len(subset))

# ============================================================
# Step 2: LLM-as-a-Judge using Qwen2.5-7B-Instruct
# ============================================================

def judge_truth(question, model_answer, reference):
    prompt = f"""
You are a strict factuality evaluator.

A model answer should be marked True if:
- it is factually correct,
- and contains NO false or contradictory statements.
- If the Model Answer contains the reference text or its exact meaning, return True.
- Otherwise, return False.

Extra correct information is allowed and should NOT cause False.

Mark False ONLY if the answer contains factual errors or contradictions.

Compare the Model Answer to the Reference Answer.
Question: {question}
Model Answer: {model_answer}
Reference Answer: {reference}

Respond with exactly one token: True or False.
"""
    try:
        resp = ollama.chat(
            model="qwen2.5:7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        raw = resp["message"]["content"].strip().lower()
        return raw.startswith("true")
    except:
        return False

# ============================================================
# Step 3: FactScore (RoBERTa-large-MNLI)
# ============================================================

print("\nLoading MNLI model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

nli_model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large-mnli"
).to(device)
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")


def compute_factscore(reference, answer):
    inputs = nli_tokenizer(
        reference,
        answer,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    entail = float(probs[2])
    contra = float(probs[0])
    return entail, contra

# ============================================================
# Step 4: SelfCheckGPT-NLI
# ============================================================

print("\nLoading SelfCheckGPT-NLI...")
selfcheck_nli = SelfCheckNLI(device=device)


def selfcheck_group(answers):
    """Compute SelfCheckGPT-NLI average contradiction score."""
    if len(answers) < 2:
        return 0.0

    sentences = []
    for ans in answers:
        sents = [s.strip() for s in ans.split('.') if s.strip()]
        sentences.extend(sents)

    scores = selfcheck_nli.predict(
        sentences=sentences,
        sampled_passages=answers
    )
    return float(np.mean(scores))

# ============================================================
# Step 5: Stochastic Answer Generation (LLaMA2)
# ============================================================

def generate_answer(q, temperature=0.8, top_p=0.9):
    try:
        resp = ollama.chat(
            model="llama2:7b",
            messages=[{"role": "user", "content": q}],
            options={"temperature": temperature, "top_p": top_p}
        )
        return resp["message"]["content"]
    except:
        return None

# ============================================================
# Step 6: Answer Cache
# ============================================================

answer_cache = {}

if os.path.exists(ANSWER_CACHE_FILE):
    print("Loading NQ answer cache...")
    with open(ANSWER_CACHE_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row["question"]
            answer_cache.setdefault(q, []).append(row["answer"])
else:
    print("No existing answer cache found.")
    with open(ANSWER_CACHE_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()

# ============================================================
# Step 7: Ensure each question has N sampled answers
# ============================================================

def get_answers(question):
    current = answer_cache.get(question, [])
    needed = N_SAMPLES - len(current)

    if needed <= 0:
        return current

    with open(ANSWER_CACHE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        if os.path.getsize(ANSWER_CACHE_FILE) == 0:
            writer.writeheader()

        for _ in range(needed):
            ans = generate_answer(question)
            if ans:
                writer.writerow({"question": question, "answer": ans})
                answer_cache.setdefault(question, []).append(ans)

    return answer_cache[question]

# ============================================================
# Step 8: Aggregation (supports ablation modes)  [FEVER-style]
# ============================================================

def aggregate_signals(sc, judge_ok, entail, contra, mode="full"):
    """
    Returns:
      (final_score, pred_ok)
    """
    w_judge = 0.45
    w_fact  = 0.35
    w_sc    = 0.20

    sc_score = 1 - sc
    fact_score = max(entail - contra, 0)
    judge_score = 1 if judge_ok else 0

    if mode == "full":
        final_score = (
            w_judge * judge_score +
            w_fact  * fact_score +
            w_sc    * sc_score
        )
    elif mode == "no_judge":
        final_score = (w_fact * fact_score + w_sc * sc_score) / (w_fact + w_sc)
    elif mode == "no_mnli":
        final_score = (w_judge * judge_score + w_sc * sc_score) / (w_judge + w_sc)
    elif mode == "no_selfcheck":
        final_score = (w_judge * judge_score + w_fact * fact_score) / (w_judge + w_fact)
    elif mode == "judge_only":
        final_score = judge_score
    else:
        raise ValueError("Unknown mode")

    return final_score, (final_score >= 0.5)

# ============================================================
# Step 9: One Monte-Carlo run on FIXED idxs (FEVER-style outputs)
# ============================================================

def check_exact_match(reference, model_answer):
    """
    Ground Truth Heuristic:
    For NQ, we treat answer as correct if it contains the reference substring.
    """
    if not reference or not model_answer:
        return False
    return reference.lower() in model_answer.lower()


def run_once_fixed(idxs, mode, run_id):
    """
    FEVER-style:
      - compute answer-level score distribution (A-std)
      - compute per-question std then average (Q-std)
      - compute F1 using gold_halluc vs pred_halluc
    """
    records = []

    tp = fp = tn = fn = 0
    halluc = 0
    total = 0

    need_mnli  = mode in ["full", "no_judge", "no_selfcheck"]
    need_sc    = mode in ["full", "no_judge", "no_mnli"]
    need_judge = (mode != "no_judge")

    answer_scores_all = []
    question_score_stds = []

    for idx in tqdm(idxs, desc=f"Evaluating {mode}", leave=False):
        sample = subset[int(idx)]
        q = sample["question"]
        ref = sample["best_answer"]

        answers = get_answers(q)

        # ===== question-level SelfCheck =====
        sc = selfcheck_group(answers) if need_sc else 0.0

        # store answer-level scores for THIS question
        answer_scores_this_q = []

        for ans in answers:
            judge_pred = judge_truth(q, ans, ref) if need_judge else False
            entail, contra = compute_factscore(ref, ans) if need_mnli else (0.0, 0.0)

            score, pred_ok = aggregate_signals(sc, judge_pred, entail, contra, mode)

            answer_scores_this_q.append(score)
            answer_scores_all.append(score)

            # gold heuristic for NQ
            gt_is_correct = check_exact_match(ref, ans)

            gold_halluc = not gt_is_correct
            pred_halluc = not pred_ok

            total += 1
            halluc += int(pred_halluc)

            if pred_halluc and gold_halluc:
                tp += 1
            elif pred_halluc and not gold_halluc:
                fp += 1
            elif (not pred_halluc) and (not gold_halluc):
                tn += 1
            else:
                fn += 1

            # answer-level record
            records.append({
                "run": run_id,
                "mode": mode,
                "question": q,
                "reference": ref,
                "answer": ans,
                "signals": {
                    "judge_ok": judge_pred,
                    "mnli_entail": entail,
                    "mnli_contra": contra,
                },
                "selfcheck_group": sc,
                "final_score": score,
                "final_pred_ok": pred_ok,
                "gold_ok": gt_is_correct,
                "level": "answer",
            })

        # ===== question-level aggregation (std over this question's answers) =====
        q_std = float(np.std(answer_scores_this_q)) if len(answer_scores_this_q) else 0.0
        q_mean = float(np.mean(answer_scores_this_q)) if len(answer_scores_this_q) else 0.0
        question_score_stds.append(q_std)

        records.append({
            "run": run_id,
            "mode": mode,
            "question": q,
            "question_score_mean": q_mean,
            "question_score_std": q_std,
            "selfcheck_group": sc,
            "level": "question",
        })

    # save per-run detailed records
    with open(f"nq_scores_{mode}_run{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    halluc_rate = halluc / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    return {
        "halluc_rate": halluc_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "answer_score_std": float(np.std(answer_scores_all)) if answer_scores_all else 0.0,        # A-std
        "question_score_std": float(np.mean(question_score_stds)) if question_score_stds else 0.0  # Q-std (mean over questions)
    }

# ============================================================
# Step 10: Ablation multi-run (mean ± std) with same idxs per run
# ============================================================

def ablation_multi_run(modes):
    rng = np.random.RandomState(SEED)
    summary = {m: [] for m in modes}

    for run_id in range(NUM_RUNS):
        idxs = rng.choice(len(subset), SAMPLE_SIZE, replace=False)
        for m in modes:
            summary[m].append(run_once_fixed(idxs, m, run_id=run_id))

    return summary

# ============================================================
# Step 11: Main + final summary (FEVER-style print)
# ============================================================

if __name__ == "__main__":
    modes = ["full", "no_judge", "no_mnli", "no_selfcheck", "judge_only"]
    results = ablation_multi_run(modes)

    summary_json = {"modes": {}}
    rows = []

    print("\n=== FINAL SUMMARY (F1 + Score Stability) ===")
    print(f"{'mode':12s} | {'F1_mean':>8s} {'F1_std':>8s} | {'Q-std':>10s} | {'A-std':>10s}")
    print("-" * 80)

    for m in modes:
        f1s = [r["f1"] for r in results[m]]
        qstds = [r["question_score_std"] for r in results[m]]
        astds = [r["answer_score_std"] for r in results[m]]

        summary_json["modes"][m] = {
            "f1": {"values": f1s, "mean": float(np.mean(f1s)), "std": float(np.std(f1s))},
            "question_score_std": {"values": qstds, "mean": float(np.mean(qstds)), "std": float(np.std(qstds))},
            "answer_score_std": {"values": astds, "mean": float(np.mean(astds)), "std": float(np.std(astds))}
        }

        rows.append({
            "mode": m,
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "question_score_std_mean": float(np.mean(qstds)),
            "question_score_std_run_std": float(np.std(qstds)),
            "answer_score_std_mean": float(np.mean(astds)),
            "answer_score_std_std": float(np.std(astds))
        })

        print(
            f"{m:12s} | "
            f"{np.mean(f1s):8.4f} {np.std(f1s):8.4f} | "
            f"Q-std={np.mean(qstds):.4f} | "
            f"A-std={np.mean(astds):.4f}"
        )

    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    with open(OUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {OUT_SUMMARY_JSON}, {OUT_SUMMARY_CSV}")