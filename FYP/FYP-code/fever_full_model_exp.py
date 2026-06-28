import os
import ssl
import json
import csv
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
# Step 0: Basic configuration
# ============================================================

ANSWER_CACHE_FILE = "fever_full_model_exp_answer_cache.csv"

N_SAMPLES = 5  
NUM_RUNS = 3
SAMPLE_SIZE = 500
SEED = 42

os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface/models"
ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================
# Step 1: Load FEVER and perform stratified sampling (1000 examples)
# ============================================================

print("Loading FEVER dataset...")

raw_dataset = load_dataset(
    "fever/fever",
    "v1.0",
    cache_dir="D:/huggingface/datasets"
)["labelled_dev"]

labels = [s["label"] for s in raw_dataset]

_, selected_idx = train_test_split(
    list(range(len(raw_dataset))),
    test_size=1000,
    stratify=labels,
    random_state=42
)

sampled = raw_dataset.select(selected_idx)


def extract_evidence(item):
    """
    Extract the first evidence sentence from the FEVER item.
    FEVER evidence format: ['wiki_url', id, 'evidence sentence']
    """
    evid_list = item.get("evidence", [])
    if not evid_list:
        return ""
    first = evid_list[0]
    if len(first) >= 3:
        return first[2]
    return ""


subset = []
for item in sampled:
    ev_text = extract_evidence(item)
    subset.append({
        "question": item["claim"],
        "best_answer": ev_text,
        "label": item["label"]
    })

print(f"FEVER stratified subset size: {len(subset)}")


# ============================================================
# Step 2: Qwen-as-a-Judge (LLM-as-a-Judge)
# ============================================================

def judge_truth(question, model_answer, reference_label, evidence):
    """
    Judge whether the model_answer correctly predicts
    SUPPORTED / REFUTED / NOT ENOUGH INFO,
    allowing extra valid explanation.
    """
    prompt = f"""
You are a strict FEVER evaluator. Your ONLY job is to check if the Model's predicted label matches the Reference Label.

Logic:
1. Identify the label in the Model Answer (SUPPORTED / REFUTED / NOT ENOUGH INFO).
2. Compare it to the Reference Label.
3. If they match, return True.
4. If they do not match, return False.
5. If the Model Answer contains factual hallucinations contradicting the evidence, return False.

A model answer is True if:
- It predicts the correct FEVER label (SUPPORTED / REFUTED / NOT ENOUGH INFO),
- It may include explanation, but MUST NOT contain incorrect factual statements.

Claim: {question}
Evidence: {evidence}
Reference Label: {reference_label}
Model Answer: {model_answer}

Respond with exactly one token: True or False.
"""
    try:
        resp = ollama.chat(
            model="qwen2.5:7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        return resp["message"]["content"].strip().lower().startswith("true")
    except:
        return False


# ============================================================
# Step 3: FactScore-like NLI (RoBERTa-large-MNLI)
# ============================================================

print("\nLoading MNLI model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

nli_model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large-mnli"
).to(device)
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")


def compute_factscore(evidence, answer):
    """
    Compute entailment and contradiction probabilities using RoBERTa-large-MNLI.
    Returns (P_entail, P_contra).
    """
    inputs = nli_tokenizer(
        evidence,
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
    """
    Use SelfCheckGPT-NLI to compute an average contradiction score
    over a group of sampled answers.
    """
    if len(answers) < 2:
        return 0.0

    sentences = []
    for a in answers:
        sents = [s.strip() for s in a.split('.') if s.strip()]
        sentences.extend(sents)

    scores = selfcheck_nli.predict(
        sentences=sentences,
        sampled_passages=answers
    )

    return float(np.mean(scores))


# ============================================================
# Step 5: Answer generation (LLaMA2)
# ============================================================

def generate_answer(question, evidence, temperature=0.8, top_p=0.9):
    """
    Generate a FEVER-style classification answer using LLaMA2-7B.
    """
    prompt = f"""
Claim: {question}
Evidence: {evidence}

Classify this claim into:
SUPPORTED / REFUTED / NOT ENOUGH INFO.

Explain briefly.
"""
    try:
        resp = ollama.chat(
            model="llama2:7b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "top_p": top_p}
        )
        return resp["message"]["content"]
    except Exception:
        return None


def extract_fever_label(answer_text):
    """
    Extract FEVER label from LLM output.
    Returns one of: SUPPORTED / REFUTED / NOT ENOUGH INFO / None
    """
    text = answer_text.upper()

    if "SUPPORTED" in text:
        return "SUPPORTED"
    if "REFUTED" in text:
        return "REFUTED"
    if "NOT ENOUGH INFO" in text or "NEI" in text:
        return "NOT ENOUGH INFO"

    return None

# ============================================================
# Step 6: Answer cache (for re-use across multiple runs)
# ============================================================

answer_cache = {}

if os.path.exists(ANSWER_CACHE_FILE):
    print("Loading FEVER answer cache...")
    with open(ANSWER_CACHE_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Backward compatibility: older files might use "claim" instead of "question"
        for row in reader:
            q = row.get("question") or row.get("claim")
            if q:
                answer_cache.setdefault(q, []).append(row["answer"])
else:
    print("No existing answer cache found.")


# ============================================================
# Step 7: Ensure each claim has N_SAMPLES answers
# ============================================================

def ensure_full_answers(question, evidence):
    """
    Ensure the given question has N_SAMPLES generated answers in the cache.
    If not, generate and append new answers to the cache and CSV file.
    """
    cur = answer_cache.get(question, [])
    needed = N_SAMPLES - len(cur)

    with open(ANSWER_CACHE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        if os.path.getsize(ANSWER_CACHE_FILE) == 0:
            writer.writeheader()

        for _ in range(needed):
            ans = generate_answer(question, evidence)
            if ans:
                writer.writerow({"question": question, "answer": ans})
                f.flush()
                answer_cache.setdefault(question, []).append(ans)

    return answer_cache[question]


# ============================================================
# Step 8: Aggregation (with ablation modes)
# ============================================================

def aggregate_signals(sc, judge_ok, entail, contra, mode="full"):
    """
    mode:
      - "full"
      - "no_judge"
      - "no_mnli"
      - "no_selfcheck"
      - "judge_only"
    """
    # weights — same as TQA version
    w_judge = 0.35
    w_fact  = 0.45
    w_sc    = 0.20

    # normalized scores
    judge_score = 1 if judge_ok else 0
    sc_score = 1 - sc
    fact_score = max(entail - contra, 0)

    if mode == "full":
        score = (
            w_judge * judge_score +
            w_fact  * fact_score +
            w_sc    * sc_score
        )

    elif mode == "no_judge":
        # renormalize to keep score in [0,1] scale comparable
        score = (
            w_fact * fact_score +
            w_sc   * sc_score
        ) / (w_fact + w_sc)

    elif mode == "no_mnli":
        score = (
            w_judge * judge_score +
            w_sc    * sc_score
        ) / (w_judge + w_sc)

    elif mode == "no_selfcheck":
        score = (
            w_judge * judge_score +
            w_fact  * fact_score
        ) / (w_judge + w_fact)

    elif mode == "judge_only":
        score = judge_score

    else:
        raise ValueError("Unknown mode")

    return score, (score >= 0.5)

# ============================================================
# Step 9: One Monte-Carlo run on FIXED idxs (supports ablation + compute only needed signals)
# ============================================================

def run_once_fixed(idxs, mode, run_id):
    """
    One Monte-Carlo run on fixed question indices.
    Produces BOTH:
      - answer-level records
      - question-level aggregated records
    """
    records = []

    tp = fp = tn = fn = 0
    halluc = 0
    total = 0

    judge_only = 0
    mnli_only = 0
    selfcheck_only = 0
    overlap = 0

    need_mnli = mode in ["full", "no_judge", "no_selfcheck"]
    need_sc   = mode in ["full", "no_judge", "no_mnli"]

    for idx in tqdm(idxs, desc=f"Evaluating {mode}", leave=False):
        idx = int(idx)
        item = subset[idx]

        q = item["question"]
        evidence = item["best_answer"]
        gold_label = item["label"]

        answers = ensure_full_answers(q, evidence)

        # ===== question-level SelfCheck =====
        sc = selfcheck_group(answers) if need_sc else 0.0
        sc_flag = sc > 0.5

        judge_hit = False
        mnli_hit = False

        # ===== store answer-level scores for THIS question =====
        answer_scores_this_q = []

        for ans in answers:
            judge_ok = judge_truth(q, ans, gold_label, evidence)

            entail, contra = (
                compute_factscore(evidence, ans) if need_mnli else (0.0, 0.0)
            )

            score, pred_ok = aggregate_signals(
                sc, judge_ok, entail, contra, mode
            )

            answer_scores_this_q.append(score)

            pred_label = extract_fever_label(ans)
            gold_ok = (pred_label == gold_label)

            gold_halluc = not gold_ok
            pred_halluc = not pred_ok

            total += 1
            halluc += int(pred_halluc)

            if pred_halluc and gold_halluc:
                tp += 1
            elif pred_halluc and not gold_halluc:
                fp += 1
            elif not pred_halluc and not gold_halluc:
                tn += 1
            else:
                fn += 1

            j = not judge_ok
            m = contra > 0.5 if need_mnli else False

            judge_hit |= j
            mnli_hit |= m

            # ===== answer-level record =====
            records.append({
                "run": run_id,
                "mode": mode,
                "question_id": idx,
                "question": q,
                "gold_label": gold_label,
                "answer": ans,
                "signals": {
                    "judge_ok": judge_ok,
                    "mnli_entail": entail,
                    "mnli_contra": contra,
                },
                "selfcheck_group": sc,
                "final_score": score,
                "final_pred_ok": pred_ok,
                "level": "answer",
            })

        # ===== question-level aggregation =====
        question_score_mean = float(np.mean(answer_scores_this_q))
        question_score_std  = float(np.std(answer_scores_this_q))

        records.append({
            "run": run_id,
            "mode": mode,
            "question_id": idx,
            "question": q,
            "question_score_mean": question_score_mean,
            "question_score_std": question_score_std,
            "selfcheck_group": sc,
            "level": "question",
        })

        # ===== question-level signal attribution =====
        if judge_hit and not mnli_hit and not sc_flag:
            judge_only += 1
        elif mnli_hit and not judge_hit and not sc_flag:
            mnli_only += 1
        elif sc_flag and not judge_hit and not mnli_hit:
            selfcheck_only += 1
        elif judge_hit or mnli_hit or sc_flag:
            overlap += 1

    # ===== save per-run detailed records =====
    with open(f"fever_scores_{mode}_run{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # ===== metrics =====
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    # ===== stability metrics =====
    answer_scores = [
        r["final_score"]
        for r in records
        if r["level"] == "answer"
    ]

    question_score_stds = [
        r["question_score_std"]
        for r in records
        if r["level"] == "question"
    ]

    question_score_mean = float(np.mean(question_score_stds))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # Answer-level variability (expected HIGH for full)
        "answer_score_std": float(np.std(answer_scores)),
        # Question-level stability (expected LOW for full)
        "question_score_std": question_score_mean,
        "signal_breakdown": {
            "judge_only": judge_only,
            "mnli_only": mnli_only,
            "selfcheck_only": selfcheck_only,
            "overlap": overlap,
        }
    }
    
# ============================================================
# Step 10: Ablation multi-run (mean ± std) with same idxs per run
# ============================================================

def ablation_multi_run(modes):
    rng = np.random.RandomState(SEED)
    summary = {m: [] for m in modes}

    for r in range(NUM_RUNS):
        idxs = rng.choice(len(subset), SAMPLE_SIZE, replace=False)
        for m in modes:
            summary[m].append(run_once_fixed(idxs, m, run_id=r))

    return summary

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    modes = ["full", "no_judge", "no_mnli", "no_selfcheck", "judge_only"]
    results = ablation_multi_run(modes)

    # ===== Save overall summary to JSON + CSV =====
    summary_json = {"modes": {}}
    rows = []

    print("\n=== FINAL SUMMARY (F1 + Score Stability) ===")
    print(f"{'mode':12s} | {'F1_mean':>8s} {'F1_std':>8s} | {'Q-std':>10s} | {'A-std':>10s}")
    print("-"*80)

    for m in modes:
        f1s = [r["f1"] for r in results[m]]

        question_stds = [r["question_score_std"] for r in results[m]]
        answer_stds   = [r["answer_score_std"] for r in results[m]]

        summary_json["modes"][m] = {
            "f1": {"values": f1s, "mean": float(np.mean(f1s)), "std": float(np.std(f1s))},
            "question_score_std": {"values": question_stds, "mean": float(np.mean(question_stds)), "std": float(np.std(question_stds))},
            "answer_score_std": {"values": answer_stds, "mean": float(np.mean(answer_stds)), "std": float(np.std(answer_stds))}
        }

        rows.append({
            "mode": m,
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "question_score_std_mean": float(np.mean(question_stds)),
            "question_score_std_run_std": float(np.std(question_stds)),
            "answer_score_std_mean": float(np.mean(answer_stds)),
            "answer_score_std_std":  float(np.std(answer_stds))
        })

        with open("fever_overall_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False)

        with open("fever_overall_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        
        print(
            f"{m:12s} | "
            f"{np.mean(f1s):8.4f} {np.std(f1s):8.4f} | "
            f"Q-std={np.mean(question_stds):.4f} | "
            f"A-std={np.mean(answer_stds):.4f}"
        )

        print("\nSaved: fever_overall_summary.json, fever_overall_summary.csv")
        
    