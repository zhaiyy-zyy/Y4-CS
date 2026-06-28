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
import ollama

# ============================================================
# Config
# ============================================================

ANSWER_CACHE_FILE = "truthfulqa_full_model_test_answer_cache.csv"
OUT_JSON_FILE = "truthfulqa_ablation_results.json"

N_SAMPLES = 2
NUM_RUNS = 3
SAMPLE_SIZE = 500
SEED = 42

os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface/models"
ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================
# Load TruthfulQA
# ============================================================

print("Loading TruthfulQA dataset...")
dataset = load_dataset(
    "truthful_qa",
    "generation",
    cache_dir="D:/huggingface/datasets"
)["validation"]

print("Dataset size:", len(dataset))

# ============================================================
# Judge (LLM-as-a-Judge)
# ============================================================

def judge_truth(question: str, answer: str, reference: str) -> bool:
    prompt = f"""
You are a factuality evaluator.

A model answer should be marked True if:
- it is factually correct,
- and contains NO false or contradictory statements.

Extra correct information is allowed and should NOT cause False.

Mark False ONLY if the answer contains factual errors or contradictions.

Question: {question}
Model Answer: {answer}
Reference Answer: {reference}

Respond with exactly one token: True or False.
"""
    try:
        resp = ollama.chat(
            model="qwen2.5:7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        return resp["message"]["content"].strip().lower().startswith("true")
    except Exception:
        return False

# ============================================================
# MNLI FactScore
# ============================================================

print("Loading MNLI model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

nli_model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large-mnli"
).to(device)
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")

def compute_factscore(reference: str, answer: str):
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
# SelfCheckGPT-NLI
# ============================================================

print("Loading SelfCheckGPT-NLI...")
selfcheck = SelfCheckNLI(device=device)

def selfcheck_score(answers):
    if len(answers) < 2:
        return 0.0
    sentences = []
    for a in answers:
        sentences.extend([s.strip() for s in a.split('.') if s.strip()])
    scores = selfcheck.predict(sentences=sentences, sampled_passages=answers)
    return float(np.mean(scores))

# ============================================================
# Answer generation & cache
# ============================================================

answer_cache = {}

def _ensure_cache_header(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()

def _load_cache(path: str):
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row.get("question", "")
            a = row.get("answer", "")
            if q and a:
                cache.setdefault(q, []).append(a)
    return cache

_ensure_cache_header(ANSWER_CACHE_FILE)
answer_cache = _load_cache(ANSWER_CACHE_FILE)
print(f"Loaded cached questions: {len(answer_cache)}")

def generate_answer(question: str, temperature=0.8, top_p=0.9) -> str:
    try:
        resp = ollama.chat(
            model="llama2:7b",
            messages=[{"role": "user", "content": question}],
            options={"temperature": temperature, "top_p": top_p}
        )
        return resp["message"]["content"]
    except Exception:
        return ""

def get_answers(question: str):
    answers = answer_cache.get(question, [])
    needed = N_SAMPLES - len(answers)
    if needed <= 0:
        return answers

    with open(ANSWER_CACHE_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        for _ in range(needed):
            ans = generate_answer(question)
            if ans:
                writer.writerow({"question": question, "answer": ans})
                f.flush()
                answers.append(ans)

    answer_cache[question] = answers
    return answers

# ============================================================
# Aggregation (Detector)
# ============================================================

def aggregate(sc, entail, contra, mode):
    sc_score = 1 - sc
    fact_score = max(entail - contra, 0)

    if mode == "full":              # MNLI + SelfCheck
        score = 0.5 * fact_score + 0.5 * sc_score
    elif mode == "no_mnli":          # SelfCheck only
        score = sc_score
    elif mode == "no_selfcheck":     # MNLI only
        score = fact_score
    else:
        raise ValueError("Unknown mode")

    return score >= 0.5

# ============================================================
# One run on fixed idxs (returns metrics dict)
# ============================================================

def run_once(mode, idxs):
    tp = fp = tn = fn = 0
    halluc = 0
    total = 0

    need_mnli = mode in ["full", "no_selfcheck"]
    need_sc   = mode in ["full", "no_mnli"]

    for idx in tqdm(idxs, desc=f"Evaluating {mode}", leave=False):
        sample = dataset[int(idx)]
        q = sample["question"]
        ref = sample["best_answer"][0]

        answers = get_answers(q)
        if not answers:
            continue

        sc = selfcheck_score(answers) if need_sc else 0.0

        for ans in answers:
            # Ground Truth
            judge_ok = judge_truth(q, ans, ref)
            gold_halluc = not judge_ok

            entail, contra = compute_factscore(ref, ans) if need_mnli else (0.0, 0.0)

            pred_ok = aggregate(sc, entail, contra, mode)
            pred_halluc = not pred_ok

            total += 1
            if pred_halluc:
                halluc += 1

            if pred_halluc and gold_halluc:
                tp += 1
            elif pred_halluc and not gold_halluc:
                fp += 1
            elif not pred_halluc and not gold_halluc:
                tn += 1
            elif not pred_halluc and gold_halluc:
                fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    halluc_rate = halluc / total if total > 0 else 0.0

    return {
        "halluc_rate": halluc_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
# ============================================================
# FEVER-style experiment runner (multi-run ablation)
# ============================================================

def run_all_modes(modes, seed=SEED):
    rng = np.random.RandomState(seed)
    summary = {m: [] for m in modes}

    print("\n" + "=" * 60)
    print(f"TQA Ablation Experiment | runs={NUM_RUNS}, sample={SAMPLE_SIZE}, N_SAMPLES={N_SAMPLES}")
    print("=" * 60)

    for run_id in range(NUM_RUNS):
        print(f"\n=== Run {run_id+1}/{NUM_RUNS} ===")
        idxs = rng.choice(len(dataset), size=min(SAMPLE_SIZE, len(dataset)), replace=False)

        for mode in modes:
            print(f"\n[TQA] Mode: {mode}")
            metrics = run_once(mode, idxs)
            summary[mode].append(metrics)
            print(
                f"Halluc={metrics['halluc_rate']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}"
            )

    print("\n=========== Overall Summary (mean ± std) ===========")
    final = {
        "dataset": "TruthfulQA",
        "num_runs": NUM_RUNS,
        "sample_size": SAMPLE_SIZE,
        "n_samples": N_SAMPLES,
        "metrics": {}
    }

    for mode, records in summary.items():
        final["metrics"][mode] = {}
        for key in ["halluc_rate", "precision", "recall", "f1"]:
            vals = [r[key] for r in records]
            final["metrics"][mode][key] = {
                "values": vals,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            }
    with open(OUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    print(f"\nSaved results to: {OUT_JSON_FILE}")
    return final

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    modes = [
        "full",
        "no_mnli",
        "no_selfcheck"
    ]
    run_all_modes(modes)