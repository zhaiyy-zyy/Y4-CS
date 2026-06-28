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

ANSWER_CACHE_FILE = "nq_full_model_test_answer_cache.csv"

N_SAMPLES = 2
NUM_RUNS = 3
SAMPLE_SIZE = 300
SEED = 42

os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface/models"
ssl._create_default_https_context = ssl._create_unverified_context

# ============================================================
# Step 1: Load Natural Questions (filter invalid items + sample 500 items)
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
subset = list(np.random.choice(valid_items, size=min(500, len(valid_items)), replace=False))

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
    with open(ANSWER_CACHE_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row["question"]
            answer_cache.setdefault(q, []).append(row["answer"])
else:
    # 如果没有文件，先创建一个空的带表头的 CSV
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

        for _ in range(needed):
            ans = generate_answer(question)
            if ans:
                writer.writerow({"question": question, "answer": ans})
                answer_cache.setdefault(question, []).append(ans)

    return answer_cache[question]


# ============================================================
# Step 8: Aggregation (supports ablation modes)  [TQA-style]
# ============================================================

def aggregate_signals(sc, judge_ok, entail, contra, mode="full"):

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
        final_score = (
            w_fact * fact_score +
            w_sc   * sc_score
        ) / (w_fact + w_sc)

    elif mode == "no_mnli":
        final_score = (
            w_judge * judge_score +
            w_sc    * sc_score
        ) / (w_judge + w_sc)

    elif mode == "no_selfcheck":
        final_score = (
            w_judge * judge_score +
            w_fact  * fact_score
        ) / (w_judge + w_fact)

    elif mode == "judge_only":
        final_score = judge_score

    else:
        raise ValueError("Unknown mode")

    return final_score >= 0.5


# ============================================================
# Step 9: One Monte-Carlo run on FIXED idxs (fair ablation)
# ============================================================
def check_exact_match(reference, model_answer):
    """
    Ground Truth Heuristic:
    如果是 NQ, 只要生成的答案包含标准答案的字符串, 就视为正确。
    """
    if not reference or not model_answer:
        return False
    return reference.lower() in model_answer.lower()

def run_once_fixed(idxs, mode="full"):
    """
    NQ hallucination detection (aligned with TQA / FEVER):
    gold_halluc = not judge_ok
    pred_halluc = not pred_ok
    """

    tp = fp = tn = fn = 0
    halluc = 0
    total = 0

    need_mnli  = mode in ["full", "no_judge", "no_selfcheck"]
    need_sc    = mode in ["full", "no_judge", "no_mnli"]
    need_judge= mode != "no_judge"

    for idx in tqdm(idxs, desc=f"Evaluating {mode}", leave=False):
        sample = subset[int(idx)]
        q = sample["question"]
        ref = sample["best_answer"]

        answers = get_answers(q)
        sc = selfcheck_group(answers) if need_sc else 0.0

        for ans in answers:
            # 1. 计算系统预测 (Prediction)
            judge_pred = judge_truth(q, ans, ref) if need_judge else False
            entail, contra = compute_factscore(ref, ans) if need_mnli else (0.0, 0.0)

            pred_ok = aggregate_signals(sc, judge_pred, entail, contra, mode)

            gt_is_correct = check_exact_match(ref, ans)

            total += 1
            if not pred_ok:
                halluc += 1

            gold_halluc = not gt_is_correct
            pred_halluc = not pred_ok

            if pred_halluc and gold_halluc:
                tp += 1
            elif pred_halluc and not gold_halluc:
                fp += 1
            elif not pred_halluc and not gold_halluc:
                tn += 1
            elif not pred_halluc and gold_halluc:
                fn += 1

    halluc_rate = halluc / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    return {
        "halluc_rate": halluc_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
# ============================================================
# Step 10: Ablation multi-run (mean ± std)  [TQA-style]
# ============================================================

def ablation_multi_run(modes, num_runs, sample_size, seed, dataset_name="NQ"):

    rng = np.random.RandomState(seed)
    summary = {m: [] for m in modes}

    print("\n" + "=" * 60)
    print(f"{dataset_name} Ablation Experiment | runs={num_runs}, sample={sample_size}, N_SAMPLES={N_SAMPLES}")
    print("=" * 60)

    for run_id in range(num_runs):
        print(f"\n=== Run {run_id+1}/{num_runs} ===")

        idxs = rng.choice(len(subset), size=sample_size, replace=False)

        for mode in modes:
            print(f"\n[{dataset_name}] Mode: {mode}")
            metrics = run_once_fixed(idxs, mode=mode)
            summary[mode].append(metrics)
            print(
                f"Halluc={metrics['halluc_rate']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}"
            )
            
    print("\n=========== Overall Summary (mean ± std) ===========")
    final = {
        "dataset": dataset_name,
        "num_runs": num_runs,
        "sample_size": sample_size,
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

    out_json = f"{dataset_name.lower()}_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    
    print(f"\nSaved results to: {out_json}")
    return final

def print_conclusion(results):
    metrics = []
    for m, data in results["metrics"].items():
        metrics.append({"mode": m, "mean": data["f1"]["mean"], "std": data["f1"]["std"]})
    
    metrics.sort(key=lambda x: x["mean"], reverse=True)
    best = metrics[0]
    full = next((x for x in metrics if x["mode"] == "full"), None)
    
    print("\n" + "="*60 + "\nFINAL CONCLUSION\n" + "="*60)
    print(f"{'Rank':<5} {'Mode':<15} {'F1 Mean':<10} {'F1 Std':<10}")
    for i, m in enumerate(metrics):
        print(f"{i+1:<5} {m['mode']:<15} {m['mean']:.4f}     {m['std']:.4f}")
    
    if full and best["mode"] == "full":
        print("\n SUCCESS: 'Full' model is the winner!")
# ============================================================
# Step 11: Run
# ============================================================

if __name__ == "__main__":

    modes = [
        "full",
        "no_judge",
        "no_mnli",
        "no_selfcheck",
        "judge_only"
    ]

    results=ablation_multi_run(
        modes=modes,
        num_runs=NUM_RUNS,
        sample_size=SAMPLE_SIZE,
        seed=SEED,
        dataset_name="NQ"
    )
    print_conclusion(results)