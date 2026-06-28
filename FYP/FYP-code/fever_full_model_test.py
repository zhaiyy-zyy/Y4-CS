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

ANSWER_CACHE_FILE = "fever_full_model_test_answer_cache.csv"

N_SAMPLES = 2  
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
        # renormalize to keep score in [0,1] scale comparable
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
# Step 9: One Monte-Carlo run on FIXED idxs (supports ablation + compute only needed signals)
# ============================================================

def run_once_fixed(idxs, mode="full"):
    """
    Hallucination detection on FEVER (aligned with TQA):
    gold_halluc = not judge_ok   (judge always used as oracle)
    pred_halluc = not pred_ok
    """

    tp = fp = tn = fn = 0
    halluc = 0
    total = 0

    need_mnli  = mode in ["full", "no_judge", "no_selfcheck"]
    need_sc    = mode in ["full", "no_judge", "no_mnli"]

    for idx in tqdm(idxs, desc=f"Evaluating {mode}", leave=False):
        item = subset[int(idx)]
        q = item["question"]
        evidence = item["best_answer"]
        gold_label = item["label"]

        answers = ensure_full_answers(q, evidence)
        sc = selfcheck_group(answers) if need_sc else 0.0

        for ans in answers:
            # judge ALWAYS computed (oracle)
            judge_ok = judge_truth(q, ans, gold_label, evidence)

            entail, contra = compute_factscore(evidence, ans) if need_mnli else (0.0, 0.0)

            # aggregation respects ablation
            pred_ok = aggregate_signals(
                sc=sc,
                judge_ok=judge_ok if mode != "no_judge" else False,
                entail=entail,
                contra=contra,
                mode=mode
            )

            total += 1
            if not pred_ok:
                halluc += 1

            pred_label = extract_fever_label(ans)
            gold_ok = (pred_label == gold_label)
            gold_halluc = not gold_ok
            pred_halluc = not pred_ok

            if pred_halluc and gold_halluc:
                tp += 1
            elif pred_halluc and not gold_halluc:
                fp += 1
            elif not pred_halluc and not gold_halluc:
                tn += 1
            elif not pred_halluc and gold_halluc:
                fn += 1

    halluc_rate = halluc / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "halluc_rate": halluc_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
# ============================================================
# Step 10: Ablation multi-run (mean ± std) with same idxs per run
# ============================================================

def ablation_multi_run(
    modes,
    num_runs,
    sample_size,
    seed,
    dataset_name="FEVER"
):
    """
    For each run:
      - sample ONE idxs set
      - evaluate ALL modes on same idxs
    Finally print mean ± std for each mode.
    """

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
            print(f"Halluc={metrics['halluc_rate']:.4f}, " 
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
   
# ============================================================
# Step 11: Automated Analysis & Conclusion
# ============================================================

def print_automated_conclusion(final_results):
    metrics = final_results["metrics"]
    summary = []
    for mode, data in metrics.items():
        summary.append({
            "mode": mode,
            "f1_mean": data["f1"]["mean"],
            "f1_std": data["f1"]["std"]
        })
    
    # 排序：准确性(降序), 稳定性(升序)
    ranked_by_acc = sorted(summary, key=lambda x: x["f1_mean"], reverse=True)
    ranked_by_stab = sorted(summary, key=lambda x: x["f1_std"], reverse=False)
    
    full_model = next((item for item in summary if item["mode"] == "full"), None)
    best_acc = ranked_by_acc[0]
    best_stab = ranked_by_stab[0]
    
    print("\n" + "="*60)
    print("                  AUTOMATED ANALYSIS REPORT                  ")
    print("="*60)
    print(f"{'Rank':<5} | {'Mode':<15} | {'F1 Mean (Accuracy)':<20} | {'F1 Std (Stability)':<20}")
    print("-" * 70)
    for i, item in enumerate(ranked_by_acc):
        mark = " (Best)" if i == 0 else ""
        print(f"{i+1:<5} | {item['mode']:<15} | {item['f1_mean']:.4f}{mark:<7} | {item['f1_std']:.4f}")
    print("-" * 70)
    
    if full_model:
        is_acc = (best_acc["mode"] == "full")
        is_stab = (best_stab["mode"] == "full")
        
        print(f"\n>>> FINAL VERDICT FOR 'FULL' MODEL:")
        print(f"[{'✅' if is_acc else '❌'}] Most Accurate?  {'YES' if is_acc else 'NO'} (Rank: {ranked_by_acc.index(full_model)+1})")
        print(f"[{'✅' if is_stab else '❌'}] Most Stable?    {'YES' if is_stab else 'NO'} (Rank: {ranked_by_stab.index(full_model)+1})")
        
        if is_acc and is_stab:
            print("\n SUCCESS: The Full Model is proven to be the most accurate AND stable!")
        else:
            print("\n RESULT: Full Model did not win both categories. Check weights or prompts.")
    print("="*60 + "\n")

# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    modes = [
        "full",
        "no_judge",
        "no_mnli",
        "no_selfcheck",
        "judge_only"
    ]

    results = ablation_multi_run(
        modes=modes,
        num_runs=NUM_RUNS,
        sample_size=SAMPLE_SIZE,
        seed=SEED,
        dataset_name="FEVER"
    )
    
    # 自动打印证明结果
    print_automated_conclusion(results)
