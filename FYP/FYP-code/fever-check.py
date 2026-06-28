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

PROGRESS_FILE = "fever_progress_multi_sample2.csv"
ANSWER_DETAIL_FILE = "fever_answer_details_sample2.csv"
FINAL_FILE = "fever_results_multi_sample2.csv"
CHECKPOINT = "fever_multi_run_checkpoint_sample2.json"

N_SAMPLES = 2  # number of sampled answers per claim (for SelfCheck)

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
    trust_remote_code=True,
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
You are a FEVER fact verification evaluator.

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


# ============================================================
# Step 6: Answer cache (for re-use across multiple runs)
# ============================================================

answer_cache = {}

if os.path.exists(ANSWER_DETAIL_FILE):
    print("Loading FEVER answer cache...")
    with open(ANSWER_DETAIL_FILE, "r", encoding="utf-8") as f:
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

    with open(ANSWER_DETAIL_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        if os.path.getsize(ANSWER_DETAIL_FILE) == 0:
            writer.writeheader()

        for _ in range(needed):
            ans = generate_answer(question, evidence)
            if ans:
                writer.writerow({"question": question, "answer": ans})
                f.flush()
                answer_cache.setdefault(question, []).append(ans)

    return answer_cache[question]


# ============================================================
# Step 8: Main experiment (multi-layer hallucination detection)
# ============================================================
def aggregate_signals(sc, judge_ok, entail, contra):
    # weights — same as TQA version
    w_judge = 0.45
    w_fact  = 0.35
    w_sc    = 0.20
    
    # normalize
    sc_score = 1 - sc
    fact_score = max(entail - contra, 0)
    judge_score = 1 if judge_ok else 0

    final_score = (
        w_judge * judge_score +
        w_fact  * fact_score +
        w_sc    * sc_score
    )

    final_ok = final_score >= 0.5
    return final_ok

def run_main_experiment():
    """
    Run the full FEVER experiment:
    - Ensure N_SAMPLES answers per claim
    - Compute SelfCheck, Judge, FactScore
    - Save per-answer records to CSV with resume support
    """
    print("\nRunning FEVER main experiment...\n")

    done_questions = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            done_questions = {row["question"] for row in csv.DictReader(f)}
        out = open(PROGRESS_FILE, "a", newline="", encoding="utf-8")
    else:
        out = open(PROGRESS_FILE, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(out, fieldnames=[
            "question", "answer", "reference", "label",
            "selfcheck", "judge", "entail", "contra", "final"
        ])
        writer.writeheader()
        out.flush()

    writer = csv.DictWriter(out, fieldnames=[
        "question", "answer", "reference", "label",
        "selfcheck", "judge", "entail", "contra", "final"
    ])

    for item in tqdm(subset):

        q = item["question"]
        evidence = item["best_answer"]
        label = item["label"]

        if q in done_questions:
            continue

        # A. generate or load multiple answers
        answers = ensure_full_answers(q, evidence)

        # B. SelfCheck - one score per question
        sc = selfcheck_group(answers)
       
        # C. Judge + NLI for each answer
        for ans in answers:
            judge_ok = judge_truth(q, ans, label, evidence)
            entail, contra = compute_factscore(evidence, ans)

            final_ok = aggregate_signals(sc, judge_ok, entail, contra)

            writer.writerow({
                "question": q,
                "answer": ans,
                "reference": evidence,
                "label": label,
                "selfcheck": sc,
                "judge": judge_ok,
                "entail": entail,
                "contra": contra,
                "final": final_ok
            })
            out.flush()

    out.close()
    os.replace(PROGRESS_FILE, FINAL_FILE)

    print("FEVER main experiment finished.")
    print(f"Results saved to: {FINAL_FILE}")


# ============================================================
# Step 9: Single Monte-Carlo run with random subsampling
# ============================================================

def run_once(sample_size=300):
    """
    Perform one Monte-Carlo run:
    - Randomly sample 'sample_size' claims
    - Use cached/generated answers
    - Compute hallucination rate over all answer instances
    """
    idxs = np.random.choice(len(subset), size=sample_size, replace=False)

    halluc = 0
    total = 0

    for idx in idxs:
        item = subset[idx]
        q = item["question"]
        evidence = item["best_answer"]
        label = item["label"]

        answers = ensure_full_answers(q, evidence)

        sc = selfcheck_group(answers)

        for ans in answers:
            total += 1
            judge_ok = judge_truth(q, ans, label, evidence)
            entail, contra = compute_factscore(evidence, ans)
            
            final_ok = aggregate_signals(sc, judge_ok, entail, contra)

            if not final_ok:
                halluc += 1

    return halluc / total if total > 0 else 0.0


# ============================================================
# Step 10: Multi-run Monte-Carlo estimation (mean ± std)
# ============================================================

def multi_run(num_runs=3, sample_size=300):
    """
    Run multiple Monte-Carlo trials with checkpointing.
    Returns (mean, std) hallucination rate.
    """
    if os.path.exists(CHECKPOINT):
        ckpt = json.load(open(CHECKPOINT, "r"))
    else:
        ckpt = {"num_runs": num_runs, "done": 0, "rates": []}

    for i in range(ckpt["done"], num_runs):
        print(f"\nRun {i + 1}/{num_runs}...")
        rate = run_once(sample_size)
        ckpt["rates"].append(rate)
        ckpt["done"] += 1
        json.dump(ckpt, open(CHECKPOINT, "w"), indent=2)
        print(f"Hallucination Rate = {rate:.4f}")

    mean = np.mean(ckpt["rates"])
    std = np.std(ckpt["rates"], ddof=1)

    print(f"\nFinal Hallucination Rate = {mean:.4f} ± {std:.4f}")
    os.remove(CHECKPOINT)

    return mean, std


# ============================================================
# Step 11: Entry point
# ============================================================

if __name__ == "__main__":
    run_main_experiment()
    multi_run()