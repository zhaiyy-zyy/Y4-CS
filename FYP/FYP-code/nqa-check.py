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
# Step 0: Basic Configuration
# ============================================================

PROGRESS_FILE = "nq_progress_multi_sample2.csv"
ANSWER_DETAIL_FILE = "nq_answer_details_sample2.csv"
FINAL_FILE = "nq_results_multi_sample2.csv"
CHECKPOINT = "nq_multi_run_checkpoint_sample2.json"

N_SAMPLES = 2

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

np.random.seed(42)
subset = list(np.random.choice(valid_items, size=min(500, len(valid_items)), replace=False))

print("Final subset size:", len(subset))


# ============================================================
# Step 2: LLM-as-a-Judge using Qwen2.5-7B-Instruct
# ============================================================

def judge_truth(question, model_answer, reference):
    prompt = f"""
You are a factuality evaluator.

A model answer should be marked True if:
- it is factually correct,
- and contains NO false or contradictory statements.

Extra correct information is allowed and should NOT cause False.

Mark False ONLY if the answer contains factual errors or contradictions.

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


def compute_factscore(evidence, answer):
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

if os.path.exists(ANSWER_DETAIL_FILE):
    with open(ANSWER_DETAIL_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row["question"]
            answer_cache.setdefault(q, []).append(row["answer"])


# ============================================================
# Step 7: Ensure each question has N sampled answers
# ============================================================

def ensure_full_answers(question):
    current = answer_cache.get(question, [])
    needed = N_SAMPLES - len(current)

    with open(ANSWER_DETAIL_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        if os.path.getsize(ANSWER_DETAIL_FILE) == 0:
            writer.writeheader()

        for _ in range(needed):
            ans = generate_answer(question)
            if ans:
                writer.writerow({"question": question, "answer": ans})
                answer_cache.setdefault(question, []).append(ans)

    return answer_cache[question]


# ============================================================
# Step 8: Main Experiment (SelfCheck + Judge + FactScore)
# ============================================================
def aggregate_signals(sc, judge_ok, entail, contra):

    # 与 TQA 完全相同的权重
    w_judge = 0.45
    w_fact  = 0.35
    w_sc    = 0.20

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

    print("\nRunning main experiment...\n")

    done_questions = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            done_questions = {row["question"] for row in csv.DictReader(f)}
        out = open(PROGRESS_FILE, "a", newline="", encoding="utf-8")
    else:
        out = open(PROGRESS_FILE, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(out, fieldnames=[
            "question", "answer", "reference",
            "selfcheck", "judge", "entail", "contra", "final"
        ])
        writer.writeheader()

    writer = csv.DictWriter(out, fieldnames=[
        "question", "answer", "reference",
        "selfcheck", "judge", "entail", "contra", "final"
    ])

    for sample in tqdm(subset):
        q = sample["question"]
        ref = sample["best_answer"]

        if q in done_questions:
            continue

        answers = ensure_full_answers(q)
        sc = selfcheck_group(answers)

        for ans in answers:
            judge_ok = judge_truth(q, ans, ref)
            entail, contra = compute_factscore(ref, ans)

            final_ok = aggregate_signals(
                sc, judge_ok, entail, contra
            )


            writer.writerow({
                "question": q,
                "answer": ans,
                "reference": ref,
                "selfcheck": sc,
                "judge": judge_ok,
                "entail": entail,
                "contra": contra,
                "final": final_ok
            })

        out.flush()

    out.close()
    os.replace(PROGRESS_FILE, FINAL_FILE)
    print("Main experiment finished:", FINAL_FILE)


# ============================================================
# Step 9: Random-sample Evaluation
# ============================================================

def run_once(sample_size=200):

    idxs = np.random.choice(len(subset), size=sample_size, replace=False)

    halluc = 0
    total = 0

    for idx in idxs:
        sample = subset[idx]
        q = sample["question"]
        ref = sample["best_answer"]

        answers = ensure_full_answers(q)
        sc = selfcheck_group(answers)
        
        for ans in answers:
            total += 1
            judge_ok = judge_truth(q, ans, ref)
            entail, contra = compute_factscore(ref, ans)
            final_ok = aggregate_signals(
                sc=sc,
                judge_ok=judge_ok,
                entail=entail,
                contra=contra
            )

            if not final_ok:
                halluc += 1

    return halluc / total if total > 0 else 0.0


# ============================================================
# Step 10: Multi-run Monte Carlo Estimation
# ============================================================

def multi_run(num_runs=3, sample_size=200):

    if os.path.exists(CHECKPOINT):
        ckpt = json.load(open(CHECKPOINT, "r"))
        if ckpt["num_runs"] != num_runs:
            ckpt = {"num_runs": num_runs, "done": 0, "rates": []}
    else:
        ckpt = {"num_runs": num_runs, "done": 0, "rates": []}

    for i in range(ckpt["done"], num_runs):
        print(f"Run {i+1}/{num_runs}...")
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
# Step 11: Run
# ============================================================

if __name__ == "__main__":
    run_main_experiment()
    multi_run()