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
# CONFIG
# ============================================================

SEED = 42
N_SAMPLES = 5

SUBSET_SIZE = 1000
POOL_SIZE = 800
VAL_RATIO = 0.2

NUM_RUNS = 3
SAMPLE_SIZE = 200

MITI_K = 5

GEN_MODEL = "llama2:7b"
JUDGE_MODEL = "qwen2.5:7b-instruct"

DETECTOR_NLI_MODEL = "roberta-large-mnli"

GOLD_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

W_JUDGE = 0.35
W_FACT = 0.45
W_SC = 0.20

ANSWER_CACHE = "f_answers_cache.csv"
CASES_FILE = "f_mitigation_cases.csv"
SUMMARY_JSON = "f_experiment_summary.json"

# ============================================================
# ENV
# ============================================================

os.environ["HF_HOME"] = "D:/huggingface"
ssl._create_default_https_context = ssl._create_unverified_context

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD DATA
# ============================================================

print("Loading FEVER dataset...")

dataset = load_dataset(
    "fever/fever",
    "v1.0"
)["labelled_dev"]

labels = [x["label"] for x in dataset]

idx = list(range(len(dataset)))

_, subset_idx = train_test_split(
    idx,
    test_size=SUBSET_SIZE,
    stratify=labels,
    random_state=SEED
)

dataset = dataset.select(subset_idx)


def extract_evidence(item):

    evid = item.get("evidence")

    if isinstance(evid, list) and evid:
        e = evid[0]

        if isinstance(e, list) and e and isinstance(e[0], (list, tuple)):
            e = e[0]

        if isinstance(e, (list, tuple)) and len(e) >= 3:
            return str(e[2])

    return ""


subset = []

for item in dataset:

    subset.append(
        {
            "question": item["claim"],
            "evidence": extract_evidence(item),
            "label": item["label"]
        }
    )

print("Subset size:", len(subset))

# ============================================================
# NLI MODELS
# ============================================================


def load_nli(model):

    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForSequenceClassification.from_pretrained(model).to(device)

    mdl.eval()

    return tok, mdl


print("Loading detector NLI...")
det_tok, det_nli = load_nli(DETECTOR_NLI_MODEL)

print("Loading gold NLI...")
gold_tok, gold_nli = load_nli(GOLD_NLI_MODEL)


@torch.no_grad()
def nli_score(tok, mdl, premise, hypothesis):

    x = tok(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    p = F.softmax(mdl(**x).logits, dim=-1)[0].cpu().numpy()

    ent = p[2]
    con = p[0]

    return ent, con


# ============================================================
# SELFCHECK
# ============================================================

print("Loading SelfCheckGPT...")

selfcheck = SelfCheckNLI(device=device)


def selfcheck_group(answers):

    if len(answers) < 2:
        return 0

    sents = []

    for a in answers:
        sents += [x.strip() for x in a.split(".") if x.strip()]

    scores = selfcheck.predict(
        sentences=sents,
        sampled_passages=answers
    )

    return float(np.mean(scores))


# ============================================================
# GENERATION
# ============================================================

def generate_answer(question, evidence):

    prompt = f"""
Claim: {question}
Evidence: {evidence}

Classify claim as:

SUPPORTED
REFUTED
NOT ENOUGH INFO

Then explain briefly.
"""

    r = ollama.chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.8}
    )

    return r["message"]["content"]


# ============================================================
# JUDGE
# ============================================================

def judge(question, answer, label, evidence):

    prompt = f"""
Claim: {question}
Evidence: {evidence}
Reference Label: {label}
Model Answer: {answer}

Is the model answer correct?

Answer True or False.
"""

    try:

        r = ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        return r["message"]["content"].lower().startswith("true")

    except:
        return False


# ============================================================
# SCORE
# ============================================================

def compute_score(question, evidence, answer, label):

    judge_ok = judge(question, answer, label, evidence)

    ent, con = nli_score(det_tok, det_nli, evidence, answer)

    sc = 0

    judge_score = 1 if judge_ok else 0
    fact_score = max(ent - con, 0)
    sc_score = 1 - sc

    score = W_JUDGE * judge_score + W_FACT * fact_score + W_SC * sc_score

    return float(score)


# ============================================================
# REWRITE
# ============================================================

def rewrite(question, evidence, prev, temp):

    prompt = f"""
Claim: {question}
Evidence: {evidence}

Correct the previous answer if needed.

Previous answer:
{prev}
"""

    r = ollama.chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temp}
    )

    return r["message"]["content"]


# ============================================================
# MITIGATION
# ============================================================

def mitigate(question, evidence, label, base, t_low, t_high):

    base_score = compute_score(question, evidence, base, label)

    if base_score >= t_high:
        return base, "keep"

    candidates = []

    for _ in range(MITI_K):
        candidates.append(rewrite(question, evidence, base, 0.2))

    sc = selfcheck_group(candidates)

    best = base
    best_score = base_score

    for c in candidates:

        s = compute_score(question, evidence, c, label)

        if s > best_score:
            best = c
            best_score = s

    if best_score < t_low:

        nei = "NOT ENOUGH INFO"

        return nei, "abstain"

    return best, "rewrite"


# ============================================================
# EXPERIMENT
# ============================================================

print("Running experiment...")

cases = []

for item in tqdm(subset[:200]):

    q = item["question"]
    ev = item["evidence"]
    label = item["label"]

    base = generate_answer(q, ev)

    final, action = mitigate(
        q,
        ev,
        label,
        base,
        0.18,
        0.23
    )

    cases.append(
        {
            "question": q,
            "base_answer": base,
            "final_answer": final,
            "action": action
        }
    )

with open(CASES_FILE, "w", newline="", encoding="utf-8") as f:

    w = csv.DictWriter(
        f,
        fieldnames=["question", "base_answer", "final_answer", "action"]
    )

    w.writeheader()
    w.writerows(cases)

print("Cases saved:", CASES_FILE)

summary = {
    "dataset": "FEVER",
    "samples": len(cases),
    "mitigation": "detector conditioned rewrite"
}

with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print("Done.")