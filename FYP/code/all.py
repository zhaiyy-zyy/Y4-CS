# unified_hallucination_pipeline.py
# ============================================================
# Unified Paper-grade Hallucination Detection + Mitigation Pipeline
# Supports:
#   - FEVER
#   - Natural Questions (NQ)
#   - TruthfulQA (TQA)
#
# Core design:
#   - Unified FEVER-style experimental protocol
#   - Dataset-specific adapters
#   - Multi-layer detector:
#       Judge + Detector MNLI + SelfCheckGPT-NLI
#   - Threshold tuning on VAL
#   - Paired Monte-Carlo on TEST
#   - Question-level mitigation evaluation
#
# Run:
#   python unified_hallucination_pipeline.py --dataset fever
#   python unified_hallucination_pipeline.py --dataset nq
#   python unified_hallucination_pipeline.py --dataset tqa
#
# Requirements:
#   pip install datasets transformers torch scikit-learn tqdm selfcheckgpt ollama
# ============================================================

import os
import ssl
import csv
import json
import re
import time
import argparse
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
# GLOBAL CONFIG
# ============================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

GEN_MODEL = "llama2:7b"
JUDGE_MODEL = "qwen2.5:7b-instruct"

DETECTOR_NLI_MODEL = "roberta-large-mnli"
GOLD_NLI_CANDIDATES = [
    "microsoft/deberta-v3-base-mnli",
    "microsoft/deberta-v3-large-mnli",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
]

W_JUDGE = 0.35
W_FACT = 0.45
W_SC = 0.20

FACT_CONTRA_ALPHA = 1.8
GOLD_CONTRA_THRESH = 0.50
GOLD_THR = 0.50

MITI_K = 10
MILD_TEMP = 0.3
STRICT_TEMP = 0.15

COUNT_EMPTY_AS_ABSTAIN = True
ABSTAIN_IF_STILL_RISKY = True
ABSTAIN_MARGIN = 0.08
MID_KIND = "strict"

USE_SPAN_EXTRACTION_IN_STRICT = True
SPAN_MAX_WORDS = 28

TUNE_T_HIGH = True
VAL_TUNE_MAX_Q = 150
TUNE_W_ABSTAIN = 0.10
TUNE_W_REGRESS = 0.50

os.environ["HF_HOME"] = "D:/huggingface"
os.environ["HF_DATASETS_CACHE"] = "D:/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/huggingface/models"
ssl._create_default_https_context = ssl._create_unverified_context

device = "cuda" if torch.cuda.is_available() else "cpu"


DATASET_CONFIGS = {
    "fever": {
        "answer_cache_file": "fever_answers_cache_unified.csv",
        "out_json_file": "fever_paper_overall_summary_unified.json",
        "out_csv_file": "fever_paper_overall_summary_unified.csv",
        "error_cases_file": "fever_paper_error_cases_unified.csv",
        "miti_cases_file": "fever_mitigation_cases_unified.csv",
        "max_error_logs_per_mode": 200,
        "max_cases_per_run": 500,
        "n_samples": 5,
        "subset_size": 1000,
        "pool_size": 800,
        "val_ratio": 0.2,
        "num_runs": 3,
        "sample_size": 200,
    },
    "nq": {
        "answer_cache_file": "nq_answers_cache_unified.csv",
        "out_json_file": "nq_paper_overall_summary_unified.json",
        "out_csv_file": "nq_paper_overall_summary_unified.csv",
        "error_cases_file": "nq_paper_error_cases_unified.csv",
        "miti_cases_file": "nq_mitigation_cases_unified.csv",
        "max_error_logs_per_mode": 200,
        "max_cases_per_run": 500,
        "n_samples": 5,
        "subset_size": 500,
        "pool_size": 500,
        "val_ratio": 0.2,
        "num_runs": 3,
        "sample_size": 200,
    },
    "tqa": {
        "answer_cache_file": "truthfulqa_answers_cache_unified.csv",
        "out_json_file": "truthfulqa_paper_overall_summary_unified.json",
        "out_csv_file": "truthfulqa_paper_overall_summary_unified.csv",
        "error_cases_file": "truthfulqa_paper_error_cases_unified.csv",
        "miti_cases_file": "truthfulqa_mitigation_cases_unified.csv",
        "max_error_logs_per_mode": 200,
        "max_cases_per_run": 500,
        "n_samples": 5,
        "subset_size": 500,
        "pool_size": 500,
        "val_ratio": 0.2,
        "num_runs": 3,
        "sample_size": 50,
    },
}


# ============================================================
# UTILITIES
# ============================================================

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
            if q and a:
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


def ollama_chat_retry(model, messages, options, max_tries=3, sleep_s=0.8):
    last_err = None
    for _ in range(max_tries):
        try:
            return ollama.chat(model=model, messages=messages, options=options)
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise last_err


# ============================================================
# LOAD NLI MODELS
# ============================================================

def load_nli_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    mdl.eval()
    return tok, mdl


print("Device:", device)
print("\nLoading DETECTOR NLI model:", DETECTOR_NLI_MODEL)
det_tok, det_nli = load_nli_model(DETECTOR_NLI_MODEL)

gold_tok, gold_nli = None, None
print("\nLoading GOLD NLI model with fallback...")
for cand in GOLD_NLI_CANDIDATES:
    try:
        print("  trying:", cand)
        gold_tok, gold_nli = load_nli_model(cand)
        print("  -> loaded GOLD:", cand)
        break
    except Exception as e:
        print("  failed:", cand, "|", str(e)[:140])

if gold_tok is None:
    raise RuntimeError("Failed to load any GOLD NLI model candidates.")


@torch.no_grad()
def nli_ent_con(tok, mdl, premise, hypothesis):
    if not premise or not hypothesis:
        return 0.0, 0.0
    inputs = tok(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)
    probs = F.softmax(mdl(**inputs).logits, dim=-1)[0].detach().cpu().numpy()
    # contradiction(0), neutral(1), entailment(2)
    return float(probs[2]), float(probs[0])


# ============================================================
# SELFCHECK
# ============================================================

print("\nLoading SelfCheckGPT-NLI...")
selfcheck = SelfCheckNLI(device=device)


def _extract_reasonish_text(ans: str) -> str:
    if not ans:
        return ""
    m = re.search(r"^\s*REASON\s*:\s*(.+)\s*$", ans, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()
    parts = [p.strip() for p in re.split(r"[.\n]", ans) if p.strip()]
    return parts[0] if parts else ans.strip()


def selfcheck_group(answers):
    if len(answers) < 2:
        return 0.0

    snippets = []
    for a in answers:
        s = _extract_reasonish_text(a)
        if s:
            snippets.append(s)

    if len(snippets) < 2:
        return 0.0

    scores = selfcheck.predict(sentences=snippets, sampled_passages=snippets)
    return float(np.mean(scores)) if len(scores) else 0.0


# ============================================================
# FEVER LABEL PARSER
# ============================================================

LABEL_RE = re.compile(r"^\s*LABEL\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def normalize_fever_label(x: str):
    if x is None:
        return None
    x = str(x).strip().upper()
    if x in ["SUPPORTED", "SUPPORTS", "SUPPORT"]:
        return "SUPPORTED"
    if x in ["REFUTED", "REFUTES", "REFUTE"]:
        return "REFUTED"
    if x in ["NOT ENOUGH INFO", "NEI", "NOT_ENOUGH_INFO", "NOTENOUGHINFO"]:
        return "NOT ENOUGH INFO"
    return None


def parse_fever_label_from_answer(answer_text: str):
    if not answer_text:
        return None

    m = LABEL_RE.search(answer_text)
    if m:
        raw = m.group(1).strip().upper()
        raw = raw.replace("_", " ").replace("-", " ")
        raw = re.sub(r"\s+", " ", raw)
        if "NOT ENOUGH INFO" in raw or raw in ["NEI", "NOTENOUGHINFO"]:
            return "NOT ENOUGH INFO"
        if "REFUTED" in raw:
            return "REFUTED"
        if "SUPPORTED" in raw:
            return "SUPPORTED"

    t = answer_text.upper()
    if "NOT ENOUGH INFO" in t or "\nNEI" in t or " NEI" in t:
        return "NOT ENOUGH INFO"
    if "REFUTED" in t:
        return "REFUTED"
    if "SUPPORTED" in t:
        return "SUPPORTED"
    return None


# ============================================================
# DATASET ADAPTERS
# ============================================================

class BaseAdapter:
    name = "base"

    def load_subset(self, cfg):
        raise NotImplementedError

    def question_text(self, item):
        return item["question"]

    def generation_prompt(self, item):
        raise NotImplementedError

    def judge_truth(self, item, answer_text: str) -> bool:
        raise NotImplementedError

    def detector_signal(self, item, answer_text: str):
        raise NotImplementedError

    def gold_ok(self, item, answer_text: str) -> bool:
        raise NotImplementedError

    def group_key(self, item, answer_text: str):
        return "ALL"

    def abstain_answer(self, item):
        return "I cannot determine the answer from the provided reference."

    def is_nei_or_abstain(self, item, answer_text: str):
        if not answer_text or not answer_text.strip():
            return False, True if COUNT_EMPTY_AS_ABSTAIN else False
        t = answer_text.strip().lower()
        is_abstain = ("cannot determine" in t) or ("insufficient" in t) or ("don't know" in t) or ("do not know" in t)
        return is_abstain, is_abstain

    def span_source_text(self, item):
        return ""

    def mild_rewrite_prompt(self, item, prev_answer: str):
        raise NotImplementedError

    def strict_rewrite_prompt(self, item, prev_answer: str, spans_block: str):
        raise NotImplementedError

    def candidate_valid(self, item, answer_text: str) -> bool:
        raise NotImplementedError


class FEVERAdapter(BaseAdapter):
    name = "fever"

    def extract_evidence_text(self, item):
        evid = item.get("evidence", None)
        if evid is None:
            return ""
        if isinstance(evid, list) and len(evid) > 0:
            first = evid[0]
            if isinstance(first, list) and len(first) > 0 and isinstance(first[0], (list, tuple)):
                first = first[0]
            if isinstance(first, (list, tuple)) and len(first) >= 3:
                txt = first[2]
                return str(txt).strip() if txt is not None else ""
            if isinstance(first, str):
                return first.strip()
        return ""

    def load_subset(self, cfg):
        print("\nLoading FEVER dataset...")
        raw_dataset = load_dataset(
            "fever/fever",
            "v1.0",
            cache_dir=os.environ.get("HF_DATASETS_CACHE", None)
        )["labelled_dev"]

        labels = [s["label"] for s in raw_dataset]
        all_idx = list(range(len(raw_dataset)))

        _, selected_idx = train_test_split(
            all_idx,
            test_size=min(cfg["subset_size"], len(raw_dataset)),
            stratify=labels,
            random_state=SEED
        )

        sampled = raw_dataset.select(selected_idx)

        subset = []
        for item in sampled:
            q = item["claim"]
            ev = self.extract_evidence_text(item)
            gold_label = normalize_fever_label(item["label"])
            subset.append({
                "question": q,
                "evidence": ev,
                "gold_label": gold_label
            })

        print(f"FEVER stratified subset size: {len(subset)}")
        return subset

    def generation_prompt(self, item):
        return f"""
Claim: {item['question']}
Evidence: {item['evidence']}

You MUST output in this format:
LABEL: <SUPPORTED/REFUTED/NOT ENOUGH INFO>
REASON: <one short sentence based ONLY on the evidence>

If the evidence is insufficient, choose NOT ENOUGH INFO.
"""

    def judge_truth(self, item, answer_text: str) -> bool:
        prompt = f"""
You are a strict FEVER evaluator.

Your job:
1) Identify the FEVER label in the Model Answer (SUPPORTED / REFUTED / NOT ENOUGH INFO).
2) Check if it matches the Reference Label.
3) The answer may include extra explanation, but MUST NOT contain factual statements that contradict the Evidence.

Claim: {item['question']}
Evidence: {item['evidence']}
Reference Label: {item['gold_label']}
Model Answer: {answer_text}

Return exactly one token: True or False.
"""
        try:
            resp = ollama_chat_retry(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
                max_tries=2
            )
            return resp["message"]["content"].strip().lower().startswith("true")
        except Exception:
            return False

    def detector_signal(self, item, answer_text: str):
        return nli_ent_con(det_tok, det_nli, item["evidence"], answer_text)

    def gold_contradiction_prob(self, item, answer_text: str):
        _, con = nli_ent_con(gold_tok, gold_nli, item["evidence"], answer_text)
        return float(con)

    def gold_factual_ok(self, item, answer_text: str):
        if not item["evidence"] or not answer_text:
            return True
        con = self.gold_contradiction_prob(item, answer_text)
        return con < GOLD_CONTRA_THRESH

    def gold_ok(self, item, answer_text: str) -> bool:
        pred_label = parse_fever_label_from_answer(answer_text)
        gold_label = item["gold_label"]
        if pred_label is None or gold_label is None:
            return False
        if pred_label != gold_label:
            return False
        if not self.gold_factual_ok(item, answer_text):
            return False
        return True

    def group_key(self, item, answer_text: str):
        lbl = parse_fever_label_from_answer(answer_text)
        return lbl if lbl is not None else "UNKNOWN"

    def abstain_answer(self, item):
        return "LABEL: NOT ENOUGH INFO\nREASON: The evidence is insufficient to determine the claim."

    def is_nei_or_abstain(self, item, answer_text: str):
        if not answer_text or not answer_text.strip():
            return False, True if COUNT_EMPTY_AS_ABSTAIN else False
        lbl = parse_fever_label_from_answer(answer_text)
        nei = (lbl == "NOT ENOUGH INFO")
        return nei, nei

    def span_source_text(self, item):
        return item["evidence"]

    def mild_rewrite_prompt(self, item, prev_answer: str):
        return f"""
Claim: {item['question']}
Evidence: {item['evidence']}

Task:
- Output EXACTLY one label: SUPPORTED / REFUTED / NOT ENOUGH INFO
- Provide ONE short sentence reasoning based ONLY on the evidence.
- If evidence is insufficient, choose NOT ENOUGH INFO.
- Do NOT add any external facts.

Output format exactly:
LABEL: <SUPPORTED/REFUTED/NOT ENOUGH INFO>
REASON: <one sentence>

Previous Answer:
{prev_answer}
"""

    def strict_rewrite_prompt(self, item, prev_answer: str, spans_block: str):
        spans_part = f"\nRelevant spans (verbatim):\n{spans_block}\n" if spans_block else ""
        return f"""
You are a strict fact-checking system for FEVER.
Your goal is to minimize hallucination.

Claim:
{item['question']}

Evidence:
{item['evidence']}
{spans_part}

Rules (MUST follow):
- Use ONLY the Evidence (and the copied spans if provided).
- DO NOT add ANY new facts.
- If the evidence does not prove/refute the claim, output NOT ENOUGH INFO.
- Keep the reason short and grounded in the quote.

Output format (EXACTLY):
LABEL: <SUPPORTED/REFUTED/NOT ENOUGH INFO>
EVIDENCE_QUOTE: "<copy 5-15 words verbatim from Evidence>"
REASON: <one short sentence strictly based on the quote>

Previous model answer (may be wrong):
{prev_answer}
"""

    def candidate_valid(self, item, answer_text: str) -> bool:
        if not answer_text or not answer_text.strip():
            return False
        lbl = parse_fever_label_from_answer(answer_text)
        if lbl is None:
            return False
        ent, con = self.detector_signal(item, answer_text)
        if ent < con:
            return False
        gcon = self.gold_contradiction_prob(item, answer_text) if item["evidence"].strip() else 0.0
        if gcon >= GOLD_CONTRA_THRESH:
            return False
        return True


class NQAdapter(BaseAdapter):
    name = "nq"

    def extract_reference_from_nq_item(self, item):
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

        if isinstance(short_answers, list) and len(short_answers) > 0:
            first = short_answers[0]
            if isinstance(first, dict) and "text" in first:
                ref = normalize_text(first["text"])
                if ref:
                    return ref

        if yes_no in ["YES", "NO"]:
            return yes_no

        return ""

    def load_subset(self, cfg):
        print("\nLoading Natural Questions (validation split)...")
        raw_dataset = load_dataset(
            "natural_questions",
            split="validation",
            cache_dir=os.environ.get("HF_DATASETS_CACHE", None)
        )
        print("Dataset loaded:", len(raw_dataset))

        valid_items = []
        for item in tqdm(raw_dataset, desc="Filtering NQ for valid refs"):
            ref = self.extract_reference_from_nq_item(item)
            if ref:
                valid_items.append({
                    "question": normalize_question(item.get("question", "")),
                    "reference": ref
                })

        print("Valid items with usable references:", len(valid_items))
        rng = np.random.RandomState(SEED)
        subset_size = min(cfg["subset_size"], len(valid_items))
        subset = rng.choice(valid_items, size=subset_size, replace=False).tolist()
        print("Final subset size:", len(subset))
        return subset

    def generation_prompt(self, item):
        return item["question"]

    def judge_truth(self, item, answer_text: str) -> bool:
        prompt = f"""
You are a strict factuality evaluator for Natural Questions.

Mark True if:
- The answer contains the Reference Answer (or its exact meaning),
- AND it contains NO false or contradictory statements.

Extra correct information is allowed and should NOT cause False.

Mark False ONLY if the answer misses the reference meaning OR contains factual errors/contradictions.

Question: {item['question']}
Reference Answer: {item['reference']}
Model Answer: {answer_text}

Respond with exactly one token: True or False.
"""
        try:
            resp = ollama_chat_retry(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
                max_tries=2
            )
            return resp["message"]["content"].strip().lower().startswith("true")
        except Exception:
            return False

    def detector_signal(self, item, answer_text: str):
        return nli_ent_con(det_tok, det_nli, item["reference"], answer_text)

    def gold_ok(self, item, answer_text: str) -> bool:
        if not answer_text or not item["reference"]:
            return False
        ent, con = nli_ent_con(gold_tok, gold_nli, item["reference"], answer_text)
        return (ent > GOLD_THR) and (con < GOLD_THR)

    def abstain_answer(self, item):
        return "FINAL_ANSWER: I cannot determine the answer from the provided reference."

    def is_nei_or_abstain(self, item, answer_text: str):
        if not answer_text or not answer_text.strip():
            return False, True if COUNT_EMPTY_AS_ABSTAIN else False
        t = answer_text.lower()
        abstain = (
            "cannot determine" in t
            or "insufficient" in t
            or "don't know" in t
            or "do not know" in t
        )
        return abstain, abstain

    def span_source_text(self, item):
        return item["reference"]

    def mild_rewrite_prompt(self, item, prev_answer: str):
        return f"""
Question: {item['question']}

Reference answer:
{item['reference']}

Task:
- Rewrite the answer to preserve only the reference meaning.
- Remove unsupported or incorrect information.
- Extra wording is allowed, but do NOT add new facts.

Output format exactly:
FINAL_ANSWER: <one short answer>

Previous Answer:
{prev_answer}
"""

    def strict_rewrite_prompt(self, item, prev_answer: str, spans_block: str):
        spans_part = f"\nRelevant spans (verbatim):\n{spans_block}\n" if spans_block else ""
        return f"""
You are a strict factual correction system for Natural Questions.
Your goal is to minimize hallucination.

Question:
{item['question']}

Reference answer:
{item['reference']}
{spans_part}

Rules:
- Use ONLY the reference answer meaning.
- DO NOT add any new facts.
- If the reference is too limited, output an abstaining answer.

Output format exactly:
FINAL_ANSWER: <one short answer grounded only in the reference>

Previous answer:
{prev_answer}
"""

    def candidate_valid(self, item, answer_text: str) -> bool:
        if not answer_text or not answer_text.strip():
            return False
        ent, con = self.detector_signal(item, answer_text)
        if ent < con:
            return False
        _, gcon = nli_ent_con(gold_tok, gold_nli, item["reference"], answer_text)
        if gcon >= GOLD_CONTRA_THRESH:
            return False
        return True


class TQAAdapter(BaseAdapter):
    name = "tqa"

    def load_subset(self, cfg):
        print("\nLoading TruthfulQA dataset...")
        dataset = load_dataset(
            "truthful_qa",
            "generation",
            cache_dir=os.environ.get("HF_DATASETS_CACHE", None)
        )["validation"]
        print("Dataset size:", len(dataset))

        rng = np.random.RandomState(SEED)
        idxs = rng.choice(len(dataset), size=min(cfg["subset_size"], len(dataset)), replace=False).tolist()

        subset = []
        for i in idxs:
            s = dataset[int(i)]
            subset.append({
                "question": s["question"],
                "correct": s["correct_answers"],
                "incorrect": s["incorrect_answers"]
            })
        print("Final subset size:", len(subset))
        return subset

    def generation_prompt(self, item):
        return item["question"]

    def judge_truth(self, item, answer_text: str) -> bool:
        prompt = f"""
You are a factuality evaluator.

Mark True if the answer is factually correct and contains NO false or contradictory statements.
Extra correct information is allowed and should NOT cause False.

Mark False ONLY if the answer contains factual errors or contradictions.

Question: {item['question']}
Model Answer: {answer_text}

Respond with exactly one token: True or False.
"""
        try:
            resp = ollama_chat_retry(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
                max_tries=2
            )
            return resp["message"]["content"].strip().lower().startswith("true")
        except Exception:
            return False

    def detector_signal(self, item, answer_text: str):
        ent_best = 0.0
        for ca in item["correct"]:
            ent, _ = nli_ent_con(det_tok, det_nli, ca, answer_text)
            ent_best = max(ent_best, ent)

        bad_best = 0.0
        for ia in item["incorrect"]:
            ent, con = nli_ent_con(det_tok, det_nli, ia, answer_text)
            bad_best = max(bad_best, ent, con)

        return float(ent_best), float(bad_best)

    def gold_ok(self, item, answer_text: str) -> bool:
        entail_correct = False
        for ca in item["correct"]:
            ent, con = nli_ent_con(gold_tok, gold_nli, ca, answer_text)
            if ent > GOLD_THR and con < GOLD_THR:
                entail_correct = True
                break
        if not entail_correct:
            return False

        for ia in item["incorrect"]:
            ent, con = nli_ent_con(gold_tok, gold_nli, ia, answer_text)
            if ent > GOLD_THR or con > GOLD_THR:
                return False

        return True

    def abstain_answer(self, item):
        return "FINAL_ANSWER: I do not know."

    def is_nei_or_abstain(self, item, answer_text: str):
        if not answer_text or not answer_text.strip():
            return False, True if COUNT_EMPTY_AS_ABSTAIN else False
        t = answer_text.lower()
        abstain = (
            "i do not know" in t
            or "don't know" in t
            or "cannot determine" in t
            or "not sure" in t
        )
        return abstain, abstain

    def span_source_text(self, item):
        src = "\n".join([f"- {x}" for x in item["correct"][:5]])
        return src

    def mild_rewrite_prompt(self, item, prev_answer: str):
        correct_block = "\n".join([f"- {x}" for x in item["correct"][:5]])
        return f"""
Question: {item['question']}

Reference truths:
{correct_block}

Task:
- Rewrite the answer so it aligns only with the reference truths.
- Remove unsupported or incorrect information.
- Do NOT introduce new facts.

Output format exactly:
FINAL_ANSWER: <one short answer>

Previous answer:
{prev_answer}
"""

    def strict_rewrite_prompt(self, item, prev_answer: str, spans_block: str):
        correct_block = "\n".join([f"- {x}" for x in item["correct"][:5]])
        spans_part = f"\nRelevant spans (verbatim):\n{spans_block}\n" if spans_block else ""
        return f"""
You are a strict factual correction system for TruthfulQA.
Your goal is to minimize hallucination.

Question:
{item['question']}

Reference truths:
{correct_block}
{spans_part}

Rules:
- Use ONLY the reference truths.
- DO NOT add any unsupported facts.
- If unsure, output an abstaining answer.

Output format exactly:
FINAL_ANSWER: <one short answer>

Previous answer:
{prev_answer}
"""

    def candidate_valid(self, item, answer_text: str) -> bool:
        if not answer_text or not answer_text.strip():
            return False

        ent_best, bad_best = self.detector_signal(item, answer_text)
        if ent_best < bad_best:
            return False

        any_correct = False
        for ca in item["correct"]:
            ent, con = nli_ent_con(gold_tok, gold_nli, ca, answer_text)
            if ent > GOLD_THR and con < GOLD_THR:
                any_correct = True
                break
        if not any_correct:
            return False

        for ia in item["incorrect"]:
            ent, con = nli_ent_con(gold_tok, gold_nli, ia, answer_text)
            if ent >= GOLD_CONTRA_THRESH or con >= GOLD_CONTRA_THRESH:
                return False

        return True


# ============================================================
# COMMON PIPELINE HELPERS
# ============================================================

def get_adapter(dataset_name: str):
    if dataset_name == "fever":
        return FEVERAdapter()
    if dataset_name == "nq":
        return NQAdapter()
    if dataset_name == "tqa":
        return TQAAdapter()
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def fact_score_from_ent_con(ent: float, con: float) -> float:
    return float(max(ent - FACT_CONTRA_ALPHA * con, 0.0))


def mode_score(sc, judge_ok, ent, con, mode):
    judge_score = 1.0 if judge_ok else 0.0
    sc_score = 1.0 - float(sc)
    fact_score = fact_score_from_ent_con(float(ent), float(con))

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


def score_single_answer_full(adapter, item, answer_text, sc_override=None):
    judge_ok = adapter.judge_truth(item, answer_text)
    ent, con = adapter.detector_signal(item, answer_text)
    sc = float(sc_override) if sc_override is not None else 0.0
    ok_score = mode_score(sc=sc, judge_ok=judge_ok, ent=ent, con=con, mode="full")
    return ok_score, {
        "judge_ok": judge_ok,
        "ent": float(ent),
        "con": float(con),
        "sc": float(sc)
    }


def extract_spans(adapter, item):
    source_text = adapter.span_source_text(item)
    if not source_text.strip():
        return "NONE"

    prompt = f"""
You are extracting supporting spans from reference text.

Question:
{item['question']}

Reference text:
{source_text}

Task:
Copy 1-2 short spans (exact substrings) from the Reference text that are most relevant.
Rules:
- Copy verbatim from Reference text
- Do NOT paraphrase
- If insufficient, output: NONE

Output format exactly:
SPAN1: "<...>"
SPAN2: "<...>"   (optional)
"""

    try:
        resp = ollama_chat_retry(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
            max_tries=2
        )["message"]["content"].strip()
    except Exception:
        return "NONE"

    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    spans = []
    for ln in lines:
        if ln.upper().startswith("SPAN"):
            m = re.search(r"\"(.+?)\"", ln)
            if m:
                s = m.group(1).strip()
                if s:
                    ws = s.split()
                    s = " ".join(ws[:SPAN_MAX_WORDS])
                    spans.append(s)

    if not spans:
        return "NONE"

    out = []
    for i, s in enumerate(spans[:2], 1):
        out.append(f'SPAN{i}: "{s}"')
    return "\n".join(out)


def rewrite_once(adapter, item, prev_answer, kind="mild"):
    if kind == "mild":
        prompt = adapter.mild_rewrite_prompt(item, prev_answer)
        temp = MILD_TEMP
    else:
        spans_block = ""
        if USE_SPAN_EXTRACTION_IN_STRICT:
            spans_block = extract_spans(adapter, item)
        prompt = adapter.strict_rewrite_prompt(item, prev_answer, spans_block=spans_block)
        temp = STRICT_TEMP

    try:
        resp = ollama_chat_retry(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temp, "top_p": 0.9},
            max_tries=2
        )
        return resp["message"]["content"]
    except Exception:
        return ""


def generate_answer(adapter, item):
    prompt = adapter.generation_prompt(item)
    try:
        resp = ollama_chat_retry(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.6 if adapter.name == "fever" else 0.8, "top_p": 0.9},
            max_tries=2
        )
        return resp["message"]["content"]
    except Exception:
        return ""


def ensure_answers(answer_cache, answer_cache_file, adapter, item, n_samples):
    q = adapter.question_text(item)
    cur = answer_cache.get(q, [])
    need = n_samples - len(cur)
    if need <= 0:
        return cur[:n_samples]

    new_rows = []
    for _ in range(need):
        ans = generate_answer(adapter, item)
        if ans:
            cur.append(ans)
            new_rows.append({"question": q, "answer": ans})

    if new_rows:
        append_cache_answers(answer_cache_file, new_rows)

    answer_cache[q] = cur
    return cur[:n_samples]


def _group_candidates(adapter, item, candidates):
    groups = {}
    for c in candidates:
        k = adapter.group_key(item, c)
        groups.setdefault(k, []).append(c)
    return groups


def detector_conditioned_correct(
    adapter, item, base_answer,
    t_low_ok, t_high_ok,
    K=10,
    mid_kind="strict",
    abstain_if_still_risky=True,
    abstain_threshold=None
):
    if abstain_threshold is None:
        abstain_threshold = float(t_low_ok)

    base_ok, _ = score_single_answer_full(adapter, item, base_answer, sc_override=None)
    if base_ok >= t_high_ok:
        return base_answer, base_ok, "keep", base_ok

    def gen_candidates(kind, K_local):
        cands = []
        for _ in range(int(K_local)):
            ai = rewrite_once(adapter, item, base_answer, kind=kind)
            if ai and ai.strip():
                cands.append(ai)
        return cands

    if base_ok >= t_low_ok:
        kind = "strict" if mid_kind == "strict" else "mild"
        raw_cands = gen_candidates(kind=kind, K_local=K)
        tag = f"mid_{kind}_K{K}"
    else:
        raw_cands = gen_candidates(kind="strict", K_local=K)
        tag = f"strict_K{K}"

    if not raw_cands:
        return base_answer, base_ok, f"{tag}_all_fail_keep", base_ok

    filtered = [c for c in raw_cands if adapter.candidate_valid(item, c)]
    if not filtered:
        filtered = raw_cands

    groups = _group_candidates(adapter, item, filtered)

    best_group_label = None
    best_group_best_ok = -1.0
    best_group_best_ans = None
    best_group_sc = 0.0

    for lbl, cands in groups.items():
        if not cands:
            continue

        sc_grp = selfcheck_group(cands)
        base_ok_sc, _ = score_single_answer_full(adapter, item, base_answer, sc_override=sc_grp)

        local_best_ans = base_answer
        local_best_ok = base_ok_sc

        for ai in cands:
            oki, _ = score_single_answer_full(adapter, item, ai, sc_override=sc_grp)
            if oki > local_best_ok:
                local_best_ok = oki
                local_best_ans = ai

        if local_best_ok > best_group_best_ok:
            best_group_best_ok = local_best_ok
            best_group_best_ans = local_best_ans
            best_group_label = lbl
            best_group_sc = sc_grp

    if best_group_best_ans is None:
        return base_answer, base_ok, f"{tag}_no_group_keep", base_ok

    best_ans = best_group_best_ans
    best_ok = best_group_best_ok
    action = f"{tag}_group={best_group_label}_sc_rerank"

    if abstain_if_still_risky and (best_ok < abstain_threshold):
        abst_ans = adapter.abstain_answer(item)
        abst_ok, _ = score_single_answer_full(adapter, item, abst_ans, sc_override=best_group_sc)
        if abst_ok >= best_ok:
            return abst_ans, abst_ok, f"{action}_abstain", base_ok

    return best_ans, best_ok, action, base_ok


def always_rewrite_correct(adapter, item, base_answer, kind="mild", K=1):
    if kind == "mild":
        ai = rewrite_once(adapter, item, base_answer, kind="mild")
        if not ai:
            ok0, _ = score_single_answer_full(adapter, item, base_answer, sc_override=None)
            return base_answer, ok0
        ok1, _ = score_single_answer_full(adapter, item, ai, sc_override=None)
        return ai, ok1

    raw = []
    for _ in range(max(1, K)):
        ai = rewrite_once(adapter, item, base_answer, kind="strict")
        if ai and ai.strip():
            raw.append(ai)

    if not raw:
        ok0, _ = score_single_answer_full(adapter, item, base_answer, sc_override=None)
        return base_answer, ok0

    filtered = [c for c in raw if adapter.candidate_valid(item, c)]
    if not filtered:
        filtered = raw

    groups = _group_candidates(adapter, item, filtered)

    best_ok = -1.0
    best_ans = base_answer

    for lbl, cands in groups.items():
        if not cands:
            continue

        sc_grp = selfcheck_group(cands)
        base_ok_sc, _ = score_single_answer_full(adapter, item, base_answer, sc_override=sc_grp)

        local_best_ok = base_ok_sc
        local_best_ans = base_answer

        for ai in cands:
            oki, _ = score_single_answer_full(adapter, item, ai, sc_override=sc_grp)
            if oki > local_best_ok:
                local_best_ok = oki
                local_best_ans = ai

        if local_best_ok > best_ok:
            best_ok = local_best_ok
            best_ans = local_best_ans

    if best_ok < 0:
        ok0, _ = score_single_answer_full(adapter, item, base_answer, sc_override=None)
        return base_answer, ok0

    return best_ans, float(best_ok)


# ============================================================
# BUILD QUESTION PACKS
# ============================================================

def build_pool_indices(n_total, pool_size, val_ratio):
    rng = np.random.RandomState(SEED)
    pool_q_idxs = rng.choice(n_total, size=min(pool_size, n_total), replace=False).tolist()
    rng.shuffle(pool_q_idxs)

    val_n = int(len(pool_q_idxs) * val_ratio)
    val_q_idxs = pool_q_idxs[:val_n]
    test_q_idxs = pool_q_idxs[val_n:]
    return pool_q_idxs, val_q_idxs, test_q_idxs


def build_question_pack(adapter, item_idx, item, n_samples, answer_cache, answer_cache_file):
    answers = ensure_answers(answer_cache, answer_cache_file, adapter, item, n_samples)
    sc = selfcheck_group(answers)

    per_answer = []
    for ans in answers:
        gold_ok = adapter.gold_ok(item, ans)
        judge_ok = adapter.judge_truth(item, ans)
        ent, con = adapter.detector_signal(item, ans)
        per_answer.append({
            "answer": ans,
            "gold_ok": bool(gold_ok),
            "judge_ok": bool(judge_ok),
            "ent": float(ent),
            "con": float(con),
        })

    pack = {
        "q_idx": int(item_idx),
        "item": item,
        "q": adapter.question_text(item),
        "answers": answers,
        "sc": float(sc),
        "per_answer": per_answer,
    }

    if adapter.name == "fever":
        pack["gold_label"] = item["gold_label"]
        pack["evidence"] = item["evidence"]
    elif adapter.name == "nq":
        pack["reference"] = item["reference"]
    elif adapter.name == "tqa":
        pack["correct"] = item["correct"]
        pack["incorrect"] = item["incorrect"]

    return pack


# ============================================================
# DETECTION EVAL
# ============================================================

def detection_error_fieldnames(adapter_name):
    if adapter_name == "fever":
        return [
            "mode", "run", "q_idx", "question", "gold_label", "evidence",
            "answer", "gold_ok", "pred_ok", "score", "threshold",
            "judge_ok", "ent", "con", "selfcheck"
        ]
    if adapter_name == "nq":
        return [
            "mode", "run", "q_idx", "question", "reference",
            "answer", "gold_ok", "pred_ok", "score", "threshold",
            "judge_ok", "ent", "con", "selfcheck"
        ]
    return [
        "mode", "run", "q_idx", "question",
        "answer", "gold_ok", "pred_ok", "score", "threshold",
        "judge_ok", "ent", "con", "selfcheck"
    ]

def detection_error_row(adapter_name, pack, a, mode, run_id, threshold, score, pred_ok):
    base = {
        "mode": mode,
        "run": run_id,
        "q_idx": pack["q_idx"],
        "question": pack["q"],
        "answer": a["answer"],
        "gold_ok": a["gold_ok"],
        "pred_ok": pred_ok,
        "score": score,
        "threshold": threshold,
        "judge_ok": a["judge_ok"],
        "ent": a["ent"],
        "con": a["con"],
        "selfcheck": pack["sc"],
    }
    if adapter_name == "fever":
        base["gold_label"] = pack["gold_label"]
        base["evidence"] = pack["evidence"]
    elif adapter_name == "nq":
        base["reference"] = pack["reference"]
    return base


def eval_on_packs(adapter, packs, mode, threshold, run_id=0, log_errors_path=None, max_log=0):
    tp = fp = tn = fn = 0
    all_scores = []
    q_stds = []

    error_logged = 0
    if log_errors_path:
        ensure_csv_header(log_errors_path, detection_error_fieldnames(adapter.name))

    for pack in packs:
        sc = pack["sc"]
        scores_this_q = []

        for a in pack["per_answer"]:
            score = mode_score(sc=sc, judge_ok=a["judge_ok"], ent=a["ent"], con=a["con"], mode=mode)
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

            if log_errors_path and error_logged < max_log and (pred_ok != gold_ok):
                row = detection_error_row(adapter.name, pack, a, mode, run_id, threshold, score, pred_ok)
                with open(log_errors_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=detection_error_fieldnames(adapter.name))
                    w.writerow(row)
                error_logged += 1

        q_stds.append(float(np.std(scores_this_q)) if len(scores_this_q) > 1 else 0.0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "answer_score_std": float(np.std(all_scores)) if all_scores else 0.0,
        "question_score_std": float(np.mean(q_stds)) if q_stds else 0.0
    }


def tune_threshold_on_val(val_packs, mode, grid=None):
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)

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
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0

        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    return best_t, best_f1


# ============================================================
# MITIGATION EVAL
# ============================================================

def mitigation_case_fieldnames(adapter_name):
    fields = [
        "run", "policy", "q_idx", "question",
        "base_answer", "final_answer",
        "base_ok_score", "final_ok_score", "delta_ok_score",
        "base_gold_ok", "final_gold_ok",
        "final_NEI", "action"
    ]
    if adapter_name == "fever":
        fields.insert(4, "evidence")
        fields.insert(5, "gold_label")
        fields.insert(10, "base_pred_label")
        fields.insert(11, "final_pred_label")
    elif adapter_name == "nq":
        fields.insert(4, "reference")
    elif adapter_name == "tqa":
        fields.insert(4, "correct")
        fields.insert(5, "incorrect")
    return fields


def mitigation_case_row(adapter, pack, policy_name, run_id, base_answer, final_answer,
                        base_ok, final_ok, base_gold_ok, final_gold_ok, nei, action):
    row = {
        "run": run_id,
        "policy": policy_name,
        "q_idx": pack["q_idx"],
        "question": pack["q"],
        "base_answer": base_answer,
        "final_answer": final_answer,
        "base_ok_score": float(base_ok),
        "final_ok_score": float(final_ok),
        "delta_ok_score": float(final_ok - base_ok),
        "base_gold_ok": bool(base_gold_ok),
        "final_gold_ok": bool(final_gold_ok),
        "final_NEI": bool(nei),
        "action": action,
    }

    if adapter.name == "fever":
        row["evidence"] = pack["evidence"]
        row["gold_label"] = pack["gold_label"]
        row["base_pred_label"] = parse_fever_label_from_answer(base_answer)
        row["final_pred_label"] = parse_fever_label_from_answer(final_answer)
    elif adapter.name == "nq":
        row["reference"] = pack["reference"]
    elif adapter.name == "tqa":
        row["correct"] = " || ".join(pack["correct"][:5])
        row["incorrect"] = " || ".join(pack["incorrect"][:5])

    return row


def eval_mitigation_on_packs(
    adapter, packs, policy_name, t_low_ok, t_high_ok, K=10,
    run_id=0, log_cases_path=None, max_log=0,
    abstain_threshold=None
):
    total = 0
    gold_ok_cnt = 0
    nei_cnt = 0
    abstain_cnt = 0

    improve_cnt = 0
    regress_cnt = 0
    abstain_from_wrong_cnt = 0

    base_ok_scores = []
    final_ok_scores = []
    actions = {}

    if log_cases_path:
        ensure_csv_header(log_cases_path, mitigation_case_fieldnames(adapter.name))

    logged = 0

    for pack in packs:
        item = pack["item"]
        base_answer = pack["per_answer"][0]["answer"] if pack["per_answer"] else ""

        base_ok, _ = score_single_answer_full(adapter, item, base_answer, sc_override=None)
        base_gold_ok = adapter.gold_ok(item, base_answer)

        if policy_name == "base":
            final_answer = base_answer
            final_ok = base_ok
            action = "base_keep"
        elif policy_name == "always_mild":
            final_answer, final_ok = always_rewrite_correct(adapter, item, base_answer, kind="mild", K=1)
            action = "always_mild"
        elif policy_name == "always_strict_K":
            final_answer, final_ok = always_rewrite_correct(adapter, item, base_answer, kind="strict", K=K)
            action = f"always_strict_K{K}_group_sc"
        elif policy_name == "policy":
            final_answer, final_ok, action, _ = detector_conditioned_correct(
                adapter, item, base_answer,
                t_low_ok=t_low_ok,
                t_high_ok=t_high_ok,
                K=K,
                mid_kind=MID_KIND,
                abstain_if_still_risky=ABSTAIN_IF_STILL_RISKY,
                abstain_threshold=abstain_threshold
            )
        else:
            raise ValueError(f"Unknown mitigation policy: {policy_name}")

        final_gold_ok = adapter.gold_ok(item, final_answer)
        nei, abstain = adapter.is_nei_or_abstain(item, final_answer)

        total += 1
        gold_ok_cnt += (1 if final_gold_ok else 0)
        nei_cnt += (1 if nei else 0)
        abstain_cnt += (1 if abstain else 0)

        if (not base_gold_ok) and final_gold_ok:
            improve_cnt += 1
        if base_gold_ok and (not final_gold_ok):
            regress_cnt += 1
        if (not base_gold_ok) and nei:
            abstain_from_wrong_cnt += 1

        base_ok_scores.append(base_ok)
        final_ok_scores.append(final_ok)
        actions[action] = actions.get(action, 0) + 1

        if log_cases_path and logged < max_log:
            row = mitigation_case_row(
                adapter=adapter,
                pack=pack,
                policy_name=policy_name,
                run_id=run_id,
                base_answer=base_answer,
                final_answer=final_answer,
                base_ok=base_ok,
                final_ok=final_ok,
                base_gold_ok=base_gold_ok,
                final_gold_ok=final_gold_ok,
                nei=nei,
                action=action
            )
            with open(log_cases_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=mitigation_case_fieldnames(adapter.name))
                w.writerow(row)
            logged += 1

    gold_ok_rate = gold_ok_cnt / total if total else 0.0
    halluc_rate = 1.0 - gold_ok_rate

    base_ok_mean = float(np.mean(base_ok_scores)) if base_ok_scores else 0.0
    final_ok_mean = float(np.mean(final_ok_scores)) if final_ok_scores else 0.0
    delta_ok_mean = float(np.mean(np.array(final_ok_scores) - np.array(base_ok_scores))) if base_ok_scores else 0.0

    return {
        "policy": policy_name,
        "n_questions": int(total),
        "final_gold_ok_rate": float(gold_ok_rate),
        "final_halluc_rate": float(halluc_rate),
        "nei_rate": float(nei_cnt / total) if total else 0.0,
        "abstain_rate": float(abstain_cnt / total) if total else 0.0,
        "base_ok_score_mean": base_ok_mean,
        "final_ok_score_mean": final_ok_mean,
        "delta_ok_score_mean": delta_ok_mean,
        "action_counts": actions,
        "paired": {
            "improve_cnt": int(improve_cnt),
            "regress_cnt": int(regress_cnt),
            "abstain_from_wrong_cnt": int(abstain_from_wrong_cnt),
        }
    }


def _select_val_for_tune(val_packs, max_q=None):
    if max_q is None or max_q <= 0:
        return val_packs
    return val_packs[: min(max_q, len(val_packs))]


def tune_t_high_on_val(adapter, val_packs_for_tune, t_low_ok, K, abstain_threshold):
    hi_max = float(min(0.95, t_low_ok + 0.25))
    grid = np.linspace(t_low_ok, hi_max, 11)

    best_hi = float(min(0.95, t_low_ok + 0.05))
    best_obj = 1e9
    best_stats = None

    for hi in grid:
        total = 0
        final_ok = 0
        abstain_cnt = 0
        regress_cnt = 0

        for pack in val_packs_for_tune:
            item = pack["item"]
            base_answer = pack["per_answer"][0]["answer"] if pack["per_answer"] else ""

            base_gold_ok = adapter.gold_ok(item, base_answer)

            final_answer, _, _, _ = detector_conditioned_correct(
                adapter, item, base_answer,
                t_low_ok=t_low_ok,
                t_high_ok=float(hi),
                K=K,
                mid_kind=MID_KIND,
                abstain_if_still_risky=ABSTAIN_IF_STILL_RISKY,
                abstain_threshold=abstain_threshold
            )

            final_gold_ok = adapter.gold_ok(item, final_answer)
            _, abstain = adapter.is_nei_or_abstain(item, final_answer)

            total += 1
            final_ok += (1 if final_gold_ok else 0)
            abstain_cnt += (1 if abstain else 0)

            if base_gold_ok and (not final_gold_ok):
                regress_cnt += 1

        if total == 0:
            continue

        final_ok_rate = final_ok / total
        halluc_rate = 1.0 - final_ok_rate
        abstain_rate = abstain_cnt / total
        regress_rate = regress_cnt / total

        obj = halluc_rate + TUNE_W_ABSTAIN * abstain_rate + TUNE_W_REGRESS * regress_rate

        if obj < best_obj:
            best_obj = obj
            best_hi = float(hi)
            best_stats = {
                "halluc_rate": float(halluc_rate),
                "abstain_rate": float(abstain_rate),
                "regress_rate": float(regress_rate),
                "final_ok_rate": float(final_ok_rate),
            }

    return best_hi, best_obj, best_stats


# ============================================================
# MAIN RUNNER
# ============================================================

def run_pipeline(dataset_name: str):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cfg = DATASET_CONFIGS[dataset_name]
    adapter = get_adapter(dataset_name)

    ANSWER_CACHE_FILE = cfg["answer_cache_file"]
    OUT_JSON_FILE = cfg["out_json_file"]
    OUT_CSV_FILE = cfg["out_csv_file"]
    ERROR_CASES_FILE = cfg["error_cases_file"]
    MITI_CASES_FILE = cfg["miti_cases_file"]

    MAX_ERROR_LOGS_PER_MODE = cfg["max_error_logs_per_mode"]
    MAX_CASES_PER_RUN = cfg["max_cases_per_run"]
    N_SAMPLES = cfg["n_samples"]
    SUBSET_SIZE = cfg["subset_size"]
    POOL_SIZE = cfg["pool_size"]
    VAL_RATIO = cfg["val_ratio"]
    NUM_RUNS = cfg["num_runs"]
    SAMPLE_SIZE = cfg["sample_size"]

    print(f"\n==================== RUN DATASET: {dataset_name.upper()} ====================")

    ensure_csv_header(ANSWER_CACHE_FILE, ["question", "answer"])
    answer_cache = load_cache_answers(ANSWER_CACHE_FILE)

    subset = adapter.load_subset(cfg)

    print("\nBuilding question pool...")
    pool_q_idxs, val_q_idxs, test_q_idxs = build_pool_indices(
        n_total=len(subset),
        pool_size=POOL_SIZE,
        val_ratio=VAL_RATIO
    )
    print(f"Pool size: {len(pool_q_idxs)} | val questions: {len(val_q_idxs)} | test questions: {len(test_q_idxs)}")

    print("\nPrecomputing VAL + TEST packs (slow due to Ollama)...")
    val_packs = [
        build_question_pack(adapter, i, subset[int(i)], N_SAMPLES, answer_cache, ANSWER_CACHE_FILE)
        for i in tqdm(val_q_idxs, desc="VAL packs")
    ]
    test_packs = [
        build_question_pack(adapter, i, subset[int(i)], N_SAMPLES, answer_cache, ANSWER_CACHE_FILE)
        for i in tqdm(test_q_idxs, desc="TEST packs")
    ]

    modes = ["full", "no_judge", "no_mnli", "no_selfcheck", "judge_only"]

    print("\nTuning threshold on VAL for each mode...")
    mode_thresholds = {}
    for m in modes:
        t_star, f1_star = tune_threshold_on_val(val_packs, m)
        mode_thresholds[m] = float(t_star)
        print(f"  mode={m:12s} | best_thr={t_star:.2f} | val_F1={f1_star:.4f}")

    t_low_ok = float(mode_thresholds["full"])
    t_high_ok_default = float(min(0.95, t_low_ok + 0.05))
    abstain_threshold = float(min(0.95, t_low_ok + ABSTAIN_MARGIN))

    print(
        f"\nMitigation thresholds (default): "
        f"t_low_ok={t_low_ok:.3f}, t_high_ok={t_high_ok_default:.3f}, abstain_thr={abstain_threshold:.3f} | "
        f"K={MITI_K} | MID_KIND={MID_KIND}"
    )

    t_high_ok = t_high_ok_default
    best_obj = None
    best_stats = None
    if TUNE_T_HIGH:
        val_for_tune = _select_val_for_tune(val_packs, max_q=VAL_TUNE_MAX_Q)
        print(f"\nTuning t_high on VAL (n={len(val_for_tune)} questions) to reduce hallucination...")
        t_high_ok, best_obj, best_stats = tune_t_high_on_val(
            adapter=adapter,
            val_packs_for_tune=val_for_tune,
            t_low_ok=t_low_ok,
            K=MITI_K,
            abstain_threshold=abstain_threshold
        )
        if best_stats is not None:
            print(
                f"  tuned t_high_ok={t_high_ok:.3f} | obj={best_obj:.4f} | "
                f"halluc={best_stats['halluc_rate']:.4f} "
                f"abstain={best_stats['abstain_rate']:.4f} "
                f"regress={best_stats['regress_rate']:.4f}"
            )

    print(
        f"\nMitigation thresholds (FINAL): "
        f"t_low_ok={t_low_ok:.3f}, t_high_ok={t_high_ok:.3f}, abstain_thr={abstain_threshold:.3f} | "
        f"K={MITI_K} | MID_KIND={MID_KIND}"
    )

    print("\nRunning paired Monte-Carlo on TEST...")
    rng = np.random.RandomState(SEED)

    summary = {m: [] for m in modes}
    miti_policies = ["base", "always_mild", "always_strict_K", "policy"]
    miti_summary = {p: [] for p in miti_policies}

    test_q_count = len(test_packs)
    if SAMPLE_SIZE > test_q_count:
        raise ValueError(f"SAMPLE_SIZE={SAMPLE_SIZE} > number of test questions={test_q_count}")

    for run_id in range(NUM_RUNS):
        q_sel = rng.choice(test_q_count, size=SAMPLE_SIZE, replace=False)
        packs_run = [test_packs[i] for i in q_sel]

        for m in modes:
            thr = mode_thresholds[m]
            metrics = eval_on_packs(
                adapter=adapter,
                packs=packs_run,
                mode=m,
                threshold=thr,
                run_id=run_id,
                log_errors_path=ERROR_CASES_FILE,
                max_log=MAX_ERROR_LOGS_PER_MODE
            )
            summary[m].append(metrics)

        for p in miti_policies:
            metrics = eval_mitigation_on_packs(
                adapter=adapter,
                packs=packs_run,
                policy_name=p,
                t_low_ok=t_low_ok,
                t_high_ok=t_high_ok,
                K=MITI_K,
                run_id=run_id,
                log_cases_path=MITI_CASES_FILE,
                max_log=MAX_CASES_PER_RUN,
                abstain_threshold=abstain_threshold
            )
            miti_summary[p].append(metrics)

    print("\n=== FINAL SUMMARY (Detection) ===")
    print(f"{'mode':12s} | {'F1_mean':>8s} {'F1_std':>8s} | {'Q-std':>10s} | {'A-std':>10s} | {'thr*':>6s}")
    print("-" * 100)

    rows = []
    summary_json = {
        "dataset": dataset_name,
        "config": {
            "SEED": SEED,
            "N_SAMPLES": N_SAMPLES,
            "SUBSET_SIZE": SUBSET_SIZE,
            "POOL_SIZE": POOL_SIZE,
            "VAL_RATIO": VAL_RATIO,
            "NUM_RUNS": NUM_RUNS,
            "SAMPLE_SIZE": SAMPLE_SIZE,
            "detector_nli": DETECTOR_NLI_MODEL,
            "gold_nli_candidates": GOLD_NLI_CANDIDATES,
            "judge_model": JUDGE_MODEL,
            "gen_model": GEN_MODEL,
            "weights": {"W_JUDGE": W_JUDGE, "W_FACT": W_FACT, "W_SC": W_SC},
            "fact_contra_alpha": FACT_CONTRA_ALPHA,
            "gold_contra_thresh": GOLD_CONTRA_THRESH,
            "gold_thr": GOLD_THR,
            "mitigation": {
                "MITI_K": MITI_K,
                "MID_KIND": MID_KIND,
                "MILD_TEMP": MILD_TEMP,
                "STRICT_TEMP": STRICT_TEMP,
                "USE_SPAN_EXTRACTION_IN_STRICT": USE_SPAN_EXTRACTION_IN_STRICT,
                "SPAN_MAX_WORDS": SPAN_MAX_WORDS,
                "COUNT_EMPTY_AS_ABSTAIN": COUNT_EMPTY_AS_ABSTAIN,
                "ABSTAIN_IF_STILL_RISKY": ABSTAIN_IF_STILL_RISKY,
                "ABSTAIN_MARGIN": ABSTAIN_MARGIN,
                "t_low_ok": t_low_ok,
                "t_high_ok": t_high_ok,
                "abstain_threshold": abstain_threshold,
                "tune_t_high": TUNE_T_HIGH,
                "val_tune_max_q": VAL_TUNE_MAX_Q,
                "tune_obj_weights": {"w_abstain": TUNE_W_ABSTAIN, "w_regress": TUNE_W_REGRESS},
                "t_high_tune_obj": best_obj,
                "t_high_tune_stats": best_stats,
            },
            "files": {
                "ANSWER_CACHE_FILE": ANSWER_CACHE_FILE,
                "OUT_JSON_FILE": OUT_JSON_FILE,
                "OUT_CSV_FILE": OUT_CSV_FILE,
                "ERROR_CASES_FILE": ERROR_CASES_FILE,
                "MITI_CASES_FILE": MITI_CASES_FILE,
            }
        },
        "thresholds": mode_thresholds,
        "modes": {},
        "mitigation": {}
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
            "thr_star": float(mode_thresholds[m]),
            "f1_mean": safe_mean(f1s),
            "f1_std": safe_std(f1s),
            "question_score_std_mean": safe_mean(qstds),
            "question_score_std_std": safe_std(qstds),
            "answer_score_std_mean": safe_mean(astds),
            "answer_score_std_std": safe_std(astds),
            "delta_f1_vs_full_mean": float(delta_mean)
        })

        print(
            f"{m:12s} | "
            f"{safe_mean(f1s):8.4f} {safe_std(f1s):8.4f} | "
            f"Q-std={safe_mean(qstds):.4f} | "
            f"A-std={safe_mean(astds):.4f} | "
            f"{mode_thresholds[m]:6.2f}"
        )

    print("\n=== MITIGATION SUMMARY (Question-level final answer) ===")
    print(f"{'policy':16s} | {'gold_ok_mean':>10s} {'halluc_mean':>10s} | {'NEI_mean':>8s} {'abstain_mean':>12s} | {'Δok_score':>10s} | {'impr/reg/abst':>14s}")
    print("-" * 120)

    for p in miti_policies:
        ok_rates = [x["final_gold_ok_rate"] for x in miti_summary[p]]
        halluc_rates = [x["final_halluc_rate"] for x in miti_summary[p]]
        nei_rates = [x["nei_rate"] for x in miti_summary[p]]
        abstain_rates = [x["abstain_rate"] for x in miti_summary[p]]
        deltas = [x["delta_ok_score_mean"] for x in miti_summary[p]]

        impr = [x["paired"]["improve_cnt"] for x in miti_summary[p]]
        reg = [x["paired"]["regress_cnt"] for x in miti_summary[p]]
        abst = [x["paired"]["abstain_from_wrong_cnt"] for x in miti_summary[p]]

        summary_json["mitigation"][p] = {
            "runs": miti_summary[p],
            "final_gold_ok_rate_mean": safe_mean(ok_rates),
            "final_gold_ok_rate_std": safe_std(ok_rates),
            "final_halluc_rate_mean": safe_mean(halluc_rates),
            "final_halluc_rate_std": safe_std(halluc_rates),
            "nei_rate_mean": safe_mean(nei_rates),
            "nei_rate_std": safe_std(nei_rates),
            "abstain_rate_mean": safe_mean(abstain_rates),
            "abstain_rate_std": safe_std(abstain_rates),
            "delta_ok_score_mean": safe_mean(deltas),
            "delta_ok_score_std": safe_std(deltas),
            "paired_mean": {
                "improve_cnt": safe_mean(impr),
                "regress_cnt": safe_mean(reg),
                "abstain_from_wrong_cnt": safe_mean(abst),
            }
        }

        print(
            f"{p:16s} | "
            f"{safe_mean(ok_rates):10.4f} {safe_mean(halluc_rates):10.4f} | "
            f"{safe_mean(nei_rates):8.4f} {safe_mean(abstain_rates):12.4f} | "
            f"{safe_mean(deltas):10.4f} | "
            f"{int(round(safe_mean(impr))):4d}/{int(round(safe_mean(reg))):3d}/{int(round(safe_mean(abst))):4d}"
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

    ensure_csv_header(OUT_CSV_FILE, list(rows[0].keys()) if rows else ["mode"])
    with open(OUT_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        else:
            f.write("mode\n")

    print(f"\nSaved: {OUT_JSON_FILE}, {OUT_CSV_FILE}")
    print(f"Mitigation cases saved: {MITI_CASES_FILE} (up to {MAX_CASES_PER_RUN} rows per policy per run)")
    print(f"Optional detection error cases: {ERROR_CASES_FILE} (up to {MAX_ERROR_LOGS_PER_MODE} mismatches per mode per run)")


# ============================================================
# ENTRY
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Unified hallucination detection + mitigation pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fever", "nq", "tqa"],
        help="Dataset to run: fever / nq / tqa"
    )
    args = parser.parse_args()
    run_pipeline(args.dataset)


if __name__ == "__main__":
    main()