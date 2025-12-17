#!/usr/bin/env python3
"""
Standalone KC classification evaluator with embedded prompt.
- No hardcoded API keys or base URLs. Supply them via CLI flags or environment variables.
- Works for both zero-shot and few-shot (optional examples file).
- Outputs predictions and metrics to a chosen run directory.

Example (zero-shot):
  python llm_validation_standalone.py \
    --dataset /path/to/test_split.csv \
    --base-url https://api.example.com/v1/chat/completions \
    --model-id your-model-id \
    --api-key-env MY_MODEL_KEY

Example (few-shot, limit 20 for smoke test):
  python llm_validation_standalone.py \
    --dataset /path/to/test_split.csv \
    --base-url https://api.example.com/v1/chat/completions \
    --model-id your-model-id \
    --api-key-env MY_MODEL_KEY \
    --mode few \
    --fewshot-file /path/to/fewshot.txt \
    --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import requests
from tqdm.auto import tqdm

SYSTEM_PROMPT = """Knowledge Contribution Classification Decision Tree and Detailed Category Definitions

START: Analyze the citation sentence

Step 1: Determine the Citation Subject (Core Branching)
Question: Is the actor in the sentence "we" (the authors) or "others"?
- "We" (subject contains we, our, us, or clearly refers to the authors' work) → Proceed to Step 2 (Author Action Branch)
- "Others" (subject is the cited authors or generic reference) → Proceed to Step 3 (Descriptive Branch)
- Special Rule: When a compound sentence has different actors in the main clause and subordinate clause, determine based on the actor of the sub-clause where [CITATION] directly appears. Even if the main clause uses "we", if [CITATION] is in a subordinate clause describing others' work, enter the Descriptive Branch.

Step 2: Author Action Branch (Determining "our" usage purpose)
When the authors are the actors, judge in the following A, B, C order:

Question A: Are we using [CITATION] to conduct research (Category 1)?
- Core Judgment: Do the authors directly use [CITATION]'s method, technique, or system? Do not guess.
- Judgment Content: Algorithms, models, technical architecture, evaluation metrics, systems, etc.
- Keywords: based on, built upon, we use...as backbone, evaluate using...
- Yes → Output: Category 1 (Method Contribution)
- No → Proceed to Question B

Question B: Are we using [CITATION] as a "resource" (Category 2)?
- Core Judgment: Do the authors directly use [CITATION]'s dataset, corpus, or toolkit? Do not guess.
- Judgment Content: dataset, corpus, explicitly labeled toolkit/toolbox.
- Yes → Output: Category 2 (Resource Tool Contribution)
- No → Proceed to Question C

Question C: Are we using [CITATION] as a "yardstick" for comparison or evaluation (Category 3)?
- Core Judgment: Do the authors explicitly compare their own work as one party with [CITATION] for the purpose of positioning themselves, highlighting characteristics, or evaluating performance?
- Judgment Clues:
  - Explicit comparison: in contrast to, unlike, different from, whereas, while, but
  - Similarity comparison: similarly, likewise, as in [CITATION], same as
  - Special Check: This rule only applies when authors compare their own work with others (e.g., "our method is similar to [CITATION]"). If the sentence describes similarity between two other studies (e.g., "[CITATION A] is similar to [CITATION B]"), its core function is providing background, not this category—answer "No" in final judgment.
  - Performance comparison: outperforms, compared to, baseline, benchmark
  - Parallel display: Listed side by side in tables/lists, "A: X [CITATION], B: Y" format
- Yes → Output: Category 3 (Evidential Finding Contribution)
- No → (Final Decision) → Output: Category 4 (Background Contribution)

Step 3: Descriptive Branch (Determining "others'" usage purpose)
When authors are describing others' work, perform the following "Knowledge Contribution Type Test":

Core Question: Does [CITATION] contribute a specific "argument" usable in the authors' logical chain, or a macro "map" for understanding background?

Checklist: Does [CITATION] provide any of the following types of specific arguments?

1. Provides "verifiable findings" for statement or comparison?
- Core Judgment: Does the author cite a specific, verifiable finding (e.g., "...is most effective", numerical results, clear conclusions) and integrate it as factual evidence into their own argument? This can be for direct comparison or simply to state an important empirical fact.
- Judgment Clues: Stating facts—look for reporting verbs (e.g., show, demonstrate, find, report) + specific conclusions. Direct comparison—look for comparative context (e.g., unlike..., whereas...).
- Examples (Category 3):
  - "BERT scored 90.65% [CITATION], RNN scored 78.63%"
  - "[CITATION] only considers verbs, while we consider all parts of speech"
  - "Unlike [CITATION]'s batch processing, we adopt streaming processing"
  - "[CITATION] demonstrated that using text is most effective"
- Yes → Output: Category 3 (Evidential Finding Contribution)

2. Provides "methodological justification" to support authors' action?
- Core Judgment: This rule requires simultaneously meeting the following two conditions:
  - Condition A (Citation Premise): Does the author cite a finding or premise as a reason?
  - Condition B (Leading to Action): Does the author explicitly connect this reason with a specific research decision or action they took?
- Judgment Clues: Look for a clear "[CITATION]'s finding → our action" logical chain.
- Keywords: Because [CITATION] showed..., Based on the finding that... [CITATION], we chose to...
- Both conditions met → Yes → Output: Category 3 (Evidential Finding Contribution)

3. Reveals "critical flaws or conflicts" supporting authors' action?
- Core Judgment: This rule requires simultaneously meeting the following two conditions:
  - Condition A (Pointing Out Flaws): Does the author explicitly point out specific, serious flaws, failures, or direct conflicts in viewpoint in [CITATION]?
  - Condition B (Leading to Action): Does the author immediately use this flaw as an explicit reason to introduce a specific subsequent action of their own?
- Judgment Clues: Look for a clear "problem-our solution" logical chain. If only describing flaws without leading to the author's explicit action, this condition is not met—should be categorized as Category 4.
- Keywords:
  - Flaw/Conflict Signals: suffer from, fail to, ignore the problem, incorrect, a contested area, conflicting results
  - Subsequent Action Signals: Therefore, we propose..., To address this issue, we..., This motivates our work on...
- Both conditions met → Yes → Output: Category 3 (Evidential Finding Contribution)

Final Decision:
- If at least one of the above three checks is "Yes" → Output: Category 3 (Evidential Finding Contribution)
- If all three checks are "No" → Output: Category 4 (Background Contribution)

Category Descriptions

Category 1: Method Contribution
Category 2: Resource Tool Contribution
Category 3: Evidential Finding Contribution
Category 4: Background Contribution

You are an academic literature knowledge contribution classification expert. Please strictly follow the decision tree above to classify citations.

Input Format:
- id: Citation identifier
- citation_section: Section where the citation appears (for context understanding)
- citation_context: Complete sentence containing [CITATION]

Output Format (Strictly Follow):
- Format: id,category_number
- Category Numbers: 1=Method Contribution, 2=Resource Tool Contribution, 3=Evidential Finding Contribution, 4=Background Contribution
- Do NOT output any explanations, reasons, or extra text

Classification Process:
1. Identify citation subject: Determine if the actor is "we" or "others"
2. If "we": Check in A→B→C order (Method→Resource→Comparison→Background)
3. If "others": Check if specific arguments are provided (verifiable findings/methodological justification/critical flaws)
4. Strictly follow every judgment condition in the decision tree

Key Notes:
- In compound sentences, judge based on the actor of the sub-clause where [CITATION] appears
- "Others use X" description ≠ "We use X"
- All three Category 3 check rules (methodological justification, critical flaws) require simultaneously meeting both "Condition A" and "Condition B"
- Merely stating historical facts (e.g., "[CITATION] proposed BERT") → Category 4
- Stating empirical findings (e.g., "[CITATION] showed BERT achieves 92% accuracy") → Category 3
"""

USER_PROMPT_TEMPLATE = """id: {id}\ncitation_section: \"{section}\"\ncitation_context: \"{context}\"\n\nOutput:"""

OUTPUT_REGEX = re.compile(r"(\d+),([1234])")
VALID_LABELS = {"1", "2", "3", "4"}


def extract_prediction(text: str, expected_id: str) -> Optional[str]:
    text = text.strip()
    match = OUTPUT_REGEX.search(text)
    if match:
        resp_id, code = match.groups()
        if resp_id != expected_id:
            print(f"[warn] ID mismatch expected {expected_id} got {resp_id}; using code anyway.", file=sys.stderr)
        return code

    patterns = [
        r"Final\s+Answer:\s*\d+,([1-4])",
        r"(?:Output|Answer|Classification|Category):\s*([1-4])",
        r"(?:Output|Answer|Classification|Category)\s*([1-4])",
        r"\b(?:is|should be|classified as|categorized as)\s+([1-4])\b",
        r"→\s*Output:\s*([1-4])",
        r"Final\s+(?:answer|classification|category):\s*([1-4])",
    ]
    for pattern in patterns:
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            return found.group(1)

    digits = re.findall(r"\b([1-4])\b", text)
    if digits:
        return digits[-1]
    return None


@dataclass
class Prediction:
    sample_id: str
    gold: str
    predicted: Optional[str]
    raw_response: str


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(gold_labels: List[str], predicted_labels: List[str]) -> Dict[str, object]:
    if len(gold_labels) != len(predicted_labels):
        raise ValueError("Gold and predicted labels must align.")

    paired = [(g, p) for g, p in zip(gold_labels, predicted_labels) if p in VALID_LABELS]
    invalid_count = len(gold_labels) - len(paired)
    if not paired:
        raise ValueError("No valid predictions to compute metrics.")

    gold_labels = [g for g, _ in paired]
    predicted_labels = [p for _, p in paired]

    labels = sorted(set(gold_labels + predicted_labels))
    confusion = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}

    for gold, pred in zip(gold_labels, predicted_labels):
        if gold == pred:
            confusion[gold]["tp"] += 1
        else:
            confusion[pred]["fp"] += 1
            confusion[gold]["fn"] += 1

    per_class = {}
    total_tp = total_fp = total_fn = 0
    for label, counts in confusion.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_precision = sum(v["precision"] for v in per_class.values()) / len(per_class)
    macro_recall = sum(v["recall"] for v in per_class.values()) / len(per_class)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)

    micro_precision = safe_div(total_tp, total_tp + total_fp)
    micro_recall = safe_div(total_tp, total_tp + total_fn)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    accuracy = sum(1 for g, p in zip(gold_labels, predicted_labels) if g == p) / len(gold_labels)

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "accuracy": accuracy,
        "per_class_metrics": per_class,
        "confusion_matrix": confusion,
        "invalid_predictions": invalid_count,
    }


def build_prompt(sample_id: str, section: str, context: str, fewshot: Optional[str]) -> str:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        id=sample_id,
        section=section.replace('"', '\\"'),
        context=context.replace('"', '\\"'),
    )
    if fewshot:
        return f"{fewshot}\n\n{user_prompt}"
    return user_prompt


def chat_completion(base_url: str, model_id: str, api_key: str, system_prompt: str, user_prompt: str, timeout: float) -> str:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    try:
        message = data["choices"][0]["message"]
        content = message.get("content", "").strip()
        return content
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response payload: {json.dumps(data, ensure_ascii=False)}") from exc


def parse_fewshot(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Few-shot file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def save_metrics(run_dir: Path, metrics: Dict[str, object]) -> None:
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        "=" * 60,
        "KC EVALUATION RESULTS",
        "=" * 60,
        f"Dataset: {metrics['dataset']}",
        f"Total samples: {metrics['total_samples']}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        "",
        "Macro Metrics:",
        f"  Precision: {metrics['macro_precision']:.4f}",
        f"  Recall:    {metrics['macro_recall']:.4f}",
        f"  F1:        {metrics['macro_f1']:.4f}",
        "",
        "Micro Metrics:",
        f"  Precision: {metrics['micro_precision']:.4f}",
        f"  Recall:    {metrics['micro_recall']:.4f}",
        f"  F1:        {metrics['micro_f1']:.4f}",
        "",
        f"Unparsed / invalid predictions: {metrics['invalid_predictions']}",
    ]
    summary_path = results_dir / "evaluation_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


def run(df: pd.DataFrame, run_dir: Path, *, base_url: str, model_id: str, api_key: str, mode: str, fewshot: Optional[str], timeout: float, limit: Optional[int], sleep: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.csv"
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)

    if limit is not None:
        df = df.head(limit)
        print(f"[info] Limiting evaluation to {len(df)} samples")

    preds: List[Prediction] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        sample_id = str(row["id"])
        gold = str(row["KC"])
        section = str(row.get("citation_section", ""))
        context = str(row.get("citation_context", ""))
        prompt = build_prompt(sample_id, section, context, fewshot)

        attempt = 0
        response_text: Optional[str] = None
        while attempt <= 3:
            try:
                response_text = chat_completion(
                    base_url=base_url,
                    model_id=model_id,
                    api_key=api_key,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    timeout=timeout,
                )
                break
            except requests.RequestException as http_err:
                attempt += 1
                if attempt > 3:
                    raise
                wait = 2 ** (attempt - 1)
                print(f"[retry] {http_err} waiting {wait}s", file=sys.stderr)
                time.sleep(wait)

        if response_text is None:
            raise RuntimeError("Model response is None after retries.")

        parsed = extract_prediction(response_text, sample_id)
        prediction = Prediction(
            sample_id=sample_id,
            gold=gold,
            predicted=parsed if parsed in VALID_LABELS else None,
            raw_response=response_text,
        )
        preds.append(prediction)

        # live save
        pd.DataFrame(
            [(p.sample_id, p.predicted if p.predicted else "") for p in preds],
            columns=["id", "predicted"],
        ).to_csv(predictions_path, index=False, encoding="utf-8")

        pd.DataFrame(
            [
                (
                    p.sample_id,
                    p.gold,
                    p.predicted if p.predicted else "",
                    "✓" if p.predicted == p.gold else "✗",
                    p.raw_response,
                )
                for p in preds
            ],
            columns=["id", "gold_label", "predicted_label", "correct", "raw_response"],
        ).to_csv(results_dir / "detailed_predictions.csv", index=False, encoding="utf-8")

        if sleep > 0:
            time.sleep(sleep)

    gold_labels = [p.gold for p in preds]
    predicted_labels = [p.predicted if p.predicted else "" for p in preds]
    metrics = compute_metrics(gold_labels, predicted_labels)
    metrics_payload = {
        "dataset": str(run_dir),
        "total_samples": len(preds),
        **metrics,
    }
    save_metrics(run_dir, metrics_payload)
    print("[done] Evaluation complete. Metrics written to", run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone KC classification evaluator (no hardcoded keys/URLs).")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the input CSV (must include columns: id, KC, citation_section, citation_context).")
    parser.add_argument("--output", type=Path, default=Path("./outputs"), help="Directory to store predictions and metrics.")
    parser.add_argument("--mode", type=str, choices=["zero", "few"], default="zero", help="zero = standard prompt, few = prepend examples from --fewshot-file.")
    parser.add_argument("--fewshot-file", type=Path, default=None, help="Optional text file whose content is prepended as few-shot examples.")
    parser.add_argument("--base-url", type=str, required=True, help="Chat completion endpoint (not logged).")
    parser.add_argument("--model-id", type=str, required=True, help="Model identifier understood by the provider.")
    parser.add_argument("--api-key", type=str, default=None, help="API key (avoid logging); overrides environment variable.")
    parser.add_argument("--api-key-env", type=str, default=None, help="Environment variable name containing the API key.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for quick tests.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between API calls.")
    return parser.parse_args()


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    if args.api_key_env:
        env_val = os.getenv(args.api_key_env)
        if env_val:
            return env_val
    raise ValueError("API key is required. Provide --api-key or --api-key-env.")


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    api_key = resolve_api_key(args)
    fewshot_text = parse_fewshot(args.fewshot_file) if args.mode == "few" else None

    run_dir = args.output / args.mode
    df = pd.read_csv(args.dataset)

    print("[config] mode=", args.mode)
    print("[config] dataset=", args.dataset)
    print("[config] output_dir=", run_dir)
    print("[config] model_id= (hidden)")
    print("[config] base_url= (hidden)")

    run(
        df=df,
        run_dir=run_dir,
        base_url=args.base_url,
        model_id=args.model_id,
        api_key=api_key,
        mode=args.mode,
        fewshot=fewshot_text,
        timeout=args.timeout,
        limit=args.limit,
        sleep=args.sleep,
    )


if __name__ == "__main__":
    main()
