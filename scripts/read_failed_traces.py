#!/usr/bin/env python3
"""
Read and display all failed traces from cache (JSON) and eval results (pkl).

Failure is defined as ANY of:
  1. trace.is_error == True
  2. trace.parse_error is not None
  3. trace.output is None (no structured output)
  4. IndexedEvalResult.error is not None (timeout/crash in eval pipeline)
  5. (optional) scorer < 0.8 when --dataset is provided (wrong answer)

Usage:
  python scripts/read_failed_traces.py                    # structural failures only
  python scripts/read_failed_traces.py --dataset .dataset/officeqa_pro.csv  # + answer correctness
  python scripts/read_failed_traces.py --all              # show all traces, mark failures
"""

import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / ".cache" / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

TOLERANCE_LEVELS = [0.05, 0.01, 0.1, 0.0, 0.025]


def score_multi_tolerance(question: str, predicted: str, ground_truth: str) -> float:
    from src.evaluation.reward import score_answer

    weighted_sum = 0.0
    weight_total = 0.0
    for tol in TOLERANCE_LEVELS:
        weight = 1.0 / (1.0 + 20.0 * tol)
        score = score_answer(predicted, ground_truth, tol)
        weighted_sum += weight * score
        weight_total += weight
    return weighted_sum / weight_total


@dataclass
class TraceRecord:
    source: str
    source_path: str
    question: str
    ground_truth: Optional[str]
    model: str
    is_error: bool
    parse_error: Optional[str]
    final_answer: Optional[str]
    reasoning: Optional[str]
    duration_ms: int
    cost_usd: float
    num_turns: int
    result_snippet: str
    failure_reasons: list[str] = field(default_factory=list)
    error_msg: Optional[str] = None
    score: Optional[float] = None

    @property
    def is_failed(self) -> bool:
        return len(self.failure_reasons) > 0


def load_cache_traces() -> list[TraceRecord]:
    records: list[TraceRecord] = []
    if not CACHE_DIR.exists():
        return records

    for json_file in sorted(CACHE_DIR.rglob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] Cannot read {json_file}: {e}")
            continue

        trace = entry.get("trace", {})
        cache_key = entry.get("cache_key", {})

        is_error = trace.get("is_error", False)
        parse_error = trace.get("parse_error")
        output = trace.get("output")

        reasons: list[str] = []
        if is_error:
            reasons.append("is_error=True")
        if parse_error:
            reasons.append(f"parse_error")
        if output is None:
            reasons.append("output=None")

        final_answer = None
        reasoning = None
        if output and isinstance(output, dict):
            final_answer = output.get("final_answer")
            reasoning = output.get("reasoning")

        result_text = str(trace.get("result", ""))[:300]

        records.append(TraceRecord(
            source="cache",
            source_path=str(json_file.relative_to(PROJECT_ROOT)),
            question=cache_key.get("question", "???"),
            ground_truth=None,
            model=trace.get("model", "???"),
            is_error=is_error,
            parse_error=parse_error,
            final_answer=final_answer,
            reasoning=reasoning,
            duration_ms=trace.get("duration_ms", 0),
            cost_usd=trace.get("total_cost_usd", 0),
            num_turns=trace.get("num_turns", 0),
            result_snippet=result_text,
            failure_reasons=reasons,
        ))

    return records


def load_results_traces() -> list[TraceRecord]:
    records: list[TraceRecord] = []
    if not RESULTS_DIR.exists():
        return records

    for pkl_file in sorted(RESULTS_DIR.glob("*.pkl")):
        try:
            with open(pkl_file, "rb") as f:
                results = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Cannot read {pkl_file}: {e}")
            continue

        for r in results:
            reasons: list[str] = []

            if r.error is not None:
                reasons.append(f"error: {r.error[:80]}")

            if r.trace is not None:
                if r.trace.is_error:
                    reasons.append("is_error=True")
                if r.trace.parse_error:
                    reasons.append("parse_error")
                if r.trace.output is None:
                    reasons.append("output=None")
            elif r.error is None:
                reasons.append("trace=None (timeout/crash)")

            trace = r.trace
            final_answer = None
            reasoning = None
            if trace and trace.output and hasattr(trace.output, "final_answer"):
                final_answer = trace.output.final_answer
            if trace and trace.output and hasattr(trace.output, "reasoning"):
                reasoning = trace.output.reasoning

            result_text = str(trace.result)[:300] if trace else ""

            records.append(TraceRecord(
                source="results",
                source_path=str(pkl_file.relative_to(PROJECT_ROOT)),
                question=r.question,
                ground_truth=r.ground_truth,
                model=trace.model if trace else "???",
                is_error=trace.is_error if trace else False,
                parse_error=trace.parse_error if trace else None,
                final_answer=final_answer,
                reasoning=reasoning,
                duration_ms=trace.duration_ms if trace else 0,
                cost_usd=trace.total_cost_usd if trace else 0,
                num_turns=trace.num_turns if trace else 0,
                result_snippet=result_text,
                failure_reasons=reasons,
                error_msg=r.error,
            ))

    return records


def enrich_with_ground_truth(
    records: list[TraceRecord], dataset_path: str
) -> None:
    """Match cache traces to dataset rows by question and score answers."""
    import pandas as pd

    df = pd.read_csv(dataset_path)
    gt_col = "ground_truth" if "ground_truth" in df.columns else "answer"
    if gt_col not in df.columns:
        print(f"[WARN] Dataset has no 'ground_truth' or 'answer' column, skipping scoring")
        return

    q_to_gt: dict[str, str] = {}
    for _, row in df.iterrows():
        q_to_gt[row["question"].strip()] = str(row[gt_col]).strip()

    for rec in records:
        if rec.ground_truth is not None:
            continue
        gt = q_to_gt.get(rec.question.strip())
        if gt is None:
            continue
        rec.ground_truth = gt
        if rec.final_answer:
            try:
                rec.score = score_multi_tolerance(
                    rec.question,
                    rec.final_answer.strip().lower(),
                    gt.strip().lower(),
                )
                if rec.score < 0.8:
                    rec.failure_reasons.append(
                        f"wrong_answer (score={rec.score:.2f})"
                    )
            except Exception as e:
                rec.failure_reasons.append(f"scoring_error: {e}")


def print_trace(i: int, t: TraceRecord, show_reasoning: bool = False) -> None:
    sep = "=" * 80
    status = "FAIL" if t.is_failed else "OK"
    print(sep)
    print(f"  [{i}] [{status}] Source: {t.source} ({t.source_path})")
    print(f"  Model: {t.model}  |  Turns: {t.num_turns}  |  "
          f"Duration: {t.duration_ms / 1000:.1f}s  |  Cost: ${t.cost_usd:.4f}")

    if t.failure_reasons:
        print(f"  Failures: {' | '.join(t.failure_reasons)}")
    if t.error_msg:
        print(f"  Error: {t.error_msg}")

    print()
    q_display = t.question[:200] + ("..." if len(t.question) > 200 else "")
    print(f"  Q: {q_display}")

    if t.ground_truth:
        print(f"  Ground Truth: {t.ground_truth}")
    if t.final_answer:
        print(f"  Agent Answer: {t.final_answer}")
    if t.score is not None:
        print(f"  Score: {t.score:.4f} {'(PASS)' if t.score >= 0.8 else '(FAIL)'}")
    if t.parse_error:
        print(f"  Parse Error: {t.parse_error[:300]}")

    if show_reasoning and t.reasoning:
        r_display = t.reasoning[:500] + ("..." if len(t.reasoning) > 500 else "")
        print(f"  Reasoning: {r_display}")

    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Read and display failed traces")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="CSV dataset path to match ground truth and score answers",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all traces (not just failures)",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Show agent reasoning in output",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  TRACE READER - Failed Trace Inspector")
    print("=" * 80)
    print(f"  Cache dir:   {CACHE_DIR}")
    print(f"  Results dir: {RESULTS_DIR}")
    if args.dataset:
        print(f"  Dataset:     {args.dataset}")
    print()

    cache_records = load_cache_traces()
    results_records = load_results_traces()
    all_records = cache_records + results_records

    if args.dataset:
        enrich_with_ground_truth(all_records, args.dataset)

    total_cache = len(list(CACHE_DIR.rglob("*.json"))) if CACHE_DIR.exists() else 0
    total_results_count = 0
    if RESULTS_DIR.exists():
        for pkl_file in RESULTS_DIR.glob("*.pkl"):
            try:
                with open(pkl_file, "rb") as f:
                    total_results_count += len(pickle.load(f))
            except Exception:
                pass

    failed = [r for r in all_records if r.is_failed]
    passed = [r for r in all_records if not r.is_failed]

    print(f"  Total traces:  cache={total_cache}, results={total_results_count}")
    print(f"  Failed:        {len(failed)}")
    print(f"  Passed:        {len(passed)}")
    print()

    if failed:
        reason_counts: dict[str, int] = {}
        for t in failed:
            for reason in t.failure_reasons:
                key = reason.split(":")[0].strip().split("(")[0].strip()
                reason_counts[key] = reason_counts.get(key, 0) + 1

        print("  Failure breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
        print()

    display_records = all_records if args.all else failed

    if not display_records:
        if args.all:
            print("No traces found at all!")
        else:
            print("No failed traces found! All traces passed.")
            if all_records:
                print(f"\nShowing {len(all_records)} passing trace(s) for reference:\n")
                for i, t in enumerate(all_records, 1):
                    print_trace(i, t, show_reasoning=args.reasoning)
        return

    print(f"{'All' if args.all else 'Failed'} traces ({len(display_records)}):\n")
    for i, t in enumerate(display_records, 1):
        print_trace(i, t, show_reasoning=args.reasoning)

    total_cost = sum(t.cost_usd for t in display_records)
    total_duration = sum(t.duration_ms for t in display_records)
    print("=" * 80)
    print(f"  SUMMARY: {len(display_records)} traces shown, "
          f"total cost: ${total_cost:.4f}, "
          f"total duration: {total_duration / 1000:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
