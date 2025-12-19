Feedback Descent
================

This repo implements the core optimization loop from https://arxiv.org/abs/2511.07919:

1) Generate a new candidate conditioned on the current best and accumulated textual feedback  
2) Evaluate candidate vs. best to get a binary preference and a rationale  
3) Append the rationale to the feedback history  
4) Replace the best when the candidate wins; stop after patience or iteration cap

Two entry points are available:
- `src.feedback_descent.FeedbackDescent`: minimal, LLM-agnostic loop you can wire to any proposer/evaluator callables.
- `src.engine.FeedbackDescentEngine`: richer loop that works with the higher-level candidate/comparator/adapter abstractions already in this repo.

Minimal scaffold
----------------
```python
import asyncio
from dataclasses import dataclass
from src import FeedbackDescent, EvaluationResult, Proposer, Evaluator

@dataclass
class Artifact:
    text: str

class MockProposer(Proposer[Artifact]):
    async def generate_initial(self, problem: str) -> Artifact:
        return Artifact(text=f"initial solution for: {problem}")

    async def propose(self, current_best: Artifact, feedback_history):
        return Artifact(text=current_best.text + " [tweak]")

class MockEvaluator(Evaluator[Artifact]):
    async def evaluate(self, current_best: Artifact, candidate: Artifact) -> EvaluationResult:
        # replace with your judge LLM / metrics
        candidate_better = len(candidate.text) > len(current_best.text)
        rationale = "longer text is preferred in this mock setup"
        return EvaluationResult(
            preference_for_candidate=candidate_better,
            rationale=rationale,
            score_best=len(current_best.text),
            score_candidate=len(candidate.text),
        )

async def main():
    loop = FeedbackDescent(
        proposer=MockProposer(),
        evaluator=MockEvaluator(),
        max_iterations=5,
        no_improvement_limit=2,
    )
    result = await loop.run("demo problem")
    print("Best:", result.best)
    print("Feedback:", [f.rationale for f in result.feedback_history])

asyncio.run(main())
```

Notes
-----
- Swap `MockProposer`/`MockEvaluator` with your LLM-backed implementations.
- `FeedbackDescent` is generic over the artifact type; it can be a prompt string, a config object, code, etc.
- If you want structured execution traces and grading, use `FeedbackDescentEngine` with `BaseAdapter`, `BaseComparator`, and `BaseProposer` in `src/`.
