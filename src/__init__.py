"""Feedback Descent: Open-Ended Text Optimization via Pairwise Comparison."""

from .feedback_descent import (
    FeedbackDescent,
    EvaluationResult,
    FeedbackEntry,
    FeedbackDescentResult,
    Proposer,
    Evaluator,
)

__all__ = [
    "FeedbackDescent",
    "EvaluationResult",
    "FeedbackEntry",
    "FeedbackDescentResult",
    "Proposer",
    "Evaluator",
]
