def score_sealqa(ground_truth: str, predicted: str) -> float:
    """Score a SEAL-QA answer. Returns 0.0 or 1.0.

    TODO: Replace with DSPy COT agent scorer.
    """
    # Placeholder: exact case-insensitive match
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0
