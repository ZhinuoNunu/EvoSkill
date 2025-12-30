"""Configuration for the self-improving loop."""

from dataclasses import dataclass


@dataclass
class LoopConfig:
    """Configuration parameters for SelfImprovingLoop.

    Attributes:
        max_iterations: Maximum number of improvement iterations.
        frontier_size: Number of top-performing programs to keep.
        no_improvement_limit: Stop early after this many iterations without improvement.
        tolerance: Tolerance for answer matching (0.0 = exact match).
        concurrency: Number of concurrent evaluations.
    """

    max_iterations: int = 5
    frontier_size: int = 3
    no_improvement_limit: int = 5
    tolerance: float = 0.0
    concurrency: int = 2
