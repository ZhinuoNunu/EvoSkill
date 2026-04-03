#!/usr/bin/env python3
"""Run self-improving agent loop on SEAL-QA dataset."""

import argparse
import asyncio

import pandas as pd

from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.agent_profiles import (
    Agent,
    sealqa_agent_options,
    make_sealqa_agent_options,
    skill_proposer_options,
    prompt_proposer_options,
    skill_generator_options,
    prompt_generator_options,
)
from src.agent_profiles.skill_generator import get_project_root
from src.evaluation.sealqa_scorer import score_sealqa
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)


def stratified_split(
    data: pd.DataFrame, train_ratio: float = 0.18, val_ratio: float = 0.12
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
    """Split data ensuring each category has at least 1 in both train and validation.

    Args:
        data: DataFrame with 'question', 'ground_truth', 'category' columns.
        train_ratio: Fraction of each category to use for training.
        val_ratio: Fraction of each category to use for validation.

    Returns:
        train_pools: Dict mapping category -> list of (question, answer) tuples.
        val_data: List of (question, answer, category) tuples for validation.
    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) cannot exceed 1.0")

    # Drop rows with missing categories
    data = data.dropna(subset=['category'])
    categories = data['category'].unique()
    train_pools: dict[str, list[tuple[str, str]]] = {}
    val_data: list[tuple[str, str, str]] = []

    for cat in categories:
        cat_data = data[data['category'] == cat].sample(frac=1, random_state=42)
        n_train = max(1, int(len(cat_data) * train_ratio))
        n_val = max(1, int(len(cat_data) * val_ratio))

        # Train comes first, then validation
        train_pools[cat] = [
            (row.question, row.ground_truth)
            for _, row in cat_data.head(n_train).iterrows()
        ]
        val_data.extend([
            (row.question, row.ground_truth, cat)
            for _, row in cat_data.iloc[n_train:n_train + n_val].iterrows()
        ])

    return train_pools, val_data


def _sealqa_scorer(question: str, predicted: str, ground_truth: str) -> float:
    """Wrapper around score_sealqa matching the runner's (question, predicted, ground_truth) signature."""
    return score_sealqa(question, ground_truth, predicted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-improving agent loop on SEAL-QA")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skill_only", "prompt_only"],
        default="skill_only",
        help="Evolution mode: 'skill_only' or 'prompt_only' (default: skill_only)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum number of improvement iterations (default: 20)",
    )
    parser.add_argument(
        "--frontier-size",
        type=int,
        default=3,
        help="Number of top-performing programs to keep (default: 3)",
    )
    parser.add_argument(
        "--no-improvement-limit",
        type=int,
        default=5,
        help="Stop after this many iterations without improvement (default: 5)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent evaluations (default: 4)",
    )
    parser.add_argument(
        "--failure-samples",
        type=int,
        default=3,
        help="Number of samples to test per iteration for pattern detection (default: 3)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable run caching",
    )
    parser.add_argument(
        "--no-reset-feedback",
        action="store_true",
        help="Don't reset feedback history on start",
    )
    parser.add_argument(
        "--continue",
        dest="continue_loop",
        action="store_true",
        help="Continue from existing frontier/branch instead of starting fresh",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to SEAL-QA CSV for ratio-based splitting (ignored when --train-dataset and --val-dataset are both set)",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default=None,
        help="Path to pre-split training CSV",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default=None,
        help="Path to pre-split validation CSV",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.18,
        help="Fraction of each category for training (only used with --dataset, default: 0.18)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.12,
        help="Fraction of each category for validation (only used with --dataset, default: 0.12)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["opus", "sonnet", "haiku"],
        default="claude-opus-4-5-20251101",
        help="Model for base agent (default: opus via SDK default)",
    )
    return parser.parse_args()


def normalize_sealqa(df: pd.DataFrame) -> pd.DataFrame:
    """Rename SEAL-QA columns to match stratified_split expectations."""
    df = df.copy()
    if "topic" in df.columns and "category" not in df.columns:
        df.rename(columns={"topic": "category"}, inplace=True)
    if "answer" in df.columns and "ground_truth" not in df.columns:
        df.rename(columns={"answer": "ground_truth"}, inplace=True)
    return df


def load_separate_datasets(
    train_path: str,
    val_path: str,
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
    """Load pre-split train/val CSVs into the formats expected by SelfImprovingLoop."""
    train_df = normalize_sealqa(pd.read_csv(train_path))
    val_df = normalize_sealqa(pd.read_csv(val_path))

    train_pools: dict[str, list[tuple[str, str]]] = {}
    for cat in train_df["category"].dropna().unique():
        cat_rows = train_df[train_df["category"] == cat]
        train_pools[cat] = [
            (row.question, row.ground_truth) for _, row in cat_rows.iterrows()
        ]

    val_data: list[tuple[str, str, str]] = [
        (row.question, row.ground_truth, row.category)
        for _, row in val_df.dropna(subset=["category"]).iterrows()
    ]

    return train_pools, val_data


async def main(args: argparse.Namespace):
    if args.train_dataset and args.val_dataset:
        train_pools, val_data = load_separate_datasets(
            args.train_dataset, args.val_dataset
        )
        print(f"Train dataset: {args.train_dataset}")
        print(f"Val dataset:   {args.val_dataset}")
    elif args.dataset:
        data = pd.read_csv(args.dataset)
        data = normalize_sealqa(data)
        train_pools, val_data = stratified_split(
            data, train_ratio=args.train_ratio, val_ratio=args.val_ratio
        )
        print(f"Dataset: {args.dataset}")
        print(
            f"Split ratios: train={args.train_ratio:.0%}, val={args.val_ratio:.0%} "
            f"(remaining {1-args.train_ratio-args.val_ratio:.0%} unused)"
        )
    else:
        raise ValueError(
            "Must provide either --train-dataset and --val-dataset, or --dataset"
        )

    categories = list(train_pools.keys())
    total_train = sum(len(pool) for pool in train_pools.values())
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print(f"Training pools: {', '.join(f'{cat}: {len(pool)}' for cat, pool in train_pools.items())}")
    print(f"Total training samples: {total_train}")
    print(f"Validation samples: {len(val_data)}")

    # Use custom model for sealqa agent if specified
    base_options = make_sealqa_agent_options(model=args.model) if args.model else sealqa_agent_options

    agents = LoopAgents(
        base=Agent(base_options, AgentResponse),
        skill_proposer=Agent(skill_proposer_options, SkillProposerResponse),
        prompt_proposer=Agent(prompt_proposer_options, PromptProposerResponse),
        skill_generator=Agent(skill_generator_options, ToolGeneratorResponse),
        prompt_generator=Agent(prompt_generator_options, PromptGeneratorResponse),
    )
    manager = ProgramManager(cwd=get_project_root())

    config = LoopConfig(
        max_iterations=args.max_iterations,
        frontier_size=args.frontier_size,
        no_improvement_limit=args.no_improvement_limit,
        concurrency=args.concurrency,
        evolution_mode=args.mode,
        failure_sample_count=args.failure_samples,
        categories_per_batch=args.failure_samples,
        cache_enabled=not args.no_cache,
        reset_feedback=not args.no_reset_feedback,
        continue_mode=args.continue_loop,
    )

    model_info = f", model={args.model}" if args.model else ""
    print(f"Running loop with evolution_mode={args.mode}{model_info}")
    loop = SelfImprovingLoop(config, agents, manager, train_pools, val_data, scorer=_sealqa_scorer)
    result = await loop.run()

    print(f"Best: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
