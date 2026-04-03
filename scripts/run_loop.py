#!/usr/bin/env python3
"""Run self-improving agent loop."""

import asyncio
from typing import Literal, Optional

import pandas as pd
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.agent_profiles import (
    Agent,
    base_agent_options,
    make_base_agent_options,
    skill_proposer_options,
    prompt_proposer_options,
    skill_generator_options,
    prompt_generator_options,
    set_sdk,
)
from src.agent_profiles.skill_generator import get_project_root
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)


class LoopSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=True,
        title="Run self-improving agent loop",
    )

    mode: Literal["skill_only", "prompt_only"] = Field(
        default="skill_only",
        description="Evolution mode: 'skill_only' or 'prompt_only'",
    )
    max_iterations: int = Field(
        default=20, description="Maximum number of improvement iterations"
    )
    frontier_size: int = Field(
        default=3, description="Number of top-performing programs to keep"
    )
    no_improvement_limit: int = Field(
        default=5, description="Stop after this many iterations without improvement"
    )
    concurrency: int = Field(default=4, description="Number of concurrent evaluations")
    failure_samples: int = Field(
        default=3,
        description="Number of samples to test per iteration for pattern detection",
    )
    cache: bool = Field(default=True, description="Enable run caching")
    reset_feedback: bool = Field(
        default=True, description="Reset feedback history on start"
    )
    continue_loop: bool = Field(
        default=False,
        description="Continue from existing frontier/branch instead of starting fresh",
    )
    dataset: Optional[str] = Field(
        default=None,
        description=(
            "CSV with question + (ground_truth or answer); "
            "category optional (uses difficulty if present, else 'default'). "
            "Used with train_ratio/val_ratio for ratio-based splitting. "
            "Ignored when train_dataset and val_dataset are both provided."
        ),
    )
    train_dataset: Optional[str] = Field(
        default=None,
        description="Path to training CSV (question, ground_truth/answer, category/difficulty)",
    )
    val_dataset: Optional[str] = Field(
        default=None,
        description="Path to validation CSV (question, ground_truth/answer, category/difficulty)",
    )
    num_samples: Optional[int] = Field(
        default=None,
        description="Use only the first N rows after loading (smoke tests)",
    )
    train_ratio: float = Field(
        default=0.18, description="Fraction of each category for training (only used with --dataset)"
    )
    val_ratio: float = Field(
        default=0.12, description="Fraction of each category for validation (only used with --dataset)"
    )
    val_count: Optional[int] = Field(
        default=None, description="Override total validation count"
    )
    model: Optional[str] = Field(
        default=None, description="Model for base agent (opus, sonnet, haiku)"
    )
    sdk: Literal["claude", "opencode"] = Field(
        default="claude",
        description="SDK to use: 'claude' or 'opencode'",
    )


def normalize_dataset_for_loop(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure question, ground_truth, category for stratified_split."""
    df = data.copy()
    if "question" not in df.columns:
        raise ValueError(
            f"Dataset must have 'question' column (found: {list(df.columns)!r})"
        )
    if "ground_truth" not in df.columns:
        if "answer" in df.columns:
            df["ground_truth"] = df["answer"]
        else:
            raise ValueError(
                f"Dataset needs 'ground_truth' or 'answer' (found: {list(df.columns)!r})"
            )
    if "category" not in df.columns:
        if "difficulty" in df.columns:
            df["category"] = df["difficulty"].astype(str)
        else:
            df["category"] = "default"
    return df


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
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) cannot exceed 1.0"
        )

    # Drop rows with missing categories
    data = data.dropna(subset=["category"])
    categories = data["category"].unique()
    train_pools: dict[str, list[tuple[str, str]]] = {}
    val_data: list[tuple[str, str, str]] = []

    for cat in categories:
        cat_data = data[data["category"] == cat].sample(frac=1, random_state=42)
        n = len(cat_data)
        if n == 0:
            continue
        if n == 1:
            row = cat_data.iloc[0]
            train_pools[cat] = [(row.question, row.ground_truth)]
            val_data.append((row.question, row.ground_truth, cat))
            continue

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(1, n - n_train)
            if n_train + n_val > n:
                n_train = n - n_val

        train_pools[cat] = [
            (row.question, row.ground_truth)
            for _, row in cat_data.head(n_train).iterrows()
        ]
        val_data.extend(
            [
                (row.question, row.ground_truth, cat)
                for _, row in cat_data.iloc[n_train : n_train + n_val].iterrows()
            ]
        )

    return train_pools, val_data


def load_separate_datasets(
    train_path: str,
    val_path: str,
    num_samples: Optional[int] = None,
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
    """Load pre-split train/val CSVs into the formats expected by SelfImprovingLoop."""
    train_df = normalize_dataset_for_loop(pd.read_csv(train_path))
    val_df = normalize_dataset_for_loop(pd.read_csv(val_path))

    if num_samples is not None:
        train_df = train_df.head(num_samples)
        val_df = val_df.head(num_samples)

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


async def main(settings: LoopSettings):
    # Set SDK based on CLI argument
    set_sdk(settings.sdk)

    if settings.train_dataset and settings.val_dataset:
        train_pools, val_data = load_separate_datasets(
            settings.train_dataset,
            settings.val_dataset,
            num_samples=settings.num_samples,
        )
        ds_note = f" (head {settings.num_samples})" if settings.num_samples else ""
        print(f"Train dataset: {settings.train_dataset}{ds_note}")
        print(f"Val dataset:   {settings.val_dataset}{ds_note}")
    elif settings.dataset:
        data = pd.read_csv(settings.dataset)
        data = normalize_dataset_for_loop(data)
        if settings.num_samples is not None:
            data = data.head(settings.num_samples)
        train_pools, val_data = stratified_split(
            data, train_ratio=settings.train_ratio, val_ratio=settings.val_ratio
        )
        ds_note = f" (head {settings.num_samples})" if settings.num_samples else ""
        print(f"Dataset: {settings.dataset}{ds_note}")
        print(
            f"Split ratios: train={settings.train_ratio:.0%}, val={settings.val_ratio:.0%} (remaining {1 - settings.train_ratio - settings.val_ratio:.0%} unused)"
        )
    else:
        raise ValueError(
            "Must provide either --train_dataset and --val_dataset, or --dataset"
        )

    categories = list(train_pools.keys())
    total_train = sum(len(pool) for pool in train_pools.values())
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print(
        f"Training pools: {', '.join(f'{cat}: {len(pool)}' for cat, pool in train_pools.items())}"
    )
    print(f"Total training samples: {total_train}")
    print(f"Validation samples: {len(val_data)}")

    # Use custom model for base agent if specified
    base_options = (
        make_base_agent_options(model=settings.model)
        if settings.model
        else base_agent_options
    )

    agents = LoopAgents(
        base=Agent(base_options, AgentResponse),
        skill_proposer=Agent(skill_proposer_options, SkillProposerResponse),
        prompt_proposer=Agent(prompt_proposer_options, PromptProposerResponse),
        skill_generator=Agent(skill_generator_options, ToolGeneratorResponse),
        prompt_generator=Agent(prompt_generator_options, PromptGeneratorResponse),
    )
    manager = ProgramManager(cwd=get_project_root())

    config = LoopConfig(
        max_iterations=settings.max_iterations,
        frontier_size=settings.frontier_size,
        no_improvement_limit=settings.no_improvement_limit,
        concurrency=settings.concurrency,
        evolution_mode=settings.mode,
        failure_sample_count=settings.failure_samples,
        categories_per_batch=settings.failure_samples,  # Sample from N different categories
        cache_enabled=settings.cache,
        reset_feedback=settings.reset_feedback,
        continue_mode=settings.continue_loop,
    )

    model_info = f", model={settings.model}" if settings.model else ""
    print(f"Running loop with evolution_mode={settings.mode}{model_info}")
    loop = SelfImprovingLoop(config, agents, manager, train_pools, val_data)
    result = await loop.run()

    print(f"Best: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")


if __name__ == "__main__":
    settings = LoopSettings()
    asyncio.run(main(settings))
