#!/usr/bin/env python3
"""Run self-improving agent loop on a dataset."""

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from src.agent_profiles import (
    Agent,
    base_agent_options,
    proposer_options,
    skill_generator_options,
    prompt_generator_options,
)
from src.agent_profiles.skill_generator import get_project_root
from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    ProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)


async def main():
    parser = argparse.ArgumentParser(
        description="Run self-improving agent loop on OfficeQA dataset"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        required=True,
        help="Path to dataset CSV file",
    )
    parser.add_argument(
        "--max-iterations",
        "-n",
        type=int,
        default=5,
        help="Maximum number of improvement iterations (default: 5)",
    )
    parser.add_argument(
        "--frontier-size",
        "-f",
        type=int,
        default=3,
        help="Number of top programs to keep in frontier (default: 3)",
    )
    parser.add_argument(
        "--no-improvement-limit",
        type=int,
        default=5,
        help="Stop after this many iterations without improvement (default: 5)",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "hard", "all"],
        default="hard",
        help="Filter dataset by difficulty (default: hard)",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=5,
        help="Number of training samples to use (default: 5)",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=3,
        help="Number of validation samples to use (default: 3)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=2,
        help="Max concurrent evaluations (default: 2)",
    )
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=0.0,
        help="Tolerance for answer matching (default: 0.0)",
    )
    args = parser.parse_args()

    # Load dataset
    dataset_path = args.dataset.expanduser()
    data = pd.read_csv(dataset_path)

    # Filter by difficulty
    if args.difficulty != "all":
        data = data[data["difficulty"] == args.difficulty]

    print(f"Dataset: {len(data)} samples ({args.difficulty})")

    # Prepare train/val splits
    train_data = [
        (row["question"], row["answer"])
        for _, row in data.head(args.train_samples).iterrows()
    ]
    val_data = [
        (row["question"], row["answer"])
        for _, row in data.tail(args.val_samples).iterrows()
    ]

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    # Create agents
    agents = LoopAgents(
        base=Agent(base_agent_options, AgentResponse),
        proposer=Agent(proposer_options, ProposerResponse),
        skill_generator=Agent(skill_generator_options, ToolGeneratorResponse),
        prompt_generator=Agent(prompt_generator_options, PromptGeneratorResponse),
    )

    # Create manager
    manager = ProgramManager(cwd=get_project_root())

    # Create config
    config = LoopConfig(
        max_iterations=args.max_iterations,
        frontier_size=args.frontier_size,
        no_improvement_limit=args.no_improvement_limit,
        tolerance=args.tolerance,
        concurrency=args.concurrency,
    )

    # Run loop
    loop = SelfImprovingLoop(config, agents, manager, train_data, val_data)
    result = await loop.run()

    # Print results
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Iterations completed: {result.iterations_completed}")
    print(f"\nFrontier (best agents):")
    for name, score in result.frontier:
        print(f"  - {name}: {score:.4f}")
    print(f"\nBest agent: {result.best_program}")
    print(f"Best score: {result.best_score:.4f}")

    # Switch back to main
    manager._git_checkout("main")
    print(f"\nSwitched to: {manager._git_current_branch()}")


if __name__ == "__main__":
    asyncio.run(main())
