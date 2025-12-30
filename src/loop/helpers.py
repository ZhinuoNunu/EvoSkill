"""Helper functions for the self-improving loop."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent_profiles.base import AgentTrace
    from src.schemas import ProposerResponse


def build_proposer_query(
    trace: "AgentTrace", ground_truth: str, feedback_history: str
) -> str:
    """Build the query for the proposer agent.

    Args:
        trace: The agent trace from the failed attempt.
        ground_truth: The correct answer.
        feedback_history: Previous feedback history.

    Returns:
        Formatted query string for the proposer.
    """
    if trace.output is None:
        agent_answer = f"[PARSE FAILED: {trace.parse_error}]"
    else:
        agent_answer = trace.output.final_answer

    return f"""## Previous Attempts Feedback
{feedback_history}

## Current Attempt
{trace.summarize()}

Agent Answer: {agent_answer}
Ground Truth: {ground_truth}"""


def build_skill_query(proposer_trace: "AgentTrace[ProposerResponse]") -> str:
    """Build the query for the skill generator agent.

    Args:
        proposer_trace: The trace from the proposer agent.

    Returns:
        Formatted query string for the skill generator.
    """
    return f"""Proposed tool or skill (high level description): {proposer_trace.output.proposed_skill_or_prompt}

Justification: {proposer_trace.output.justification}"""


def build_prompt_query(
    proposer_trace: "AgentTrace[ProposerResponse]", original_prompt: str
) -> str:
    """Build the query for the prompt generator agent.

    Args:
        proposer_trace: The trace from the proposer agent.
        original_prompt: The original system prompt to optimize.

    Returns:
        Formatted query string for the prompt generator.
    """
    return f"""## Original Prompt
{original_prompt}

## Proposed Change
{proposer_trace.output.proposed_skill_or_prompt}

## Justification
{proposer_trace.output.justification}"""


def append_feedback(
    path: Path, iteration: str, skill: str, justification: str
) -> None:
    """Append feedback entry to history file.

    Args:
        path: Path to the feedback history file.
        iteration: Iteration identifier (e.g., "iter-1").
        skill: The skill or prompt that was proposed.
        justification: Why this change was proposed.
    """
    entry = f"""
## {iteration}
**Skill or Prompt**: {skill}
**Justification**: {justification}

"""
    with open(path, "a") as f:
        f.write(entry)


def read_feedback_history(path: Path) -> str:
    """Read feedback history or return default message.

    Args:
        path: Path to the feedback history file.

    Returns:
        Contents of feedback file or default message.
    """
    if path.exists():
        return path.read_text()
    return "No previous attempts."


def update_prompt_file(file_path: Path, new_prompt: str) -> None:
    """Write the new prompt to prompt.txt.

    The Agent reads this file at runtime on each run().

    Args:
        file_path: Path to the prompt file.
        new_prompt: The new prompt content.
    """
    file_path.write_text(new_prompt.strip())
