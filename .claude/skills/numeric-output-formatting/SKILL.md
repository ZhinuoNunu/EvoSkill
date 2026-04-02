---
name: numeric-output-formatting
description: Format numeric answers concisely based on question context. Use when outputting numbers, currency values, percentages, or any quantitative answer to ensure proper formatting. Triggers on any query requesting numeric data, financial figures, statistics, or measurements.
---

# Numeric Output Formatting

Guidelines for formatting numeric answers based on what the question asks.

## Default Rule

Output ONLY the numeric value unless the question explicitly asks for units or descriptive text.

**Correct:**
- Question: "What were total expenditures?" → Answer: `507`
- Question: "What was the amount in millions?" → Answer: `507`
- Question: "What is the percentage?" → Answer: `12.5`

## Context Detection

Before outputting, determine what the question asks for:

| Question asks for... | Output format | Example |
|---------------------|---------------|---------|
| Raw value / number / amount | Just digits | `507` |
| Value "in millions/billions" | Just digits (units implied) | `507` |
| Amount with units | Number + units | `507 million dollars` |
| Descriptive answer | Full sentence | `The expenditure was 507 million dollars.` |

## Financial Data Conventions

For treasury/fiscal/budget data:
- Output just the number when units are implied by context (millions/billions)
- No currency symbols ($, €, £) unless specifically requested
- No labels like "dollars" or "million" unless required by the question

**Correct:**
- Question: "What were total outlays?" → Answer: `2847` (context: millions)
- Question: "State the amount with units" → Answer: `2847 million dollars`

## Verification Step

Before final output, ask: **"Does this answer contain ONLY what was asked for?"**

Remove:
- Extra words ("The total is...", "The answer is...")
- Unrequested units ("dollars", "million")
- Unrequested symbols ("$", "%")
- Explanations unless asked

## Quick Reference

| Question pattern | Output |
|-----------------|--------|
| "What is/was/are..." | Just the number |
| "How much..." | Just the number |
| "State the amount with units" | Number + units |
| "Express the answer as..." | Follow the specified format |
