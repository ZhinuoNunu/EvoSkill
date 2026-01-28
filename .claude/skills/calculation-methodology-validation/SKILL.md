---
name: calculation-methodology-validation
description: Verify calculation methodology AFTER data extraction but BEFORE reporting final answers. Catches formula selection errors (log vs simple growth, median vs mean deviation), multi-step calculation errors, and regression validity issues. Use when performing statistical metrics (mean deviation, variance), growth rates, regression analysis, or any multi-step calculation.
---

# Calculation Methodology Validation

## Purpose

Catch methodology errors where correct data extraction leads to wrong answers due to incorrect formulas, wrong calculation methods, or invalid statistical procedures.

**Position in workflow:** After data-extraction-verification, before final answer.

## Formula-Question Alignment (Critical)

Before calculating, verify formula matches what question asks:

| Question asks for | Use | NOT |
|------------------|-----|-----|
| "log growth rate" | ln(V2/V1) | (V2-V1)/V1 |
| "simple growth rate" | (V2-V1)/V1 | ln(V2/V1) |
| "CAGR" | (V2/V1)^(1/n) - 1 | simple or log |
| "mean deviation from median" | Σ\|xi - median\|/n | Σ\|xi - mean\|/n |
| "mean deviation from mean" | Σ\|xi - mean\|/n | Σ\|xi - median\|/n |
| "variance" | Σ(xi - mean)²/(n-1) for sample | dividing by n |

State explicitly: "Question asks for [X], using formula [Y]"

## Value Count Verification

Before computing:
- Count values: "Working with n=[X] data points"
- List values explicitly: "[v1, v2, v3, ...]"
- If calculating a statistic, verify all expected values are present

## Magnitude Sanity Checks

### Growth Rates
- Input ratio ~2.4 → log growth ~87%, simple growth ~140%
- Input ratio ~1.5 → log growth ~41%, simple growth ~50%
- Sign check: Growth positive if V2 > V1

### Deviations
- Mean deviation must be < range of data
- If median=8.7, values span 4.8-10.0, deviations sum < 5.2 × n

### General
- If result seems implausible for context, STOP and re-verify inputs

## Regression Validity

**Critical: n=2 is NOT a statistical regression**
- With 2 points, the line passes through both exactly—this is interpolation, not fitting
- With n<5, note limited statistical validity

Before reporting regression results:
- State: "Regression based on n=[X] points"
- If n=2: "Note: exact fit through 2 points, not statistical model"
- Verify coefficients are contextually plausible

## Multi-Step Calculation Tracking

For calculations with >2 sequential operations:

1. Document each intermediate result
2. Before proceeding to next step, verify current result is plausible
3. State the chain: "Step 1: [result] → Step 2: [result] → Final: [result]"

Example for counterfactual analysis:
```
Step 1: Calculate growth rate → [X]%
Step 2: Apply to base value → [Y]
Step 3: Compute difference → [Z]
Verify: Each intermediate is plausible before proceeding
```

## Quick Checklist

```
□ Formula matches what question asked? (log/simple, median/mean)
□ All input values present? Count = [n]
□ For regression: n > 2? If n=2, noted as exact fit?
□ Intermediate calculations documented?
□ Each intermediate result plausible?
□ Final result magnitude consistent with inputs?
```

Complete this checklist before reporting any calculated answer.
