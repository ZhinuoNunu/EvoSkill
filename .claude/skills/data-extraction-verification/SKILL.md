---
name: data-extraction-verification
description: Rigorous protocol for extracting numerical data from Treasury Bulletin tables. ALWAYS use this skill when extracting ANY value from treasury_bulletins_parsed tables. Prevents common extraction errors including adjacent cell misreads, wrong metric selection, and incorrect time granularity. Required by brainstorming skill for all data lookups.
---

# Data Extraction Verification Protocol

Mandatory verification steps before, during, and after reading values from Treasury Bulletin tables.

## Pre-Extraction: Metric Identification

Before extracting ANY value, explicitly document:

1. **Table identification**: Exact table name/title (e.g., "Table SB-2: Sales and Redemptions by Periods")
2. **Column header**: Exact text as it appears in the table
3. **Row label**: Exact text as it appears in the table
4. **Metric match check**: Verify the column measures EXACTLY what the question asks
   - "Amount outstanding" ≠ "Sales and redemptions"
   - "Calendar year average" ≠ "Specific month value"
   - "Regional aggregate" ≠ "Individual country"

5. **Semantic definition verification** (MANDATORY before any extraction):

   Before extracting, explicitly define BOTH what the question asks for AND what the column measures:

   a) **Question semantic**: What does the question ask for?
      - "Sales and redemptions" = FLOWS (new issuances + redemptions in a period)
      - "Amount outstanding" = STOCK (cumulative balance at a point in time)
      - "Specific month value" = exact value for that month
      - "Annual average" = mean of 12 monthly values

   b) **Column semantic**: What does the column header actually measure?
      - Read column header exactly as written
      - Define what that metric type represents

   c) **Semantic match verification**:
      - Do the semantics match? Answer YES or NO explicitly
      - If NO: STOP. Find the correct column.
      - If uncertain: STOP. Re-read table structure.

   Common semantic confusions to guard against:
   | Question asks for | WRONG column | RIGHT column |
   |-------------------|--------------|--------------|
   | Sales and redemptions | Amount outstanding | Sales OR Redemptions columns |
   | Specific month value | Annual average | Monthly data row |
   | Individual country | Regional aggregate | Country-specific row |
   | Flow (period total) | Stock (point-in-time balance) | Period totals column |

⚠️ **STOP CONDITION**: If semantic match verification fails, DO NOT proceed with extraction. Re-examine the table to find the column that semantically matches the question's requirement.

## Extraction: Cell Verification

When reading the value:

1. **Record cell position**: "Row: '[exact label]', Column: '[exact header]'"
2. **Read value with units**: Note millions, billions, percent, etc.
3. **Read adjacent context**: Values one row above AND one row below
4. **Triple check**:
   - Is this the correct row?
   - Is this the correct column?
   - Are the units correct?

## Post-Extraction: Cross-Verification

After extraction:

1. **Compare against totals**: Does value fit within any table subtotals/totals?
2. **Check progression**: For time series, does value follow expected pattern?
3. **Re-extract**: Read the table again independently to confirm the value
4. **Alternative source**: If available, verify against a different table

## Verification Output Format

```
## Data Extraction Verification
Table: [exact table name]
Target Metric: [what question asks for]
Question Semantic: [FLOW/STOCK/POINT-VALUE/AVERAGE - define what's requested]
Column Selected: [exact column header]
Column Semantic: [FLOW/STOCK/POINT-VALUE/AVERAGE - define what column measures]
SEMANTIC MATCH: [YES/NO - explicit verification]
Row Selected: [exact row label]
Extracted Value: [value with units]
Adjacent Context:
  - Row above: [value]
  - Row below: [value]
Cross-Verification: [how verified - totals, re-read, alternative source]
```

## Common Pitfalls

| Error Type | Example | Prevention |
|------------|---------|------------|
| Adjacent cell | UK: 103,235 vs 103,375 | Always record and verify adjacent rows |
| Wrong metric | "Amount outstanding" when asked for "Sales and redemptions" | Explicit metric match check |
| Wrong granularity | Calendar year average vs specific month | Verify time period matches question |
| Wrong aggregation | Regional total vs individual country | Confirm geographic scope |
| Count mismatch | 119 values extracted instead of 120 | Track running count, verify against expected |
| Cumulative drift | Small errors compound to 4% aggregate error | Verify against subtotals at checkpoints |
| Block navigation | Misaligned row/column in multi-block table | Map table structure explicitly before extraction |

---

## Bulk Extraction Protocol (for aggregation tasks)

For scenarios requiring extraction of many values (>10) that will be aggregated.

### Pre-Bulk Extraction Setup

1. **Count expected values**: Calculate before starting (e.g., "120 months = 10 years × 12 months")
2. **Identify checkpoints**: Find any subtotals or totals in the table that can serve as verification points
3. **Map table structure**: Document explicitly (e.g., "3 row blocks covering years X-Y, 4 columns per block")

### Incremental Verification During Extraction

1. **Track running count**: Maintain count of extracted values
2. **Verify at logical groups**: After each year, row block, or section, verify count matches expected
3. **Verify against subtotals**: If table has subtotals, verify running sum matches before proceeding

### Sample Verification

For large extractions (>20 values):
1. **Spot-check at least 3 random values** by re-extracting independently
2. **If any spot-check fails**, re-extract the entire dataset for that section

### Post-Aggregation Sanity Checks

1. **Compare against expected range**: E.g., "average yield spread typically 0.5-1.5%"
2. **Verify count**: "I extracted N values, expected M" - mismatch indicates navigation error
3. **Cross-validate paths**: If multiple aggregation paths exist (e.g., sum of row totals vs sum of individual cells), verify they match

### Bulk Extraction Verification Output Format

```
## Bulk Extraction Verification
Table: [table name]
Expected count: [N values]
Extracted count: [N values] ✓/✗
Subtotal checkpoints: [list checkpoint verifications]
Sample verification: [3 spot-checks with pass/fail]
Final aggregate: [value]
Sanity check: [reasonable range? Y/N]
```
