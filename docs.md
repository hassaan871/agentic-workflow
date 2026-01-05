# QC Pipeline - Developer Documentation

## Table of Contents
1. [Overview](#overview)
2. [Feature Description](#feature-description)
3. [Architecture & Flow](#architecture--flow)
4. [Code Structure](#code-structure)
5. [Key Components](#key-components)
6. [Data Flow](#data-flow)
7. [Implementation Details](#implementation-details)
8. [Usage](#usage)
9. [Output Format](#output-format)
10. [Taxonomy Selection](#taxonomy-selection)

---

## Overview

The QC (Question Correction) Pipeline is an agentic workflow system designed to generate and refine prompts that test model-breaking scenarios. The system uses an iterative refinement approach where prompts are continuously improved until they successfully break the model (cause 3+ criteria to fail 3+ times out of 4 attempts).

### Key Features
- **Taxonomy Selection**: Interactive prompt to select which taxonomy to use (currently Question Correction)
- **Iterative Refinement**: Automatically refines prompts that don't break the model
- **Criteria-Level Analysis**: Tracks which specific criteria pass/fail across multiple attempts
- **Progressive Improvement**: Targets specific criteria that need improvement while maintaining those already failing
- **Selective Storage**: Only saves model-breaking prompts to CSV

---

## Feature Description

### What It Does
1. **Selects Taxonomy** - Interactive prompt to choose which taxonomy to use (default: Question Correction)
2. **Generates** prompts based on selected taxonomy (e.g., Question Correction prompts with hidden flaws)
3. **Tests** each prompt by having Agent02 respond 4 times
4. **Evaluates** each response against multiple criteria using a Judge
5. **Analyzes** which criteria are passing/failing
6. **Refines** prompts that don't break the model (if < 3 criteria failing 3+ times)
7. **Saves** only model-breaking prompts to CSV

### Success Condition
A prompt is considered "model breaking" when:
- **3 or more criteria** fail **3 or more times** out of 4 attempts
- Example: C1 fails 3/4, C2 fails 4/4, C3 fails 3/4 → Model Breaking ✅

---

## Architecture & Flow

### High-Level Flow

```
START
  ↓
TAXONOMY SELECTION (Interactive Prompt)
  - User selects taxonomy (default: 1 for Question Correction)
  - System configures prompt and file based on selection
  ↓
FOR EACH RUN (--runs):
  │
  ├─ ITERATION LOOP (max: --max-iterations):
  │   │
  │   ├─ ITERATION 1:
  │   │   ├─ Agent01: Generate NEW prompt + criteria
  │   │   ├─ Agent02: Respond 4 times
  │   │   ├─ Judge: Evaluate 4 times
  │   │   ├─ Analyze: Which criteria pass/fail?
  │   │   └─ Check: Model breaking?
  │   │       ├─ YES → Save to CSV → DONE ✅
  │   │       └─ NO → Continue to Iteration 2
  │   │
  │   ├─ ITERATION 2+:
  │   │   ├─ create_refinement_feedback(): Analyze results
  │   │   ├─ Agent01: Refine prompt + criteria (using feedback)
  │   │   ├─ Agent02: Respond 4 times (to new prompt)
  │   │   ├─ Judge: Evaluate 4 times (with new criteria)
  │   │   ├─ Analyze: Which criteria pass/fail?
  │   │   └─ Check: Model breaking?
  │   │       ├─ YES → Save to CSV → DONE ✅
  │   │       └─ NO → Continue to next iteration
  │   │
  │   └─ Exit Conditions:
  │       ├─ Success: 3+ criteria failing 3+ times
  │       └─ Max Iterations: Reached --max-iterations limit
  │
  └─ NEXT RUN
```

### Detailed Iteration Flow

```
┌─────────────────────────────────────────────────────────┐
│ ITERATION {n}                                            │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 1: Generate/Refine Prompt                           │
│   - If iteration == 1: Use SYSTEM_PROMPT (selected taxonomy)│
│   - If iteration > 1: Use create_refinement_feedback()   │
│   - Agent01 outputs: prompt, criteria, correct_response  │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 2: Test Prompt (4 attempts)                        │
│   FOR attempt in [1, 2, 3, 4]:                           │
│     - Agent02 responds to prompt                        │
│     - Judge evaluates response against criteria         │
│     - Store: response, judge_output, status             │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: Analyze Results                                  │
│   - Parse judge outputs: Extract per-criteria PASS/FAIL │
│   - Count failures per criteria: {C1: 2, C2: 3, ...}  │
│   - Calculate: How many criteria failing 3+ times?      │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: Decision Point                                   │
│   IF 3+ criteria failing 3+ times:                      │
│     → Save to CSV → BREAK loop → DONE ✅                 │
│   ELSE:                                                   │
│     → Prepare feedback → Continue to next iteration     │
└─────────────────────────────────────────────────────────┘
```

---

## Code Structure

### File Organization

```
qc.py
├── Imports & Configuration
│   ├── Standard libraries (os, json, csv, argparse)
│   ├── OpenAI client setup
│   └── File paths and constants
│
├── Constants & Templates
│   ├── CRITERIA_DESIGN_RULES
│   ├── PROMPT_HEADER
│   ├── SYSTEM_PROMPT_QC
│   ├── JUDGE_PROMPT_TEMPLATE
│   └── OUTPUT_FORMAT_NOTE
│
├── Helper Functions
│   ├── select_taxonomy()
│   ├── parse_criteria_from_judge()
│   ├── get_criteria_text()
│   └── create_refinement_feedback()
│
├── CLI Argument Parsing
│   ├── --runs
│   └── --max-iterations
│
├── Taxonomy Selection
│   └── Interactive prompt to select taxonomy
│
├── System Configuration
│   ├── Set SYSTEM_PROMPT based on taxonomy
│   ├── Set file_name based on taxonomy
│   └── Update file_path
│
├── CSV Setup
│   └── Initialize CSV file with headers (based on taxonomy)
│
└── Main Execution Loop
    ├── FOR each run
    │   └── WHILE iteration < max_iterations
    │       ├── Generate/Refine prompt
    │       ├── Test (4 attempts)
    │       ├── Analyze results
    │       └── Save or continue
```

### Key Constants

#### `CRITERIA_DESIGN_RULES`
- **Purpose**: Ensures criteria are non-overlapping and self-contained
- **Used in**: Both initial generation and refinement prompts
- **Rules**:
  - Each criterion evaluates single, independent behavior
  - No logical consequence between criteria
  - No rephrasing same judgment
  - Self-contained, no overlap

#### `PROMPT_HEADER`
- **Purpose**: Base template for Agent01 prompt generation
- **Contains**: Task description, output format, criteria design rules
- **Note**: Includes "Generate NEW task" instruction (only for iteration 1)

#### `OUTPUT_FORMAT_NOTE`
- **Purpose**: Output format template for refinement (no "generate NEW")
- **Used in**: `create_refinement_feedback()` function
- **Difference**: Doesn't include "generate NEW" instruction

---

## Key Components

### 1. Helper Functions

#### `select_taxonomy()`
**Purpose**: Interactive prompt for taxonomy selection

**Input**: None (reads from user input)

**Output**: Taxonomy identifier string (e.g., "qc" for Question Correction)

**How it works**:
1. Displays available taxonomies with numbers and descriptions
2. Prompts user to enter taxonomy number (default: 1)
3. Validates input - only accepts valid taxonomy numbers
4. Shows error and exits if invalid input
5. Returns taxonomy identifier for system configuration

**Example**:
```
======================================================================
                    TAXONOMY SELECTION
======================================================================

Available Taxonomies:
----------------------------------------------------------------------
  [1] Question Correction (QC)
      Questions containing logical fallacies, factual errors, or inconsistencies where all provided options are incorrect

======================================================================
Enter taxonomy number (default: 1): 1

======================================================================
✓ Selected Taxonomy: Question Correction (QC)
======================================================================
```

#### `parse_criteria_from_judge(judge_output, criteria_id)`
**Purpose**: Extract PASS/FAIL status for a specific criteria from judge output

**Input**:
- `judge_output`: String containing judge's evaluation
- `criteria_id`: Criteria ID (e.g., "C1", "C2")

**Output**: `"PASS"`, `"FAIL"`, or `None`

**How it works**:
1. Searches for "Grading Basis:" section in judge output
2. Extracts JSON object: `{"C1": "PASS", "C2": "FAIL", ...}`
3. Returns status for given criteria_id
4. Fallback: Text search if JSON parsing fails

**Example**:
```python
judge_output = """
Grading Basis:
{"C1": "PASS", "C2": "FAIL", "C3": "FAIL"}
"""
status = parse_criteria_from_judge(judge_output, "C1")
# Returns: "PASS"
```

#### `get_criteria_text(data_qc, criteria_id)`
**Purpose**: Extract criteria text from data_qc structure

**Input**:
- `data_qc`: Dictionary containing prompt data
- `criteria_id`: Criteria ID to find

**Output**: Criteria text string or None

**How it works**:
1. Searches `response_reference` array in data_qc
2. Finds criteria with matching `id`
3. Returns criteria text (tries multiple possible keys: "criteria", "criteria1", etc.)

#### `create_refinement_feedback(data_qc, criteria_failures, judge_responses, nemotron_responses)`
**Purpose**: Create feedback prompt for Agent01 to refine prompt and criteria

**Input**:
- `data_qc`: Current prompt, criteria, correct_response
- `criteria_failures`: Dict of {criteria_id: fail_count}
- `judge_responses`: List of 4 judge outputs
- `nemotron_responses`: List of 4 Agent02 responses

**Output**: Formatted feedback prompt string

**How it works**:
1. **Separates criteria into two groups**:
   - `needs_improvement`: Criteria failing < 3 times
   - `keep_intact`: Criteria failing 3+ times

2. **Builds feedback prompt**:
   - Current prompt, criteria, correct_response
   - Test results analysis
   - Instructions for each criteria group
   - Criteria design rules
   - Task instructions

3. **Returns**: Complete feedback prompt for Agent01

**Example Output Structure**:
```
You are refining a Question Correction prompt...

CURRENT PROMPT: [prompt text]
CURRENT CRITERIA: [criteria JSON]
CURRENT CORRECT RESPONSE: [response text]

CRITERIA TO IMPROVE:
- C1: Currently failing 2/4 times
  Action: Make prompt harder, make criteria stricter

CRITERIA TO KEEP INTACT:
- C2: Currently failing 3/4 times ✅
  Action: Keep prompt constraints, keep criteria unchanged

[CRITERIA_DESIGN_RULES]

YOUR TASK:
1. For criteria needing improvement: Update prompt + make criteria stricter
2. For criteria working well: Keep unchanged
3. Update correct_response
4. Follow design rules
```

### 2. Main Execution Loop

#### Iteration Logic

```python
for run_idx in range(RUNS):
    iteration = 0
    best_criteria_count = 0
    previous_total_failing = 0
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        
        # Step 1: Generate or refine
        if iteration == 1:
            agent01_input = SYSTEM_PROMPT  # Generate new (based on selected taxonomy)
        else:
            agent01_input = create_refinement_feedback(...)  # Refine
        
        # Step 2: Get prompt from Agent01
        response = client.responses.create(model="...", input=agent01_input)
        data_qc = json.loads(response.output_text)
        
        # Step 3: Test prompt (4 attempts)
        # ... testing code ...
        
        # Step 4: Analyze criteria-level failures
        criteria_failures = {}
        for criteria_id in criteria_list:
            count = 0
            for judge_output in judge_responses:
                if parse_criteria_from_judge(judge_output, criteria_id) == "FAIL":
                    count += 1
            criteria_failures[criteria_id] = count
        
        # Step 5: Check success
        total_failing = sum(1 for c in criteria_failures.values() if c >= 3)
        if total_failing >= 3:
            # Success! Save and break
            save_to_csv()
            break
        else:
            # Continue to next iteration
            continue
```

---

## Data Flow

### Data Structures

#### `data_qc` (Agent01 Output)
```python
{
    "taxonomy": "Question Correction",
    "prompt": "Which vitamin is crucial for...",
    "correct_response": "The correct answer is Vitamin D...",
    "response_reference": [
        {"id": "C1", "criteria": "Does the response state..."},
        {"id": "C2", "criteria": "Does the response provide..."},
        ...
    ]
}
```

#### `criteria_failures` (Analysis Result)
```python
{
    "C1": 2,  # Failed 2 out of 4 times
    "C2": 3,  # Failed 3 out of 4 times
    "C3": 4,  # Failed 4 out of 4 times
    "C4": 1   # Failed 1 out of 4 times
}
```

#### CSV Row Structure
```python
{
    "prompt": "Which vitamin...",
    "correct_response": "The correct answer...",
    "response_reference": "[{\"id\": \"C1\", ...}]",  # JSON string
    "model": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
    "nemotron_response": "{\"attempt_1\": \"...\", \"attempt_2\": \"...\", ...}",  # JSON string
    "judge_response": "{\"attempt_1\": {\"judge_output\": \"...\", \"status\": \"FAIL\"}, ...}",  # JSON string
    "status": "FAIL"  # Always "FAIL" (only model-breaking saved)
}
```

### Data Transformation Flow

```
Agent01 Output (JSON)
    ↓
data_qc (dict)
    ↓
    ├─→ prompt → Agent02 (4 times)
    ├─→ response_reference → Judge (4 times)
    └─→ correct_response → (reference)
            ↓
    Agent02 Responses (4 strings)
            ↓
    Judge Outputs (4 strings)
            ↓
    parse_criteria_from_judge() → criteria_failures (dict)
            ↓
    Analysis → total_failing_criteria (int)
            ↓
    IF total_failing >= 3:
        → Save to CSV
    ELSE:
        → create_refinement_feedback() → New Agent01 Input
            ↓
        (Loop back to Agent01)
```

---

## Implementation Details

### Criteria-Level Analysis

**Process**:
1. For each of 4 attempts, judge evaluates response
2. Judge outputs per-criteria PASS/FAIL: `{"C1": "PASS", "C2": "FAIL", ...}`
3. `parse_criteria_from_judge()` extracts status for each criteria
4. Count failures across all 4 attempts
5. Result: `{"C1": 2, "C2": 3, "C3": 4, "C4": 1}`

**Example**:
```
Attempt 1: C1=PASS, C2=FAIL, C3=FAIL, C4=PASS
Attempt 2: C1=FAIL, C2=PASS, C3=FAIL, C4=PASS
Attempt 3: C1=FAIL, C2=FAIL, C3=FAIL, C4=FAIL
Attempt 4: C1=PASS, C2=FAIL, C3=FAIL, C4=PASS

Result:
C1: Failed 2/4 times
C2: Failed 3/4 times ✅
C3: Failed 4/4 times ✅
C4: Failed 1/4 times

Total criteria failing 3+ times: 2 (C2, C3)
→ Not model breaking (need 3+)
→ Refine prompt
```

### Refinement Strategy

**Targeted Improvement**:
- **Improve**: Criteria failing < 3 times
  - Make prompt harder (add complexity, misleading context)
  - Make criteria stricter (harder to pass)
- **Maintain**: Criteria failing 3+ times
  - Keep prompt constraints that make them fail
  - Keep criteria unchanged

**Example Refinement**:
```
Iteration 1:
  C1: 2/4 failures → Needs improvement
  C2: 3/4 failures → Keep intact
  C3: 4/4 failures → Keep intact
  C4: 1/4 failures → Needs improvement

Feedback to Agent01:
  "Improve C1 and C4. Maintain C2 and C3."

Iteration 2:
  C1: 3/4 failures ✅ (improved!)
  C2: 3/4 failures ✅ (maintained)
  C3: 4/4 failures ✅ (maintained)
  C4: 3/4 failures ✅ (improved!)

Result: 4 criteria failing 3+ times → Model Breaking! ✅
```

### Progress Tracking

**Metrics Tracked**:
- `total_failing_criteria`: Number of criteria failing 3+ times
- `previous_total_failing`: Previous iteration's count
- `improvement`: Difference between iterations

**Console Output**:
```
📈 PROGRESS TRACKING:
  Criteria failing consistently (3+): 2
  ✅ Improvement: +1 criteria now failing

🎯 TARGETING FOR NEXT ITERATION:
  Improve: C1, C4
  Maintain: C2, C3
```

---

## Usage

### Command Line Arguments

```bash
python qc.py [--runs N] [--max-iterations M]
```

**Arguments**:
- `--runs`: Number of times to execute the QC pipeline (default: 1)
- `--max-iterations`: Maximum iterations to refine prompt until model breaking (default: 10)

### Examples

```bash
# Single run with default settings (max 10 iterations)
python qc.py

# Run 5 times, max 10 iterations each
python qc.py --runs 5

# Run 3 times, max 5 iterations each (faster testing)
python qc.py --runs 3 --max-iterations 5

# Run 10 times, max 15 iterations each (more attempts)
python qc.py --runs 10 --max-iterations 15
```

### Expected Console Output

```
======================================================================
                    TAXONOMY SELECTION
======================================================================

Available Taxonomies:
----------------------------------------------------------------------
  [1] Question Correction (QC)
      Questions containing logical fallacies, factual errors, or inconsistencies where all provided options are incorrect

======================================================================
Enter taxonomy number (default: 1): 1

======================================================================
✓ Selected Taxonomy: Question Correction (QC)
======================================================================

data.csv already exists.

========== RUN 1/5 ==========

============================================================
🔄 ITERATION 1/10
============================================================
Layer 1: Generating initial prompt...
Agent01 Response: 
{...}

Layer 2: Nemotron Solve the QC task (4 attempts)
--- Attempt 1/4 ---
...

────────────────────────────────────────────────────────────
📊 ITERATION 1 - TEST RESULTS
────────────────────────────────────────────────────────────
Overall: 2/4 attempts failed
Individual Statuses: ['FAIL', 'FAIL', 'PASS', 'PASS']

📋 CRITERIA-LEVEL ANALYSIS:
  ❌ C1: Failed 2/4 times
  ✅ C2: Failed 3/4 times
  ✅ C3: Failed 4/4 times
  ❌ C4: Failed 1/4 times

📈 PROGRESS TRACKING:
  Criteria failing consistently (3+): 2

🎯 TARGETING FOR NEXT ITERATION:
  Improve: C1, C4
  Maintain: C2, C3

────────────────────────────────────────────────────────────
❌ Not model breaking yet (2 criteria failing consistently).
   Refining prompt for iteration 2...
============================================================

============================================================
🔄 ITERATION 2/10
============================================================
Layer 1: Refining prompt (iteration 2)...
...

────────────────────────────────────────────────────────────
📊 ITERATION 2 - TEST RESULTS
────────────────────────────────────────────────────────────
Overall: 3/4 attempts failed
Individual Statuses: ['FAIL', 'FAIL', 'FAIL', 'PASS']

📋 CRITERIA-LEVEL ANALYSIS:
  ✅ C1: Failed 3/4 times
  ✅ C2: Failed 3/4 times
  ✅ C3: Failed 4/4 times
  ✅ C4: Failed 3/4 times

📈 PROGRESS TRACKING:
  Criteria failing consistently (3+): 4
  ✅ Improvement: +2 criteria now failing

────────────────────────────────────────────────────────────
✅ SUCCESS! Model breaking achieved after 2 iteration(s)!
   Saving to CSV...
============================================================

✓ Task saved to data.csv with status FAIL.
```

---

## Output Format

### CSV File Structure

**File**: `data.csv` (shared across all taxonomies)

**Columns**:
- `taxonomy`: Taxonomy identifier (e.g., "qc" for Question Correction)
- `prompt`: The prompt text
- `correct_response`: Ideal correct response
- `response_reference`: JSON string of criteria array
- `model`: Model identifier
- `nemotron_response`: JSON string with all 4 attempts
- `judge_response`: JSON string with all 4 judge evaluations
- `status`: Always "FAIL" (only model-breaking prompts saved)

### JSON Fields in CSV

#### `nemotron_response` (JSON string)
```json
{
  "attempt_1": "Response text from attempt 1...",
  "attempt_2": "Response text from attempt 2...",
  "attempt_3": "Response text from attempt 3...",
  "attempt_4": "Response text from attempt 4..."
}
```

#### `judge_response` (JSON string)
```json
{
  "attempt_1": {
    "judge_output": "Grading Basis: {...}\nScore: 0 point\n...",
    "status": "FAIL"
  },
  "attempt_2": {
    "judge_output": "Grading Basis: {...}\nScore: 1 point\n...",
    "status": "PASS"
  },
  "attempt_3": {
    "judge_output": "Grading Basis: {...}\nScore: 0 point\n...",
    "status": "FAIL"
  },
  "attempt_4": {
    "judge_output": "Grading Basis: {...}\nScore: 0 point\n...",
    "status": "FAIL"
  }
}
```

---

## Taxonomy Selection

### Overview
The system supports multiple taxonomies (currently Question Correction, with more to be added). Users interactively select which taxonomy to use at the start of execution.

### How It Works

1. **Interactive Prompt**: When the script starts, it displays available taxonomies
2. **User Selection**: User enters taxonomy number (default: 1 for Question Correction)
3. **Validation**: System validates input and shows error if invalid
4. **Configuration**: System configures prompt template and output file based on selection

### Available Taxonomies

Currently Available:
- **1. Question Correction (QC)**: Questions containing logical fallacies, factual errors, or inconsistencies where all provided options are incorrect

Future Taxonomies (to be added):
- Additional taxonomies will be added to the `select_taxonomy()` function

### Implementation Details

**Function**: `select_taxonomy()`
- Location: Before CLI argument parsing
- Displays numbered options with descriptions
- Validates user input
- Returns taxonomy identifier ("qc", "itf", "mim", etc.)

**System Configuration**:
```python
if TAXONOMY == "qc":
    SYSTEM_PROMPT = SYSTEM_PROMPT_QC
elif TAXONOMY == "itf":
    SYSTEM_PROMPT = SYSTEM_PROMPT_ITF  # Future

# All taxonomies use the same CSV file (data.csv)
file_name = "data.csv"
```

**CSV Storage**:
- All taxonomies write to the same `data.csv` file
- Each row includes a `taxonomy` field to identify which taxonomy was used
- This allows easy filtering and analysis across different taxonomies

**Error Handling**:
- Invalid input: Shows error message and exits
- Empty input: Defaults to Question Correction (1)
- Keyboard interrupt: Gracefully exits

### Adding New Taxonomies

To add a new taxonomy:

1. Add to `taxonomies` dictionary in `select_taxonomy()`:
```python
taxonomies = {
    "1": {"id": "qc", "name": "Question Correction (QC)", ...},
    "2": {"id": "itf", "name": "Intentional Textual Flaws (ITF)", ...}  # New
}
```

2. Add system prompt constant:
```python
SYSTEM_PROMPT_ITF = f"""
    {PROMPT_HEADER}
    Intentional Textual Flaws (ITF):
    ...
"""
```

3. Update configuration section:
```python
if TAXONOMY == "itf":
    SYSTEM_PROMPT = SYSTEM_PROMPT_ITF
# Note: All taxonomies use the same CSV file (data.csv)
# The taxonomy field in CSV will automatically store "itf"
```

---

## Key Design Decisions

### 1. Why 4 Attempts?
- Provides statistical significance
- Allows criteria-level analysis
- Reduces variance from single attempts

### 2. Why 3+ Criteria Failing 3+ Times?
- Ensures consistent model breaking (not random)
- Multiple criteria failing = more robust test
- 3+ threshold balances strictness with achievability

### 3. Why Iterative Refinement?
- Some prompts need adjustment to break model
- Progressive improvement ensures convergence
- Targets specific criteria that need work

### 4. Why Update Both Prompt and Criteria?
- Keeps them aligned (prompt changes require criteria updates)
- Criteria can be made stricter for passing ones
- Maintains Question Correction category

### 5. Why Only Save Model-Breaking?
- Focuses on valuable test cases
- Reduces CSV size
- Only successful refinements are stored

---

## Troubleshooting

### Common Issues

#### 1. Judge Output Parsing Fails
**Symptom**: `parse_criteria_from_judge()` returns None
**Solution**: Check judge output format, ensure "Grading Basis:" section exists

#### 2. Criteria Not Improving
**Symptom**: Same criteria failing after multiple iterations
**Solution**: Check refinement feedback, may need to adjust prompt strategy

#### 3. Max Iterations Reached
**Symptom**: Loop exits without success
**Solution**: Increase `--max-iterations` or check if prompt is fundamentally too easy

#### 4. CSV Not Created
**Symptom**: No CSV file after run
**Solution**: Only model-breaking prompts are saved. Check console for "Not model breaking" messages.

---

## Future Enhancements

### Potential Improvements
1. **Convergence Detection**: Stop early if no improvement for N iterations
2. **Best Attempt Saving**: Save best result even if not perfect
3. **Multiple Strategies**: Try different refinement approaches when stuck
4. **Parallel Testing**: Test multiple prompt variants simultaneously
5. **Metrics Dashboard**: Track success rates, iteration counts, etc.

---

## Version History

### Current Version
- **Feature**: Iterative prompt refinement with criteria-level analysis
- **Success Condition**: 3+ criteria failing 3+ times
- **Max Iterations**: Configurable via CLI (default: 10)
- **Storage**: Only model-breaking prompts saved to CSV

---

## Contact & Support

For questions or issues, refer to the code comments or this documentation.

