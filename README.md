## Agentic Workflow – Developer Documentation

### Table of Contents
1. [Overview](#overview)
2. [Feature Description](#feature-description)
3. [Architecture & Flow](#architecture--flow)
4. [Code Structure](#code-structure)
5. [Key Components](#key-components)
6. [Data Flow](#data-flow)
7. [Usage](#usage)
8. [Output Format](#output-format)
9. [Key Design Decisions](#key-design-decisions)

---

## Overview

The agentic workflow in `updated_workflow.py` is a multi-agent, multi-taxonomy system for generating **Inverse IFEval**-style tasks and testing whether a target model can be reliably broken under those tasks.

### Key Features
- **Multi-taxonomy support**: `qc` (Question Correction), `itf` (Intentional Textual Flaws), `mim` (Mid‑Turn Instruction Modification) via a single script.
- **Non-interactive CLI**: Taxonomies, runs, and max iterations are all configured via CLI only (no interactive selection).
- **Agent01 validation layer**: A separate **Agent01 Judge** (GPT‑5) validates Agent01’s `correct_response` against the `response_reference`.
- **Three-agent evaluation loop**:
  - **Agent01**: task generator/refiner (Nemotron).
  - **Agent01 Judge**: validates `correct_response` (GPT‑5).
  - **Agent02 + Judge**: solver (Nemotron) evaluated by a grading Judge (GPT‑5).
- **Iterative feedback loop**: Taxonomy-aware refinement via `create_refinement_feedback(...)` for QC, ITF, and MIM.
- **Model-breaking detection (attempt-based)**: Breaks the model when **3 or 4 out of 4 attempts** from Agent02 fail.
- **Unified CSV + embeddings**: All taxonomies write to a single CSV (`updated_workflow_data.csv`) including prompt embeddings and max similarity with existing prompts.

---

## Feature Description

### What It Does
For each selected taxonomy and run:
- **Generate / refine task**
  - Agent01 produces a JSON object with:
    - `taxonomy`
    - `prompt`
    - `correct_response`
    - `response_reference` (criteria list).
- **Validate Agent01**
  - Agent01 Judge checks whether `correct_response` actually aligns with `response_reference`.
  - Returns JSON:  
    `{"status": "PASS" | "FAIL", "remarks": "..."}`.
- **Test Agent02 (4 attempts)**
  - Agent02 gets the same `prompt` 4 times.
  - Each attempt is graded by the Judge using the criteria.
- **Analyze**
  - Per-criteria failure counts across 4 attempts.
  - Per-attempt PASS/FAIL, summarized as `fail_count`.
- **Refine (feedback loop)**
  - If model-breaking is not achieved and iterations remain, `create_refinement_feedback(...)` builds a taxonomy-aware refinement instruction for Agent01.
- **Save**
  - When model-breaking is achieved, the task, metadata, and embeddings are appended to `updated_workflow_data.csv`.

### Model-Breaking Condition (Current Logic)

- Let `individual_statuses` be the 4 attempt-level statuses: `["PASS" | "FAIL", ...]`.
- Let `fail_count = individual_statuses.count("FAIL")`.
- **Model-breaking** is declared when:
  - **`fail_count >= 3`** (i.e., 3 or 4 attempts fail).
- This condition:
  - Triggers CSV saving and stops the iteration loop for that run.
  - Replaces the old “3+ criteria failing consistently” condition.

---

## Architecture & Flow

### High-Level Flow

```text
START
  ↓
CLI PARSING (no interactive input)
  - --runs "qc:8,itf:5,mim:3"
  - --max-iterations "qc:15,itf:10,mim:5" (optional, defaults = 1)
  ↓
FOR EACH taxonomy_id IN TAXONOMY_RUNS:
  SYSTEM_PROMPT = TAXONOMY_PROMPTS[taxonomy_id]
  MAX_ITERATIONS = MAX_ITERATIONS_DICT[taxonomy_id]
  ↓
  FOR run_idx IN [1..num_runs_for_taxonomy]:
    iteration = 0
    best_criteria_count, best_result tracked for diagnostics
    ↓
    WHILE iteration < MAX_ITERATIONS:
      iteration += 1

      1) Agent01 (Generator / Refiner)
         - iteration == 1:
             agent01_input = SYSTEM_PROMPT_<taxonomy>
         - iteration > 1:
             agent01_input = create_refinement_feedback(..., taxonomy=taxonomy_id)
         - Output JSON → data_qc

      2) Agent01 Judge (Validation)
         - Validates correct_response vs response_reference
         - Outputs JSON: status + remarks

      3) Agent02 + Judge (4 attempts)
         - For attempt in 1..4:
             Agent02 answers data_qc["prompt"]
             Judge grades response vs response_reference
             → attempt-level PASS/FAIL

      4) Analysis
         - individual_statuses: 4 attempt outcomes
         - fail_count: number of FAIL attempts
         - criteria_failures: per-criteria fail counts across 4 attempts

      5) Decision
         - IF fail_count ≥ 3:
             → compute embedding + max_similarity
             → append CSV row
             → break (end run)
         - ELSE IF iteration < MAX_ITERATIONS:
             → refine via create_refinement_feedback(...)
         - ELSE:
             → end run without saving
```

### Agent Roles

- **Agent01 – Generator / Refiner**
  - Model: `openrouter/nvidia/nemotron-3-nano-30b-a3b`.
  - Inputs:
    - Iteration 1: taxonomy-specific system prompt:
      - `SYSTEM_PROMPT_QC`
      - `SYSTEM_PROMPT_ITF`
      - `SYSTEM_PROMPT_MIM`
    - Iteration > 1: feedback prompt built by `create_refinement_feedback(...)`.
  - Output JSON:
    - `taxonomy`
    - `prompt`
    - `correct_response`
    - `response_reference` (criteria array).

- **Agent01 Judge – Validation Layer**
  - Model: `gpt-5`.
  - Prompt: `AGENT01_VALIDATION_PROMPT_TEMPLATE`.
  - Checks:
    - Whether `correct_response` logically aligns with and satisfies the criteria in `response_reference`.
  - Output JSON:
    - `status`: `"PASS"` or `"FAIL"`.
    - `remarks`: reasoning and diagnostics.

- **Agent02 – Solver**
  - Model: `openrouter/nvidia/nemotron-3-nano-30b-a3b`.
  - Input: task `prompt` from `data_qc`.
  - Four independent attempts per iteration.

- **Judge – Scoring Agent for Agent02**
  - Model: `gpt-5`.
  - Prompt: `JUDGE_PROMPT_TEMPLATE`.
  - Responsibilities:
    - Per-criterion grading via a “Grading Basis” JSON section.
    - Overall score: `Score: 1 point` (PASS) or `Score: 0 point` (FAIL).
  - Implementation detail:
    - Attempt-level status is derived by searching for `"1 point"` in judge output.

---

## Code Structure

### Main File

- **File**: `updated_workflow.py`

High-level layout:
- **Imports & configuration**
  - OpenAI client (`OpenAI`) with hard-coded `API_KEY` and `BASE_URL`.
  - Embedding model: `SentenceTransformer('all-MiniLM-L6-v2')`.
  - CSV config:
    - `file_name = "updated_workflow_data.csv"`.
    - `file_path` derived from script directory.
- **Prompt and taxonomy constants**
  - `CRITERIA_DESIGN_RULES`
  - `PROMPT_HEADER`
  - `SYSTEM_PROMPT_QC`
  - `SYSTEM_PROMPT_ITF`
  - `SYSTEM_PROMPT_MIM`
  - `TAXONOMY_PROMPTS`
  - `VALID_TAXONOMIES`
  - `JUDGE_PROMPT_TEMPLATE`
  - `AGENT01_VALIDATION_PROMPT_TEMPLATE`
  - `OUTPUT_FORMAT_NOTE`
- **Helper functions**
  - `parse_criteria_from_judge(judge_output, criteria_id)`
  - `get_criteria_text(data_qc, criteria_id)`
  - `create_refinement_feedback(data_qc, criteria_failures, judge_responses, nemotron_responses, taxonomy="qc")`
  - Embedding helpers:
    - `get_prompt_embedding(prompt)`
    - `load_existing_embeddings_from_csv(file_path)`
    - `calculate_max_similarity(new_embedding, existing_embeddings)`
- **CLI parsing**
  - `parse_taxonomy_runs(runs_str)` → `TAXONOMY_RUNS`
  - `parse_max_iterations(iterations_str, taxonomy_runs)` → `MAX_ITERATIONS_DICT`
  - `argparse` with:
    - `--runs` (required, string).
    - `--max-iterations` (optional, string, default empty).
- **CSV setup**
  - Create `updated_workflow_data.csv` with the new header if it doesn’t exist.
- **Main execution loop**
  - For each `(taxonomy_id, num_runs)`:
    - Set `SYSTEM_PROMPT` and `MAX_ITERATIONS`.
    - Run nested iteration loop with generation → validation → solving → judging → analysis → refine/save.

---

## Key Components

### CLI Parsing

- **`parse_taxonomy_runs(runs_str)`**
  - Input: string like `"qc:8,itf:5,mim:2"` or `"qc,itf"`.
  - Behavior:
    - Splits by comma, then by optional colon.
    - Default count is 1 if no number given (`"qc"` → `qc:1`).
    - Validates taxonomy against `VALID_TAXONOMIES = {"qc", "itf", "mim"}`.
    - Returns dict: `{"qc": 8, "itf": 5, "mim": 2}`.

- **`parse_max_iterations(iterations_str, taxonomy_runs)`**
  - Input: string like `"qc:15,itf:10"` (optional).
  - Behavior:
    - Initializes every taxonomy in `taxonomy_runs` with default `1`.
    - Overrides where explicitly provided (e.g., `qc:15`).
    - Returns dict: `{"qc": 15, "itf": 10, "mim": 1}` (if `mim` not listed).

### Agent01 Validation Layer

- **Prompt template**: `AGENT01_VALIDATION_PROMPT_TEMPLATE`
  - Injects:
    - `CORRECT_RESPONSE = data_qc["correct_response"]`
    - `RESPONSE_REFERENCE = json.dumps(data_qc["response_reference"])`
  - The Judge is instructed to:
    - Check logical alignment between `correct_response` and each criterion.
    - Return strict JSON with `status` and `remarks`.
- **Usage in main loop**:
  - Immediately after Agent01’s JSON is parsed.
  - Logs validation status and remarks to console.
  - Values persisted in CSV:
    - `agent_01_judge_model = "gpt-5"`
    - `agent_01_judge_model_remarks`
    - `agent_01_correct_response_status`

### Refinement Feedback (`create_refinement_feedback`)

- **Inputs**:
  - `data_qc`: the current task’s JSON.
  - `criteria_failures`: per-criteria fail counts across 4 attempts.
  - `judge_responses`, `nemotron_responses`: for context (not parsed deeply today but passed through).
  - `taxonomy`: one of `"qc"`, `"itf"`, `"mim"`.

- **Logic**:
  - For each criteria id:
    - If `fail_count < 3` → **needs_improvement** (prompt should be refined to make this criterion harder to pass).
    - If `fail_count >= 3` → **keep_intact** (constraints are working; preserve their behavior).
  - Builds a large system prompt including:
    - Current prompt, criteria, and correct response.
    - Criteria design rules (`CRITERIA_DESIGN_RULES`).
    - Taxonomy-specific refinement techniques:
      - QC: hidden flaws, subtle inconsistencies, better distractors.
      - ITF: constraint entanglement, ambiguous counts, surface simplicity with hidden fragility.
      - MIM: strong initial instruction, explicit mid-turn modifications, clear distinct final instruction.
    - Critical constraints:
      - Do **not** break logical consistency.
      - Do **not** artificially make criteria stricter independently of the refined prompt.
      - Keep the task valid within its taxonomy.

---

## Data Flow

### Core Structures

- **Agent01 output (`data_qc`)**
  - `taxonomy`: `"qc"`, `"itf"`, or `"mim"`.
  - `prompt`: text shown to Agent02.
  - `correct_response`: ideal answer under the intended constraints.
  - `response_reference`: array of criteria objects:
    - Each has `"id": "C1"`, and one text field (`criteria`, `criteria1`, …).

- **Per-iteration evaluation data**
  - `nemotron_responses`: list of 4 solver responses.
  - `judge_responses`: list of 4 judge outputs.
  - `individual_statuses`: list of `"PASS"`/`"FAIL"` for each attempt.
  - `criteria_failures`: `{criteria_id: fail_count_across_4_attempts}`.
  - `fail_count`: scalar `0..4`, number of failed attempts.
  - `previous_total_failing` and `best_criteria_count` used only for progress reporting / diagnostics.

- **Embeddings**
  - New prompt embedding: `get_prompt_embedding(prompt)` → list of floats.
  - Existing embeddings: loaded from the `embedding` column of `updated_workflow_data.csv`.
  - Max similarity with history: `calculate_max_similarity(new_embedding, existing_embeddings)`.

### End-to-End Flow (Per Run)

```text
Agent01 (Nemotron) → data_qc (JSON)
  ↓
Agent01 Judge (GPT‑5) → validation JSON (status + remarks)
  ↓
Agent02 (Nemotron, 4 attempts) → 4 responses
  ↓
Judge (GPT‑5, 4 times) → 4 graded outputs
  ↓
individual_statuses + criteria_failures + fail_count
  ↓
IF fail_count ≥ 3:
  → compute new prompt embedding
  → load existing embeddings
  → compute max_similarity
  → append row to updated_workflow_data.csv
ELSE IF iteration < MAX_ITERATIONS:
  → build refinement prompt via create_refinement_feedback(..., taxonomy)
  → feed into Agent01 for next iteration
ELSE:
  → run ends without CSV write
```

---

## Usage

### Command Line Interface

```bash
python updated_workflow.py --runs <spec> [--max-iterations <spec>]
```

- **`--runs` (required)**:
  - Format:
    - `qc:8,itf:5,mim:3`
    - `qc,itf` (equivalent to `qc:1,itf:1`).
  - Each `taxonomy:count` pair defines how many runs to execute for that taxonomy.

- **`--max-iterations` (optional)**:
  - Same syntax, but for iteration limits per taxonomy.
  - Any taxonomy present in `--runs` that’s missing here defaults to `1` iteration.
  - Examples:
    - `--max-iterations qc:15,itf:10`
    - `--max-iterations qc:10,itf:8,mim:5`

### Example Commands

```bash
# Only QC, 8 runs, up to 15 iterations per run
python updated_workflow.py --runs qc:8 --max-iterations qc:15

# QC + ITF, each 5 and 3 runs respectively, default 1 iteration each
python updated_workflow.py --runs qc:5,itf:3

# QC + ITF + MIM, each with custom iteration caps
python updated_workflow.py --runs qc:4,itf:4,mim:2 --max-iterations qc:10,itf:8,mim:5
```

If `--runs` is missing or invalid, the script prints usage via `argparse` and exits; it does not prompt interactively.

---

## Output Format

### CSV File Structure

- **File**: `updated_workflow_data.csv`
- **Columns (in order)**:
  - `taxonomy`
  - `agent_01_model`
  - `prompt`
  - `correct_response`
  - `response_reference`
  - `agent_01_judge_model`
  - `agent_01_judge_model_remarks`
  - `agent_01_correct_response_status`
  - `agent_02_model`
  - `agent_02_response`
  - `judge_response`
  - `status`
  - `embedding`
  - `max_similarity`

### Column Semantics

- **`taxonomy`**: one of `qc`, `itf`, `mim` (as used in CLI).
- **`agent_01_model`**: string identifier of Agent01 model (currently Nemotron).
- **`prompt`**: final task prompt that produced model-breaking behavior.
- **`correct_response`**: Agent01’s ideal answer text.
- **`response_reference`**: JSON string of the criteria array (the `response_reference` list serialized).
- **`agent_01_judge_model`**: string identifier of Agent01 Judge model (currently `gpt-5`).
- **`agent_01_judge_model_remarks`**: free-text explanation from the validation agent.
- **`agent_01_correct_response_status`**: `"PASS"` or `"FAIL"` from Agent01 Judge.
- **`agent_02_model`**: string identifier of Agent02 model (currently Nemotron).
- **`agent_02_response`**: JSON string with all 4 attempts:

```json
{
  "attempt_1": "response text ...",
  "attempt_2": "response text ...",
  "attempt_3": "response text ...",
  "attempt_4": "response text ..."
}
```

- **`judge_response`**: JSON string with 4 judge outputs:

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

- **`status`**:
  - Currently always `"FAIL"` for saved rows (they are model-breaking tasks).
- **`embedding`**:
  - JSON array of floats (MiniLM embedding for the prompt).
- **`max_similarity`**:
  - Float (as string with 4 decimals), highest cosine similarity vs previous embeddings in this CSV.

---

## Key Design Decisions

### Visual Workflow Diagram

The diagram below summarizes the full agentic architecture from CLI to CSV:

```text
               CLI
  python updated_workflow.py
        --runs / --max-iterations
                       │
                       ▼
          +------------------------+
          |  Taxonomy Controller   |
          | (TAXONOMY_RUNS / MAX_) |
          +------------------------+
           │    │          │
           │    │          │
           ▼    ▼          ▼
         qc   itf         mim
   (one loop per taxonomy-id)

For each taxonomy-id:

  FOR each run:
    ┌───────────────────────────────────────────────────────┐
    │  Iteration Loop (1 .. MAX_ITERATIONS[taxonomy-id])   │
    └───────────────────────────────────────────────────────┘
                 │
                 ▼
        ┌───────────────────────┐
        │ Agent01 (Nemotron)    │
        │  - iteration 1:       │
        │      SYSTEM_PROMPT_*  │
        │  - iteration >1:      │
        │      create_refinement│
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ data_qc JSON          │
        │  - taxonomy           │
        │  - prompt             │
        │  - correct_response   │
        │  - response_reference │
        └───────────┬───────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ Agent01 Judge (GPT‑5)       │
        │  - validates correct_       │
        │    response vs criteria     │
        │  - outputs {status,remarks} │
        └───────────┬────────────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ Agent02 + Judge (4 attempts)│
        │  LOOP 4x:                   │
        │    Agent02 (Nemotron)       │
        │      → answer(prompt)       │
        │    Judge (GPT‑5)            │
        │      → grade vs criteria    │
        │    → attempt PASS / FAIL    │
        └───────────┬────────────────┘
                    │
                    ▼
        ┌─────────────────────────────┐
        │ Analysis                    │
        │  - individual_statuses (4)  │
        │  - fail_count = FAILs       │
        │  - criteria_failures        │
        └───────────┬────────────────┘
                    │
     fail_count ≥ 3 │     fail_count < 3
        (3+ fails)  │
                    │
        ▼           ▼
  ┌───────────────────────────┐      ┌───────────────────────────┐
  │ Save & Exit Run           │      │ More Iterations?          │
  │  - compute embedding      │      │  - if iteration < max:    │
  │  - compute max_similarity │      │      create_refinement    │
  │  - append CSV row         │      │      (taxonomy-aware)     │
  └───────────────────────────┘      │      → next iteration     │
                                     │  - else: stop run         │
                                     └───────────────────────────┘
```

### Multi-Taxonomy, Single Script

- All taxonomy prompts (`SYSTEM_PROMPT_QC`, `SYSTEM_PROMPT_ITF`, `SYSTEM_PROMPT_MIM`) live directly in `updated_workflow.py`.
- `TAXONOMY_PROMPTS` and `VALID_TAXONOMIES` drive:
  - System prompt selection.
  - CLI validation.
  - Iteration behavior in the main loop.

### Non-Interactive Configuration

- Workflow is fully controlled by CLI:
  - No runtime taxonomy selection prompts.
  - If `--runs` is not specified, `argparse` shows an error and exits.
- This makes the script easier to automate and compose into bigger pipelines.

### Attempt-Based Model-Breaking

- Final decision uses **attempt-level** failure rate:
  - `fail_count >= 3` over the 4 attempts.
- Criteria-level stats are still computed and used for refinement, but they do **not** gate the “model-breaking” decision anymore.
- This matches the requirement: **if 3 or more out of 4 individual attempts fail, it is model-breaking**.

### Taxonomy-Aware Refinement

- `create_refinement_feedback` is shared but its instructions depend on `taxonomy`:
  - QC: hidden flaw strengthening, distractors, subtle inconsistencies.
  - ITF: error constraints, ambiguity, entanglement.
  - MIM: mid-turn instruction modifications and final-instruction focus.
- Keeps one feedback engine while respecting differences between taxonomies.

### Validation Layer for Agent01

- Agent01 Judge ensures that:
  - `correct_response` is actually a good gold answer relative to the criteria.
  - Bad or misaligned gold answers can be flagged via `status="FAIL"` and detailed `remarks`.
- These signals are stored for each saved prompt, enabling later quality analysis.

---

## Notes for Future Work

- Possible extensions:
  - Early stopping when there’s no improvement in criteria behavior across N iterations.
  - Optionally saving “best so far” runs even if `fail_count < 3`.
  - Adding more taxonomies by filling `TAXONOMY_PROMPTS` and reusing the existing control flow.
  - More sophisticated aggregation of judge outputs (beyond simple `"1 point"` string search).


