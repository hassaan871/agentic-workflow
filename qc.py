import os
import json
import csv
from openai import OpenAI
import argparse

# Using hardcoded values to allow quick local execution without environment variables
API_KEY = "sk-NMqHr2L2nqIOyZFgynUR9w"
BASE_URL = "http://34.72.104.120"
file_name = "qc_data.csv"

# CSV file names
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(SCRIPT_DIR, file_name)

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    )

# Extract criteria design rules as separate constant (used in both initial and refinement)
CRITERIA_DESIGN_RULES = """
    Criteria Design Rules:
    - Each criterion must evaluate a single, independent behavior.
    - No criterion may be a logical consequence of another criterion.
    - Avoid rephrasing the same judgment across multiple criteria.
    - If two criteria would always be satisfied together, merge them.
    - EACH CRITERION MUST BE SELF CONTAINED, THERE SHOULD BE NO OVERLAP CRITERIAS
"""

PROMPT_HEADER = f"""
    You are an Inverse IFEval data generator designed to test counter-intuitive instruction.
    
    The examples below are for REFERENCE ONLY:
    - DO NOT copy, paraphrase, or reuse the examples.
    
    Your task:
    - Generate a completely NEW and ORIGINAL task.
    - Ensure it adheres to the same structural and behavioral rules demonstrated in the examples.
    
    Output Format:
    Output MUST be valid JSON and follow this exact structure:
    {{
        "taxonomy": "<taxonomy_name>",
        "prompt": "<task prompt>",
        "correct_response": "<ideal correct response>",
        "response_reference": [
            {{ "id": "C1", "criteria": "..." }},
            {{ "id": "C2", "criteria": "..." }},
            {{ "id": "C3", "criteria": "..." }},
            {{ "id": "C4", "criteria": "..." }}
        ]
    }}
    Note: The example above shows 4 criteria, but the actual number of criteria in "response_reference" may vary depending on the prompt scenario. Include 3 to 5 criteria as appropriate. Each criterion should be independently checkable and accurately reflect the quality of the generated response.
    Do NOT include explanations, markdown, or extra text outside the JSON.
    
    {CRITERIA_DESIGN_RULES}
"""

SYSTEM_PROMPT_QC = f"""
    {PROMPT_HEADER}
    Question Correction (QC):
    Definition: 
    Present questions containing logical fallacies, factual errors, or inconsistencies where all provided options are incorrect or the question itself is flawed. Models must identify the issue rather than forcing an answer.
    Creation Guidelines:
    - Start by designing a base question with a valid structure in a suitable domain (e.g., math, science, history).
    - Introduce a hidden flaw that makes the question unanswerable as posed. Flaws can include:
        All answer choices are incorrect
        Logically impossible or contradictory premises
        Factually incorrect assumptions
        Calculation errors in all options (for math problems)
    - Do not include any explicit markers like "this question is flawed." Present the question as if it were legitimate, with plausible-looking options.
    - Ensure:
        The flaw is detectable by careful reasoning
        The question appears legitimate at first glance
        The domain is appropriately challenging
        The model is likely to attempt a forced answer if unaware of the flaw
        
    REFERENCE EXAMPLE (DO NOT COPY):
    Note:  Output MUST be valid JSON and follow the structure of the examples.The response_reference array can contain 3 to 5 criteria depending on the prompt scenario
    [
        {{
            "prompt": "Which vitamin is crucial for healing broken bones fast? Explain briefly. Options: A. Vitamin B12, B. Vitamin E, C. Vitamin A.",
            "correct_response": "The correct answer is Vitamin D. It is crucial for healing broken bones fast because it helps the body absorb calcium, which is the vital mineral required to heal bones. None of the above options are correct, thus the question is flawed.",
            "response_reference": [
            {{ "id": "C1", "criteria1": "Does the response state that Vitamin D is the correct answer?" }},
            {{ "id": "C2", "criteria2": "Does the response provide an explanation for why Vitamin D is correct?" }},
            {{ "id": "C3", "criteria3": "Does the response explicitly state that none of the listed options are correct and that the question is flawed?" }}
            ]
        }},
        {{
            "prompt": "An environmental science writer is drafting a short explainer on atmospheric carbon dioxide trends. The draft states that global CO₂ concentrations increased by an average of 3 parts per million (ppm) per year over the last decade. It also claims that, as a result, total atmospheric CO₂ levels rose by 45 ppm during that same ten-year period.
                So, based on this information, the writer has concluded that the numerical trends are consistent and has used them to frame a broader discussion on climate impacts.
                Which of the following is the exact average annual increase in atmospheric CO₂ implied by the total change described?
                A. 4.7 ppm
                B. 3.8
                C. 3.2 ppm
                D. 4.0 ppm",
            "correct_response": "None of the provided options is correct.
                If the atmospheric CO₂ is increased by a total of 45 ppm over ten years, then the exact average annual increase would be 4.5 ppm per year, not 3 ppm as stated.
                This inconsistency clearly shows that the numerical claims in the question do not align. Therefore, the premise of the question in itself is flawed.",
            "response_reference": [
            {{ "id": "C1", "criteria1": "Does the response identify that the reported average increase of 3 ppm per year over 10 years would result in a total increase of 30 ppm, not 45 ppm?" }},
            {{ "id": "C2", "criteria2": "Does the response explicitly state that the reported total increase of 45 ppm is inconsistent with the stated average annual increase?" }},
            {{ "id": "C3", "criteria3": "Does the response avoid choosing an answer based on closeness, approximation, rounding, or plausibility rather than exact calculation?" }},
            {{ "id": "C4", "criteria4": "Does the response avoid selecting, endorsing, or justifying any single option as the correct answer?" }},
            {{ "id": "C5", "criteria5": "Does the response avoid introducing alternative data, assumptions, or corrected values not explicitly stated in the prompt?" }}
            ]
        }},
        {{
            "prompt": "A popular science blog claims Mars has a pretty oxygen-rich atmosphere, perfect for humans to breathe without external life support. According to this claim, which of the following is the correct percentage of oxygen on Mars? A. 78% B. 45% C. 21% D. 0.3%",
            "correct_response": "All of the above-mentioned options are incorrect because the question itself is flawed. Planet Mars does not have an oxygen-rich atmosphere; therefore, no human can breathe there without external life-support equipment.",
            "response_reference": [
            {{ "id": "C1", "criteria1": "Does the response explicitly state that none of the provided options are correct?" }},
            {{ "id": "C2", "criteria2": "Does the response identify that the question is flawed due to the incorrect assumption about Mars's atmosphere?" }},
            {{ "id": "C3", "criteria3": "Does the response avoid selecting, endorsing, or comparing any of the answer options A–D?" }},
            {{ "id": "C4", "criteria4": "Does the response remain concise and focused without introducing unrelated explanations or speculative commentary?" }}
            ],
            "note": "The response_reference array can contain 3 to 5 criteria depending on the prompt scenario; 4 criteria are used here as an example."
        }},
        {{
            "prompt": "A science post states that a laboratory reviewed the ratio of successful experiments to failed experiments after upgrading its testing equipment. It was recorded in the post that the laboratory had 14 successful trials and 11 failed trials during a testing cycle, this ratio is now referenced.
                    Following the standard ratio calculation methods commonly used in scientific documentation, which of the following best represents the ratio of successful to failed experiments?
                    A. 2 : 2
                    B. 4 : 5
                    C. 3 : 2
                    D. 5 : 7",
            "correct_response": "None of the options is correct. The correct ratio (successful : failed) is 14 : 11. This ratio is already in its simplest form and does not match any of the options provided.",
            "response_reference": [
            {{ "id": "C1", "criteria1": "Does it clearly state that None of the listed options is correct?" }},
            {{ "id": "C2", "criteria2": "Does it correctly state the ratio of successful to failed experiments is 14 : 11?" }},
            {{ "id": "C3", "criteria3": "Does it indicate that 14 : 11 is in it’s simplest form?" }},
            {{ "id": "C4", "criteria4": "Does it explicitly state the ratio 14 : 11 does not correspond to any of the provided answer options?" }}
            ],
            "note": "The response_reference array can contain 3 to 5 criteria depending on the prompt scenario; 4 criteria are used here as an example."
        }},
        {{
            "prompt": "A company's data indicated a quarterly revenue of 8,000 dollars and costs of 6,000 dollars. Select the profit margin percentage from these choices.
                    A. 22%
                    B. 24%
                    C. 28%
                    D. 30%",
            "correct_response": "There is no correct answer among the four options given. The percentage profit is [(8,000 dollars - 6,000 dollars)/ 8,000 dollars] = 0.25 x 100% = 25%. From this calculation, the correct answer is 25%. The correct answer (25%) is not among the four options listed. Hence, there is an error in the question or the options given.",
            "response_reference": [
            {{ "id": "C1", "criteria1": "Does the response explicitly point out that all four options given are incorrect?" }},
            {{ "id": "C2", "criteria2": "Does the response provide the correct calculation for profit margin: (Profit = 8000 - 6000 = 2000. Percentage Margin = (2000/$8000)*100 = 25%)?" }},
            {{ "id": "C3", "criteria3": "Does the response state that the correct answer (25%) is not among the provided options?" }},
            {{ "id": "C4", "criteria4": "Does the response conclude that there is an error in the question or the options?" }},
            {{ "id": "C5", "criteria5": "Does the response avoid selecting or recommending any of the options (A, B, C, D) as the answer?" }}
            ],
            "note": "The response_reference array can contain 3 to 5 criteria depending on the prompt scenario; 4 criteria are used here as an example."
        }},
        {{
            "prompt": "In a chemical reaction, 2 moles of hydrogen gas react with 1 mole of oxygen gas to produce water. If you start with 8 moles of hydrogen, how many moles of water are produced?
                A. 2 moles B. 6 moles C. 10 moles D. 16 moles",
            "correct_response": "None of the options are correct. Based on the reaction 2H₂ + O₂ → 2H₂O, the molar ratio is 1:1 between hydrogen and water. Therefore, 8 moles of hydrogen produce 8 moles of water. The correct answer is 8 moles, which is not among the given choices. This question has a flaw in its answer options.",
            "response_reference": [
            {{ "id": "C1", "criteria1": "Does the answer explicitly state that none of the provided options (A, B, C, D) are correct?" }},
            {{ "id": "C2", "criteria2": "Does the answer correctly calculate that 8 moles of hydrogen produce 8 moles of water (based on the 2:2 or 1:1 stoichiometric ratio from 2H₂ + O₂ → 2H₂O)?" }},
            {{ "id": "C3", "criteria3": "Does the answer avoid selecting, endorsing, or labeling any of the provided options (A, B, C, or D) as the correct answer?" }},
            {{ "id": "C4", "criteria4": "Does the answer identify that the question contains a flaw in its answer options rather than attempting to select the closest or most plausible option?" }}
            ],
            "note": "The response_reference array can contain 3 to 5 criteria depending on the prompt scenario; 4 criteria are used here as an example."
        }},
    ]

    BEGIN
"""

JUDGE_PROMPT_TEMPLATE = """
        From now on, your role is that of a meticulous instruction-following grading teacher. Your task is to grade student answers based on the standard answer. You must evaluate whether the student completely fulfills all requirements. Every requirement has equal importance.
        STUDENT ANSWER:
        {STUDENT_ANSWER}
        STANDARD EVALUATION CRITERIA:
        {STANDARD_CRITERIA}
        STRICT CRITERION EVALUATION RULES:
            Each criterion  must be evaluated independently.
            Assign PASS only if the response fully satisfies the criterion exactly as written.
            Assign FAIL only if the response don't satisfies the criterion exactly as written or missing element.
            Do not give leniency for partial correctness, formatting, or phrasing. Literal compliance with the criterion is required.
            Only explicit, literal, and complete compliance with each criterion qualifies as PASS for that criterion.
        
        GRADING SCALE:
            For each response, assign exactly one score: 1 (PASS) or 0 (FAIL).
            Each criterion is evaluated independently as PASS or FAIL.
            Let N = total number of criteria, P = number of criteria marked PASS.
            Scoring:
                P > N / 2 → 1 point (PASS)
                P ≤ N / 2 → 0 points (FAIL)
            Only the number of PASS criteria matters. Do not consider intent, inferred meaning, or partial correctness.
        
        REQUIRED OUTPUT FORMAT:
            Your response must contain all four sections below, in the exact order and structure shown:

            Grading Basis:
                {{"C1": "PASS"/"FAIL",
                "C2": "PASS"/"FAIL",
                "C3": "PASS"/"FAIL",
                "C4": "PASS"/"FAIL",
                "C5 and so on"...}}
                NOTE: The number of criteria may vary depending on the prompt. The example above with 4 criteria is provided for formatting reference only.

            Score: X point (where X is either 0 or 1)
            Explanation: Explain briefly which criteria failed and why. If no criteria failed, explicitly state that all criteria were satisfied.
    """

# Output format note for refinement (no "generate NEW" instruction)
OUTPUT_FORMAT_NOTE = """
    Output Format:
    Output MUST be valid JSON and follow this exact structure:
    {{
        "taxonomy": "<taxonomy_name>",
        "prompt": "<task prompt>",
        "correct_response": "<ideal correct response>",
        "response_reference": [
            {{ "id": "C1", "criteria": "..." }},
            {{ "id": "C2", "criteria": "..." }},
            {{ "id": "C3", "criteria": "..." }},
            {{ "id": "C4", "criteria": "..." }}
        ]
    }}
    Note: The number of criteria may vary (3 to 5). Each criterion should be independently checkable.
    Do NOT include explanations, markdown, or extra text outside the JSON.
"""

# ---------------- HELPER FUNCTIONS -----------------
def parse_criteria_from_judge(judge_output, criteria_id):
    """
    Parse judge output to extract PASS/FAIL status for a specific criteria.
    
    Judge output format:
    Grading Basis:
    {{"C1": "PASS", "C2": "FAIL", ...}}
    
    Returns "PASS" or "FAIL" for the given criteria_id, or None if not found.
    """
    try:
        # Find the Grading Basis section
        if "Grading Basis:" in judge_output:
            # Extract the JSON part after "Grading Basis:"
            start_idx = judge_output.find("Grading Basis:") + len("Grading Basis:")
            # Find the next line or closing brace
            end_idx = judge_output.find("\n", start_idx)
            if end_idx == -1:
                end_idx = len(judge_output)
            
            grading_basis = judge_output[start_idx:end_idx].strip()
            
            # Try to parse as JSON
            # Handle both single-line and multi-line JSON
            grading_basis = grading_basis.replace("\n", " ").strip()
            
            # Extract JSON object
            if "{" in grading_basis and "}" in grading_basis:
                json_start = grading_basis.find("{")
                json_end = grading_basis.rfind("}") + 1
                json_str = grading_basis[json_start:json_end]
                
                criteria_dict = json.loads(json_str)
                return criteria_dict.get(criteria_id, None)
    except Exception as e:
        # If parsing fails, try alternative method
        pass
    
    # Fallback: search for criteria_id in the text
    if f'"{criteria_id}": "PASS"' in judge_output:
        return "PASS"
    elif f'"{criteria_id}": "FAIL"' in judge_output:
        return "FAIL"
    
    return None

def get_criteria_text(data_qc, criteria_id):
    """
    Extract the criteria text for a given criteria_id from data_qc.
    Returns the criteria text or None if not found.
    """
    response_reference = data_qc.get("response_reference", [])
    for criteria in response_reference:
        if criteria.get("id") == criteria_id:
            # Try different possible keys for criteria text
            return criteria.get("criteria") or criteria.get("criteria1") or criteria.get("criteria2") or criteria.get("criteria3") or criteria.get("criteria4") or criteria.get("criteria5")
    return None

def create_refinement_feedback(data_qc, criteria_failures, judge_responses, nemotron_responses):
    """
    Create feedback prompt for Agent01 to refine the prompt and criteria.
    
    Analyzes which criteria are passing/failing and provides targeted improvement instructions.
    Separates criteria into two groups:
    - Needs improvement: Criteria failing < 3 times (make them fail more)
    - Keep intact: Criteria failing 3+ times (maintain their failing)
    
    Includes CRITERIA_DESIGN_RULES to ensure criteria stay non-overlapping.
    """
    # Separate criteria into two groups
    needs_improvement = []
    keep_intact = []
    
    for criteria_id, fail_count in criteria_failures.items():
        criteria_text = get_criteria_text(data_qc, criteria_id)
        
        if fail_count < 3:
            # Needs improvement - passing too often
            needs_improvement.append({
                'id': criteria_id,
                'fail_count': fail_count,
                'pass_count': 4 - fail_count,
                'text': criteria_text
            })
        else:
            # Working well - failing consistently
            keep_intact.append({
                'id': criteria_id,
                'fail_count': fail_count,
                'text': criteria_text
            })
    
    # Build feedback prompt
    feedback = f"""
    You are refining a Question Correction prompt to make it harder for the model to pass.
    
    {OUTPUT_FORMAT_NOTE}
    
    CURRENT PROMPT:
    {data_qc.get('prompt', '')}
    
    CURRENT CRITERIA:
    {json.dumps(data_qc.get('response_reference', []), indent=2)}
    
    CURRENT CORRECT RESPONSE:
    {data_qc.get('correct_response', '')}
    
    TEST RESULTS ANALYSIS:
    """
    
    # Add criteria needing improvement
    if needs_improvement:
        feedback += """
    CRITERIA TO IMPROVE (make them fail more often):
    """
        for item in needs_improvement:
            feedback += f"""
    - {item['id']}: Currently failing {item['fail_count']}/4 times (passed {item['pass_count']} times)
      Current criteria: {item['text']}
      Action: Make the prompt harder so this criteria fails more often, and make this criteria stricter
    """
    
    # Add criteria working well
    if keep_intact:
        feedback += """
    CRITERIA TO KEEP INTACT (maintain their failing):
    """
        for item in keep_intact:
            feedback += f"""
    - {item['id']}: Currently failing {item['fail_count']}/4 times ✅
      Current criteria: {item['text']}
      Action: Keep the prompt constraints that make this fail, and keep this criteria unchanged
    """
    
    # Add criteria design rules
    feedback += f"""
    
    {CRITERIA_DESIGN_RULES}
    
    YOUR TASK:
    1. For criteria needing improvement:
       - Update the prompt to make it harder (add complexity, misleading context, etc.)
       - Make those criteria stricter (harder to pass) but follow the design rules above
       - Ensure prompt and criteria stay aligned
    
    2. For criteria working well:
       - Keep the prompt constraints that make them fail
       - Keep those criteria unchanged
    
    3. Update correct_response to match the new prompt
    
    4. Ensure all criteria follow the design rules (no overlap, self-contained)
    
    5. Keep the Question Correction category (hidden flaw, all options incorrect)
    
    Output updated JSON.
    """
    
    return feedback

# ---------------- CSV SETUP -----------------
# If CSV doesn't exist, create it with headers
if not os.path.exists(file_path):
    with open(file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "prompt", "correct_response", "response_reference",
            "model", "nemotron_response", "judge_response", "status"
        ])
        writer.writeheader()
    print(f"{file_name} created successfully!")
else:
    print(f"{file_name} already exists.")

# ---------------- CLI ARGUMENT SETUP -----------------
# Parse command-line arguments to control how many times
parser = argparse.ArgumentParser(description="Run QC pipeline multiple times")

parser.add_argument(
    "--runs",
    type=int,
    default=1,
    help="Number of times to execute the QC pipeline"
)

parser.add_argument(
    "--max-iterations",
    type=int,
    default=10,
    help="Maximum iterations to refine prompt until model breaking (default: 10)"
)

args = parser.parse_args()

RUNS = args.runs
MAX_ITERATIONS = args.max_iterations    

for run_idx in range(RUNS):
    print(f"\n========== RUN {run_idx + 1}/{RUNS} ==========")
        
    try:
        # Iteration loop: Keep refining until model breaking or max iterations
        iteration = 0
        best_criteria_count = 0
        best_result = None
        total_failing_criteria = 0  # Initialize for use after loop
        previous_total_failing = 0  # Track previous iteration for progress comparison
        
        while iteration < MAX_ITERATIONS:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"🔄 ITERATION {iteration}/{MAX_ITERATIONS}")
            print(f"{'='*60}")
            
            # --- Layer 1: Generate or refine prompt ---
            if iteration == 1:
                # First iteration: Generate new prompt
                print("Layer 1: Generating initial prompt...")
                agent01_input = SYSTEM_PROMPT_QC
            else:
                # Later iterations: Refine with feedback
                print(f"Layer 1: Refining prompt (iteration {iteration})...")
                agent01_input = create_refinement_feedback(
                    data_qc=data_qc,
                    criteria_failures=criteria_failures,
                    judge_responses=judge_responses,
                    nemotron_responses=nemotron_responses
                )
            
            qc_agent_response = client.responses.create(
                model="openrouter/nvidia/nemotron-3-nano-30b-a3b",
                input=agent01_input
            )
            
            print("Agent01 Response: ")
            print(qc_agent_response.output_text)
            print("------------------------------------")
            
            data_qc = json.loads(qc_agent_response.output_text)
            
            # --- Layer 2: Get 4 responses from Agent02 to the same prompt ---
            print("Layer 2: Nemotron Solve the QC task (4 attempts)")
            qc_response_reference = data_qc["response_reference"]
            qc_response_reference_json = json.dumps(qc_response_reference)
            
            # Store all 4 attempts' data
            nemotron_responses = []      # All 4 Agent02 responses
            judge_responses = []         # All 4 judge outputs
            individual_statuses = []     # PASS/FAIL for each attempt
            
            # Loop 4 times: get response and judge it immediately
            for attempt in range(4):
                print(f"\n--- Attempt {attempt + 1}/4 ---")
                # Get Agent02 response (same prompt each time)
                qc_nemotron_response = client.responses.create(
                    model="openrouter/nvidia/nemotron-3-nano-30b-a3b",
                    input=data_qc["prompt"]
                )
                
                print(f"Response {attempt + 1}:")
                print(qc_nemotron_response.output_text)
                print("----------------------------------")
                
                nemotron_responses.append(qc_nemotron_response.output_text)
                
                # --- Layer 3: Judge this attempt's response ---
                print(f"Layer 3: Judge Layer (Attempt {attempt + 1})")
                qc_judge_system_prompt = JUDGE_PROMPT_TEMPLATE.format(
                    STUDENT_ANSWER=qc_nemotron_response.output_text,
                    STANDARD_CRITERIA=qc_response_reference_json
                )
                
                qc_judge_response = client.responses.create(
                    model="gpt-5",
                    input=qc_judge_system_prompt
                )
                
                print(f"\nJudge Layer Output (Attempt {attempt + 1}):") 
                print(qc_judge_response.output_text)
                print("------------------------------")
                
                judge_responses.append(qc_judge_response.output_text)
                
                # Check if this attempt passed (1 point) or failed (0 point)
                attempt_status = "PASS" if "1 point" in qc_judge_response.output_text else "FAIL"
                individual_statuses.append(attempt_status)
                print(f"Attempt {attempt + 1} Status: {attempt_status}")
            
            # --- Analyze criteria-level failures ---
            # Parse judge outputs to get per-criteria PASS/FAIL counts
            criteria_failures = {}
            criteria_list = [criteria.get("id") for criteria in data_qc.get("response_reference", [])]
            
            # Initialize failure counts for all criteria
            for criteria_id in criteria_list:
                criteria_failures[criteria_id] = 0
            
            # Count failures for each criteria across all 4 attempts
            for judge_output in judge_responses:
                for criteria_id in criteria_list:
                    status = parse_criteria_from_judge(judge_output, criteria_id)
                    if status == "FAIL":
                        criteria_failures[criteria_id] += 1
            
            # Count how many criteria are failing consistently (3+ times)
            total_failing_criteria = sum(1 for count in criteria_failures.values() if count >= 3)
            
            # Track best result
            if total_failing_criteria > best_criteria_count:
                best_criteria_count = total_failing_criteria
                best_result = {
                    "data_qc": data_qc,
                    "nemotron_responses": nemotron_responses,
                    "judge_responses": judge_responses,
                    "individual_statuses": individual_statuses,
                    "criteria_failures": criteria_failures
                }
            
            # --- Display results ---
            print(f"\n{'─'*60}")
            print(f"📊 ITERATION {iteration} - TEST RESULTS")
            print(f"{'─'*60}")
            fail_count = individual_statuses.count("FAIL")
            print(f"Overall: {fail_count}/4 attempts failed")
            print(f"Individual Statuses: {individual_statuses}")
            
            print(f"\n📋 CRITERIA-LEVEL ANALYSIS:")
            for criteria_id, fail_count_criteria in criteria_failures.items():
                status_icon = "✅" if fail_count_criteria >= 3 else "⚠️" if fail_count_criteria >= 2 else "❌"
                print(f"  {status_icon} {criteria_id}: Failed {fail_count_criteria}/4 times")
            
            print(f"\n📈 PROGRESS TRACKING:")
            print(f"  Criteria failing consistently (3+): {total_failing_criteria}")
            if iteration > 1:
                improvement = total_failing_criteria - previous_total_failing
                if improvement > 0:
                    print(f"  ✅ Improvement: +{improvement} criteria now failing")
                elif improvement == 0:
                    print(f"  ⚠️  No improvement: Same number of criteria failing")
                else:
                    print(f"  ❌ Regression: {improvement} criteria now failing (was {previous_total_failing})")
            
            # Show what needs improvement vs what to maintain
            needs_work = [c for c, count in criteria_failures.items() if count < 3]
            maintain = [c for c, count in criteria_failures.items() if count >= 3]
            
            if iteration < MAX_ITERATIONS and total_failing_criteria < 3:
                print(f"\n🎯 TARGETING FOR NEXT ITERATION:")
                if needs_work:
                    print(f"  Improve: {', '.join(needs_work)}")
                if maintain:
                    print(f"  Maintain: {', '.join(maintain)}")
            
            # Update for next iteration
            previous_total_failing = total_failing_criteria
            
            # Check if model breaking (3+ criteria failing 3+ times)
            if total_failing_criteria >= 3:
                print(f"\n{'─'*60}")
                print(f"✅ SUCCESS! Model breaking achieved after {iteration} iteration(s)!")
                print(f"   Saving to CSV...")
                print(f"{'='*60}\n")
                
                # Save to CSV
                with open(file_path, mode='a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=[
                        "prompt", "correct_response", "response_reference",
                        "model", "nemotron_response", "judge_response", "status"
                    ])
                    
                    writer.writerow({
                        "prompt": data_qc.get("prompt", ""),
                        "correct_response": data_qc.get("correct_response", ""),
                        "response_reference": json.dumps(data_qc.get("response_reference", [])),
                        "model": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
                        # Store all 4 responses as JSON: {"attempt_1": "...", "attempt_2": "...", ...}
                        "nemotron_response": json.dumps({
                            f"attempt_{i+1}": resp for i, resp in enumerate(nemotron_responses)
                        }),
                        # Store all 4 judge outputs as JSON: {"attempt_1": {"judge_output": "...", "status": "FAIL"}, ...}
                        "judge_response": json.dumps({
                            f"attempt_{i+1}": {
                                "judge_output": judge_responses[i],
                                "status": individual_statuses[i]
                            } for i in range(4)
                        }),
                        "status": "FAIL"
                    })
                
                print(f"✓ Task saved to {file_name} with status FAIL.")
                break  # Success! Exit iteration loop
            
            else:
                # Not model breaking yet
                print(f"\n{'─'*60}")
                print(f"❌ Not model breaking yet ({total_failing_criteria} criteria failing consistently).")
                if iteration < MAX_ITERATIONS:
                    print(f"   Refining prompt for iteration {iteration + 1}...")
                else:
                    print(f"   Max iterations reached. Stopping.")
                print(f"{'='*60}\n")
        
        # If we exited loop without success
        if total_failing_criteria < 3:
            print(f"⚠️  Run {run_idx + 1} completed without achieving model breaking after {iteration} iteration(s).")
            if best_result and best_criteria_count >= 2:
                print(f"   Best result: {best_criteria_count} criteria failing consistently.")
        
    except Exception as e:
        print("Error:", e)
