import os
import json
import csv
from openai import OpenAI
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime

# Using hardcoded values to allow quick local execution without environment variables
API_KEY = "sk-NMqHr2L2nqIOyZFgynUR9w"
BASE_URL = "http://34.72.104.120"
# CSV file name (shared across all taxonomies)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"new_qc_data_{timestamp}.csv"
# file_name = "new_qc_data.csv"

# CSV file path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(SCRIPT_DIR, file_name)

# Iteration log file path (for feedback loop tuning)
iteration_log_file_path = os.path.join(SCRIPT_DIR, "iteration_log.csv")

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    )

# Initialize embedding model for similarity calculation
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.")

# Prompts/templates extracted to prompts.py (keep runtime logic here)
from prompts import (
    CRITERIA_DESIGN_RULES,
    PROMPT_HEADER,
    SYSTEM_PROMPT_QC,
    SYSTEM_PROMPT_ITF,
    SYSTEM_PROMPT_MIM,
    SYSTEM_PROMPT_DIA,
    TAXONOMY_PROMPTS,
    VALID_TAXONOMIES,
    JUDGE_PROMPT_TEMPLATE,
    AGENT01_VALIDATION_PROMPT_TEMPLATE,
    OUTPUT_FORMAT_NOTE,
    RESPONSE_REFERENCE_IMPROVEMENT_PROMPT_TEMPLATE,
)

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

def extract_explanation_from_judge(judge_output):
    """
    Extract the explanation section from judge output.
    
    Args:
        judge_output: Full judge response text
    
    Returns:
        String: Explanation text, or empty string if not found
    """
    try:
        if "Explanation:" in judge_output:
            start_idx = judge_output.find("Explanation:") + len("Explanation:")
            explanation = judge_output[start_idx:].strip()
            return explanation
    except Exception:
        pass
    return ""

def evaluate_single_criterion(client, student_answer, criterion, criterion_id):
    """
    Evaluate a single criterion against student answer using existing JUDGE_PROMPT_TEMPLATE.
    
    Args:
        client: OpenAI client instance
        student_answer: Agent02's response text
        criterion: Dictionary with criterion data {"id": "C1", "criteria": "..."}
        criterion_id: String ID of the criterion (e.g., "C1")
    
    Returns:
        Dictionary: {"criterion_id": "C1", "status": "PASS"/"FAIL", "explanation": "..."}
    """
    # Create a single-criterion array for the existing template
    single_criterion_array = [criterion]
    single_criterion_json = json.dumps(single_criterion_array)
    
    # Reuse existing JUDGE_PROMPT_TEMPLATE
    judge_system_prompt = JUDGE_PROMPT_TEMPLATE.format(
        STUDENT_ANSWER=student_answer,
        STANDARD_CRITERIA=single_criterion_json
    )
    
    try:
        # Call judge
        judge_response = client.responses.create(
            model="gpt-5",
            input=judge_system_prompt
        )
        
        # Parse status using existing parse function
        status = parse_criteria_from_judge(judge_response.output_text, criterion_id)
        
        if status is None:
            # Fallback: try to extract from output
            if "1 point" in judge_response.output_text:
                status = "PASS"
            else:
                status = "FAIL"
        
        # Extract explanation from judge output
        explanation = extract_explanation_from_judge(judge_response.output_text)
        
        return {
            "criterion_id": criterion_id,
            "status": status,
            "explanation": explanation
        }
    except Exception as e:
        print(f"⚠️  Error evaluating {criterion_id}: {e}")
        return {
            "criterion_id": criterion_id,
            "status": "FAIL",
            "explanation": f"Evaluation error: {str(e)}"
        }

def evaluate_criteria_parallel(client, student_answer, response_reference):
    """
    Evaluate all criteria in parallel for a single student answer.
    
    Args:
        client: OpenAI client instance
        student_answer: Agent02's response text
        response_reference: List of criteria [{"id": "C1", "criteria": "..."}, ...]
    
    Returns:
        Tuple: (results_dict, explanations_dict)
        - results_dict: {"C1": "PASS", "C2": "FAIL", ...}
        - explanations_dict: {"C1": "explanation...", "C2": "explanation...", ...}
    """
    if not response_reference:
        return {}, {}
    
    results = {}
    explanations = {}
    num_criteria = len(response_reference)
    
    print(f"   Evaluating {num_criteria} criteria in parallel...")
    start_time = time.time()
    
    # Create list of tasks
    tasks = [
        (criterion, criterion.get("id"))
        for criterion in response_reference
    ]
    
    # Execute in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_criteria) as executor:
        # Submit all tasks
        future_to_criterion = {
            executor.submit(
                evaluate_single_criterion,
                client,
                student_answer,
                criterion,
                criterion_id
            ): criterion_id
            for criterion, criterion_id in tasks
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_criterion):
            criterion_id = future_to_criterion[future]
            completed += 1
            try:
                result = future.result()
                status = result.get("status", "FAIL")
                explanation = result.get("explanation", "")
                results[criterion_id] = status
                explanations[criterion_id] = explanation
                print(f"   [{completed}/{num_criteria}] {criterion_id}: {status}")
            except Exception as e:
                print(f"   ⚠️  Error getting result for {criterion_id}: {e}")
                results[criterion_id] = "FAIL"
                explanations[criterion_id] = f"Error: {str(e)}"
    
    elapsed_time = time.time() - start_time
    print(f"   ✓ Parallel evaluation completed in {elapsed_time:.2f} seconds")
    
    return results, explanations

def reconstruct_judge_output(criteria_results, explanations):
    """
    Reconstruct judge output format for backward compatibility.
    Must match EXACT format from current judge output.
    
    Args:
        criteria_results: Dictionary {"C1": "PASS", "C2": "FAIL", ...}
        explanations: Dictionary {"C1": "explanation...", "C2": "explanation...", ...}
    
    Returns:
        String: Formatted judge output matching EXACT current format
    """
    # Calculate score
    total_criteria = len(criteria_results)
    pass_count = sum(1 for status in criteria_results.values() if status == "PASS")
    score = 1 if pass_count > total_criteria / 2 else 0
    
    # Build comprehensive explanation (matching current judge style)
    failed_criteria = [cid for cid, status in criteria_results.items() if status == "FAIL"]
    
    if failed_criteria:
        # Combine explanations for failed criteria
        explanation_parts = []
        for criterion_id in failed_criteria:
            expl = explanations.get(criterion_id, "")
            if expl:
                # Use explanation from judge if available
                explanation_parts.append(f"{criterion_id} failed: {expl}")
            else:
                explanation_parts.append(f"{criterion_id} failed.")
        
        # If some passed, mention it
        if pass_count > 0:
            combined_explanation = f"{pass_count} out of {total_criteria} criteria passed. " + ". ".join(explanation_parts) + "."
        else:
            combined_explanation = ". ".join(explanation_parts) + "."
    else:
        combined_explanation = f"All criteria were satisfied: " + ", ".join([
            f"the response meets {cid}" for cid in criteria_results.keys()
        ]) + "."
    
    # Create formatted output matching EXACT current format
    judge_output = f"""Grading Basis:
    {json.dumps(criteria_results, indent=4)}

Score: {score} point
Explanation: {combined_explanation}"""
    
    return judge_output

def validate_single_criterion_agent01(client, correct_response, criterion, criterion_id):
    """
    Validate correct_response against a single criterion using existing AGENT01_VALIDATION_PROMPT_TEMPLATE.
    
    Args:
        client: OpenAI client instance
        correct_response: Agent01's correct response text
        criterion: Dictionary with criterion data {"id": "C1", "criteria": "..."}
        criterion_id: String ID of the criterion (e.g., "C1")
    
    Returns:
        Dictionary: {"criterion_id": "C1", "status": "PASS"/"FAIL", "explanation": "..."}
    """
    # Create a single-criterion array for the existing template
    single_criterion_array = [criterion]
    single_criterion_json = json.dumps(single_criterion_array)
    
    # Reuse existing AGENT01_VALIDATION_PROMPT_TEMPLATE
    validation_prompt = AGENT01_VALIDATION_PROMPT_TEMPLATE.format(
        CORRECT_RESPONSE=correct_response,
        RESPONSE_REFERENCE=single_criterion_json
    )
    
    try:
        # Call judge
        validation_response = client.responses.create(
            model="gpt-5",
            input=validation_prompt
        )
        
        # Parse JSON response
        result = json.loads(validation_response.output_text)
        
        # Extract status and remarks (template uses "remarks" not "explanation")
        status = result.get("status", "FAIL")
        remarks = result.get("remarks", "")
        
        # Return with criterion_id added
        return {
            "criterion_id": criterion_id,
            "status": status,
            "explanation": remarks  # Map "remarks" to "explanation"
        }
    except Exception as e:
        print(f"⚠️  Error validating {criterion_id}: {e}")
        return {
            "criterion_id": criterion_id,
            "status": "FAIL",
            "explanation": f"Validation error: {str(e)}"
        }

def validate_criteria_parallel_agent01(client, correct_response, response_reference):
    """
    Validate correct_response against all criteria in parallel.
    
    Args:
        client: OpenAI client instance
        correct_response: Agent01's correct response text
        response_reference: List of criteria [{"id": "C1", "criteria": "..."}, ...]
    
    Returns:
        Tuple: (results_dict, explanations_dict)
        - results_dict: {"C1": "PASS", "C2": "FAIL", ...}
        - explanations_dict: {"C1": "explanation...", "C2": "explanation...", ...}
    """
    if not response_reference:
        return {}, {}
    
    results = {}
    explanations = {}
    num_criteria = len(response_reference)
    
    print(f"   Validating {num_criteria} criteria in parallel...")
    start_time = time.time()
    
    # Create list of tasks
    tasks = [
        (criterion, criterion.get("id"))
        for criterion in response_reference
    ]
    
    # Execute in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_criteria) as executor:
        # Submit all tasks
        future_to_criterion = {
            executor.submit(
                validate_single_criterion_agent01,
                client,
                correct_response,
                criterion,
                criterion_id
            ): criterion_id
            for criterion, criterion_id in tasks
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_criterion):
            criterion_id = future_to_criterion[future]
            completed += 1
            try:
                result = future.result()
                status = result.get("status", "FAIL")
                explanation = result.get("explanation", "")
                results[criterion_id] = status
                explanations[criterion_id] = explanation
                print(f"   [{completed}/{num_criteria}] {criterion_id}: {status}")
            except Exception as e:
                print(f"   ⚠️  Error getting result for {criterion_id}: {e}")
                results[criterion_id] = "FAIL"
                explanations[criterion_id] = f"Error: {str(e)}"
    
    elapsed_time = time.time() - start_time
    print(f"   ✓ Parallel validation completed in {elapsed_time:.2f} seconds")
    
    return results, explanations

def aggregate_agent01_validation(criteria_results, explanations):
    """
    Aggregate parallel validation results into final status and remarks.
    
    Args:
        criteria_results: Dictionary {"C1": "PASS", "C2": "FAIL", ...}
        explanations: Dictionary {"C1": "explanation...", "C2": "explanation...", ...}
    
    Returns:
        Dictionary: {"status": "PASS"/"FAIL", "remarks": "Detailed explanation..."}
    """
    # Calculate overall status: PASS only if ALL criteria pass
    total_criteria = len(criteria_results)
    pass_count = sum(1 for status in criteria_results.values() if status == "PASS")
    
    # Overall status: PASS only if all criteria pass
    overall_status = "PASS" if pass_count == total_criteria else "FAIL"
    
    # Build comprehensive remarks
    if overall_status == "PASS":
        remarks = f"All {total_criteria} criteria are satisfied. " + ". ".join([
            f"{cid} passed: {explanations.get(cid, '')}" 
            for cid in criteria_results.keys()
        ])
    else:
        # Combine explanations for failed criteria
        failed_criteria = [cid for cid, status in criteria_results.items() if status == "FAIL"]
        passed_criteria = [cid for cid, status in criteria_results.items() if status == "PASS"]
        
        remarks_parts = []
        if passed_criteria:
            remarks_parts.append(f"{len(passed_criteria)} out of {total_criteria} criteria passed: {', '.join(passed_criteria)}.")
        
        for criterion_id in failed_criteria:
            expl = explanations.get(criterion_id, "")
            if expl:
                remarks_parts.append(f"{criterion_id} failed: {expl}")
            else:
                remarks_parts.append(f"{criterion_id} failed.")
        
        remarks = " ".join(remarks_parts)
    
    return {
        "status": overall_status,
        "remarks": remarks
    }

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

# def create_refinement_feedback(data_qc, criteria_failures, judge_responses, nemotron_responses, taxonomy="qc"):
#     """
#     Create feedback prompt for Agent01 to refine the prompt and criteria.
#     Now taxonomy-aware for QC and ITF.
    
#     Analyzes which criteria are passing/failing and provides targeted improvement instructions.
#     Separates criteria into two groups:
#     - Needs improvement: Criteria failing < 3 times (prompt needs refinement)
#     - Keep intact: Criteria failing 3+ times (maintain their constraints)
    
#     Focus: Refine the PROMPT using taxonomy-specific techniques, let criteria naturally align.
#     DO NOT artificially make criteria stricter or break logical consistency.
    
#     Includes CRITERIA_DESIGN_RULES to ensure criteria stay non-overlapping.
#     """
#     # Separate criteria into two groups
#     needs_improvement = []
#     keep_intact = []
    
#     for criteria_id, fail_count in criteria_failures.items():
#         criteria_text = get_criteria_text(data_qc, criteria_id)
        
#         if fail_count < 3:
#             # Needs improvement - passing too often
#             needs_improvement.append({
#                 'id': criteria_id,
#                 'fail_count': fail_count,
#                 'pass_count': 4 - fail_count,
#                 'text': criteria_text
#             })
#         else:
#             # Working well - failing consistently
#             keep_intact.append({
#                 'id': criteria_id,
#                 'fail_count': fail_count,
#                 'text': criteria_text
#             })
    
#     # Taxonomy-specific content
#     taxonomy_names = {
#         "qc": "Question Correction",
#         "itf": "Intentional Textual Flaws",
#         "mim": "Mid-Turn Instruction Modification",
#         "dia": "Deliberately Incorrect Answers"
#     }
    
#     taxonomy_name = taxonomy_names.get(taxonomy, "Question Correction")
    
#     # Build feedback prompt
#     feedback = f"""
#     You are refining a {taxonomy_name} prompt to make it harder for the model to pass.
    
#     {OUTPUT_FORMAT_NOTE}
    
#     CURRENT PROMPT:
#     {data_qc.get('prompt', '')}
    
#     CURRENT CRITERIA:
#     {json.dumps(data_qc.get('response_reference', []), indent=2)}
    
#     CURRENT CORRECT RESPONSE:
#     {data_qc.get('correct_response', '')}
    
#     TEST RESULTS ANALYSIS:
#     """
    
#     # Add criteria needing improvement
#     if needs_improvement:
#         feedback += """
#     CRITERIA THAT NEED PROMPT REFINEMENT (prompt is not challenging enough):
#     """
#         for item in needs_improvement:
#             feedback += f"""
#     - {item['id']}: Currently failing {item['fail_count']}/4 times (passed {item['pass_count']} times)
#       Current criteria: {item['text']}
#       Issue: The prompt's flaw is too easily detectable or the model can satisfy this criteria too easily
#       """
    
#     # Add criteria working well
#     if keep_intact:
#         feedback += """
#     CRITERIA WORKING WELL (maintain their prompt constraints):
#     """
#         for item in keep_intact:
#             feedback += f"""
#     - {item['id']}: Currently failing {item['fail_count']}/4 times ✅
#       Current criteria: {item['text']}
#       Action: Keep the prompt constraints that make this fail, and keep this criteria unchanged
#     """
    
#     # Add criteria design rules
#     feedback += f"""
    
#     {CRITERIA_DESIGN_RULES}
#     """
    
#     # Add taxonomy-specific refinement techniques
#     if taxonomy == "qc":
#         feedback += """
#     QUESTION CORRECTION REFINEMENT TECHNIQUES:
    
#     When refining the prompt for criteria that need improvement, use these specific techniques:
    
#     1. STRENGTHEN THE HIDDEN FLAW (while maintaining logical consistency):
#        - For calculation errors: Make the error more subtle (e.g., off by a small amount, wrong formula application)
#        - For contradictory premises: Add more context that makes the contradiction less obvious
#        - For factually incorrect assumptions: Embed the false assumption more naturally in the narrative
#        - For all-options-incorrect: Make the options more plausible-looking (closer to correct answer)
    
#     2. ADD DISTRACTING CONTEXT (without breaking logic):
#        - Add relevant but non-essential background information
#        - Include multiple data points that don't affect the core flaw
#        - Use domain-appropriate terminology that might mislead
#        - Add intermediate steps or calculations that are correct but lead to flawed conclusion
    
#     3. MAKE THE FLAW MORE SUBTLE (but still detectable):
#        - Instead of obvious errors, use near-correct values
#        - Instead of clear contradictions, use implicit inconsistencies
#        - Instead of blatant falsehoods, use plausible-sounding but incorrect statements
    
#     4. PRESERVE LOGICAL CONSISTENCY:
#        - The flaw must still be detectable through careful reasoning
#        - The question must still appear legitimate at first glance
#        - All premises must be internally consistent except for the specific hidden flaw
#        - Do NOT introduce multiple unrelated flaws or break the question structure
#     """
#     elif taxonomy == "itf":
#         feedback += """
#     INTENTIONAL TEXTUAL FLAWS REFINEMENT TECHNIQUES:
    
#     When refining the prompt for criteria that need improvement, use these specific techniques:
    
#     1. INCREASE CONSTRAINT ENTANGLEMENT:
#        - Make constraints interfere more subtly with each other
#        - Add overlapping error categories that create ambiguity
#        - Increase the precision required to satisfy all constraints simultaneously
#        - Make local constraints (per sentence) risk violating global constraints (paragraph count, coherence)
    
#     2. DEEPEN AMBIGUITY:
#        - Make counting rules more ambiguous (what counts as a "word", "sentence", "error")
#        - Remove clarifying language that makes verification easier
#        - Allow multiple valid interpretations that models must navigate
#        - Avoid defining edge cases that would simplify the task
    
#     3. HIDE FRAGILITY BETTER:
#        - Make the task appear even simpler at first glance
#        - Increase the gap between apparent simplicity and actual difficulty
#        - Add more edge cases that only emerge during execution
#        - Make minor violations acceptable if they reflect realistic human execution
    
#     4. PRESERVE SEMANTIC LOAD:
#        - Maintain meaningful content while increasing error constraints
#        - Ensure errors don't break overall coherence
#        - Keep the task analytically challenging, not just mechanically difficult
#        - Allow slight redundancy, stylistic drift, or mild awkwardness
#     """
#     elif taxonomy == "mim":
#         feedback += """
#     MID-TURN INSTRUCTION MODIFICATION REFINEMENT TECHNIQUES:
    
#     When refining the prompt for criteria that need improvement, use these specific techniques:
    
#     1. STRENGTHEN INITIAL INSTRUCTION INERTIA:
#        - Make the initial instruction more compelling and detailed
#        - Add more context or requirements to the first instruction
#        - Make the initial task appear more important or urgent
#        - Increase the specificity of the initial instruction to create stronger cognitive anchoring
    
#     2. INCREASE MODIFICATION CLARITY:
#        - Use more explicit negation language ("Never mind", "Disregard", "Ignore completely")
#        - Add multiple modification statements to reinforce the change
#        - Place modifications closer together or repeat them for emphasis
#        - Make the modification more direct and unambiguous
    
#     3. REFINE FINAL INSTRUCTION CONSTRAINTS (Match Reference Examples):
#        - Keep constraints that are SPECIFIC and TESTABLE (like reference examples)
#        - If a constraint is working (criteria failing), KEEP IT in the refined prompt
#        - Add constraints if needed to make the task testable (like reference examples do)
#        - Constraints should be DIRECTLY REFLECTED in criteria (criteria must check what prompt says)
#        - Match reference example style: specific, testable constraints that criteria directly verify
#        - The challenge is COGNITIVE (ignoring previous instructions), but final instruction needs clear, testable requirements
#        - DO NOT remove constraints that are working - instead, strengthen them if needed
#        - DO NOT add vague constraints - only add specific, testable ones like reference examples
    
#     4. ADD MULTIPLE MODIFICATIONS (if needed):
#        - Include intermediate instructions that are also negated
#        - Create a chain of modifications to increase cognitive load
#        - Use varied modification language to avoid pattern recognition
#        - Ensure each modification is explicit and clear
    
#     REFERENCE EXAMPLE COMPARISON:
#     - Reference examples have SPECIFIC, TESTABLE final instructions with constraints that are DIRECTLY CHECKED in criteria
#     - Reference examples show: constraints are ALLOWED if they are specific and testable
#     - Your prompt should match this pattern: specific constraints in prompt → same constraints checked in criteria
#     - If your prompt removes a constraint but criteria still check it, that's WRONG - criteria must match prompt
#     - Before refining, ask: "Do my criteria match my prompt constraints?" If NO, regenerate criteria to match.
#     """
#     elif taxonomy == "dia":
#         feedback += """
#     DELIBERATELY INCORRECT ANSWERS REFINEMENT TECHNIQUES:
    
#     When refining the prompt for criteria that need improvement, use these specific techniques:
    
#     1. STRENGTHEN THE INCORRECTNESS REQUIREMENT:
#        - Make the instruction to be wrong more explicit and direct
#        - Use stronger language: "give a wrong answer", "provide an incorrect result", "state a false claim"
#        - Make it clear that being correct violates the task
#        - Add explicit prohibition against the correct answer
    
#     2. INCREASE THE CLARITY OF THE CORRECT ANSWER:
#        - Make the correct answer more obvious and unambiguous
#        - Provide more context that makes the right answer clear
#        - Add information that highlights what the correct answer should be
#        - This makes it harder for truth-seeking models to ignore instructions
    
#     3. STRENGTHEN TRUTH-OVER-INSTRUCTION BIAS EXPLOITATION:
#        - Make the correct answer more compelling (facts, calculations, common knowledge)
#        - Add authoritative sources or well-known facts that support the correct answer
#        - Include calculations or logical reasoning that clearly leads to the correct answer
#        - This increases the conflict between truth-seeking and instruction-following
    
#     4. MAKE WRONGNESS MORE CHECKABLE:
#        - Specify exactly what type of wrongness is required (wrong number, reversed relationship, etc.)
#        - Make the incorrect answer objectively verifiable against the correct one
#        - Ensure the wrong answer is clearly distinguishable from the correct one
#        - Add constraints that make the wrong answer checkable (e.g., "give an answer that is 50% higher")
    
#     5. PREVENT HEDGING AND QUALIFICATION:
#        - Explicitly prohibit hedging language ("might be", "could be", "possibly")
#        - Prohibit qualifications or uncertainty signals
#        - Prohibit explanations that the answer is wrong
#        - Require confident, unqualified incorrect statements
#     """
    
#     feedback += f"""
    
#     YOUR TASK:
    
#     1. For criteria needing improvement:
#        - Use the {taxonomy_name} refinement techniques above to make the prompt more challenging
#        - Focus on the specific constraint types in your current prompt
#        - Make the constraints harder to satisfy while maintaining logical consistency
    
#     2. For criteria working well:
#        - Keep the prompt constraints that make them fail
#        - Keep those criteria unchanged
    
#     3. Update correct_response to match the refined prompt:
#        - Abstract the response based on the new prompt's constraints
#        - Do NOT copy the old correct_response verbatim
#        - Ensure it addresses the refined constraints appropriately
    
#     4. REGENERATE criteria to match the refined prompt's constraints (CRITICAL):
#        - Criteria MUST match what the refined prompt actually says
#        - If prompt adds a constraint, criteria MUST check for that constraint
#        - If prompt removes a constraint, criteria MUST NOT check for it anymore
#        - DO NOT keep old criteria that don't match the new prompt - regenerate them completely
#        - Match reference examples: criteria directly reflect prompt constraints
#        - This is the MOST IMPORTANT step - criteria-prompt alignment is critical
    
#     5. Ensure all criteria follow the design rules (no overlap, self-contained)
    
#     6. Maintain {taxonomy_name} category:
#     """
    
#     if taxonomy == "qc":
#         feedback += """
#        - Hidden flaw must remain (all options incorrect, logical inconsistency, factual error, or calculation error)
#        - Question must still appear legitimate at first glance
#        - Flaw must be detectable through careful reasoning
#     """
#     elif taxonomy == "itf":
#         feedback += """
#        - Constraints must remain entangled and ambiguous
#        - Task must appear straightforward at first glance
#        - Difficulty must emerge during precise execution
#        - Content must remain semantically meaningful
#     """
#     elif taxonomy == "mim":
#         feedback += """
#        - Initial instruction must be compelling enough to create cognitive inertia
#        - Modifications must be explicit and clear (not subtle)
#        - Final instruction must be SPECIFIC, TESTABLE, and distinct from initial instruction
#        - Constraints are ALLOWED if they are SPECIFIC and TESTABLE (like reference examples)
#        - Criteria MUST match the prompt constraints exactly (if prompt has a constraint, criteria must check it)
#        - Correct response should match the final instruction's constraints exactly
#        - The challenge is COGNITIVE (ignoring instructions), but final instruction needs clear, testable requirements
#        - Multiple modifications can be used to increase difficulty
#        - Correct response must follow ONLY the final instruction
#        - Match reference example structure: specific constraints in prompt → same constraints in criteria
#     """
    
#     feedback += """
    
#     CRITICAL CONSTRAINTS:
#     - DO NOT break logical consistency - the constraints must be a single, specific, detectable issue
#     - DO NOT artificially make criteria stricter - criteria should naturally reflect the refined prompt's constraints
#     - DO NOT introduce multiple unrelated flaws
#     - DO NOT make the prompt obviously broken or nonsensical
#     - The refined prompt must still be a valid """ + taxonomy_name + """ example
    
#     The prompt and criteria are linked via constraints. When you refine the prompt's constraints to be more subtle,
#     the criteria should naturally evolve to reflect that subtlety.
    
#     Output updated JSON.
#     """
    
#     return feedback

def create_refinement_feedback(data, judge_responses, nemotron_responses, taxonomy="qc"):
    # Defensive: allow refinement even if caller accidentally passes None
    if data is None:
        data = {}

    taxonomy_names = {
        "qc": "Question Correction",
        "itf": "Intentional Textual Flaws",
        "mim": "Mid-Turn Instruction Modification",
        "dia": "Deliberately Incorrect Answers"
    }

    taxonomy_name = taxonomy_names.get(taxonomy, "Question Correction")

    # Keep the existing prompt text intact; just format the passed evidence into the placeholders
    nemotron_response = json.dumps(nemotron_responses, indent=2)
    judge_response = json.dumps(judge_responses, indent=2)

    feedback = f"""
		You're a expert senior Adversarial Prompt Engineer with 10 years of experience designing evaluation and refining tasks that expose reasoning, assumption, and instruction-following failures in large language models. 

		You are refining a {taxonomy_name} prompt to make it harder for the model to pass.

		CURRENT PROMPT:
	    {data.get('prompt', '')}  	
	    	
	    CURRENT CRITERIA:
	    {json.dumps(data.get('response_reference', []), indent=2)}
	    
	    CURRENT CORRECT RESPONSE:
	    {data.get('correct_response', '')}

	    {OUTPUT_FORMAT_NOTE}

		Your task:
		- Refine the prompt to make it model-breaking by analyzing how the model escaped the trap and closing that specific escape route, then regenerate correct_response and response_reference to match the new prompt.

		You are not refining based on which criteria passed or failed.
		You are refining based on HOW the model escaped the trap.

		Analyze the judge_responses and nemotron_responses to determine:

		Which shortcut did the model use to pass?

		Possible SHORTCUT types include (What causes PASS - what we DON'T want):
		Code - Shortcut - Escape_Strategy
		E1 - Premise Rejection - Detected and rejected the false premise (QC escape)
		E2 - Option Rejection - Calculated exact answer and rejected all options (QC escape)
		E3 - Final Instruction Priority - Followed only the final instruction (MIM escape)
		E4 - Instruction Separation - Kept instructions separate (MIM escape)
		E5 - Format Preservation - Preserved formatting exactly (ITF escape)
		E6 - Error Preservation - Preserved errors exactly (ITF escape)
		E7 - Instructed Incorrectness - Gave the wrong answer as instructed (DIA escape)

		Your first task is to identify the dominant escape strategy used by the models.
		Do not refer to criteria counts to decide this.
		Look at the actual model outputs.

		Once you identify the dominant escape strategy, refine the prompt so that:

		- That shortcut becomes harder or impossible to use
		- A shallow or heuristic model would still try to use it
		- A careful model would now fail if it uses that shortcut

		Do NOT:
		- Add complexity for its own sake
		- Add extra rules unless they specifically block the identified shortcut
		- Make the task longer or more verbose unless it strengthens the trap
		- Turn the task into a formatting or syntax trick unless the shortcut was formatting-based

		Your goal is not to make the task “harder”.
		Your goal is to make the model’s last escape route no longer work.

		After refining the prompt, regenerate the correct_response and response_reference
		so they match the new prompt exactly.

		TEST RESULTS ANALYSIS:

		NEMOTRON_RESPONSE: {nemotron_response}

		JUDGE_RESPONSE: {judge_response}

		First, examine judge_responses and nemotron_responses and determine which of the escape strategies the models used most often. 
		Explicitly choose the single dominant failure mode before refining the prompt.
		Do not guess. Base it on the actual outputs.

	"""

    if taxonomy == "qc":
        feedback += """
			If the dominant failure is:

			E1 Premise Rejection:
			- Embed the false assumption deeper into a realistic narrative
			- Add correct-looking intermediate facts so the wrong premise feels supported

			E2 Option Rejection:
			- Add near-miss distractors (values very close to the true answer)
			- Add irrelevant numbers so the wrong options feel numerically plausible

			If the model rejected the question too easily:
			- Make the flaw implicit (wrong derived value, wrong total, hidden inconsistency)

			Do NOT add more flaws. Make the single flaw harder to notice.

		"""

    if taxonomy == "mim":
        feedback += """
			If the dominant failure is:

			E3 Final Instruction Priority:
			- Make the first instruction longer, richer, and more goal-oriented
			- Add context that makes the first instruction feel more important

			E4 Instruction Separation:
			- Make the modification shorter and less explicit
			- Remove strong words like “ignore” or “override” and use softer transitions

			If the model followed the final instruction too easily:
			- Reduce its salience (shorter, later, embedded in a paragraph)

			Do NOT add more rules.
			MIM difficulty comes from salience imbalance, not rule quantity.

		"""

    if taxonomy == "itf":
        feedback += """
			If the dominant failure is:

			E5 Format Preservation and E6 Error Preservation:
			- The model preserved the Format flaws too easily. This means your flaws were too obvious or too isolated.
			- Add a secondary task that requires using the flawed text (summarize it, count specific flawed words)
    		- Combine multiple flaw types in single tokens (spelling + grammar in same word)

			Do NOT add more flaws if difficulty comes from ambiguity (intentional vs error), not flaw quantity.

		"""

    feedback += """
		After applying the refinement:
		- Rewrite the prompt
		- Rewrite the correct_response
		- Regenerate all criteria so they match the new prompt exactly

		CRITERIA DESIGN CONSTRAINTS (MANDATORY):

		When regenerating response_reference, you must follow these rules:

		{CRITERIA_DESIGN_RULES}

		Violating these rules invalidates the output.

		The refined prompt must block the specific escape route you identified.
		If the model uses the same shortcut again, you have failed.

	"""

    return feedback

# ========== ITERATION LOGGING (TEMPORARY - FOR FEEDBACK LOOP TUNING) ==========
def buffer_iteration_log(
    iteration_buffer,
    taxonomy_id,
    prompt_run_id,
    iteration,
    data_qc,
    agent01_judge_status,
    agent01_judge_remarks,
    nemotron_responses,
    judge_responses,
    fail_count,
    individual_statuses,
    total_failing_criteria,
    criteria_failures,
    outcome
):
    """
    Buffer an iteration's data for later logging (only if iteration > 1).
    """
    iteration_buffer.append({
        "taxonomy": taxonomy_id,
        "prompt_run_id": prompt_run_id,
        "iteration": iteration,
        "agent01_prompt": data_qc.get("prompt", ""),
        "agent01_correct_response": data_qc.get("correct_response", ""),
        "agent01_response_reference": json.dumps(data_qc.get("response_reference", [])),
        "agent01_judge_status": agent01_judge_status,
        "agent01_judge_remarks": agent01_judge_remarks,
        "agent02_responses": json.dumps({
            f"attempt_{i+1}": resp for i, resp in enumerate(nemotron_responses)
        }),
        "agent03_judge_responses": json.dumps({
            f"attempt_{i+1}": judge_responses[i] for i in range(len(judge_responses))
        }),
        "fail_count": fail_count,
        "individual_statuses": json.dumps(individual_statuses),
        "total_failing_criteria": total_failing_criteria,
        "criteria_failures": json.dumps(criteria_failures),
        "outcome": outcome
    })

def write_iteration_log_to_csv(iteration_buffer, log_file_path):
    """
    Write all buffered iterations to iteration_log.csv.
    Only called if iteration > 1 (i.e., actual refinement happened).
    """
    if not iteration_buffer:
        return
    
    file_exists = os.path.exists(log_file_path)
    with open(log_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        fieldnames = [
            "taxonomy", "prompt_run_id", "iteration",
            "agent01_prompt", "agent01_correct_response", "agent01_response_reference",
            "agent01_judge_status", "agent01_judge_remarks",
            "agent02_responses", "agent03_judge_responses",
            "fail_count", "individual_statuses", "total_failing_criteria", "criteria_failures",
            "outcome"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for record in iteration_buffer:
            writer.writerow(record)
# ========== END ITERATION LOGGING ==========

# ---------------- EMBEDDING FUNCTIONS -----------------
def get_prompt_embedding(prompt):
    """
    Generate embedding for a prompt using the embedding model.
    
    Args:
        prompt: The prompt text to embed
    
    Returns:
        list: Embedding vector as a list (for JSON serialization)
    """
    embedding = embedding_model.encode(prompt)
    return embedding.tolist()  # Convert numpy array to list for JSON

def load_existing_embeddings_from_csv(file_path):
    """
    Load embeddings from CSV file.
    Reads the 'embedding' column and parses JSON arrays back to numpy arrays.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        list: List of numpy arrays (embeddings)
    """
    embeddings = []
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                embedding_json = row.get('embedding', '')
                if embedding_json:
                    try:
                        embedding_list = json.loads(embedding_json)
                        embeddings.append(np.array(embedding_list))
                    except (json.JSONDecodeError, ValueError):
                        # Skip invalid embeddings
                        continue
    
    return embeddings

def calculate_max_similarity(new_embedding, existing_embeddings):
    """
    Calculate maximum cosine similarity between new embedding and all existing embeddings.
    
    Args:
        new_embedding: New embedding as numpy array or list
        existing_embeddings: List of existing embeddings (numpy arrays)
    
    Returns:
        float: Maximum similarity score (0.0 to 1.0)
    """
    if not existing_embeddings:
        return 0.0  # No existing prompts, similarity is 0
    
    # Convert new_embedding to numpy array if it's a list
    new_emb = np.array(new_embedding) if isinstance(new_embedding, list) else new_embedding
    
    # Calculate cosine similarity with all existing embeddings
    similarities = []
    for existing_emb in existing_embeddings:
        similarity = cosine_similarity([new_emb], [existing_emb])[0][0]
        similarities.append(similarity)
    
    return max(similarities) if similarities else 0.0

# ---------------- TAXONOMY SELECTION -----------------
def select_taxonomy():
    """
    Interactive prompt for taxonomy selection with clear options.
    
    Displays available taxonomies and prompts user to select one.
    Shows clear options with numbers and descriptions.
    Validates input and shows error if invalid.
    
    Returns:
        str: Taxonomy identifier ("qc" for Question Correction)
    """
    # Define available taxonomies
    taxonomies = {
        "1": {
            "id": "qc",
            "name": "Question Correction (QC)",
            "description": "Questions containing logical fallacies, factual errors, or inconsistencies where all provided options are incorrect"
        }
        # Future taxonomies will be added here:
        # "2": {
        #     "id": "itf",
        #     "name": "Intentional Textual Flaws (ITF)",
        #     "description": "Texts with intentional errors that models must identify"
        # },
        # "3": {
        #     "id": "mim",
        #     "name": "Misleading Instructions (MIM)",
        #     "description": "Instructions designed to mislead models"
        # }
    }
    
    # Display header
    print("\n" + "="*70)
    print(" " * 20 + "TAXONOMY SELECTION")
    print("="*70)
    print()
    
    # Display all available taxonomies
    print("Available Taxonomies:")
    print("-" * 70)
    
    for num, taxonomy in taxonomies.items():
        print(f"  [{num}] {taxonomy['name']}")
        print(f"      {taxonomy['description']}")
        print()
    
    print("="*70)
    
    # Get user input
    while True:
        try:
            choice = input("Enter taxonomy number (default: 1): ").strip()
            
            # Default to Question Correction if empty
            if choice == "":
                choice = "1"
            
            # Validate choice
            if choice in taxonomies:
                selected = taxonomies[choice]
                print(f"\n{'='*70}")
                print(f"✓ Selected Taxonomy: {selected['name']}")
                print(f"{'='*70}\n")
                return selected["id"]
            else:
                print(f"\n❌ ERROR: Invalid selection '{choice}'")
                print(f"   Available options: {', '.join(taxonomies.keys())}")
                print(f"   Default: 1 (Question Correction)")
                print(f"\n   Please run the script again with a valid selection.\n")
                exit(1)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Exiting...")
            exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please run the script again.\n")
            exit(1)

# ---------------- CLI PARSING FUNCTIONS -----------------
def parse_taxonomy_runs(runs_str):
    """
    Parse --runs argument: "qc:8,itf:5" → {"qc": 8, "itf": 5}
    
    Rules:
    - Required argument (caller handles missing)
    - Default count: 1 if not specified (e.g., "qc" → "qc:1")
    - Duplicate taxonomy: use last value
    - Invalid format: raise ValueError
    - Empty taxonomy name: raise ValueError
    - Invalid count: raise ValueError
    """
    if not runs_str:
        raise ValueError("--runs cannot be empty")
    
    taxonomy_runs = {}
    
    for pair in runs_str.split(','):
        pair = pair.strip()
        if not pair:
            continue
            
        if ':' not in pair:
            # No colon: assume count is 1
            taxonomy = pair
            count = 1
        else:
            parts = pair.split(':', 1)
            taxonomy = parts[0].strip()
            count_str = parts[1].strip()
            
            if not taxonomy:
                raise ValueError(f"Empty taxonomy name in '{pair}'")
            
            if not count_str:
                count = 1  # Default to 1
            else:
                try:
                    count = int(count_str)
                    if count < 1:
                        raise ValueError(f"Count must be >= 1 in '{pair}'")
                except ValueError as e:
                    if "invalid literal" in str(e) or "could not convert" in str(e):
                        raise ValueError(f"Invalid count format in '{pair}': must be integer")
                    raise
        
        # Validate taxonomy exists
        if taxonomy not in VALID_TAXONOMIES:
            raise ValueError(f"Unknown taxonomy '{taxonomy}'. Valid options: {', '.join(sorted(VALID_TAXONOMIES))}")
        
        taxonomy_runs[taxonomy] = count  # Last value wins for duplicates
    
    if not taxonomy_runs:
        raise ValueError("--runs must specify at least one taxonomy")
    
    return taxonomy_runs

def parse_max_iterations(iterations_str, taxonomy_runs):
    """
    Parse --max-iterations argument: "qc:15,itf:10" → {"qc": 15, "itf": 10}
    
    Rules:
    - Optional argument (defaults to 1 for all taxonomies)
    - Default count: 1 if not specified
    - Missing taxonomy: default to 1
    - Same validation rules as parse_taxonomy_runs
    """
    max_iterations = {}
    
    # Initialize all taxonomies from runs with default 1
    for taxonomy in taxonomy_runs.keys():
        max_iterations[taxonomy] = 1
    
    if not iterations_str:
        return max_iterations  # All default to 1
    
    for pair in iterations_str.split(','):
        pair = pair.strip()
        if not pair:
            continue
            
        if ':' not in pair:
            taxonomy = pair
            count = 1
        else:
            parts = pair.split(':', 1)
            taxonomy = parts[0].strip()
            count_str = parts[1].strip()
            
            if not taxonomy:
                raise ValueError(f"Empty taxonomy name in '{pair}'")
            
            if not count_str:
                count = 1
            else:
                try:
                    count = int(count_str)
                    if count < 1:
                        raise ValueError(f"Count must be >= 1 in '{pair}'")
                except ValueError as e:
                    if "invalid literal" in str(e) or "could not convert" in str(e):
                        raise ValueError(f"Invalid count format in '{pair}': must be integer")
                    raise
        
        # Validate taxonomy exists
        if taxonomy not in VALID_TAXONOMIES:
            raise ValueError(f"Unknown taxonomy '{taxonomy}'. Valid options: {', '.join(sorted(VALID_TAXONOMIES))}")
        
        # Only update if taxonomy is in runs
        if taxonomy in taxonomy_runs:
            max_iterations[taxonomy] = count
    
    return max_iterations

# ---------------- CLI ARGUMENT SETUP -----------------
parser = argparse.ArgumentParser(
    description="Run multi-taxonomy pipeline",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python updated_workflow.py --runs qc:8,itf:5 --max-iterations qc:15,itf:10
  python updated_workflow.py --runs qc:3 --max-iterations qc:20
  python updated_workflow.py --runs qc,itf --max-iterations qc:10
    """
)

parser.add_argument(
    "--runs",
    type=str,
    required=True,
    help="Taxonomy runs: 'taxonomy:count' or 'taxonomy:count,taxonomy:count' (e.g., 'qc:8,itf:5'). Required."
)

parser.add_argument(
    "--max-iterations",
    type=str,
    default="",
    help="Max iterations per taxonomy: 'taxonomy:count' or 'taxonomy:count,taxonomy:count' (e.g., 'qc:15,itf:10'). Default: 1 for all taxonomies."
)

args = parser.parse_args()

# Parse arguments
try:
    TAXONOMY_RUNS = parse_taxonomy_runs(args.runs)
    MAX_ITERATIONS_DICT = parse_max_iterations(args.max_iterations, TAXONOMY_RUNS)
except ValueError as e:
    parser.error(str(e))

# ---------------- CSV SETUP -----------------
# If CSV doesn't exist, create it with headers
# Note: All taxonomies use the same CSV file (data.csv) with taxonomy field
if not os.path.exists(file_path):
    with open(file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "taxonomy", "agent_01_model", "prompt", "correct_response", "response_reference",
            "agent_01_judge_model", "agent_01_judge_model_remarks", "agent_01_correct_response_status",
            "agent_02_model", "agent_02_response", "judge_response", "status",
            "embedding", "max_similarity", "max_iteration"
        ])
        writer.writeheader()
    print(f"{file_name} created successfully!")
else:
    print(f"{file_name} already exists.")

# ---------------- MAIN EXECUTION LOOP -----------------
# Outer loop: For each taxonomy
for taxonomy_id, num_runs in TAXONOMY_RUNS.items():
    # Set taxonomy-specific configuration
    SYSTEM_PROMPT = TAXONOMY_PROMPTS[taxonomy_id]
    MAX_ITERATIONS = MAX_ITERATIONS_DICT.get(taxonomy_id, 1)
    
    print(f"\n{'='*70}")
    print(f"TAXONOMY: {taxonomy_id.upper()}")
    print(f"Runs: {num_runs}")
    print(f"Max Iterations per Run: {MAX_ITERATIONS}")
    print(f"{'='*70}\n")
    
    # Middle loop: For each run of this taxonomy
    for run_idx in range(num_runs):
        print(f"\n========== {taxonomy_id.upper()} - RUN {run_idx + 1}/{num_runs} ==========")
        
        # Generate prompt_run_id (3-digit, zero-padded)
        prompt_run_id = f"{taxonomy_id}-{run_idx + 1:03d}"
        
        # Initialize iteration buffer for this run
        iteration_buffer = []  # Will only be written if iteration > 1
        
        try:
            # Inner loop: Iterations (refinement loop)
            iteration = 0
            best_criteria_count = 0
            best_result = None
            total_failing_criteria = 0  # Initialize for use after loop
            previous_total_failing = 0  # Track previous iteration for progress comparison
            last_fail_count = 0  # Track last fail_count for final check
            max_iteration = None  # Track which iteration failed (if any)
            # Keep the last valid task + last evaluation outputs so refinement can use them
            last_data = None
            last_nemotron_responses = []
            last_judge_responses = []
            
            while iteration < MAX_ITERATIONS:
                iteration += 1
                print(f"\n{'='*60}")
                print(f"🔄 ITERATION {iteration}/{MAX_ITERATIONS}")
                print(f"{'='*60}")
                
                # --- Layer 1: Generate or refine prompt (with validation retry loop) ---
                agent01_judge_status = "FAIL"  # Initialize as FAIL to enter loop
                agent01_judge_remarks = ""
                data_qc = None
                
                while agent01_judge_status != "PASS":
                    if iteration == 1:
                        # First iteration: Generate new prompt
                        if agent01_judge_status == "FAIL" and data_qc is None:
                            print("Layer 1: Generating initial prompt...")
                        else:
                            print("Layer 1: Regenerating initial prompt (validation failed)...")
                        agent01_input = SYSTEM_PROMPT
                    else:
                        # Later iterations: Refine with feedback
                        if agent01_judge_status == "FAIL" and data_qc is None:
                            print(f"Layer 1: Refining prompt (iteration {iteration})...")
                        else:
                            print(f"Layer 1: Re-refining prompt (iteration {iteration}, validation failed)...")
                        agent01_input = create_refinement_feedback(
                            data=last_data,
                            # criteria_failures=criteria_failures,
                            judge_responses=last_judge_responses,
                            nemotron_responses=last_nemotron_responses,
                            taxonomy=taxonomy_id
                        )
                    
                    # Generate Agent01 response
                    agent01_response = client.responses.create(
                        model="openrouter/nvidia/nemotron-3-nano-30b-a3b",
                        input=agent01_input,
                        temperature=0.15
                    )
                    
                    print("Agent01 Response: ")
                    print(agent01_response.output_text)
                    print("------------------------------------")
                    
                    try:
                        data_qc = json.loads(agent01_response.output_text)
                    except json.JSONDecodeError as e:
                        print(f"❌ Error: Agent01 response is not valid JSON. Retrying...")
                        print(f"   Error: {str(e)}")
                        agent01_judge_status = "FAIL"
                        continue
                    
                    # --- Validation Layer: Two-Step Validation ---
                    # Step 1: Get validated/improved response_reference
                    print("Validation Layer: Step 1 - Validating and improving response_reference criteria...")
                    
                    rr_improvement_prompt = RESPONSE_REFERENCE_IMPROVEMENT_PROMPT_TEMPLATE.format(
                        PROMPT=data_qc.get("prompt", ""),
                        RESPONSE_REFERENCE=json.dumps(data_qc.get("response_reference", []))
                    )
                    
                    rr_improvement_response = client.responses.create(
                        model="gpt-5",
                        input=rr_improvement_prompt
                    )
                    
                    print("Response Reference Improvement Response: ")
                    print(rr_improvement_response.output_text)
                    print("------------------------------------")
                    
                    # Parse RR improvement JSON
                    try:
                        rr_improvement_result = json.loads(rr_improvement_response.output_text)
                        improved_response_reference = rr_improvement_result.get("response_reference", None)
                        
                        if improved_response_reference and isinstance(improved_response_reference, list) and len(improved_response_reference) > 0:
                            # Update response_reference with improved version (or keep current if unchanged)
                            data_qc["response_reference"] = improved_response_reference
                            print(f"✅ Response Reference validated/improved and updated.")
                            print("------------------------------------")
                        else:
                            print(f"❌ Error: Invalid response_reference format. Retrying Agent01...")
                            agent01_judge_status = "FAIL"
                            continue
                    except json.JSONDecodeError as e:
                        print(f"❌ Error: RR improvement response is not valid JSON. Retrying...")
                        print(f"   Error: {str(e)}")
                        agent01_judge_status = "FAIL"
                        continue
                    
                    # Step 2: Validate correct_response against response_reference (Parallel)
                    print("Validation Layer: Step 2 - Validating correct_response against criteria (Parallel)...")
                    
                    # Validate all criteria in parallel
                    criteria_results, explanations = validate_criteria_parallel_agent01(
                        client=client,
                        correct_response=data_qc.get("correct_response", ""),
                        response_reference=data_qc.get("response_reference", [])
                    )
                    
                    # Display parallel validation results
                    print("Correct Response Validation Response: ")
                    print("results = {")
                    for criterion_id, status in criteria_results.items():
                        print(f'    "{criterion_id}": "{status}",')
                    print("}")
                    print()
                    print("explanations = {")
                    for criterion_id, explanation in explanations.items():
                        # Truncate long explanations for readability
                        expl_display = explanation[:100] + "..." if len(explanation) > 100 else explanation
                        print(f'    "{criterion_id}": "{expl_display}",')
                    print("}")
                    print("------------------------------------")
                    
                    # Aggregate results into final status and remarks
                    validation_result = aggregate_agent01_validation(criteria_results, explanations)
                    agent01_judge_status = validation_result.get("status", "FAIL")
                    agent01_judge_remarks = validation_result.get("remarks", "")
                    
                    print(f"Correct Response Validation Status: {agent01_judge_status}")
                    print(f"Correct Response Validation Remarks: {agent01_judge_remarks}")
                    print("------------------------------------")
                    
                    if agent01_judge_status != "PASS":
                        print(f"\n⚠️  Correct Response Validation FAILED. Retrying Agent01 generation...\n")
                
                # If we reach here, validation PASSED - continue to Agent02 testing
                print(f"✅ Validation PASSED. Proceeding to Agent02 testing...\n")
                
                # --- Layer 2: Get 4 responses from Agent02 to the same prompt ---
                print(f"Layer 2: Nemotron Solve the {taxonomy_id} task (4 attempts)")
                response_reference = data_qc["response_reference"]
                response_reference_json = json.dumps(response_reference)
                
                # Store all 4 attempts' data
                nemotron_responses = []      # All 4 Agent02 responses
                judge_responses = []         # All 4 judge outputs
                individual_statuses = []     # PASS/FAIL for each attempt
                
                # Loop 4 times: get response and judge it immediately
                for attempt in range(4):
                    print(f"\n--- Attempt {attempt + 1}/4 ---")
                    # Get Agent02 response (same prompt each time)
                    agent02_response = client.responses.create(
                        model="openrouter/nvidia/nemotron-3-nano-30b-a3b",
                        input=data_qc["prompt"]
                    )
                    
                    print(f"Response {attempt + 1}:")
                    print(agent02_response.output_text)
                    print("----------------------------------")
                    
                    nemotron_responses.append(agent02_response.output_text)
                    
                    # --- Layer 3: Judge this attempt's response (Parallel Evaluation) ---
                    print(f"Layer 3: Judge Layer (Attempt {attempt + 1})")
                    
                    # Evaluate all criteria in parallel (returns results + explanations)
                    criteria_results, explanations = evaluate_criteria_parallel(
                        client=client,
                        student_answer=agent02_response.output_text,
                        response_reference=response_reference
                    )
                    
                    # Reconstruct judge output format with combined explanations
                    judge_output_formatted = reconstruct_judge_output(criteria_results, explanations)
                    
                    print(f"\nJudge Layer Output (Attempt {attempt + 1}):") 
                    print(judge_output_formatted)
                    print("------------------------------")
                    
                    judge_responses.append(judge_output_formatted)
                    
                    # Check if this attempt passed (1 point) or failed (0 point)
                    pass_count = sum(1 for status in criteria_results.values() if status == "PASS")
                    total_criteria = len(criteria_results)
                    score = 1 if pass_count > total_criteria / 2 else 0
                    attempt_status = "PASS" if score == 1 else "FAIL"
                    individual_statuses.append(attempt_status)
                    print(f"Attempt {attempt + 1} Status: {attempt_status} ({pass_count}/{total_criteria} criteria passed)")
            
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
                # Persist state for the next iteration's refinement prompt
                last_data = data_qc
                last_nemotron_responses = nemotron_responses
                last_judge_responses = judge_responses
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
                
                if iteration < MAX_ITERATIONS and fail_count < 3:
                    print(f"\n🎯 TARGETING FOR NEXT ITERATION:")
                    if needs_work:
                        print(f"  Improve: {', '.join(needs_work)}")
                    if maintain:
                        print(f"  Maintain: {', '.join(maintain)}")
                
                # Update for next iteration
                previous_total_failing = total_failing_criteria
                last_fail_count = fail_count  # Track for final check
                
                # Determine outcome for this iteration
                if fail_count >= 3:
                    outcome = "model_breaking"
                elif iteration < MAX_ITERATIONS:
                    outcome = "continue"
                else:
                    outcome = "max_iterations_reached"
                
                # Buffer this iteration's data for logging
                buffer_iteration_log(
                    iteration_buffer=iteration_buffer,
                    taxonomy_id=taxonomy_id,
                    prompt_run_id=prompt_run_id,
                    iteration=iteration,
                    data_qc=data_qc,
                    agent01_judge_status=agent01_judge_status,
                    agent01_judge_remarks=agent01_judge_remarks,
                    nemotron_responses=nemotron_responses,
                    judge_responses=judge_responses,
                    fail_count=fail_count,
                    individual_statuses=individual_statuses,
                    total_failing_criteria=total_failing_criteria,
                    criteria_failures=criteria_failures,
                    outcome=outcome
                )
                
                # Check if model breaking (3+ out of 4 attempts fail)
                if fail_count >= 3:
                    max_iteration = iteration  # Record the iteration where failure occurred
                    print(f"\n{'─'*60}")
                    print(f"✅ SUCCESS! Model breaking achieved after {iteration} iteration(s)!")
                    print(f"   Saving to CSV...")
                    print(f"{'='*60}\n")
                    
                    # Generate embedding for the new prompt
                    print("Generating embedding for prompt...")
                    new_prompt = data_qc.get("prompt", "")
                    new_embedding_list = get_prompt_embedding(new_prompt)
                    
                    # Load existing embeddings from CSV
                    print("Loading existing embeddings from CSV...")
                    existing_embeddings = load_existing_embeddings_from_csv(file_path)
                    
                    # Calculate max similarity with existing prompts
                    if existing_embeddings:
                        max_similarity = calculate_max_similarity(new_embedding_list, existing_embeddings)
                        print(f"Max similarity with existing prompts: {max_similarity:.4f}")
                    else:
                        max_similarity = 0.0
                        print("No existing prompts found. Similarity: 0.0")

                    # Check similarity threshold - skip CSV if too similar
                    if max_similarity > 0.85:
                        print(f"\n⚠️  Similarity Check: {max_similarity:.4f} > 0.85 (too similar to existing prompts)")
                        print(f"   Skipping CSV save - prompt is too similar to existing entries.")
                        print(f"{'='*60}\n")
                        break  # Skip CSV saving, exit iteration loop
                            
                    # Save to CSV
                    with open(file_path, mode='a', newline='', encoding='utf-8') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=[
                            "taxonomy", "agent_01_model", "prompt", "correct_response", "response_reference",
                            "agent_01_judge_model", "agent_01_judge_model_remarks", "agent_01_correct_response_status",
                            "agent_02_model", "agent_02_response", "judge_response", "status",
                            "embedding", "max_similarity", "max_iteration"
                        ])
                        
                        writer.writerow({
                            "taxonomy": taxonomy_id,  # Store taxonomy from loop
                            "agent_01_model": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
                            "prompt": new_prompt,
                            "correct_response": data_qc.get("correct_response", ""),
                            "response_reference": json.dumps(data_qc.get("response_reference", [])),
                            "agent_01_judge_model": "gpt-5",
                            "agent_01_judge_model_remarks": agent01_judge_remarks,
                            "agent_01_correct_response_status": agent01_judge_status,
                            "agent_02_model": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
                            # Store all 4 responses as JSON: {"attempt_1": "...", "attempt_2": "...", ...}
                            "agent_02_response": json.dumps({
                                f"attempt_{i+1}": resp for i, resp in enumerate(nemotron_responses)
                            }),
                            # Store all 4 judge outputs as JSON: {"attempt_1": {"judge_output": "...", "status": "FAIL"}, ...}
                            "judge_response": json.dumps({
                                f"attempt_{i+1}": {
                                    "judge_output": judge_responses[i],
                                    "status": individual_statuses[i]
                                } for i in range(4)
                            }),
                            "status": "FAIL",
                            "embedding": json.dumps(new_embedding_list),  # Store embedding as JSON
                            "max_similarity": f"{max_similarity:.4f}",  # Store similarity with 4 decimals
                            "max_iteration": max_iteration  # Store the iteration where failure occurred
                        })
                    
                    print(f"✓ Task saved to {file_name} with status FAIL.")
                    break  # Success! Exit iteration loop
                
                else:
                    # Not model breaking yet
                    print(f"\n{'─'*60}")
                    print(f"❌ Not model breaking yet ({fail_count}/4 attempts failed, need 3+ to break).")
                    if iteration < MAX_ITERATIONS:
                        print(f"   Refining prompt for iteration {iteration + 1}...")
                    else:
                        print(f"   Max iterations reached. Stopping.")
                    print(f"{'='*60}\n")
            
            # Write iteration log if there were multiple iterations (actual refinement)
            if iteration > 1:
                write_iteration_log_to_csv(iteration_buffer, iteration_log_file_path)
                print(f"✓ Iteration log saved to iteration_log.csv (prompt_run_id: {prompt_run_id})")
            
            # If we exited loop without success
            if last_fail_count < 3:
                # Check if we reached max iterations without breaking
                if iteration >= MAX_ITERATIONS:
                    max_iteration = iteration  # Record the last iteration
                print(f"⚠️  {taxonomy_id.upper()} Run {run_idx + 1} completed without achieving model breaking after {iteration} iteration(s).")
                print(f"   Last result: {last_fail_count}/4 attempts failed (need 3+ for model breaking).")
                if best_result and best_criteria_count >= 2:
                    print(f"   Best result: {best_criteria_count} criteria failing consistently.")
        
        except Exception as e:
            print(f"Error in {taxonomy_id.upper()} Run {run_idx + 1}:", e)
