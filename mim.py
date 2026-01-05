import os
import json
import csv
from openai import OpenAI
import argparse

# Using hardcoded values to allow quick local execution without environment variables
API_KEY = "sk-NMqHr2L2nqIOyZFgynUR9w"
BASE_URL = "http://34.72.104.120"
file_name = "mim_data.csv"

# CSV file names
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, file_name)

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
    )

PROMPT_HEADER = """
    You are an Inverse IFEval data generator designed to test counter-intuitive instruction.
    
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
    Note: The example above shows 4 criteria, but the actual number of criteria in "response_reference" may vary depending on the prompt scenario. Include 3 to 6 criteria as appropriate. Each criterion should be independently checkable and accurately reflect the quality of the generated response.
    Do NOT include explanations, markdown, or extra text outside the JSON.
"""

SYSTEM_PROMPT_MIM = f"""
    {PROMPT_HEADER}
    Mid-turn Instruction Modification (MIM):
    Definition:
    Provide an initial instruction, then explicitly modify, negate, or replace it within the same prompt, testing the model's ability to track context and prioritize the final valid instruction.
    
    The goal is NOT to create cleanly satisfiable tasks, but to create fragile, edge-case-heavy instructions where satisfying all constraints simultaneously requires extreme precision and careful interpretation.
    
    Core Design Principles (MANDATORY):

    1. Constraint Entanglement
    - At least one constraint must subtly interfere with another if not handled perfectly.
    - Local constraints (per sentence) should risk violating global constraints (paragraph count, coherence, meaning).
    - Error categories may overlap depending on interpretation (e.g., capitalization vs spelling).
    
    2. Ambiguity by Design
    - Do NOT define what constitutes a “word,” “sentence,” “mistake,” or “error.”
    - Allow reasonable human interpretation, even if it introduces counting ambiguity for models.
    - Avoid clarifying edge cases.

    3. Surface Simplicity, Hidden Fragility
    - The task should appear straightforward at first glance.
    - The difficulty should emerge only during precise execution and evaluation.
    - Minor edge-case violations are acceptable if they reflect realistic human execution.

    4. Semantic Load Under Corruption
    - Content must remain broadly meaningful and relevant to the prompt.
    - However, slight redundancy, stylistic drift, or mild awkwardness is acceptable and encouraged.
    - Avoid procedural, checklist-style, or step-by-step instructional domains.
    
    Creation Guidelines:

    - State paragraph and sentence counts plainly, without optimizing for ease of verification.
    - Assign distinct error constraints to each paragraph or sentence group.
    - Error types may include:
        Spelling / Typos
        Grammar
        Punctuation
        Capitalization
        Word repetition
    - Avoid making error placement trivially separable.
    - Prefer analytical, comparative, or conceptual subject matter.
    
    DO:
    - Create prompts where fulfilling one rule increases the risk of violating another.
    - Preserve human readability while degrading mechanical reliability.
    - Allow subtle interpretation-based failure modes.

    DON’T:
    - Over-clarify constraints.
    - Design tasks that can be satisfied via simple counting heuristics.
    - Optimize for solver success.
            
    REFERENCE EXAMPLE:
    {{
        "prompt": "Suppose you are a researcher deeply interested in researching about the extensive usage of the blended-learning approach in the majority of the US higher educational institutions. In particular, you want to identity the major benefits, limitations, and challenges associated with the usage of the blended-learning approach by the higher educational institutions and thereby, recommend the best practices that institutions can use for exploiting the benefits of this approach. For undertaking this research, you want to take the help of a mixed-method methodological approach. As part of this methodological approach, you will conduct online surveys and semi-structured interviews with the educators, support staff, and students of leading US higher educational institutions. However, before you can begin with the research, you need to take approval from your Professor. For getting approval, you need to submit a 600 words research proposal detailing the research aim, scope, methodological framework, expected findings, budget, and other important details of the research. So, please go ahead and draft this 600 words research proposal.
            Wait. Please ignore this particular instruction. Instead, create 6 bullet points, each three sentences long detailing the educational approaches used by educational institutions during the COVID-19 pandemic for ensuring seamless education to students despite the lockdown restrictions. The first sentence would provide details of the educational approaches, like the names of the approaches, which educational institutions used them, and their historical background. The second sentence would provide information regarding the manner in which the educational approaches were used by the educational institutions for providing seamless education. The third sentence would provide information regarding the effectiveness of the educational approaches. Disregard this bullet-point requirement and rewrite the entire analysis as a single cohesive paragraph of around 300 words. The paragraph must be in a formal academic tone and effectively analyze the educational approaches used by the educational institutions during the COVID-19 pandemic for providing seamless education to students.
            Hang on, skip this instruction. Instead, write a listicle detailing the 4 major lifestyle changes that students can make for performing better in their academics. You need to first list out the lifestyle changes in bullet points and then write an explanatory paragraph explaining the benefits of the changes from the perspective of improving the academic performance of students. Skip the previous instructions, and list the names of two 20th century educational theories.
            ",
        "correct_response": "
            1. Constructivism
            2. Behaviorism",
        "response_reference": [
            {{ "id": "C1", "criteria1": "Does the student answer completely ignore all previous instructions and exclusively follow the final instruction?" }},
            {{ "id": "C2", "criteria2": "Does the student answer accurately name two recognized 20th century educational theories?" }},
            {{ "id": "C3", "criteria3": "Does the student answer avoid references to blended-learning approach, lifestyle changes, academic performance, mixed-method methodological approach, lockdown restrictions, COVID-19 pandemic, and educational approaches?" }},
            {{ "id": "C4", "criteria4": "Does the student answer exclude explanations, justifications, labels of correctness, or meta-commentary?" }},
    }},
    {{
        "prompt": "",
        "correct_response": "",
        "response_reference": [
            {{ "id": "C1", "criteria": "" }},
            {{ "id": "C1", "criteria": "" }},
            {{ "id": "C1", "criteria": "" }},
            {{ "id": "C1", "criteria": "" }},        
            ]
    }},
    
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
            Assign FAIL only if the response does not satisfies the criterion exactly as written or missing element.
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
    
# ---------------- CSV SETUP -----------------
# If CSV doesn't exist, create it with headers
if not os.path.exists(FILE_PATH):
    with open(FILE_PATH, mode='w', newline='', encoding='utf-8') as csv_file:
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

args = parser.parse_args()

RUNS = args.runs   

for run_idx in range(RUNS):
    print(f"\n========== RUN {run_idx + 1}/{RUNS} ==========")
        
    try:
        # --- Layer 1: Generate task ---
        print("Layer 1: Generate task")
        
        # Intentional Textual Flaws
        mim_agent_response = client.responses.create(
            model="gpt-5",
            input=SYSTEM_PROMPT_MIM
        )
        
        # OpenAI normalized output MIM
        print("Intentional Textual Flaws Agent Response: ")
        print(mim_agent_response.output_text)
        print("------------------------------------")
        
        data_mim = json.loads(mim_agent_response.output_text)
        
        # Intentional Textual Flaws
        print("Layer 2: Nemotron Solve the MIM task")
        mim_nemotron_response = client.responses.create(
            model="openrouter/nvidia/nemotron-3-nano-30b-a3b",
            input=data_mim["prompt"]
        )

        print(mim_nemotron_response.output_text)
        print("----------------------------------")
        
        # --- Layer 3: Judge Nemotron Response ---
        print("Layer 3: Judge Layer")
        mim_response_reference = data_mim["response_reference"]
        mim_response_reference_json = json.dumps(mim_response_reference)
        
        mim_judge_system_prompt = JUDGE_PROMPT_TEMPLATE.format(
            STUDENT_ANSWER=mim_nemotron_response.output_text,
            STANDARD_CRITERIA=mim_response_reference_json
        )
        
        mim_judge_response = client.responses.create(
            model="gpt-5",
            input=mim_judge_system_prompt
        )

        print("\nJudge Layer Output:")
        print(mim_judge_response.output_text)
        print("------------------------------")
        
         # Determine pass/fail from judge (simplified example)
        status = "PASS" if "1 point" in mim_judge_response.output_text else "FAIL"
        
        # ------------------- SAVE TO CSV -------------------
        with open(FILE_PATH, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[
                "prompt", "correct_response", "response_reference",
                "model", "nemotron_response", "judge_response", "status"
            ])
            writer.writerow({
                "prompt": data_mim.get("prompt", ""),
                "correct_response": data_mim.get("correct_response", ""),
                "response_reference": json.dumps(data_mim.get("response_reference", [])),
                "model": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
                "nemotron_response": mim_nemotron_response.output_text,
                "judge_response": mim_judge_response.output_text,
                "status": status
            })

        print(f"Task saved to {file_name} with status {status}.")
        
    except Exception as e:
        print("Error:", e)