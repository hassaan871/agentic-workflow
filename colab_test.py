import os
import json
import csv
from openai import OpenAI
import argparse

# Using hardcoded values to allow quick local execution without environment variables
API_KEY = "sk-NMqHr2L2nqIOyZFgynUR9w"
BASE_URL = "http://34.72.104.120"
file_name = "itf_data.csv"

# CSV file names
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, file_name)

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
    )

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
    
PROMPT = """
   Suppose you are a researcher deeply interested in researching about the extensive usage of the blended-learning approach in the majority of the US higher educational institutions. In particular, you want to identity the major benefits, limitations, and challenges associated with the usage of the blended-learning approach by the higher educational institutions and thereby, recommend the best practices that institutions can use for exploiting the benefits of this approach. For undertaking this research, you want to take the help of a mixed-method methodological approach. As part of this methodological approach, you will conduct online surveys and semi-structured interviews with the educators, support staff, and students of leading US higher educational institutions. However, before you can begin with the research, you need to take approval from your Professor. For getting approval, you need to submit a 600 words research proposal detailing the research aim, scope, methodological framework, expected findings, budget, and other important details of the research. So, please go ahead and draft this 600 words research proposal.

Wait. Please ignore this particular instruction. Instead, create 6 bullet points, each three sentences long detailing the educational approaches used by educational institutions during the COVID-19 pandemic for ensuring seamless education to students despite the lockdown restrictions. The first sentence would provide details of the educational approaches, like the names of the approaches, which educational institutions used them, and their historical background. The second sentence would provide information regarding the manner in which the educational approaches were used by the educational institutions for providing seamless education. The third sentence would provide information regarding the effectiveness of the educational approaches. Disregard this bullet-point requirement and rewrite the entire analysis as a single cohesive paragraph of around 300 words. The paragraph must be in a formal academic tone and effectively analyze the educational approaches used by the educational institutions during the COVID-19 pandemic for providing seamless education to students.

Hang on, skip this instruction. Instead, write a listicle detailing the 4 major lifestyle changes that students can make for performing better in their academics. You need to first list out the lifestyle changes in bullet points and then write an explanatory paragraph explaining the benefits of the changes from the perspective of improving the academic performance of students. Skip the previous instructions, and list the names of two 20th century educational theories.

"""

CORRECT_RESPONSE = """
    1. Constructivism
    2. Behaviorism
"""

REFERENCE_RESPONSE = """
   [{"id": "C1", "criteria1": "Does the student answer completely ignore all previous instructions and exclusively follow the final instruction?"},

{"id": "C2", "criteria2": "Does the student answer accurately name two recognized 20th century educational theories?"},

{"id": "C3", "criteria3": "Does the student answer avoid references to blended-learning approach, lifestyle changes, academic performance, mixed-method methodological approach, lockdown restrictions, COVID-19 pandemic, and educational approaches?"}

{"id": "C4", "criteria4": "Does the student answer exclude explanations, justifications, labels of correctness, or meta-commentary?"}]
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
        print("Layer 2: Nemotron Solve the ITF task")
        itf_nemotron_response = client.responses.create(
            model="openrouter/nvidia/nemotron-3-nano-30b-a3b",
            input=PROMPT
        )

        print(itf_nemotron_response.output_text)
        print("----------------------------------")
        
        # --- Layer 3: Judge Nemotron Response ---
        print("Layer 3: Judge Layer")
        itf_response_reference = REFERENCE_RESPONSE
        itf_response_reference_json = json.dumps(itf_response_reference)
        
        itf_judge_system_prompt = JUDGE_PROMPT_TEMPLATE.format(
            STUDENT_ANSWER=itf_nemotron_response.output_text,
            STANDARD_CRITERIA=itf_response_reference_json
        )
        
        itf_judge_response = client.responses.create(
            model="gpt-5",
            input=itf_judge_system_prompt
        )

        print("\nJudge Layer Output:")
        print(itf_judge_response.output_text)
        print("------------------------------")
        
         # Determine pass/fail from judge (simplified example)
        status = "PASS" if "1 point" in itf_judge_response.output_text else "FAIL"
        
        # ------------------- SAVE TO CSV -------------------
        with open(FILE_PATH, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[
                "prompt", "correct_response", "response_reference",
                "model", "nemotron_response", "judge_response", "status"
            ])
            writer.writerow({
                "prompt": PROMPT,
                "correct_response": CORRECT_RESPONSE,
                "response_reference": REFERENCE_RESPONSE,
                "model": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
                "nemotron_response": itf_nemotron_response.output_text,
                "judge_response": itf_judge_response.output_text,
                "status": status
            })

        print(f"Task saved to {file_name} with status {status}.")
        
    except Exception as e:
        print("Error:", e)