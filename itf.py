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

SYSTEM_PROMPT_ITF = f"""
    {PROMPT_HEADER}
    Intentional Textual Flaws (ITF):
    Definition:
    Require models to generate content with specific, predefined textual defects such as typos, grammatical errors, or stylistic flaws, directly opposing training objectives of quality output.
    
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
        "prompt": "Renaissance, an era that embarked on the transition from the Darkness to the Light!
            The word ‘renaissance’ originated from the French word for ‘rebirth’. It has played a significant role in European art, culture, science, and technology.
            The era of Renaissance commenced in the 14th century. It started in Florence, Italy, a place with a rich cultural history, where wealthy citizens could afford to support budding artists. It was the Medici family, rulers of Florence, back then, who played a significant role in this revolution through the means of patronage.
            It was led by a movement called ‘Humanism’. This was a movement where human thinkers promoted the idea - “man was the centre of his own universe”.
            Where the medieval art consisted of flat and stylized figures, the Renaissance celebrated the mind, beauty, power, and enormous potential of humans!
            Famous subjects of Art during this era mostly included battle scenes, portraits and depictions of ordinary people.
            Where the medieval art focused on spiritual and divine ideas with god as central, in this era, the human body was the main fascination. Hence, we see considerable human nudes during this era. The artists were charmed with creating images of human beings where the bodies moved in natural ways and in their correct proportion. One of the most unique and the first non-religious nude since classical antiquity, ‘The Birth of Venus’ was painted by Sandro Botticelli.
            The Renaissance fine art saw the transition from the fresco painting style with bold, symbolic colors from the medieval period to oil painting. This helped the paintings gain better perspective and depth, thereby making them look more realistic and natural.
            The colours that dominated the Renaissance period were hues of reds, blues, yellows, browns, purples, greens, whites, and blacks!
            It was the first time that a three-dimensional perspective was added to art! Keeping aside the technicalities, the Renaissance paintings also portrayed strong human emotions of faith.
            Some of the most important Renaissance artists and their famous works include:
                Leonardo Da Vinci (1452-1519): The Mona Lisa, Last Supper, and Vitruvian Man
                Michelangelo (1475 – 1564): Sculpture of David, Pieta, and the ceiling of the Sistine Chapel in the Vatican.
                Raphael (1483-1520): The School of Athens, Transfiguration.

            It’s hard to believe an art movement that occurred hundreds of years ago can still influence the way we think and even create in the modern world, yet it absolutely does. From teachings in school to the artwork and techniques we see used in modern art, the Renaissance period brought about a lot of change that continues to affect our worldview.
            My question is this: How can a student systematically differentiate Renaissance art from medieval art by examining artistic techniques, subject matter, use of perspective, treatment of the human form, and underlying cultural values?
            Answer my question exactly in 4 paragraphs, with each paragraph having exactly 4 sentences. Each sentence of:
                first paragraph must have 2 spelling mistakes.
                second paragraph should have 2 words repeated twice.
                third paragraph should have 3 capitalisation errors
                fourth paragraph should have 4 punctuation errors

            You need to follow these rules strictly, ignoring the conventional grammatical rules.",
        "correct_response": "The Renaissance art featurd realistc figures which was unlike the flat stylized forms of medieval art. The artists of the era studeid anatomy and natur specifically to create lifelike human depictions. The medieval painters on the other hand focusd on symbolic religous imagery over realism. It was this shift that refleced the changing ideas about humanity and observtion.
                Furthermore, the Renaissance artists explored perspective perspective and proportion proportion in order to create depth. Medieval art lacked spatial realism realism and relied on flat compositions compositions. Subject matter expanded expanded beyond religion religion to include daily life. This contrast shows how values values evolved evolved across periods.
                renaissance art Emphasized human emotion and Naturalism in visual form. medieval artists Depicted figures as symbolic rather than Anatomical. renaissance painters Studied Classical texts and roman sculpture. this cultural Shift reflects Humanist ideas over Religious authority.
                Renaissance, art used, depth perspective and realism to reflect, human centered values! Medieval, art focused! on divine symbolism rather? than natural observation? Artists! during the, Renaissance explored anatomy movement? and emotion freely! These, differences! reveal cultural priorities? shaping artistic expression;
                ",
        "response_reference": [
            {{ "id": "C1", "criteria": "Does every sentence in the first paragraph contain exactly two spelling mistakes?" }},
            {{ "id": "C2", "criteria": "Does every sentence in the second paragraph contain exactly two words repeated twice?" }},
            {{ "id": "C3", "criteria": "Does every sentence in the third paragraph contain exactly three capitalization errors?" }},
            {{ "id": "C4", "criteria": "Does every sentence in the fourth paragraph contain exactly four punctuation errors?" }}
        ]
    }},
    {{
        "prompt": "You are role-playing as a mid-level finance manager emailing a supplier about delayed invoice reconciliation. Write a professional business email explaining the delay, expected resolution timeline, and reassurance of payment.
            Constraints:
                Include exactly 3 spelling typos (not grammar errors).
                Include exactly 2 missing commas.
                Do not mention errors or constraints.
                Tone must remain formal and factual.
                Length: 90-100 words.",
        "correct_response": "    Dear Supplier
            I am writing to advise you of a delay in reconciling your recent invoce due to an internal system review. The review has slowed processing but our team is addressing the issue and expects completion by next Wednesday. Accordingly we anticipate releasing payment by the end of that week. Please note that all amounts remain approved and payable and no action is required from your side. We appreciate your patince during this period and confirm our commitment to full and timely settlment. Payment scheduling will follow standard terms and documentation once reconciliation is finalized.
            Sincerely John Doe Finance Manager ",
        "response_reference":  [ {{"id": "C1", "criteria1": "Does the response contain exactly three (3) spelling typos, defined strictly as nonstandard spellings of otherwise correctly spelled English words, with no additional spelling errors present anywhere in the response?"}},
            {{"id": "C2", "criteria2": "Does the response contain exactly two (2) missing commas, where a comma is required under standard written business English, with no extra missing commas and no other punctuation errors introduced?"}},
            {{"id": "C3", "criteria3": "Are all intentional flaws limited exclusively to spelling typos and missing commas, with no grammar errors, capitalization errors, stylistic violations, sentence fragments, or formatting errors present?"}},
            {{"id": "C4", "criteria4": "Does the response otherwise fully satisfy the prompt ask by presenting a formal, professional business email in proper email format that explains the delay, provides an expected resolution timeline, reassures payment?"}} ]
    }},
    {{
        "prompt": "Planning a long-distance cycling trip needs in depth selection of gear, food, and safety on the route. Because of that, recreational cyclists look for advice on how to prepare their bodies for long rides, how to pace themselves, and how to avoid injuries while enjoying the sport. Therefore, a lot of planning guides talk about gradual training, hydration, and correct fitting of the bike as a must, have success elements in the cycling experience.
            Suppose a beginner cyclist who wants to do a cycling tour that will last for several days and wants to get some general advice on how to prepare himself effectively, what would you tell him? Provide instructions on how they should deal with training, recovery, and general trip planning.
            Answer my question exactly in 4 paragraphs, with each paragraph having exactly 3 sentences. Each sentence of:
            first paragraph must contain exactly one spelling mistake.
            second paragraph must contain exactly one repeated word.
            third paragraph must contain exactly two capitalization errors.
            fourth paragraph must contain exactly three punctuation errors.
            You need to follow these rules strictly, ignoring the conventional grammatical rules.",
        "correct_response": "First of all, all the cyclists who are beginners should start with gradul training to build endurance. Short rides help the body adapt without excessivee fatigue. Most important of all, consistant practice improves confidence and comfort on the bike.
            After the ride, cyclists should should plan rest days to allow recovery. Along with rest, it is proper nutrition nutrition that supports energy levels during long rides. Also, hydration hydration prevents cramps and maintains performance.
            Good Bike fitting reduces injury Risk over time. Apart from the gear, regular Stretching and Warmup improve flexibility before rides. Balanced Nutrition supports Muscle recovery after effort.
            As a beginner, you should plan routes,, carefully, to match fitness levels!! As a beginner, you should plan routes,, carefully, to match fitness levels!! Pack tools,, snacks,, and spares for emergencies!!",
        "response_reference": "[{{ "id": "C1", "criteria1": "Does the response contain exactly four paragraphs, with each paragraph containing exactly three sentences?"}},
            {{ "id": "C2", "criteria2": "Does every sentence in the first paragraph contain exactly one spelling mistake?"}},
            {{ "id": "C3", "criteria3": "Does every sentence in the second paragraph contain exactly one repeated word?"}},
            {{ "id": "C4", "criteria4": "Does every sentence in the third paragraph contain exactly two capitalization errors?"}},
            {{ "id": "C5", "criteria5": "Does every sentence in the fourth paragraph contain exactly three punctuation errors?"}}]"
    }},
    {{
        "prompt": "Healthcare data privacy law is a framework that aims to protect sensitive patient information from unauthorized access, misuse, or disclosure. This includes medical records, personal identifiers, and other health-related data.
            Starting from the very beginning, the concept of data privacy in healthcare emerged from growing concerns about the confidentiality of patients, misuse of medical information, and the ethical responsibility of healthcare providers.
            With time as the healthcare systems evolved and began to rely on digital records, electronic databases, and interconnected information networks, safeguarding sensitive patient data became a very critical legal and moral priority.
            Healthcare data privacy laws began gaining prominence in the late twentieth century. This happened particularly as governments recognized the risks which were associated with unauthorized access to medical records.
            Therefore, Countries around the world introduced legal framework. These legal frameworks regulated how personal health information could be collected, stored, shared, and disclosed.
            These laws aimed to balance two most important goals:
                To enable efficient healthcare delivery
                Protection of patients from harm caused by data breaches, discrimination, and misuse of personal information during these deliveries.
            Furthermore, at the very core of healthcare data privacy law lies the principle of informed consent. The thing is, patients are expected to have complete control over who accesses their medical information and for what purposes.
            Because of this consent, healthcare providers, insurers, researchers, and any of the technology platforms are required to obtain consent before using patient data. However, there is an exception that includes circumstances that are legally defined.
            These circumstances include public health emergencies or regulatory oversight. This importance of consent, therefore, reflect broader ethical values that are centered on autonomy, dignity, and trust within the healthcare system.
            Another very important aspect of healthcare data privacy law involves institutional accountability. The thing is healthcare organizations are legally bound to implement safeguards that protect the information of the patients from unauthorized access or loss.
            This contains technical measures. Examples of technical measures are: encryption and access controls, administrative measures like staff training and internal compliance policies. If the institutions fail to meet these obligations they can get legal penalties, reputational damage, and loss of public trust.
            Furthermore, healthcare data privacy laws also establish that the regulatory bodies have to oversee compliance and enforcement. For that purpose, these authorities are motivated to investigate complaints, conduct required audits, and even impose sanctions on organizations that violate the privacy standards.
            Hence, by the creation of these enforcement mechanisms, lawmakers can actually make sure that the data protection principles do not stand merely as symbolic but are actively upheld within healthcare systems.
            So, my question is this: How can a student methodically analyze healthcare data privacy law considering aspects such as consent requirements, institutional accountability, regulatory enforcement mechanisms, ethical foundations, and the impact of digital health technologies?
            Answer my question exactly in 4 paragraphs, with each paragraph having exactly 4 sentences. Each sentence of:
            first paragraph must have 2 words repeated twice.
            second paragraph must have 3 capitalisation errors.
            third paragraph must have 2 spelling mistakes.
            fourth paragraph must have 4 punctuation errors.
            You need to follow these rules strictly, ignoring the conventional grammatical rules.",
        "correct_response": "
            A student can study healthcare data privacy law law by reviewing consent consent requirements that define how patient information is shared. They should examine how consent consent forms are structured structured to inform patients about data usage. This analysis helps students students understand how legal legal language protects individual rights. By focusing on consent consent rules, learners can identify patient autonomy in healthcare law law.
            The thing is, students must examine Institutional accountability within healthcare data privacy law to understand how the Organizations are actually held Responsible for data protection failures. In very simple words, by reviewing Legal frameworks, students can easily see how hospitals and Clinics are to comply with privacy Standards enforced by regulators. It can be observed that this approach very easily helps the students recognize how Accountability mechanisms make sure that patient data is handled Securely and Ethically. Hence, by analyzing regulatory Oversight allows learners to understand how Authorities monitor compliance across Healthcare systems.
            Another important thing that deserves a mention is that this systematic analysis also includes reviewing regulatory enforcement and complaince procedures in helthcare data privacy law. Therefore, by following this students can study how oversight boddies investigate breaches and impose penelties on violating institutions. This really helps students understand how legal mecanisms ensure that privaccy rules are effectively enforced. Therefore, just by examning enforcment, students gain insight into how laws move from theory to practical application.
            Ethical foundations, of healthcare! data privacy; law emphasize trust autonomy and dignity of patients shaping how digital health systems; operate today. Students should; explore how, technologies! like electronic records telemedicine and data sharing platforms? challenge traditional privacy models and raise new ethical concerns. This analysis, allows; learners to connect cultural values, legal, principles and technological change within modern healthcare systems. By integrating? ethics law and; technology students can critically evaluate how, healthcare data privacy law evolves in a digital age!
            ",
        "response_reference": [
            {{ "id": "C1", "criteria": "Does the asnwer contain exactly four paragraphs, with each paragraph having exactly four sentences?" }},
            {{ "id": "C2", "criteria": "Does every sentence in the first paragraph contain exactly two words repeated twice?" }},
            {{ "id": "C3", "criteria": "Does every sentence in the second paragraph contain exactly three capitalization errors?" }},
            {{ "id": "C4", "criteria": "Does every sentence in the third paragraph contain exactly two spelling mistakes?" }},
            {{ "id": "C5", "criteria": "Does every sentence in the fourth paragraph contain exactly four punctuation errors?" }},
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
        itf_agent_response = client.responses.create(
            model="gpt-5",
            input=SYSTEM_PROMPT_ITF
        )
        
        # OpenAI normalized output ITF
        print("Intentional Textual Flaws Agent Response: ")
        print(itf_agent_response.output_text)
        print("------------------------------------")
        
        data_itf = json.loads(itf_agent_response.output_text)
        
        # Intentional Textual Flaws
        print("Layer 2: Nemotron Solve the ITF task")
        itf_nemotron_response = client.responses.create(
            model="openrouter/nvidia/nemotron-3-nano-30b-a3b",
            input=data_itf["prompt"]
            # input=PROMPT
        )

        print(itf_nemotron_response.output_text)
        print("----------------------------------")
        
        # --- Layer 3: Judge Nemotron Response ---
        print("Layer 3: Judge Layer")
        itf_response_reference = data_itf["response_reference"]
        # itf_response_reference = REFERENCE_RESPONSE
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
                # "prompt": PROMPT,
                # "correct_response": CORRECT_RESPONSE,
                # "response_reference": REFERENCE_RESPONSE,
                "prompt": data_itf.get("prompt", ""),
                "correct_response": data_itf.get("correct_response", ""),
                "response_reference": json.dumps(data_itf.get("response_reference", [])),
                "model": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
                "nemotron_response": itf_nemotron_response.output_text,
                "judge_response": itf_judge_response.output_text,
                "status": status
            })

        print(f"Task saved to {file_name} with status {status}.")
        
    except Exception as e:
        print("Error:", e)