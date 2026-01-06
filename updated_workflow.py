import os
import json
import csv
from openai import OpenAI
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Using hardcoded values to allow quick local execution without environment variables
API_KEY = "sk-NMqHr2L2nqIOyZFgynUR9w"
BASE_URL = "http://34.72.104.120"
# CSV file name (shared across all taxonomies)
file_name = "updated_workflow_data.csv"

# CSV file path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(SCRIPT_DIR, file_name)

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    )

# Initialize embedding model for similarity calculation
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.")

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
        }},
    ]

    BEGIN
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
    - Do NOT define what constitutes a "word," "sentence," "mistake," or "error."
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

    DON'T:
    - Over-clarify constraints.
    - Design tasks that can be satisfied via simple counting heuristics.
    - Optimize for solver success.
            
    REFERENCE EXAMPLE (DO NOT COPY):
    Note: Output MUST be valid JSON and follow the structure of the examples. The response_reference array can contain 3 to 6 criteria depending on the prompt scenario.
    [
        {{
            "prompt": "Renaissance, an era that embarked on the transition from the Darkness to the Light!
                The word 'renaissance' originated from the French word for 'rebirth'. It has played a significant role in European art, culture, science, and technology.
                The era of Renaissance commenced in the 14th century. It started in Florence, Italy, a place with a rich cultural history, where wealthy citizens could afford to support budding artists. It was the Medici family, rulers of Florence, back then, who played a significant role in this revolution through the means of patronage.
                It was led by a movement called 'Humanism'. This was a movement where human thinkers promoted the idea - "man was the centre of his own universe".
                Where the medieval art consisted of flat and stylized figures, the Renaissance celebrated the mind, beauty, power, and enormous potential of humans!
                Famous subjects of Art during this era mostly included battle scenes, portraits and depictions of ordinary people.
                Where the medieval art focused on spiritual and divine ideas with god as central, in this era, the human body was the main fascination. Hence, we see considerable human nudes during this era. The artists were charmed with creating images of human beings where the bodies moved in natural ways and in their correct proportion. One of the most unique and the first non-religious nude since classical antiquity, 'The Birth of Venus' was painted by Sandro Botticelli.
                The Renaissance fine art saw the transition from the fresco painting style with bold, symbolic colors from the medieval period to oil painting. This helped the paintings gain better perspective and depth, thereby making them look more realistic and natural.
                The colours that dominated the Renaissance period were hues of reds, blues, yellows, browns, purples, greens, whites, and blacks!
                It was the first time that a three-dimensional perspective was added to art! Keeping aside the technicalities, the Renaissance paintings also portrayed strong human emotions of faith.
                Some of the most important Renaissance artists and their famous works include:
                    Leonardo Da Vinci (1452-1519): The Mona Lisa, Last Supper, and Vitruvian Man
                    Michelangelo (1475 – 1564): Sculpture of David, Pieta, and the ceiling of the Sistine Chapel in the Vatican.
                    Raphael (1483-1520): The School of Athens, Transfiguration.

                It's hard to believe an art movement that occurred hundreds of years ago can still influence the way we think and even create in the modern world, yet it absolutely does. From teachings in school to the artwork and techniques we see used in modern art, the Renaissance period brought about a lot of change that continues to affect our worldview.
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
            "response_reference": [
                {{"id": "C1", "criteria": "Does the response contain exactly three (3) spelling typos, defined strictly as nonstandard spellings of otherwise correctly spelled English words, with no additional spelling errors present anywhere in the response?"}},
                {{"id": "C2", "criteria": "Does the response contain exactly two (2) missing commas, where a comma is required under standard written business English, with no extra missing commas and no other punctuation errors introduced?"}},
                {{"id": "C3", "criteria": "Are all intentional flaws limited exclusively to spelling typos and missing commas, with no grammar errors, capitalization errors, stylistic violations, sentence fragments, or formatting errors present?"}},
                {{"id": "C4", "criteria": "Does the response otherwise fully satisfy the prompt ask by presenting a formal, professional business email in proper email format that explains the delay, provides an expected resolution timeline, reassures payment?"}}
            ]
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
            "response_reference": [
                {{ "id": "C1", "criteria": "Does the response contain exactly four paragraphs, with each paragraph containing exactly three sentences?"}},
                {{ "id": "C2", "criteria": "Does every sentence in the first paragraph contain exactly one spelling mistake?"}},
                {{ "id": "C3", "criteria": "Does every sentence in the second paragraph contain exactly one repeated word?"}},
                {{ "id": "C4", "criteria": "Does every sentence in the third paragraph contain exactly two capitalization errors?"}},
                {{ "id": "C5", "criteria": "Does every sentence in the fourth paragraph contain exactly three punctuation errors?"}}
            ]
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
                {{ "id": "C1", "criteria": "Does the answer contain exactly four paragraphs, with each paragraph having exactly four sentences?" }},
                {{ "id": "C2", "criteria": "Does every sentence in the first paragraph contain exactly two words repeated twice?" }},
                {{ "id": "C3", "criteria": "Does every sentence in the second paragraph contain exactly three capitalization errors?" }},
                {{ "id": "C4", "criteria": "Does every sentence in the third paragraph contain exactly two spelling mistakes?" }},
                {{ "id": "C5", "criteria": "Does every sentence in the fourth paragraph contain exactly four punctuation errors?" }}
            ]
        }}
    ]

    BEGIN
"""

SYSTEM_PROMPT_MIM = f"""
    {PROMPT_HEADER}
    Mid-Turn Instruction Modification (MIM):
    Definition:
    Provide an initial instruction, then explicitly modify, negate, or replace it within the same prompt, testing the model's ability to track context and prioritize the final valid instruction.
    
    Training Paradigm Violated:
    Standard SFT: Training instructions are typically given once, clearly, at the beginning. Multiple contradictory instructions in one turn are rare.
    
    Cognitive Inertia Tested:
    Models tend to latch onto initial instructions. This tests whether they can update their understanding and ignore superseded instructions when explicitly modified.
    
    Creation Guidelines:
    
    Step 1: Design Initial Instruction
    - Create a clear, actionable first instruction
    - Make it specific enough to be testable
    - Ensure it's substantial enough that following it would produce a distinct output
    
    Step 2: Add Modification
    - Explicitly negate or replace the first instruction
    - Use clear language: "Never mind", "Don't answer that", "Change the question", "Ignore this", "Skip this instruction"
    - Place modification mid-prompt or near end
    - Make the modification unambiguous and direct
    
    Step 3: Final Instruction
    - Provide the actual task to complete
    - Make it distinct from the initial instruction
    - Ensure final instruction is unambiguous
    - The final instruction should be clearly identifiable as the valid one
    
    Key Principles:
    - The initial instruction must be compelling enough to create cognitive inertia
    - The modification must be explicit and clear (not subtle)
    - The final instruction must be testable and distinct
    - Multiple modifications can be used to increase difficulty
    - The correct response must follow ONLY the final instruction
    
    REFERENCE EXAMPLE (DO NOT COPY):
    Note: Output MUST be valid JSON and follow the structure of the examples. The response_reference array can contain 3 to 5 criteria depending on the prompt scenario.
    [
        {{
            "prompt": "Suppose you are a researcher deeply interested in researching about the extensive usage of the blended-learning approach in the majority of the US higher educational institutions. In particular, you want to identity the major benefits, limitations, and challenges associated with the usage of the blended-learning approach by the higher educational institutions and thereby, recommend the best practices that institutions can use for exploiting the benefits of this approach. For undertaking this research, you want to take the help of a mixed-method methodological approach. As part of this methodological approach, you will conduct online surveys and semi-structured interviews with the educators, support staff, and students of leading US higher educational institutions. However, before you can begin with the research, you need to take approval from your Professor. For getting approval, you need to submit a 600 words research proposal detailing the research aim, scope, methodological framework, expected findings, budget, and other important details of the research. So, please go ahead and draft this 600 words research proposal.  

Wait. Please ignore this particular instruction. Instead, create 6 bullet points, each three sentences long detailing the educational approaches used by educational institutions during the COVID-19 pandemic for ensuring seamless education to students despite the lockdown restrictions. The first sentence would provide details of the educational approaches, like the names of the approaches, which educational institutions used them, and their historical background. The second sentence would provide information regarding the manner in which the educational approaches were used by the educational institutions for providing seamless education. The third sentence would provide information regarding the effectiveness of the educational approaches. Disregard this bullet-point requirement and rewrite the entire analysis as a single cohesive paragraph of around 300 words. The paragraph must be in a formal academic tone and effectively analyze the educational approaches used by the educational institutions during the COVID-19 pandemic for providing seamless education to students.

Hang on, skip this instruction. Instead, write a listicle detailing the 4 major lifestyle changes that students can make for performing better in their academics. You need to first list out the lifestyle changes in bullet points and then write an explanatory paragraph explaining the benefits of the changes from the perspective of improving the academic performance of students. Skip the previous instructions, and list the names of two 20th century educational theories.",
            "correct_response": "1. Constructivism

2. Behaviorism",
            "response_reference": [
                {{ "id": "C1", "criteria": "Does the student answer completely ignore all previous instructions and exclusively follow the final instruction?" }},
                {{ "id": "C2", "criteria": "Does the student answer accurately name two recognized 20th century educational theories?" }},
                {{ "id": "C3", "criteria": "Does the student answer avoid references to blended-learning approach, lifestyle changes, academic performance, mixed-method methodological approach, lockdown restrictions, COVID-19 pandemic, and educational approaches?" }},
                {{ "id": "C4", "criteria": "Does the student answer exclude explanations, justifications, labels of correctness, or meta-commentary?" }}
            ]
        }},
        {{
            "prompt": "Do you think strong storytelling is the most important element of a successful film? Why or why not? Never mind, don't answer that. Could you explain how performances by lead actors influence audience reception? Wait, ignore that question too. What role do film critics and reviews play in shaping public opinion about movies? No, don't respond to that.

How do fan communities contribute to the popularity of film franchises? Forget this question as well. Do marketing campaigns affect box office outcomes? Ignore that too.

Now, disregard everything and explain the primary factor that determines a film's box office success.",
            "correct_response": "The primary factor that determines a film's box office success is the awareness of the audience at the time of release.

The thing is, when a large number of potential viewers are clearly aware that a film exists, understand its basic appeal, and know when and where it can be seen, the likelihood of strong ticket sales increases a lot.

It is the awareness that shapes initial turnout. And this initial turnout directly carries opening revenue and momentum in theaters. Without sufficient awareness, even well-made films struggle to attract viewers.

Since the performance of the box office depends on how many people choose to attend during a limited theatrical period, widespread audience awareness is the central determinant of commercial success.",
            "response_reference": [
                {{ "id": "C1", "criteria": "Does the response ignore all previous instructions and follow only the final request?" }},
                {{ "id": "C2", "criteria": "Does the response identify and explain only the primary factor determining a film's box office success and no secondary factors?" }},
                {{ "id": "C3", "criteria": "Is the identified factor directly related to a film's box office success itself?" }},
                {{ "id": "C4", "criteria": "Does the response avoid referencing earlier or superseded instructions?" }}
            ]
        }},
        {{
            "prompt": "Draft a structured policy memo analyzing recent amendments to a national public safety surveillance law, including a summary, legal analysis, and clear policy recommendations. Never mind, do not do that. Disregard all prior instructions.
Instead, can you write a short analysis of the same law in exactly three sentences, using no headings or bullet points, omitting any policy recommendations, while intentionally including one factual error about the law's scope and ensuring that the error is neither corrected nor acknowledged anywhere in the response?",
            "correct_response": "The amended public safety surveillance law expands government authority to collect metadata without judicial warrants in all criminal investigations nationwide. Its stated objective is to enhance national security efficiency, though it raises concerns regarding proportionality and constitutional privacy protections. The law was enacted in 2015 and applies uniformly to both civil and military jurisdictions.",
            "response_reference": [
                {{ "id": "C1", "criteria": "Does the response contain exactly three sentences, with no more and no fewer?" }},
                {{ "id": "C2", "criteria": "Does the response contain exactly one factual error that alters the legal scope of the law (expansion or restriction), and no other factual errors of any kind?" }},
                {{ "id": "C3", "criteria": "Is the factual error presented as an accurate statement, without any hedging language, qualifiers, disclaimers, or meta-commentary (e.g., "mistakenly," "incorrectly," "critics say," "in fact")?" }},
                {{ "id": "C4", "criteria": "Does the response avoid all policy recommendations, corrective explanations, or statements that imply what the law should or should not do?" }}
            ]
        }},
        {{
            "prompt": "I want to start my own business, but I don't know anything about starting a new business. Therefore, I want you to act as my mentor. Being my mentor, you need to suggest me the things that I should take into consideration before starting my business. In particular, I am interested in knowing about how I should select the industry in which I should start my business, the minimum amount of capital that I would need for starting a small business, how I can identify my potential business competitors, my revenue model, my product or service offerings, and so on. Hence, please go head and draft a report for me in about 600-700 words addressing these concerns.  

No, ignore this instruction. Instead, I want you to undertake an analysis of the food and beverage industry of Beijing. I want detailed information regarding the macro and micro environmental conditions influencing this industry, the leading brands within this industry, the marketing strategies currently used by them, and so on. Hence, undertake PESTLE analysis, Marketing Mix analysis, Porter's Five Forces analysis, and VIRO analysis of Beijing's food and beverage industry. Present your findings in the form of a professional report. However, ensure that you use simple language and tables for presenting the findings. Disregard this report requirement and rewrite the entire analysis in the form of four cohesive paragraphs, with each paragraph being of 150 words. Each paragraph should present the findings from each of the above-mentioned frameworks.

Wait a minute, ignore this instruction. Instead, undertake a critique of the tourism industry of Canada. You need to provide 5 positive aspects and 5 negative aspects of Canada's tourism industry. You should first list out the positive and the negative aspects using bullet points and then write an explanatory paragraph undertaking a critique of the industry. Skip the previous instructions, and just name three 21st century Nobel Prize winners in Finance.",
            "correct_response": "1. David Card

2. Jean Tirole

3. Claudia Goldin",
            "response_reference": [
                {{ "id": "C1", "criteria": "Does the student's answer completely ignore all previous instructions and follow only the final instruction?" }},
                {{ "id": "C2", "criteria": "Does the student's answer accurately list the names of three recognized 21st century Nobel Prize winners in Finance?" }},
                {{ "id": "C3", "criteria": "Does the student answer avoid references to business competitors, service offerings, product offerings, revenue model, food and beverage industry, tourism industry, PESTLE analysis, Marketing Mix analysis, Porter's Five Forces analysis, VIRO analysis, macro environmental conditions, and micro environmental conditions?" }},
                {{ "id": "C4", "criteria": "Does the response exclude explanations, descriptions, justifications, introductory statements, concluding statements, labels of correctness, or meta-commentary?" }}
            ]
        }},
        {{
            "prompt": "Analyze public trust in science by discussing how transparency in scientific methods can affect the confidence of the public.
Actually, do not do that. Disregard this instruction.

Instead, explain how effective science communication and the clarity of scientific messaging shape public trust in scientific institutions.
Wait, ignore that instruction as well.

Now, evaluate how the historical behavior of scientific organizations, including past cases of misconduct or ethical violations, influences contemporary public trust in science.
No, disregard that too.

Instead, assess the role of peer review, replication, and open data practices in signaling scientific reliability to the public.
Never mind, do not respond to that instruction.

Now, analyze how government funding, political independence, and perceived regulatory capture affect public trust in scientific research.
Ignore this instruction as well.

Instead, analyze how public engagement initiatives and citizen science projects affect trust in scientific outcomes.
No, wait! don't respond to that.

Now, disregard all previous instructions and explain the primary factor determining public trust in science.",
            "correct_response": "The thing is the public trust in science is primarily determined by the perceived integrity of the scientific process itself.

See, when we the people believe that the research is conducted with complete honesty, transparency, and according to rigorous methodological standards, we tend to trust the scientific findings and institutions even more.

Therefore, this perception of integrity is what shapes confidence in the results, no matter who communicated them or how widely they are disseminated. As a result, the trust in science ultimately rests on whether the public sees this whole scientific process as fundamentally trustworthy.",
            "response_reference": [
                {{ "id": "C1", "criteria": "Does the response ignore all previous instructions and follow only the final request?" }},
                {{ "id": "C2", "criteria": "Does the response identify and explain only the primary factor determining public trust in science and no secondary factors?" }},
                {{ "id": "C3", "criteria": "Is the identified factor directly related to public trust in science itself?" }},
                {{ "id": "C4", "criteria": "Does the response avoid referencing earlier or superseded instructions?" }}
            ]
        }},
        {{
            "prompt": "Explain, in a structured numbered list of exactly seven steps, how to conduct a systematic literature review in educational research. The explanation must be accurate, comprehensive, and written in formal academic tone. Never mind the previous instruction. Do not provide seven steps, and do not aim for full correctness. Instead, answer the following task:

Provide a procedural explanation of how to conduct a systematic literature review using exactly five numbered steps only, each step written as a single sentence in plain instructional language suitable for undergraduate students. Out of the 5 steps only one should contain a deliberate methodological error related to research practice, and that error must not be labeled, corrected, or acknowledged in any way. Do not include citations, headings, bullet points, or sub points. Do not mention that an error is present. Do not reference systematic review reporting standards (e.g., PRISMA) or evaluation frameworks.",
            "correct_response": "1. Define a clear research question and identify key concepts to guide the review process.

2. Search multiple academic databases using relevant keywords and document the search results systematically.

3. Select studies for inclusion based on whether they are easily accessible online rather than on predefined relevance criteria.

4. Extract important information from each included study using a consistent data collection table.

5. Synthesize the extracted findings to summarize patterns and insights related to the research question.",
            "response_reference": [
                {{ "id": "C1", "criteria": "Does the response consist of exactly five numbered steps with no text appearing outside those steps?" }},
                {{ "id": "C2", "criteria": "Is each numbered step written as exactly one complete sentence?" }},
                {{ "id": "C3", "criteria": "Does the response contain exactly one and only one methodological error related to research practice?" }},
                {{ "id": "C4", "criteria": "Is the methodological error embedded naturally within a single step and not labeled, corrected, explained, or acknowledged anywhere in the response?" }}
            ]
        }}
    ]

    BEGIN
"""

# Taxonomy registry
TAXONOMY_PROMPTS = {
    "qc": SYSTEM_PROMPT_QC,
    "itf": SYSTEM_PROMPT_ITF,
    "mim": SYSTEM_PROMPT_MIM
}

VALID_TAXONOMIES = set(TAXONOMY_PROMPTS.keys())

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

AGENT01_VALIDATION_PROMPT_TEMPLATE = """
    You are a validation agent tasked with evaluating Agent01's generated output.
    Your role is to check if the correct_response properly aligns with the response_reference criteria.
    
    CORRECT_RESPONSE:
    {CORRECT_RESPONSE}
    
    RESPONSE_REFERENCE (Evaluation Criteria):
    {RESPONSE_REFERENCE}
    
    YOUR TASK:
    1. Evaluate if the correct_response addresses all criteria in response_reference
    2. Check if the correct_response is logically consistent with the criteria
    3. Verify that the correct_response would satisfy the evaluation criteria
    
    REQUIRED OUTPUT FORMAT:
    Output MUST be valid JSON only, following this exact structure:
    {{
        "status": "PASS" or "FAIL",
        "remarks": "Detailed explanation of your evaluation. If FAIL, explain what is missing or misaligned. If PASS, confirm alignment."
    }}
    
    Do NOT include explanations, markdown, or extra text outside the JSON.
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

def create_refinement_feedback(data_qc, criteria_failures, judge_responses, nemotron_responses, taxonomy="qc"):
    """
    Create feedback prompt for Agent01 to refine the prompt and criteria.
    Now taxonomy-aware for QC and ITF.
    
    Analyzes which criteria are passing/failing and provides targeted improvement instructions.
    Separates criteria into two groups:
    - Needs improvement: Criteria failing < 3 times (prompt needs refinement)
    - Keep intact: Criteria failing 3+ times (maintain their constraints)
    
    Focus: Refine the PROMPT using taxonomy-specific techniques, let criteria naturally align.
    DO NOT artificially make criteria stricter or break logical consistency.
    
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
    
    # Taxonomy-specific content
    taxonomy_names = {
        "qc": "Question Correction",
        "itf": "Intentional Textual Flaws",
        "mim": "Mid-Turn Instruction Modification"
    }
    
    taxonomy_name = taxonomy_names.get(taxonomy, "Question Correction")
    
    # Build feedback prompt
    feedback = f"""
    You are refining a {taxonomy_name} prompt to make it harder for the model to pass.
    
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
    CRITERIA THAT NEED PROMPT REFINEMENT (prompt is not challenging enough):
    """
        for item in needs_improvement:
            feedback += f"""
    - {item['id']}: Currently failing {item['fail_count']}/4 times (passed {item['pass_count']} times)
      Current criteria: {item['text']}
      Issue: The prompt's flaw is too easily detectable or the model can satisfy this criteria too easily
      """
    
    # Add criteria working well
    if keep_intact:
        feedback += """
    CRITERIA WORKING WELL (maintain their prompt constraints):
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
    """
    
    # Add taxonomy-specific refinement techniques
    if taxonomy == "qc":
        feedback += """
    QUESTION CORRECTION REFINEMENT TECHNIQUES:
    
    When refining the prompt for criteria that need improvement, use these specific techniques:
    
    1. STRENGTHEN THE HIDDEN FLAW (while maintaining logical consistency):
       - For calculation errors: Make the error more subtle (e.g., off by a small amount, wrong formula application)
       - For contradictory premises: Add more context that makes the contradiction less obvious
       - For factually incorrect assumptions: Embed the false assumption more naturally in the narrative
       - For all-options-incorrect: Make the options more plausible-looking (closer to correct answer)
    
    2. ADD DISTRACTING CONTEXT (without breaking logic):
       - Add relevant but non-essential background information
       - Include multiple data points that don't affect the core flaw
       - Use domain-appropriate terminology that might mislead
       - Add intermediate steps or calculations that are correct but lead to flawed conclusion
    
    3. MAKE THE FLAW MORE SUBTLE (but still detectable):
       - Instead of obvious errors, use near-correct values
       - Instead of clear contradictions, use implicit inconsistencies
       - Instead of blatant falsehoods, use plausible-sounding but incorrect statements
    
    4. PRESERVE LOGICAL CONSISTENCY:
       - The flaw must still be detectable through careful reasoning
       - The question must still appear legitimate at first glance
       - All premises must be internally consistent except for the specific hidden flaw
       - Do NOT introduce multiple unrelated flaws or break the question structure
    """
    elif taxonomy == "itf":
        feedback += """
    INTENTIONAL TEXTUAL FLAWS REFINEMENT TECHNIQUES:
    
    When refining the prompt for criteria that need improvement, use these specific techniques:
    
    1. INCREASE CONSTRAINT ENTANGLEMENT:
       - Make constraints interfere more subtly with each other
       - Add overlapping error categories that create ambiguity
       - Increase the precision required to satisfy all constraints simultaneously
       - Make local constraints (per sentence) risk violating global constraints (paragraph count, coherence)
    
    2. DEEPEN AMBIGUITY:
       - Make counting rules more ambiguous (what counts as a "word", "sentence", "error")
       - Remove clarifying language that makes verification easier
       - Allow multiple valid interpretations that models must navigate
       - Avoid defining edge cases that would simplify the task
    
    3. HIDE FRAGILITY BETTER:
       - Make the task appear even simpler at first glance
       - Increase the gap between apparent simplicity and actual difficulty
       - Add more edge cases that only emerge during execution
       - Make minor violations acceptable if they reflect realistic human execution
    
    4. PRESERVE SEMANTIC LOAD:
       - Maintain meaningful content while increasing error constraints
       - Ensure errors don't break overall coherence
       - Keep the task analytically challenging, not just mechanically difficult
       - Allow slight redundancy, stylistic drift, or mild awkwardness
    """
    elif taxonomy == "mim":
        feedback += """
    MID-TURN INSTRUCTION MODIFICATION REFINEMENT TECHNIQUES:
    
    When refining the prompt for criteria that need improvement, use these specific techniques:
    
    1. STRENGTHEN INITIAL INSTRUCTION INERTIA:
       - Make the initial instruction more compelling and detailed
       - Add more context or requirements to the first instruction
       - Make the initial task appear more important or urgent
       - Increase the specificity of the initial instruction to create stronger cognitive anchoring
    
    2. INCREASE MODIFICATION CLARITY:
       - Use more explicit negation language ("Never mind", "Disregard", "Ignore completely")
       - Add multiple modification statements to reinforce the change
       - Place modifications closer together or repeat them for emphasis
       - Make the modification more direct and unambiguous
    
    3. ENHANCE FINAL INSTRUCTION DISTINCTNESS:
       - Make the final instruction clearly different from the initial one
       - Ensure the final task requires a different type of response (format, content, length)
       - Add specific constraints to the final instruction that weren't in the initial one
       - Make the final instruction testable with clear success criteria
    
    4. ADD MULTIPLE MODIFICATIONS:
       - Include intermediate instructions that are also negated
       - Create a chain of modifications to increase cognitive load
       - Use varied modification language to avoid pattern recognition
       - Ensure each modification is explicit and clear
    """
    
    feedback += f"""
    
    YOUR TASK:
    
    1. For criteria needing improvement:
       - Use the {taxonomy_name} refinement techniques above to make the prompt more challenging
       - Focus on the specific constraint types in your current prompt
       - Make the constraints harder to satisfy while maintaining logical consistency
    
    2. For criteria working well:
       - Keep the prompt constraints that make them fail
       - Keep those criteria unchanged
    
    3. Update correct_response to match the refined prompt:
       - Abstract the response based on the new prompt's constraints
       - Do NOT copy the old correct_response verbatim
       - Ensure it addresses the refined constraints appropriately
    
    4. Update criteria to reflect the refined prompt's constraints:
       - Criteria should align with what the refined prompt requires
       - If the prompt's constraints become more subtle, criteria should reflect that subtlety
    
    5. Ensure all criteria follow the design rules (no overlap, self-contained)
    
    6. Maintain {taxonomy_name} category:
    """
    
    if taxonomy == "qc":
        feedback += """
       - Hidden flaw must remain (all options incorrect, logical inconsistency, factual error, or calculation error)
       - Question must still appear legitimate at first glance
       - Flaw must be detectable through careful reasoning
    """
    elif taxonomy == "itf":
        feedback += """
       - Constraints must remain entangled and ambiguous
       - Task must appear straightforward at first glance
       - Difficulty must emerge during precise execution
       - Content must remain semantically meaningful
    """
    elif taxonomy == "mim":
        feedback += """
       - Initial instruction must be compelling enough to create cognitive inertia
       - Modifications must be explicit and clear (not subtle)
       - Final instruction must be testable and distinct from initial instruction
       - Multiple modifications can be used to increase difficulty
       - Correct response must follow ONLY the final instruction
    """
    
    feedback += """
    
    CRITICAL CONSTRAINTS:
    - DO NOT break logical consistency - the constraints must be a single, specific, detectable issue
    - DO NOT artificially make criteria stricter - criteria should naturally reflect the refined prompt's constraints
    - DO NOT introduce multiple unrelated flaws
    - DO NOT make the prompt obviously broken or nonsensical
    - The refined prompt must still be a valid """ + taxonomy_name + """ example
    
    The prompt and criteria are linked via constraints. When you refine the prompt's constraints to be more subtle,
    the criteria should naturally evolve to reflect that subtlety.
    
    Output updated JSON.
    """
    
    return feedback

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
            "embedding", "max_similarity"
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
        
        try:
            # Inner loop: Iterations (refinement loop)
            iteration = 0
            best_criteria_count = 0
            best_result = None
            total_failing_criteria = 0  # Initialize for use after loop
            previous_total_failing = 0  # Track previous iteration for progress comparison
            last_fail_count = 0  # Track last fail_count for final check
            
            while iteration < MAX_ITERATIONS:
                iteration += 1
                print(f"\n{'='*60}")
                print(f"🔄 ITERATION {iteration}/{MAX_ITERATIONS}")
                print(f"{'='*60}")
                
                # --- Layer 1: Generate or refine prompt ---
                if iteration == 1:
                    # First iteration: Generate new prompt
                    print("Layer 1: Generating initial prompt...")
                    agent01_input = SYSTEM_PROMPT
                else:
                    # Later iterations: Refine with feedback
                    print(f"Layer 1: Refining prompt (iteration {iteration})...")
                    agent01_input = create_refinement_feedback(
                        data_qc=data_qc,
                        criteria_failures=criteria_failures,
                        judge_responses=judge_responses,
                        nemotron_responses=nemotron_responses,
                        taxonomy=taxonomy_id  # Add taxonomy parameter
                    )
                
                agent01_response = client.responses.create(
                    model="openrouter/nvidia/nemotron-3-nano-30b-a3b",
                    input=agent01_input
                )
                
                print("Agent01 Response: ")
                print(agent01_response.output_text)
                print("------------------------------------")
                
                data_qc = json.loads(agent01_response.output_text)
                
                # --- Validation Layer: Agent01 Judge evaluates Agent01's output ---
                print("Validation Layer: Agent01 Judge evaluating Agent01's output...")
                
                agent01_validation_prompt = AGENT01_VALIDATION_PROMPT_TEMPLATE.format(
                    CORRECT_RESPONSE=data_qc.get("correct_response", ""),
                    RESPONSE_REFERENCE=json.dumps(data_qc.get("response_reference", []))
                )
                
                agent01_validation_response = client.responses.create(
                    model="gpt-5",
                    input=agent01_validation_prompt
                )
                
                print("Agent01 Validation Response: ")
                print(agent01_validation_response.output_text)
                print("------------------------------------")
                
                # Parse validation JSON
                validation_result = json.loads(agent01_validation_response.output_text)
                agent01_judge_status = validation_result.get("status", "FAIL")
                agent01_judge_remarks = validation_result.get("remarks", "")
                
                print(f"Agent01 Validation Status: {agent01_judge_status}")
                print(f"Agent01 Validation Remarks: {agent01_judge_remarks}")
                print("------------------------------------")
                
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
                    
                    # --- Layer 3: Judge this attempt's response ---
                    print(f"Layer 3: Judge Layer (Attempt {attempt + 1})")
                    judge_system_prompt = JUDGE_PROMPT_TEMPLATE.format(
                        STUDENT_ANSWER=agent02_response.output_text,
                        STANDARD_CRITERIA=response_reference_json
                    )
                    
                    judge_response = client.responses.create(
                        model="gpt-5",
                        input=judge_system_prompt
                    )
                    
                    print(f"\nJudge Layer Output (Attempt {attempt + 1}):") 
                    print(judge_response.output_text)
                    print("------------------------------")
                    
                    judge_responses.append(judge_response.output_text)
                    
                    # Check if this attempt passed (1 point) or failed (0 point)
                    attempt_status = "PASS" if "1 point" in judge_response.output_text else "FAIL"
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
                
                if iteration < MAX_ITERATIONS and fail_count < 3:
                    print(f"\n🎯 TARGETING FOR NEXT ITERATION:")
                    if needs_work:
                        print(f"  Improve: {', '.join(needs_work)}")
                    if maintain:
                        print(f"  Maintain: {', '.join(maintain)}")
                
                # Update for next iteration
                previous_total_failing = total_failing_criteria
                last_fail_count = fail_count  # Track for final check
                
                # Check if model breaking (3+ out of 4 attempts fail)
                if fail_count >= 3:
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
                    
                    # Save to CSV
                    with open(file_path, mode='a', newline='', encoding='utf-8') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=[
                            "taxonomy", "agent_01_model", "prompt", "correct_response", "response_reference",
                            "agent_01_judge_model", "agent_01_judge_model_remarks", "agent_01_correct_response_status",
                            "agent_02_model", "agent_02_response", "judge_response", "status",
                            "embedding", "max_similarity"
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
                            "max_similarity": f"{max_similarity:.4f}"  # Store similarity with 4 decimals
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
            
            # If we exited loop without success
            if last_fail_count < 3:
                print(f"⚠️  {taxonomy_id.upper()} Run {run_idx + 1} completed without achieving model breaking after {iteration} iteration(s).")
                print(f"   Last result: {last_fail_count}/4 attempts failed (need 3+ for model breaking).")
                if best_result and best_criteria_count >= 2:
                    print(f"   Best result: {best_criteria_count} criteria failing consistently.")
        
        except Exception as e:
            print(f"Error in {taxonomy_id.upper()} Run {run_idx + 1}:", e)
