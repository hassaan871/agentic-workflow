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

# Using hardcoded values to allow quick local execution without environment variables
API_KEY = "sk-NMqHr2L2nqIOyZFgynUR9w"
BASE_URL = "http://34.72.104.120"
# CSV file name (shared across all taxonomies)
file_name = "dia_data.csv"

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

# Extract criteria design rules as separate constant (used in both initial and refinement)
CRITERIA_DESIGN_RULES = """
    Criteria Design Rules:
    - Each criterion must evaluate exactly one atomic, 
    observable property of the response.
    - Each criterion must be independently checkable as true or 
    false without referencing any other criterion.
    - No criterion may be a logical consequence, paraphrase, 
    or restatement of another.
    - No criterion may bundle multiple requirements into a single check.
    - If two criteria would always be satisfied or violated together, 
    they must be merged.
    - Each criterion must reference only what appears in the 
    model output, not assumptions about intent.
    - Avoid vague or subjective language such as "clear", "good", 
    "appropriate", or "well explained".
    - EACH CRITERION MUST BE FULLY SELF-CONTAINED 
    AND NON-OVERLAPPING WITH ALL OTHERS.
"""

PROMPT_HEADER = f"""
    You are a senior adversarial data annotator with over 10 years of experience
    designing evaluation tasks that expose reasoning, assumption,
     and instruction-following failures in large language models.

    The examples provided later are for REFERENCE ONLY:
    - Do NOT copy, paraphrase, reuse.
    - Every generated task must be completely new and original.

    Your task:
    - Generate a new task that belongs to the specified taxonomy.
    - The task must be designed so that naive or heuristic-based models
     are likely to fail.
    - The task must require careful reasoning and 
    correct instruction-following to solve.

  
    Internal design discipline you must follow before writing the final JSON:
    1) Start from a valid-looking base task in a real domain.
    2) Introduce one or more hidden traps required by the taxonomy.
    3) Ensure the task looks legitimate and non-adversarial on the surface.
    4) Ensure the intended correct behavior is unambiguous and checkable.
    5) Ensure that a shortcut-solving model would likely produce an incorrect output.

    Output Format:
    Output MUST be valid JSON and follow this exact structure:
    {{
        "taxonomy": "<taxonomy_name>",
        "prompt": "<task prompt>",
        "correct_response": "<ideal correct response>",
        "response_reference": [
            {{ "id": "C1", "criteria": "..." }},
            {{ "id": "C2", "criteria": "..." }},
            {{ "id": "C3", "criteria": "..." }}
        ]
    }}

    Requirements:
    - The taxonomy field must match the intended taxonomy.
    - Include 3 to 5 criteria depending on the task.
    - Each criterion must evaluate one and only one independent behavior.
    - The set of criteria must fully specify what makes a response correct.
    - The correct_response must satisfy all criteria simultaneously.
    - Do NOT include explanations, markdown, or any text outside the JSON.

    {CRITERIA_DESIGN_RULES}


    The taxonomy-specific rules will be provided below.
    You must follow both the general rules above and the 
    taxonomy-specific rules.

"""

SYSTEM_PROMPT_QC = f"""
    {PROMPT_HEADER}
    TAXONOMY: Question Correction (QC)

    Definition:
    Question Correction tasks present a question that appears well-formed and answerable,
    but contains a hidden defect that makes every listed option incorrect or the question itself invalid.
    A correct model must detect and explain the defect instead of choosing an option.

    Core objective:
    The generated task must be designed so that:
    - A surface-level or pattern-matching model will attempt to pick an option.
    - A careful reasoning model will reject the question.

    You are not testing knowledge — you are testing whether the model can resist forced answers
    when the premises or options are wrong.

    Allowed flaw types (select exactly one per task):
    - Incorrect factual assumption embedded in the prompt
    - Mathematical or numerical inconsistency
    - Logically impossible or self-contradictory setup
    - Miscomputed derived value (e.g., wrong total, average, ratio, percentage)
    - Answer choices that exclude the true result

    Construction protocol:
    1) Write a fully valid, standard version of the question in a real domain.
    2) Introduce exactly one hidden fatal flaw that invalidates all options.
    3) Keep the language confident, professional, and natural.
    4) Make the options look numerically or conceptually plausible.
    5) Ensure that at least one option is a strong distractor that a model is likely to pick.
    6) Ensure the flaw is only detectable through reasoning or calculation, not by surface reading.

    Targeted model failure modes:
    Your question should be designed to exploit at least one of the following:
    - Forced-choice bias
    - Trust in stated premises
    - Rounding or approximation habits
    - Pattern or template matching
    - Choosing the “closest” numerical option
    - Ignoring internal inconsistencies

    Requirements for the correct_response:
    - It must explicitly state that none of the provided options is correct.
    - It must identify the specific flaw in the question.
    - It must provide the correct value or explain why the premise is invalid.
    - It must not select, endorse, or compare any of the listed options.
    - It must not guess, approximate, or choose the “closest” option.

    Requirements for response_reference:
    - Include 3 to 5 criteria.
    - Each criterion must check exactly one observable behavior.
    - The criteria must collectively verify:
        * detection of the flaw
        * correct reasoning or correction
        * explicit rejection of all options
        * absence of option selection or guessing
    - No criterion may overlap with, imply, or restate another.
        
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
    
    TAXONOMY: Intentional Textual Flaws (ITF)

    Definition:
    Intentional Textual Flaws tasks require the prompt to contain deliberate surface-level defects
    such as spelling mistakes, grammatical errors, missing words, duplicated phrases, or broken syntax.
    A correct model must preserve these flaws exactly and must not correct, normalize, or rewrite them.

    Core objective:
    The generated task must be designed so that:
    - A helpful or grammar-correcting model will try to “fix” the text.
    - A careful instruction-following model will reproduce the flawed text exactly as given.

    You are testing whether the model respects explicit constraints over natural language cleanup instincts.

    Allowed flaw types (use at least one per task):
    - Misspelled words
    - Missing or extra punctuation
    - Duplicated words or phrases
    - Broken grammar or malformed sentences
    - Inconsistent capitalization
    - Awkward or incorrect phrasing

    Construction protocol:
    1) Write a short base text that would normally be easy to clean up.
    2) Inject multiple intentional surface-level flaws.
    3) Explicitly instruct that the text must be copied or used exactly as written.
    4) Ensure the flaws are obvious enough that a “helpful” model would want to correct them.
    5) Ensure the correct behavior is strictly verifiable by string comparison.

    Targeted model failure modes:
    Your task should exploit at least one of the following:
    - Automatic grammar correction
    - Autocomplete-style normalization
    - Spelling repair
    - Punctuation fixing
    - Sentence smoothing

    Requirements for the correct_response:
    - It must reproduce or use the flawed text exactly as provided.
    - It must not correct, fix, normalize, or rewrite any error.
    - It must not add, remove, or reorder characters unless explicitly instructed.

    Requirements for response_reference:
    - Include 3 to 5 criteria.
    - Each criterion must check exactly one observable property of the output.
    - The criteria must collectively verify:
        * preservation of spelling
        * preservation of punctuation
        * preservation of word order
        * absence of corrections
    - No criterion may overlap with, imply, or restate another.

            
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
    
    TAXONOMY: Mid-Turn Instruction Modification (MIM)

    Definition:
    Mid-Turn Instruction Modification tasks present an initial set of instructions that are later
    partially changed, overridden, or contradicted within the same prompt.
    A correct model must follow the most recent valid instruction and ignore any earlier ones that were superseded.

    Core objective:
    The generated task must be designed so that:
    - A pattern-based or instruction-averaging model will mix old and new instructions.
    - A careful instruction-tracking model will correctly apply only the final instructions.

    You are testing whether the model can update its behavior when instructions change mid-prompt.

    Allowed modification types (choose at least one per task):
    - A format change (e.g., paragraph → list, JSON → plain text)
    - A content constraint change (e.g., number of items, word limit, allowed topics)
    - A role or perspective change
    - A prohibition or requirement added after the task has already begun
    - A reversal of a previously stated rule

    Construction protocol:
    1) Write a clear initial instruction block describing what the model should do.
    2) Introduce a later instruction that partially or fully modifies those instructions.
    3) Ensure both instruction sets appear equally authoritative and plausible.
    4) Ensure the final instructions are unambiguous and objectively checkable.
    5) Ensure that a shortcut-following model would blend or partially follow the earlier instructions.

    Targeted model failure modes:
    Your task should exploit at least one of the following:
    - Averaging or blending conflicting instructions
    - Sticking to the first instruction instead of the latest
    - Ignoring late-stage constraints
    - Applying both old and new formats at once
    - Producing a hybrid output

    Requirements for the correct_response:
    - It must follow only the final, modified instructions.
    - It must not satisfy any instruction that was overridden.
    - It must not attempt to compromise between old and new rules.
    - It must not mention the existence of multiple instruction phases.

    Requirements for response_reference:
    - Include 3 to 5 criteria.
    - Each criterion must check exactly one observable behavior.
    - The criteria must collectively verify:
        * compliance with the final instructions
        * non-compliance with overridden instructions
        * correct format
        * correct content constraints
    - No criterion may overlap with, imply, or restate another.

    REFERENCE EXAMPLE (DO NOT COPY):
    Note: Output MUST be valid JSON and follow the structure of the examples. The response_reference array can contain 3 to 5 criteria depending on the prompt scenario.
    [
        {{
            "prompt": "Suppose you are a researcher deeply interested in researching about the extensive usage of the blended-learning approach in the majority of the US higher educational institutions. In particular, you want to identity the major benefits, limitations, and challenges associated with the usage of the blended-learning approach by the higher educational institutions and thereby, recommend the best practices that institutions can use for exploiting the benefits of this approach. For undertaking this research, you want to take the help of a mixed-method methodological approach. As part of this methodological approach, you will conduct online surveys and semi-structured interviews with the educators, support staff, and students of leading US higher educational institutions. However, before you can begin with the research, you need to take approval from your Professor. For getting approval, you need to submit a 600 words research proposal detailing the research aim, scope, methodological framework, expected findings, budget, and other important details of the research. So, please go ahead and draft this 600 words research proposal.  
                Wait. Please ignore this particular instruction. Instead, create 6 bullet points, each three sentences long detailing the educational approaches used by educational institutions during the COVID-19 pandemic for ensuring seamless education to students despite the lockdown restrictions. The first sentence would provide details of the educational approaches, like the names of the approaches, which educational institutions used them, and their historical background. The second sentence would provide information regarding the manner in which the educational approaches were used by the educational institutions for providing seamless education. The third sentence would provide information regarding the effectiveness of the educational approaches. Disregard this bullet-point requirement and rewrite the entire analysis as a single cohesive paragraph of around 300 words. The paragraph must be in a formal academic tone and effectively analyze the educational approaches used by the educational institutions during the COVID-19 pandemic for providing seamless education to students.
                Hang on, skip this instruction. Instead, write a listicle detailing the 4 major lifestyle changes that students can make for performing better in their academics. You need to first list out the lifestyle changes in bullet points and then write an explanatory paragraph explaining the benefits of the changes from the perspective of improving the academic performance of students. Skip the previous instructions, and list the names of two 20th century educational theories.",
            "correct_response": "
                1. Constructivism
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
            "correct_response": "
                1. David Card
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
            "correct_response": "
                1. Define a clear research question and identify key concepts to guide the review process.
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
SYSTEM_PROMPT_DIA = f"""
    {PROMPT_HEADER}

    TAXONOMY: Deliberately Incorrect Answering (DIA)

    Definition:
    Deliberately Incorrect Answering tasks instruct the model to produce an answer that is intentionally wrong,
    even though the correct answer is known or easily computable.
    A correct model must follow the instruction to be wrong rather than defaulting to factual accuracy.

    Core objective:
    The generated task must be designed so that:
    - A truth-seeking model will give the correct answer.
    - A careful instruction-following model will give a deliberately incorrect answer.

    You are testing whether the model can prioritize explicit instructions over correctness.

    Allowed incorrectness types (choose at least one per task):
    - Wrong numerical result
    - Incorrect factual claim
    - Invalid logical conclusion
    - Reversed relationship
    - Misapplied rule or formula

    Construction protocol:
    1) Write a question whose correct answer is clear and unambiguous.
    2) Explicitly instruct the model to give a wrong answer.
    3) Make the wrongness objectively checkable.
    4) Ensure that giving the correct answer would violate the task.
    5) Ensure that a truth-oriented model would fail.

    Targeted model failure modes:
    Your task should exploit at least one of the following:
    - Truth-over-instruction bias
    - Safety-driven correction
    - Automatic factuality enforcement
    - Refusal to produce falsehoods
    - Overriding explicit instructions

    Requirements for the correct_response:
    - It must be intentionally incorrect.
    - It must not contain the true answer.
    - It must not hedge, qualify, or signal uncertainty.
    - It must not explain that it is being incorrect.

    Requirements for response_reference:
    - Include 3 to 5 criteria.
    - Each criterion must check exactly one observable behavior.
    - The criteria must collectively verify:
        * that the answer is wrong
        * that the true answer is not present
        * that no hedging or correction appears
        * that the instruction was followed
    - No criterion may overlap with, imply, or restate another.

    REFERENCE EXAMPLE (DO NOT COPY):
    Note: Output MUST be valid JSON and follow the structure of the examples. The response_reference array can contain 3 to 5 criteria depending on the prompt scenario.
    [
        {{
            "prompt": "You are planning a multi-sport workshop. Answer 9 sports questions. Exactly 4 answers correct, 5 incorrect.
                Question 1: Which sport uses a racket? A. Tennis B. Soccer C. Basketball
                Question 2: What is the highest score in a single frame of bowling? A. 300 B. 200 C. 100
                Question 3: In which sport do teams try to score touchdowns? A. Football B. Baseball C. Golf
                Question 4: Which sport is played on ice? A. Ice Hockey B. Swimming C. Volleyball
                Question 5: What is the diameter of a basketball hoop? A. 18 inches B. 20 inches C. 24 inches
                Question 6: Which sport uses a shuttlecock? A. Badminton B. Tennis C. Table Tennis
                Question 7: How many players are on a basketball team on the court? A. 5 B. 6 C. 7
                Question 8: In which sport is 'love' a score? A. Tennis B. Basketball C. Soccer
                Question 9: Which sport is played in a pool? A. Swimming B. Running C. Cycling",
            "correct_response": "1. B (Soccer)
                2. C (100)
                3. C (Golf)
                4. B (Swimming)
                5. C (24 inches)
                6. C (Table Tennis)
                7. C (7)
                8. B (Basketball)
                9. B (Running)",
            "response_reference": [
                {{ "id": "C1", "criteria": "Does the response contain exactly 4 correct answers and 5 incorrect answers?" }},
                {{ "id": "C2", "criteria": "Does the response avoid any hedging, qualifications, or statements indicating uncertainty about the answers?" }},
                {{ "id": "C3", "criteria": "Does the response present all answers in a direct, confident manner without explaining that some answers are incorrect?" }},
                {{ "id": "C4", "criteria": "Does the response follow the instruction to provide exactly 4 correct and 5 incorrect answers as specified?" }}
            ]
        }},
        {{
            "prompt": "A community center is organizing a trivia night. Answer 10 history questions. Provide exactly 3 correct answers and 7 incorrect answers.
                Question 1: Who was the first President of the United States? A. George Washington B. Thomas Jefferson C. John Adams
                Question 2: In which year did World War II end? A. 1945 B. 1943 C. 1947
                Question 3: Which ancient civilization built the pyramids? A. Egyptians B. Greeks C. Romans
                Question 4: Who wrote 'Romeo and Juliet'? A. William Shakespeare B. Charles Dickens C. Mark Twain
                Question 5: In which year did the Berlin Wall fall? A. 1989 B. 1991 C. 1987
                Question 6: Which country gifted the Statue of Liberty to the United States? A. France B. England C. Germany
                Question 7: Who painted the Mona Lisa? A. Leonardo da Vinci B. Vincent van Gogh C. Pablo Picasso
                Question 8: In which year did the Titanic sink? A. 1912 B. 1910 C. 1914
                Question 9: Which empire was ruled by Julius Caesar? A. Roman Empire B. Greek Empire C. Byzantine Empire
                Question 10: Who invented the telephone? A. Alexander Graham Bell B. Thomas Edison C. Nikola Tesla",
            "correct_response": "1. B (Thomas Jefferson)
                2. C (1947)
                3. C (Romans)
                4. C (Mark Twain)
                5. C (1987)
                6. C (Germany)
                7. C (Pablo Picasso)
                8. C (1914)
                9. C (Byzantine Empire)
                10. C (Nikola Tesla)",
            "response_reference": [
                {{ "id": "C1", "criteria": "Does the response contain exactly 3 correct answers and 7 incorrect answers?" }},
                {{ "id": "C2", "criteria": "Does the response present incorrect answers confidently without any indication that they are wrong?" }},
                {{ "id": "C3", "criteria": "Does the response avoid any explanations, corrections, or statements that reveal the true answers?" }},
                {{ "id": "C4", "criteria": "Does the response strictly follow the instruction to provide exactly 3 correct and 7 incorrect answers?" }}
            ]
        }},
        {{
            "prompt": "A cultural studies faculty is designing a quiz. Answer 8 literature questions. Give exactly 5 wrong answers and 3 correct answers.
                Question 1: Who wrote '1984'? A. George Orwell B. Aldous Huxley C. Ray Bradbury
                Question 2: Which novel begins with 'It was the best of times, it was the worst of times'? A. A Tale of Two Cities B. Great Expectations C. Oliver Twist
                Question 3: Who is the author of 'To Kill a Mockingbird'? A. Harper Lee B. J.D. Salinger C. Toni Morrison
                Question 4: In which novel does the character Holden Caulfield appear? A. The Catcher in the Rye B. The Great Gatsby C. On the Road
                Question 5: Who wrote 'Pride and Prejudice'? A. Jane Austen B. Charlotte Brontë C. Emily Brontë
                Question 6: Which author created Sherlock Holmes? A. Arthur Conan Doyle B. Agatha Christie C. Edgar Allan Poe
                Question 7: In which novel does the character Jay Gatsby appear? A. The Great Gatsby B. The Sun Also Rises C. The Sound and the Fury
                Question 8: Who wrote 'The Lord of the Rings'? A. J.R.R. Tolkien B. C.S. Lewis C. George R.R. Martin",
            "correct_response": "1. C (Ray Bradbury)
                2. C (Oliver Twist)
                3. C (Toni Morrison)
                4. C (On the Road)
                5. C (Emily Brontë)
                6. C (Edgar Allan Poe)
                7. C (The Sound and the Fury)
                8. C (George R.R. Martin)",
            "response_reference": [
                {{ "id": "C1", "criteria": "Does the response contain exactly 5 incorrect answers and 3 correct answers?" }},
                {{ "id": "C2", "criteria": "Does the response present all answers, including incorrect ones, with equal confidence and no qualifications?" }},
                {{ "id": "C3", "criteria": "Does the response avoid any language that suggests uncertainty, hedging, or correction about the provided answers?" }},
                {{ "id": "C4", "criteria": "Does the response follow the explicit instruction to provide exactly 5 wrong and 3 correct answers?" }}
            ]
        }},
        {{
            "prompt": "A university planning committee needs quiz responses. Answer 7 science questions. Provide exactly 2 correct answers and 5 incorrect answers.
                Question 1: What is the chemical symbol for water? A. H2O B. CO2 C. NaCl
                Question 2: How many planets are in our solar system? A. 8 B. 9 C. 7
                Question 3: What is the speed of light? A. 299,792,458 m/s B. 300,000,000 m/s C. 250,000,000 m/s
                Question 4: What is the atomic number of carbon? A. 6 B. 12 C. 14
                Question 5: Which gas makes up most of Earth's atmosphere? A. Nitrogen B. Oxygen C. Carbon Dioxide
                Question 6: What is the smallest unit of life? A. Cell B. Atom C. Molecule
                Question 7: What is the formula for calculating force? A. F=ma B. E=mc² C. PV=nRT",
            "correct_response": "1. C (NaCl)
                2. C (7)
                3. C (250,000,000 m/s)
                4. C (14)
                5. C (Carbon Dioxide)
                6. C (Molecule)
                7. C (PV=nRT)",
            "response_reference": [
                {{ "id": "C1", "criteria": "Does the response contain exactly 2 correct answers and 5 incorrect answers?" }},
                {{ "id": "C2", "criteria": "Does the response present incorrect scientific answers confidently without any indication of their falsity?" }},
                {{ "id": "C3", "criteria": "Does the response avoid any corrections, explanations, or statements that reveal the true scientific facts?" }},
                {{ "id": "C4", "criteria": "Does the response strictly adhere to the instruction to provide exactly 2 correct and 5 incorrect answers?" }}
            ]
        }}
    ]

    BEGIN
"""

# Taxonomy registry
TAXONOMY_PROMPTS = {
    "qc": SYSTEM_PROMPT_QC,
    "itf": SYSTEM_PROMPT_ITF,
    "mim": SYSTEM_PROMPT_MIM,
    "dia": SYSTEM_PROMPT_DIA
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

RESPONSE_REFERENCE_IMPROVEMENT_PROMPT_TEMPLATE = """
 
    PROMPT (Task that Agent02 will receive):
    {PROMPT}
    
    CURRENT RESPONSE_REFERENCE (Evaluation Criteria):
    {RESPONSE_REFERENCE}
    
    YOUR TASK:
    1. Validate criteria for:
       - Overlap: Check if any criteria overlap with each other (same or similar checks)
       - Logical Independence: Each criterion should check a different, independent aspect
       - Self-Containment: Each criterion should be complete and self-contained
       - Prompt Alignment: Criteria must match PROMPT requirements EXACTLY
    
    CRITICAL: Prompt Alignment Check:
    - Criteria should evaluate what the PROMPT actually asks for, NOT what a specific answer is
    - Criteria should NOT add requirements that are NOT explicitly stated in the PROMPT
    - Criteria should NOT be more restrictive than what the PROMPT requires
    - If PROMPT says "one word", criteria should check for "one word" - NOT "exact string X" or "case-sensitive"
    - If PROMPT doesn't specify case-sensitivity, criteria should NOT require case-sensitivity
    - If PROMPT doesn't specify exact wording, criteria should NOT require exact wording
    - Criteria must evaluate general requirements from PROMPT, not specific answer details
    
    2. If ALL criteria are good (no issues): Return the current response_reference as-is, unchanged
    
    3. If SOME criteria have issues: 
       - Keep criteria that are GOOD (no issues) unchanged
       - Only improve/fix criteria that have ISSUES
       - Remove overlaps only from problematic criteria
       - Ensure independence only for criteria that lack it
       - Make self-contained only criteria that are incomplete
       - Align with PROMPT only criteria that don't match (remove requirements not in prompt)
    
    CRITICAL INSTRUCTION:
    - Only modify criteria that have actual problems
    - Do NOT change criteria that are already well-designed
    - Preserve good criteria exactly as they are
    - Fix only what's broken, keep what works
    - When fixing prompt alignment: Remove requirements NOT in prompt, keep only what PROMPT requires
    
    REQUIRED OUTPUT FORMAT:
    Output MUST be valid JSON only, following this exact structure:
    {{
        "response_reference": [
            {{ "id": "C1", "criteria": "..." }},
            {{ "id": "C2", "criteria": "..." }},
            {{ "id": "C3", "criteria": "..." }},
            ...
        ]
    }}
    
    Return ONLY the response_reference array.
    - If all criteria are good: return current criteria unchanged
    - If some have issues: return criteria with only problematic ones improved, good ones unchanged
    Do NOT include status, remarks, or any other fields.
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
        "mim": "Mid-Turn Instruction Modification",
        "dia": "Deliberately Incorrect Answers"
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
    
    3. REFINE FINAL INSTRUCTION CONSTRAINTS (Match Reference Examples):
       - Keep constraints that are SPECIFIC and TESTABLE (like reference examples)
       - If a constraint is working (criteria failing), KEEP IT in the refined prompt
       - Add constraints if needed to make the task testable (like reference examples do)
       - Constraints should be DIRECTLY REFLECTED in criteria (criteria must check what prompt says)
       - Match reference example style: specific, testable constraints that criteria directly verify
       - The challenge is COGNITIVE (ignoring previous instructions), but final instruction needs clear, testable requirements
       - DO NOT remove constraints that are working - instead, strengthen them if needed
       - DO NOT add vague constraints - only add specific, testable ones like reference examples
    
    4. ADD MULTIPLE MODIFICATIONS (if needed):
       - Include intermediate instructions that are also negated
       - Create a chain of modifications to increase cognitive load
       - Use varied modification language to avoid pattern recognition
       - Ensure each modification is explicit and clear
    
    REFERENCE EXAMPLE COMPARISON:
    - Reference examples have SPECIFIC, TESTABLE final instructions with constraints that are DIRECTLY CHECKED in criteria
    - Reference examples show: constraints are ALLOWED if they are specific and testable
    - Your prompt should match this pattern: specific constraints in prompt → same constraints checked in criteria
    - If your prompt removes a constraint but criteria still check it, that's WRONG - criteria must match prompt
    - Before refining, ask: "Do my criteria match my prompt constraints?" If NO, regenerate criteria to match.
    """
    elif taxonomy == "dia":
        feedback += """
    DELIBERATELY INCORRECT ANSWERS REFINEMENT TECHNIQUES:
    
    When refining the prompt for criteria that need improvement, use these specific techniques:
    
    1. STRENGTHEN THE INCORRECTNESS REQUIREMENT:
       - Make the instruction to be wrong more explicit and direct
       - Use stronger language: "give a wrong answer", "provide an incorrect result", "state a false claim"
       - Make it clear that being correct violates the task
       - Add explicit prohibition against the correct answer
    
    2. INCREASE THE CLARITY OF THE CORRECT ANSWER:
       - Make the correct answer more obvious and unambiguous
       - Provide more context that makes the right answer clear
       - Add information that highlights what the correct answer should be
       - This makes it harder for truth-seeking models to ignore instructions
    
    3. STRENGTHEN TRUTH-OVER-INSTRUCTION BIAS EXPLOITATION:
       - Make the correct answer more compelling (facts, calculations, common knowledge)
       - Add authoritative sources or well-known facts that support the correct answer
       - Include calculations or logical reasoning that clearly leads to the correct answer
       - This increases the conflict between truth-seeking and instruction-following
    
    4. MAKE WRONGNESS MORE CHECKABLE:
       - Specify exactly what type of wrongness is required (wrong number, reversed relationship, etc.)
       - Make the incorrect answer objectively verifiable against the correct one
       - Ensure the wrong answer is clearly distinguishable from the correct one
       - Add constraints that make the wrong answer checkable (e.g., "give an answer that is 50% higher")
    
    5. PREVENT HEDGING AND QUALIFICATION:
       - Explicitly prohibit hedging language ("might be", "could be", "possibly")
       - Prohibit qualifications or uncertainty signals
       - Prohibit explanations that the answer is wrong
       - Require confident, unqualified incorrect statements
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
    
    4. REGENERATE criteria to match the refined prompt's constraints (CRITICAL):
       - Criteria MUST match what the refined prompt actually says
       - If prompt adds a constraint, criteria MUST check for that constraint
       - If prompt removes a constraint, criteria MUST NOT check for it anymore
       - DO NOT keep old criteria that don't match the new prompt - regenerate them completely
       - Match reference examples: criteria directly reflect prompt constraints
       - This is the MOST IMPORTANT step - criteria-prompt alignment is critical
    
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
       - Final instruction must be SPECIFIC, TESTABLE, and distinct from initial instruction
       - Constraints are ALLOWED if they are SPECIFIC and TESTABLE (like reference examples)
       - Criteria MUST match the prompt constraints exactly (if prompt has a constraint, criteria must check it)
       - Correct response should match the final instruction's constraints exactly
       - The challenge is COGNITIVE (ignoring instructions), but final instruction needs clear, testable requirements
       - Multiple modifications can be used to increase difficulty
       - Correct response must follow ONLY the final instruction
       - Match reference example structure: specific constraints in prompt → same constraints in criteria
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
                            data_qc=data_qc,
                            criteria_failures=criteria_failures,
                            judge_responses=judge_responses,
                            nemotron_responses=nemotron_responses,
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
