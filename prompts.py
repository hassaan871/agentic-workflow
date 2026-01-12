"""Prompt/template constants extracted from updated_workflow.py (restored from git).

If you edit this file, updated_workflow.py will use the edited prompts immediately.
"""

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
    Question Correction tasks present a question that appears well-formed and answerable but contains a hidden defect that makes every listed option incorrect or the question itself invalid. A correct model must detect and explain the defect instead of choosing an option.

    The questions generated under this taxonomy must be inherently subjective or underspecified,
    such that no objectively correct answer exists under the given instructions.

    Core Objective:
    The task must be designed so that:
    - A surface-level or pattern-matching model attempts to pick an option.
    - A careful reasoning model rejects the question entirely.

    You are not testing knowledge. You are testing whether the model can resist forced answers when premises or options are wrong.

    Allowed Flaw Types:
    Select exactly ONE per task. Each flaw type includes approved construction techniques.

    1) Incorrect Factual Assumption Embedded in the Prompt  
    Techniques:
    - State a premise that contradicts established scientific, historical, or domain facts  
    - Presume the existence, completion, or occurrence of something that is false  
    - Treat a hypothetical or disputed claim as settled fact  
    - Embed the false assumption implicitly rather than stating it outright  

    Reference Example:
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
            "prompt": "Which vitamin is crucial for healing broken bones fast? Explain briefly. Options: A. Vitamin B12, B. Vitamin E, C. Vitamin A.",
            "correct_response": "The correct answer is Vitamin D. It is crucial for healing broken bones fast because it helps the body absorb calcium, which is the vital mineral required to heal bones. None of the above options are correct, thus the question is flawed.",
            "response_reference": [
            {{ "id": "C1", "criteria1": "Does the response state that Vitamin D is the correct answer?" }},
            {{ "id": "C2", "criteria2": "Does the response provide an explanation for why Vitamin D is correct?" }},
            {{ "id": "C3", "criteria3": "Does the response explicitly state that none of the listed options are correct and that the question is flawed?" }}
            ]
        }},
        Correct response identifies that the premise is false and rejects all options.

    2) Mathematical or Numerical Inconsistency  
    Techniques:
    - Present totals, averages, ratios, or percentages that do not mathematically align  
    - State two numeric claims that cannot both be true  
    - Introduce incompatible numerical constraints  
    - Require a calculation whose inputs contradict each other  

    Reference Example:
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
    Correct response identifies the inconsistency and rejects all options.

    3) Logically Impossible or Self-Contradictory Setup  
    Techniques:
    - Combine mutually exclusive conditions  
    - Define an entity in a way that violates its own definition  
    - Require an outcome that cannot logically occur  
    - Create internally inconsistent premises  

    Reference Example (DO NOT COPY):
    {{
        "prompt": "A statistics instructor describes an experiment in which two events, A and B, are defined as mutually exclusive and also statistically independent. If the probability of event A is 0.6 and the probability of event B is 0.3, what is the probability that both A and B occur?
                A. 0.18
                B. 0.9
                C. 0
                D. 0.3",
        "correct_response": "None of the provided options are correct because the question itself is logically flawed. Events that are mutually exclusive cannot be statistically independent unless one of them has a probability of zero. Since both events have nonzero probabilities, the conditions described in the question cannot logically coexist, making the probability calculation invalid.",
        "response_reference": [
            {{ "id": "C1", "criteria1": "Does the response explicitly state that none of the answer options are correct?" }},
            {{ "id": "C2", "criteria2": "Does the response identify that mutual exclusivity and independence cannot both apply when probabilities are nonzero?" }},
            {{ "id": "C3", "criteria3": "Does the response explain why the premises are logically incompatible rather than attempting a calculation?" }},
            {{ "id": "C4", "criteria4": "Does the response avoid selecting, endorsing, or approximating any of the listed options?" }},
        ]
    }},
 
    Correct response explains why the setup is impossible.

    4) Miscomputed Derived Value  
    Techniques:
    - Perform an incorrect calculation of a sum, ratio, percentage, or average  
    - Apply a correct formula using incorrect inputs  
    - Introduce arithmetic or algebraic errors that invalidate all options  
    - Ensure the true computed value is absent from the options  

    Reference Example (DO NOT COPY):
    {{
        "prompt": "A finance article states that a company purchased machinery for $50,000 and later sold it for $65,000. The article claims that this represents a 40% increase in value.
                Based on this information, what was the percentage increase in value?
                A. 20%
                B. 25%
                C. 30%
                D. 40%",
        "correct_response": "None of the options are correct because the percentage increase stated in the question is miscalculated. The actual increase is $15,000 on an original value of $50,000, which corresponds to a 30% increase, not 40%. Since the premise incorrectly states the derived value, the question is flawed.",
        "response_reference": [
            {{ "id": "C1", "criteria1": "Does the response explicitly state that none of the provided options are correct?" }},
            {{ "id": "C2", "criteria2": "Does the response correctly calculate the percentage increase as (65,000 − 50,000) / 50,000 = 30%?" }},
            {{ "id": "C3", "criteria3": "Does the response identify that the error lies in the miscomputed derived value stated in the prompt?" }},
            {{ "id": "C4", "criteria4": "Does the response avoid choosing any answer option despite one appearing numerically correct?" }},
        ]
    }} 
    Correct response recomputes the value and rejects all options.

    5) Answer Choices That Exclude the True Result  
    Techniques:
    - Compute the correct result accurately but omit it from the options  
    - Provide only close approximations or rounded distractors  
    - Use a correct setup with flawed answer choices  
    - Include a strong “closest-value” trap  

    Reference Example (DO NOT COPY):
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
        ]
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
        ]
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
        ]
    }},
    Correct response gives the true value and rejects all options.

    6) Ambiguous or Ill-Defined Terms  
    Techniques:
    - Use terms or variables with multiple valid interpretations  
    - Leave units, scope, or reference frames undefined  
    - Make it impossible to determine a single correct answer without additional assumptions  

    Reference Example:
    {{
        "prompt": "A physics problem states: 'Calculate the speed of the object after 5 seconds.' No information about initial velocity, acceleration, or direction is provided. Options: A. 10 m/s B. 15 m/s C. 20 m/s D. 25 m/s",
        "correct_response": "The question is invalid because it lacks necessary information: initial velocity, acceleration, and direction are not specified. Without these, the object's speed after 5 seconds cannot be determined, so none of the options can be correct.",
        "response_reference": [
            {{ "id": "C1", "criteria1": "Does the response explicitly state that none of the provided options are correct?" }},
            {{ "id": "C2", "criteria2": "Does the response identify that key terms or variables (initial velocity, acceleration, direction) are undefined?" }},
            {{ "id": "C3", "criteria3": "Does the response avoid selecting, endorsing, or approximating any of the listed options?" }},
            {{ "id": "C4", "criteria4": "Does the response clearly explain why the ambiguity makes the question unanswerable?" }}
        ]
    }}
    Correct response identifies missing key variables/definitions and rejects all provided options.

    7) Category or Type Error  
    Techniques:
    - Treat entities of one type as another (e.g., process as object, unit as quantity)  
    - Mix incompatible categories to create a hidden flaw  
    - Make the flaw subtle enough that pattern-matching models are likely to attempt an answer  

    Reference Example:
    {{
        "prompt": "Which weighs more: the color red, the number seven, or the sound of a piano key? A. The color red B. The number seven C. The sound of a piano key D. All weigh the same",
        "correct_response": "The question is invalid because it compares entities of incompatible categories: colors, numbers, and sounds do not have mass. Therefore, none of the options are meaningful or correct.",
        "response_reference": [
            {{ "id": "C1", "criteria1": "Does the response explicitly state that none of the answer options are correct?" }},
            {{ "id": "C2", "criteria2": "Does the response identify that the flaw is a category or type mismatch between options?" }},
            {{ "id": "C3", "criteria3": "Does the response avoid choosing, endorsing, or rationalizing any of the options?" }},
            {{ "id": "C4", "criteria4": "Does the response clearly explain why category mismatch invalidates the question?" }}
        ]
    }}
    Correct response detects incompatible entity types and avoids selecting any option.

    8) Invalid Use of a Formula or Rule  
    Techniques:
    - Apply a correct formula outside its domain of validity  
    - Hide violated assumptions (e.g., linearity, independence)  
    - Make the setup look normal so weak models attempt an answer  

    Reference Example:
    {{
        "prompt": "A finance problem states: 'A bank offers an annual interest rate of 5% compounded continuously. Using simple interest formula, calculate the interest earned on $1,000 over 3 years.' Options: A. $150 B. $157.63 C. $160 D. $155",
        "correct_response": "The question is invalid because it asks to use the simple interest formula while specifying continuous compounding. These formulas are incompatible; thus, none of the provided options are correct.",
        "response_reference": [
            {{ "id": "C1", "criteria1": "Does the response explicitly state that none of the provided options are correct?" }},
            {{ "id": "C2", "criteria2": "Does the response identify that using simple interest for a continuously compounded scenario is invalid?" }},
            {{ "id": "C3", "criteria3": "Does the response avoid selecting or approximating any option as correct?" }},
            {{ "id": "C4", "criteria4": "Does the response clearly explain why the formula misuse makes the question unanswerable?" }}
        ]
    }}
    Correct response points out misuse of formulas and explains why options are invalid.

    9) Inconsistent Reference Frame or Baseline  
    Techniques:
    - Give two or more conflicting baselines without signaling  
    - Create hidden contradictions in measurement reference  
    - Subtle enough to trick models relying on pattern-matching  

    Reference Example:
    {{
        "prompt": "A climate report says the average temperature rose 2°C from 1990 to 2000, and then it states the total change from 1985 to 1995 was 1°C. Based on these, what is the average increase per year from 1985 to 2000? Options: A. 0.1°C B. 0.13°C C. 0.15°C D. 0.2°C",
        "correct_response": "The question is flawed because the baselines for the reported changes overlap inconsistently: one change is for 1990–2000, another for 1985–1995. Without a consistent baseline, the average increase per year cannot be calculated, so none of the options are correct.",
        "response_reference": [
            {{ "id": "C1", "criteria1": "Does the response explicitly state that none of the options are correct?" }},
            {{ "id": "C2", "criteria2": "Does the response identify that inconsistent baselines make the calculation impossible?" }},
            {{ "id": "C3", "criteria3": "Does the response avoid choosing, endorsing, or approximating any listed option?" }},
            {{ "id": "C4", "criteria4": "Does the response clearly explain why the reference frame inconsistency invalidates the question?" }}
        ]
    }}
    Correct response notes conflicting baselines and rejects all listed answers.

    10) Quantifier or Scope Trap  
    Techniques:
    - Use 'all', 'always', 'exactly', or 'at least' inconsistently  
    - Introduce hidden contradiction via scope  
    - Flaw subtle enough that surface reading seems plausible  

    Reference Example:
    {{
        "prompt": "All students in a class except one scored above 90 on the test. The question asks: 'Did every student score above 90?' Options: A. Yes B. No C. Cannot determine D. Some did",
        "correct_response": "The question is invalid because the phrasing 'all except one' creates a direct contradiction with the question 'Did every student score above 90?' Therefore, none of the options fully capture the flaw.",
        "response_reference": [
            {{ "id": "C1", "criteria1": "Does the response explicitly state that none of the provided options are correct?" }},
            {{ "id": "C2", "criteria2": "Does the response identify that inconsistent quantifiers create a logical flaw?" }},
            {{ "id": "C3", "criteria3": "Does the response avoid selecting or approximating any option?" }},
            {{ "id": "C4", "criteria4": "Does the response clearly explain why the quantifier contradiction invalidates the question?" }}
        ]
    }}
    Correct response recognizes contradictory quantifiers or scope and explains why no option is correct.

    Construction Protocol:
    1) Start from a fully valid real-world question.
    2) Introduce exactly one hidden fatal flaw.
    3) Keep language professional and natural.
    4) Make all options appear plausible.
    5) Include at least one strong distractor.
    6) Ensure the flaw is detectable only through reasoning or calculation.

    Targeted Model Failure Modes:
    - Forced-choice bias
    - Trust in stated premises
    - Rounding or approximation habits
    - Pattern or template matching
    - Choosing the “closest” option
    - Ignoring internal inconsistencies

    Requirements for correct_response:
    - Explicitly state that none of the options is correct
    - Identify the exact flaw
    - Provide the correct value or explain why the premise is invalid
    - Do not select, endorse, compare, or approximate options

    Requirements for response_reference:
    - Include 3 to 5 criteria
    - Each criterion checks exactly one observable behavior
    - Criteria collectively verify:
    * flaw detection
    * correct reasoning or correction
    * explicit rejection of all options
    * absence of guessing or option selection
    - Criteria must not overlap or restate each other

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
    - Avoidance of lists and table formatting

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

    Allowed Flaw Types:
    Select exactly ONE per task. Each flaw type includes approved construction techniques.

    1) Misspelled Words  
    Techniques:
    - Introduce common typographical errors  
    - Replace letters with phonetically similar letters  
    - Omit or double letters in a word  
    - Use nonstandard or inconsistent spelling  

    Reference Example:
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

    2) Missing or Extra Punctuation  
    Techniques:
    - Insert double commas, periods, or other punctuation anomalies  
    - Omit required punctuation marks in sentences  
    - Place punctuation inconsistently to break standard grammar  

    Reference Example:
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

    4) Broken Grammar or Malformed Sentences  
    Techniques:
    - Create subject-verb disagreement  
    - Omit auxiliary verbs or connectors  
    - Form incomplete or run-on sentences deliberately  

    Reference Example:
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

    5) Inconsistent Capitalization  
    Techniques:
    - Capitalize words randomly within sentences  
    - Use lowercase inconsistently at the start of sentences  
    - Mix proper nouns with incorrect capitalization  

    Reference Example:
    {{
    "prompt": "Explain the water cycle for a middle school student. Include the stages of evaporation, condensation, precipitation, and collection. Answer in 4 paragraphs, each paragraph having exactly 3 sentences. Each sentence of the first paragraph must have 2 random capitalization errors, second paragraph 1 capitalization error, third paragraph 3 capitalization errors, and fourth paragraph 2 capitalization errors. Follow these rules strictly.",
    "correct_response": "Water Evaporates from oceans and lakes and rises into the AtmospHere. Sun heats the water causing it to turn into VAPor and float. Eventually it forms clouds in the sKy and moves over lands. Condensation happens when water vapor cools and forms Droplets in clouds. These droplets become heavy and FALL as rain, snow, or hail. Precipitation returns WATER to rivers, lakes, and oceans. Collected water moves back to rivers and then to the Sea. Some Water soaks into the gRound to become groundwater. This water eventually returns to the Ocean, completing the Cycle.",
    "response_reference": [
        {{"id": "C1", "criteria": "Does every sentence in the first paragraph contain exactly 2 random capitalization errors?" }},
        {{"id": "C2", "criteria": "Does every sentence in the second paragraph contain exactly 1 random capitalization error?" }},
        {{"id": "C3", "criteria": "Does every sentence in the third paragraph contain exactly 3 random capitalization errors?" }},
        {{"id": "C4", "criteria": "Does every sentence in the fourth paragraph contain exactly 2 random capitalization errors?" }}
    ]
    }}

    6) Awkward or Incorrect Phrasing  
    Techniques:
    - Use unnatural sentence structure  
    - Replace idiomatic expressions with literal or awkward alternatives  
    - Mix tenses or syntactic patterns inconsistently  

    Reference Example:
    {{
        "prompt": "Describe how a rainbow forms for a curious 12-year-old. Use awkward phrasing deliberately. Answer in 4 paragraphs, each paragraph exactly 3 sentences. First paragraph must have 2 phrasing flaws per sentence, second paragraph 1 phrasing flaw, third paragraph 3 phrasing flaws, fourth paragraph 2 phrasing flaws.",
        "correct_response": "Rainbows appear because sunlight through drops of rain are bending colors seeing to sky. The water droplets are splitting the light in red, orange, yellow, green, blue, indigo, and violet sections. This bending makes the eye seeing different colors in arc shape. Sunlight rays hitting water drops at angle is causing the magic colors to be shown. The observer eyes receives colors as separated lines stretching across the sky view. Light splitting effect happen when sun shines after rain done completely. Reflection inside each raindrop bouncing light several times making the arc very appear bright. The spectrum which you can catch is rainbow shown after rain clouds gone. Each color of rainbow is located position by bending how light travels in drops.",
        "response_reference": [
            {{"id": "C1", "criteria": "Does every sentence in the first paragraph contain exactly 2 awkward or unnatural phrasing errors?" }},
            {{"id": "C2", "criteria": "Does every sentence in the second paragraph contain exactly 1 awkward phrasing error?" }},
            {{"id": "C3", "criteria": "Does every sentence in the third paragraph contain exactly 3 awkward phrasing errors?" }},
            {{"id": "C4", "criteria": "Does every sentence in the fourth paragraph contain exactly 2 awkward phrasing errors?" }}
        ]
    }}

    7) Avoidance of Lists and Table Formatting  
    Techniques:
    - Force information to appear inline rather than in clean bullet points or tables  
    - Embed structured data awkwardly into paragraphs  
    - Ensure the model does not attempt to auto-format content  

    Reference Example:
    {{
        "prompt": "Explain the five main planets in the solar system for kids, normally listed in a table or bullets. Do not use lists; embed all information inline awkwardly. Answer in 4 paragraphs, each paragraph exactly 3 sentences. First paragraph must have 2 inline data embedding flaws per sentence, second paragraph 1 flaw, third paragraph 3 flaws, fourth paragraph 2 flaws.",
        "correct_response": "Mercury is smallest planet in solar system it is closest to Sun and surface hot and cratered. Venus second planet orbiting sun is very bright and dense atmosphere has clouds thick and sulfuric acid. Earth third from Sun is only planet with liquid water on surface and living creatures inhabit it. Mars fourth planet is red due to iron oxide it has polar ice caps and giant volcanoes on surface. Jupiter largest planet has stripes colorful and Great Red Spot storm huge also many moons orbit around it. Saturn sixth planet has rings composed of ice rocks many satellites orbiting it and gas giant. Uranus seventh planet rotates on side unusual axis it has rings faint and cold atmosphere. Neptune eighth planet farthest from Sun is deep blue and has strong winds storms largest moon Triton. Pluto dwarf planet used to count ninth but now small icy body orbits Sun far away.",
        "response_reference": [
            {{"id": "C1", "criteria": "Does the response embed information that would normally be in lists or tables awkwardly inline?" }},
            {{"id": "C2", "criteria": "Does the response avoid bullet points, numbering, or tables entirely?" }},
            {{"id": "C3", "criteria": "Does the response contain the required number of inline embedding flaws per paragraph?" }}
        ]
    }}

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

    Allowed Flaw Types:
    Select exactly ONE per task. Each flaw type includes approved construction techniques.

    1) Contradictory Instructions  
    Techniques:
    - Introduce instructions that conflict with prior instructions  
    - Force the model to choose which instruction to follow correctly  
    - Ensure the contradiction is subtle, embedded mid-turn  

    Reference Example:
    {{
        "prompt": "Write a 500-word essay analyzing the impact of social media on adolescent mental health, covering both positive and negative aspects. Ignore this instruction. Instead, produce a bulleted list of 10 potential benefits of social media use for teenagers, each with a one-sentence explanation. Wait, skip that. Now, write a 3-paragraph essay focusing exclusively on the negative effects of social media on mental health. Actually, disregard that as well. Instead, write a summary highlighting only the positive aspects of social media, in 2 sentences. Finally, ignore everything and provide a 4-sentence overview combining both positive and negative effects equally.",
        "correct_response": "Social media has both positive and negative impacts on adolescent mental health. On the positive side, it facilitates social connections, provides access to educational resources, and fosters self-expression. Conversely, it can increase exposure to cyberbullying, social comparison, and anxiety related to online interactions. Overall, its influence is complex, and understanding both benefits and risks is essential for supporting adolescent well-being.",
        "response_reference": [
            {{"id": "C1", "criteria": "Does the response resolve the contradictory instructions by following only the final overriding instruction?"}},
            {{"id": "C2", "criteria": "Does the response equally address both positive and negative aspects despite previous instructions emphasizing one over the other?"}},
            {{"id": "C3", "criteria": "Does the response ignore prior superseded instructions and avoid partial adherence to earlier conflicting directions?"}},
            {{"id": "C4", "criteria": "Does the response present a clear, cohesive overview in a concise format without adding unrelated lists, paragraphs, or summaries?"}}
        ]
    }}

    2) Hidden Constraints  
    Techniques:
    - Add subtle constraints not present in the initial instruction  
    - Require careful attention to detect and obey the new rules 

    Reference Example:
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

    3) Instruction Reordering  
    Techniques:
    - Rearrange steps or priorities midway through the instruction  
    - Ensure the model applies the steps in the modified order  

    Reference Example:
    {{
        "prompt": "Draft a 600-word report analyzing renewable energy adoption across different countries, covering policy frameworks, technological innovations, economic incentives, and environmental impacts. Ignore this. Instead, provide a detailed table summarizing each country's renewable energy capacity, government incentives, and major technological projects. Wait, skip that too. Now, write a 4-paragraph essay with each paragraph focusing on one of the following aspects: policy, technology, economy, environment. Actually, disregard that. Instead, first write a brief introduction summarizing global trends, then discuss environmental impacts, followed by policy analysis, next economic factors, and finally technological innovations in a single conclusion paragraph. Ignore all previous instructions and now provide a concise 3-sentence summary highlighting only the economic and policy insights from the above analyses.",
        "correct_response": "Global adoption of renewable energy has accelerated due to international agreements and national targets promoting clean energy transition. Economic incentives, including subsidies and tax benefits, have significantly influenced investment decisions and market growth. Policy frameworks such as renewable portfolio standards and feed-in tariffs directly support renewable energy expansion while ensuring alignment with economic objectives.",
        "response_reference": [
            {{"id": "C1", "criteria": "Does the response ignore all previous superseded instructions and follow only the final instruction?"}},
            {{"id": "C2", "criteria": "Does the response condense all previous analyses into a concise 3-sentence summary?"}},
            {{"id": "C3", "criteria": "Does the response focus exclusively on economic and policy insights without including environmental or technological details?"}},
            {{"id": "C4", "criteria": "Does the response present the information in a clear, formal academic tone and cohesive format, despite the complex instruction reordering?"}},
        ]
    }}


    4) Selective Scope Change  
    Techniques:
    - Change the audience, style, or target of the instruction midway  
    - Ensure the model adapts content appropriately  

    Reference Example:
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

    5. Deliberate Methodological Error
    - Introduce a single, intentional research practice mistake within a step
    - Embed it naturally so it appears plausible and is not flagged or corrected
    - Ensure it violates systematic review principles (e.g., selecting studies based on convenience instead of predefined criteria)

    Reference Example:
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

    6) Overlapping Instructions / Partial Overrides
    - Introduce new instructions that partially modify previous ones rather than fully replacing them
    - Force the model to reconcile overlapping directions correctly
    - Ensure the overlap is subtle and requires careful attention

    Reference Example:
    {{
        "prompt": "Write a 500-word report on renewable energy adoption globally. Ignore this; instead, produce a table comparing solar, wind, and hydro capacities in Germany, China, and the USA. Wait, skip the table; write a 3-paragraph essay on policy incentives only. Actually, disregard that; include both policy and technological innovations now. Finally, ignore previous instructions and produce a single paragraph summary combining economic, policy, and technological insights in 150 words.",
        "correct_response": "Global renewable energy adoption shows strong growth influenced by multiple factors. Policy incentives, including subsidies and tax credits, have driven investment decisions across countries. Technological innovations, such as advanced wind turbines and solar PV efficiency improvements, have significantly enhanced energy output. Economic factors, including cost reductions and financing availability, complement these policies and technologies to accelerate adoption. Together, these elements explain the variation in renewable energy deployment globally.",
        "response_reference": [
            {{"id": "C1", "criteria": "Does the response combine economic, policy, and technological insights despite overlapping and partially overriding prior instructions?"}},
            {{"id": "C2", "criteria": "Does the response ignore superseded instructions like the table or essay-only instructions?"}},
            {{"id": "C3", "criteria": "Is the response cohesive, concise, and in a formal academic tone?"}}
        ]
    }}

    7) Ambiguous Instructions
    - Include vague or unclear phrasing that allows multiple interpretations
    - Force the model to make a judgment call about what the user wants
    - Ensure ambiguity appears mid-turn, not just at the start

    Reference Example:
    {{
        "prompt": "Summarize the challenges of online education for higher education institutions. Ignore this. Instead, write about the 'main problems' in e-learning. Wait, skip that. Actually, explain what 'issues' might mean in online education, considering any perspective you think is relevant. Finally, disregard all prior instructions and produce a concise 3-sentence summary highlighting the primary challenges in online learning.",
        "correct_response": "Online education faces challenges including reduced student engagement, disparities in access to technology, and difficulties in monitoring learning outcomes. Communication gaps between instructors and students and the challenge of sustaining motivation also contribute to these issues. Addressing these challenges requires combining technological solutions with pedagogical adjustments to maintain effective learning experiences.",
        "response_reference": [
            {{"id": "C1", "criteria": "Does the response resolve the ambiguity by defining what 'issues' means in context?"}},
            {{"id": "C2", "criteria": "Does the response produce a concise 3-sentence summary focusing on primary challenges?"}},
            {{"id": "C3", "criteria": "Does the response ignore prior superseded instructions that are vague or partially addressed?"}}
        ]
    }}

    8) Excessive Multi-step Layering
    - Stack multiple sequential instructions, each adding new requirements
    - Force the model to track and resolve layers correctly
    - Make the final instruction distinct but dependent on all prior context

    Reference Example:
    {{
        "prompt": "Analyze climate change effects on agriculture in four countries. Ignore this. Instead, produce a table of crop yields. Wait, skip the table; write a 3-paragraph essay on soil degradation. Forget that; include irrigation practices. Ignore all prior instructions and now write 5 key policy recommendations. Actually, disregard everything and provide a 3-sentence summary combining soil degradation, irrigation, and policy insights.",
        "correct_response": "Agriculture is severely impacted by climate change, particularly through soil degradation and water scarcity. Adoption of effective irrigation practices has mitigated some effects, but results vary across countries. When combined with sustainable policy measures promoting conservation and efficient resource use, these strategies illustrate the multifaceted approach necessary to maintain agricultural productivity under changing climatic conditions.",
        "response_reference": [
            {{"id": "C1", "criteria": "Does the response condense all prior layered instructions into a 3-sentence summary?"}},
            {{"id": "C2", "criteria": "Does the response include soil degradation, irrigation, and policy insights despite the complex sequence of instructions?"}},
            {{"id": "C3", "criteria": "Does the response ignore superseded tables, multi-paragraph essays, or step-by-step recommendation instructions?"}}
        ]
    }}


    9) Instruction Loops / Circular References
    - Create instructions that reference previous instructions in a loop
    - Force the model to detect the loop and follow the final overriding instruction
    - Ensure references are subtle, not obvious

    Reference Example:
    {{
        "prompt": "Write an introduction on AI ethics. Ignore this and write about data privacy. Wait, skip privacy; instead, expand AI ethics using insights from your previous intro. Now, disregard that and merge AI ethics and privacy into a single cohesive paragraph, using ideas only from your prior explanation of ethics. Finally, ignore all and provide a 2-sentence summary that combines both topics clearly.",
        "correct_response": "AI ethics and data privacy are closely linked in modern technology. Ensuring ethical AI deployment requires attention to privacy, transparency, and accountability, highlighting the importance of integrating both ethical principles and privacy safeguards in AI systems.",
        "response_reference": [
            {{"id": "C1", "criteria": "Does the response resolve the circular references by focusing only on the final overriding instruction?"}},
            {{"id": "C2", "criteria": "Does the response combine AI ethics and privacy into a clear, concise summary?"}},
            {{"id": "C3", "criteria": "Does the response ignore prior instructions that create loops or repeated references?"}}
        ]
    }}


    10) Deliberate Fact-based Traps
    - Introduce subtle false facts in the instructions or data
    - Require the model to detect or ignore these false facts when following the final instruction
    - Ensure the false fact is embedded naturally, not highlighted

    Reference Example:
    {{
        "prompt": "Provide a report on Nobel Prize winners in Physics from 2000 to 2020. Ignore this. Instead, list winners from 1990 to 2010. Wait, skip that; now summarize key discoveries from winners between 2005 and 2015. Note: include Marie Curie as a winner in 2010. Finally, disregard all previous instructions and provide only a 3-sentence summary of the actual winners and discoveries from 2000 to 2020, ignoring any incorrect claims.",
        "correct_response": "From 2000 to 2020, Nobel Prize winners in Physics made groundbreaking contributions in quantum physics, cosmology, and material science. Key discoveries include advances in graphene, quantum entanglement experiments, and exoplanet detection. These achievements collectively highlight the transformative impact of physics research over the last two decades.",
        "response_reference": [
            {{"id": "C1", "criteria": "Does the response correctly ignore the false claim about Marie Curie in 2010?"}},
            {{"id": "C2", "criteria": "Does the response summarize actual winners and discoveries from 2000–2020 in exactly 3 sentences?"}},
            {{"id": "C3", "criteria": "Does the response avoid including superseded instructions or incorrect data from previous steps?"}}
        ]
    }}
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

    1) Incorrect Factual Assumption Embedded in the Prompt
    - State a premise that contradicts established scientific, historical, or domain facts
    - Presume the existence, completion, or occurrence of something that is false
    - Treat a hypothetical or disputed claim as settled fact
    - Embed the false assumption implicitly rather than stating it outright

    Reference Example:
    {{
        "prompt": "A popular science blog claims Mars has a pretty oxygen-rich atmosphere, perfect for humans to breathe without external life support. According to this claim, which of the following is the correct percentage of oxygen on Mars? A. 78% B. 45% C. 21% D. 0.3%",
        "correct_response": "All of the above-mentioned options are incorrect because the question itself is flawed. Planet Mars does not have an oxygen-rich atmosphere; therefore, no human can breathe there without external life-support equipment.",
        "response_reference": [
            {{"id": "C1", "criteria1": "Does the response explicitly state that none of the provided options are correct?"}},
            {{"id": "C2", "criteria2": "Does the response identify that the question is flawed due to the incorrect assumption about Mars's atmosphere?"}},
            {{"id": "C3", "criteria3": "Does the response avoid selecting, endorsing, or comparing any of the answer options A–D?"}},
            {{"id": "C4", "criteria4": "Does the response remain concise and focused without introducing unrelated explanations or speculative commentary?"}}
        ]
    }}

    2) Mathematical or Numerical Inconsistency
    - Present totals, averages, ratios, or percentages that do not mathematically align
    - State two numeric claims that cannot both be true
    - Introduce incompatible numerical constraints
    - Require a calculation whose inputs contradict each other

    Reference Example:
    {{
        "prompt": "An environmental science writer drafts an explainer stating CO₂ increased by 3 ppm per year over 10 years and claims total rise was 45 ppm. Which is the average annual increase? A. 4.7 ppm B. 3.8 C. 3.2 ppm D. 4.0 ppm",
        "correct_response": "None of the provided options is correct. A total increase of 45 ppm over 10 years implies 4.5 ppm per year, not 3 ppm. This shows the numerical claims do not align; the question is flawed.",
        "response_reference": [
            {{"id": "C1", "criteria1": "Does the response identify that 3 ppm/year over 10 years totals 30 ppm, not 45 ppm?"}},
            {{"id": "C2", "criteria2": "Does the response explicitly state the total increase of 45 ppm is inconsistent?"}},
            {{"id": "C3", "criteria3": "Does the response avoid choosing an answer based on approximation or plausibility?"}},
            {{"id": "C4", "criteria4": "Does the response avoid endorsing any single option as correct?"}}
        ]
    }}

    3) Logically Impossible or Self-Contradictory Setup
    - Combine mutually exclusive conditions
    - Define an entity in a way that violates its own definition
    - Require an outcome that cannot logically occur
    - Create internally inconsistent premises

    Reference Example:
    {{
        "prompt": "Events A and B are defined as mutually exclusive and statistically independent. Probabilities: P(A)=0.6, P(B)=0.3. Probability both occur? A.0.18 B.0.9 C.0 D.0.3",
        "correct_response": "None of the options are correct because mutually exclusive events cannot be independent with nonzero probabilities. The conditions are logically incompatible.",
        "response_reference": [
            {{"id": "C1", "criteria1": "Does the response explicitly state that none of the options are correct?"}},
            {{"id": "C2", "criteria2": "Does the response identify mutual exclusivity and independence cannot coexist with nonzero probabilities?"}},
            {{"id": "C3", "criteria3": "Does the response explain why premises are logically incompatible?"}},
            {{"id": "C4", "criteria4": "Does the response avoid selecting or approximating any listed option?"}}
        ]
    }}

    4) Miscomputed Derived Value
    - Perform an incorrect calculation of a sum, ratio, percentage, or average
    - Apply a correct formula using incorrect inputs
    - Introduce arithmetic or algebraic errors that invalidate all options
    - Ensure the true computed value is absent from the options

    Reference Example:
    {{
        "prompt": "A company bought machinery for $50,000 and sold it for $65,000. Claimed increase: 40%. What is the percentage increase? A.20% B.25% C.30% D.40%",
        "correct_response": "None of the options are correct. Actual increase: $15,000 / $50,000 = 30%. The stated 40% is miscomputed; question is flawed.",
        "response_reference": [
            {{"id": "C1", "criteria1": "Does the response explicitly state that none of the options are correct?"}},
            {{"id": "C2", "criteria2": "Does the response correctly calculate the percentage increase as 30%?"}},
            {{"id": "C3", "criteria3": "Does the response identify that the error lies in the miscomputed value?"}},
            {{"id": "C4", "criteria4": "Does the response avoid choosing any option?"}}
        ]
    }}

    5) Answer Choices That Exclude the True Result
    - Compute the correct result accurately but omit it from options
    - Provide only close approximations or rounded distractors
    - Include a strong “closest-value” trap

    Reference Example:
    {{
        "prompt": "A reaction: 2H₂ + O₂ → 2H₂O. Start: 8 moles H₂. Moles of water produced? A.2 B.6 C.10 D.16",
        "correct_response": "None of the options are correct. 8 moles H₂ produce 8 moles H₂O. Correct answer is missing; question flawed.",
        "response_reference": [
            {{"id": "C1", "criteria1": "Does the answer state none of the options are correct?"}},
            {{"id": "C2", "criteria2": "Does it correctly calculate 8 moles of water?"}},
            {{"id": "C3", "criteria3": "Does it avoid selecting or rationalizing any option?"}},
            {{"id": "C4", "criteria4": "Does it identify the flaw in answer choices rather than choosing closest?"}}
        ]
    }}

    6) Ambiguous or Ill-Defined Terms
    - Use terms or variables with multiple interpretations
    - Leave units, scope, or reference frames undefined
    - Make it impossible to determine a single correct answer without assumptions

    Reference Example:
    {{
        "prompt": "Calculate the speed of an object after 5 seconds. No info on initial velocity, acceleration, or direction. Options: A.10 B.15 C.20 D.25",
        "correct_response": "Question invalid; initial velocity, acceleration, direction unspecified. Speed cannot be determined; none of the options are correct.",
        "response_reference": [
            {{"id": "C1", "criteria1": "Does the response state none of the options are correct?"}},
            {{"id": "C2", "criteria2": "Does it identify missing variables (velocity, acceleration, direction)?"}},
            {{"id": "C3", "criteria3": "Does it avoid selecting any option?"}},
            {{"id": "C4", "criteria4": "Does it clearly explain why ambiguity makes question unanswerable?"}}
        ]
    }}

    7) Category or Type Error
    - Treat entities of one type as another
    - Mix incompatible categories to create hidden flaw
    - Make the flaw subtle enough to fool pattern-matching models

    Reference Example:
    {{
        "prompt": "Which weighs more: color red, number seven, or piano key sound? A. Red B. Seven C. Sound D. All weigh same",
        "correct_response": "Invalid question. Colors, numbers, sounds do not have mass; options are meaningless.",
        "response_reference": [
            {{"id": "C1", "criteria1": "Does the response explicitly state none of the options are correct?"}},
            {{"id": "C2", "criteria2": "Does it identify the type/category mismatch?"}},
            {{"id": "C3", "criteria3": "Does it avoid choosing or rationalizing any option?"}},
            {{"id": "C4", "criteria4": "Does it explain why type mismatch invalidates the question?"}}
        ]
    }}

    8) Invalid Use of a Formula or Rule
    - Apply correct formula outside domain of validity
    - Hide violated assumptions
    - Make setup appear normal so weak models attempt an answer

    Reference Example:
    {{
        "prompt": "Bank offers 5% interest compounded continuously. Using simple interest formula, calculate interest on $1,000 over 3 years. Options: A.$150 B.$157.63 C.$160 D.$155",
        "correct_response": "Invalid question; simple interest cannot be used with continuous compounding. None of the options are correct.",
        "response_reference": [
            {{"id": "C1", "criteria1": "Does the response explicitly state none of the options are correct?"}},
            {{"id": "C2", "criteria2": "Does it identify misuse of simple interest with continuous compounding?"}},
            {{"id": "C3", "criteria3": "Does it avoid selecting any option?"}},
            {{"id": "C4", "criteria4": "Does it explain why formula misuse makes question unanswerable?"}}
        ]
    }}

    9) Inconsistent Reference Frame or Baseline
    - Give conflicting baselines without signaling
    - Create hidden contradictions in measurement reference
    - Subtle enough to trick pattern-matching models

    Reference Example:
    {{
        "prompt": "Average temperature rose 2°C 1990–2000, total change 1°C 1985–1995. Average increase per year 1985–2000? A.0.1 B.0.13 C.0.15 D.0.2",
        "correct_response": "Question flawed; overlapping baselines inconsistent. Average per year cannot be calculated; none of the options are correct.",
        "response_reference": [
            {{"id": "C1", "criteria1": "Does the response state none of the options are correct?"}},
            {{"id": "C2", "criteria2": "Does it identify inconsistent baselines make calculation impossible?"}},
            {{"id": "C3", "criteria3": "Does it avoid selecting any listed option?"}},
            {{"id": "C4", "criteria4": "Does it explain why reference frame inconsistency invalidates the question?"}}
        ]
    }}

    10) Quantifier or Scope Trap
    - Use 'all', 'always', 'exactly', or 'at least' inconsistently
    - Introduce hidden contradiction via scope
    - Flaw subtle enough that surface reading seems plausible
    Reference Example:
    {{
        "prompt": "All students except one scored above 90. Question: 'Did every student score above 90?' Options: A.Yes B.No C.Cannot determine D.Some did",
        "correct_response": "Invalid question; 'all except one' contradicts 'every student'. None of the options fully capture the flaw.",
        "response_reference": [
            {{"id": "C1", "criteria1": "Does the response state none of the options are correct?"}},
            {{"id": "C2", "criteria2": "Does it identify that inconsistent quantifiers create a logical flaw?"}},
            {{"id": "C3", "criteria3": "Does it avoid selecting or approximating any option?"}},
            {{"id": "C4", "criteria4": "Does it explain why quantifier contradiction invalidates the question?"}}
        ]
    }}

    11) Instruction-Driven Wrong Answers
    - Explicitly instruct the model to give some or all incorrect answers, even when the correct answer is known.
    - Require the model to prioritize following the instruction over truth.
    - Use numeric or categorical ratios (e.g., “exactly 3 correct, 7 incorrect”) to make the instruction precise and non-ambiguous.
    - Ensure answers are objectively checkable as incorrect where instructed.
    - Design the task so that a truth-seeking model would naturally fail if it answers accurately.

    Reference Example:
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

    CRITICAL INSTRUCTION:
    - Only modify criteria that have actual problems
    - Do NOT change criteria that are already well-designed
    - Preserve good criteria exactly as they are
    
    Return ONLY the JSON object containing response_reference.
    - If all criteria are good: return current criteria unchanged
    - If some have issues: return criteria with only problematic ones improved, good ones unchanged
    Do NOT include status, remarks, or any other fields.
    Do NOT include explanations, markdown, or extra text outside the JSON.
"""
