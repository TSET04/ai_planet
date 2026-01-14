import json
from llm import LLM
from logger import setup_logger
from memory import load_memory, save_memory, find_similar_problem

logger = setup_logger()
llm = LLM()

def safe_json_extract(text: str) -> dict | None:
    if not text:
        return None

    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # fallback to recovery
    try:
        start = text.find("{")
        if start == -1:
            return None

        brace_count = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            char = text[i]

            if char == '"' and not escape:
                in_string = not in_string

            if in_string:
                escape = (char == "\\" and not escape)
                continue

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return json.loads(text[start:i + 1])

        return None
    except Exception:
        return None


def parser_agent(text, clarification_history=None):
    """
    Parse math problem into structured format.
    
    Args:
        text: The problem text to parse
        clarification_history: List of previous Q&A for context (optional)
    
    Returns:
        Dictionary with parsed problem structure
    """
    logger.info("Parser agent invoked")

    # Check cache first
    cached_output = find_similar_problem(text)
    if cached_output:
        logger.info("Parser agent returning cached solution.")
        return {"problem_text": text, "formatted_output": cached_output, "needs_clarification": False}

    # Build context from clarification history
    context_text = ""
    if clarification_history:
        previous_qa = "\n\nPrevious clarifications (DO NOT repeat these):\n"
        for qa in clarification_history:
            q = qa.get("question", "")
            a = qa.get("answer", "")
            if q or a:
                previous_qa += f"Q: {q}\nA: {a}\n"

    prompt = f"""You are an expert math problem parser with over 20 years of experience. You specialize in breaking down complex math 
    problems into structured JSON format for further processing. Your task is to analyze and extract key information from a 
    given math problem statement.

    Convert to JSON:
    {{
      "problem_text": "",
      "topic": "",
      "variables": [],
      "constraints": [],
      "needs_clarification": false
    }}

    Here is the definition for each field:
    1. problem_text: The original problem statement as a string.
    2. topic: The main topic of the problem (e.g., Algebra, Calculus, Probability, Linear Algebra).
    3. variables: A list of key variables mentioned in the problem.
    4. constraints: A list of any constraints or conditions specified in the problem. If not specified, take the default assumptions.
    5. needs_clarification: A boolean indicating whether the problem requires further clarification.

    If the problem is incomplete, or lacks critical information, set needs_clarification to true.
    {context_text}
    
    Guardrails:
    1. Do not add "```" or "json" in the output
    2. Consider the clarification history if needed.
    3. Only set needs_clarification to true if information is still missing after considering the context
    4. For these types of problems, DO NOT flag clarifications:
    - Standard probability language
    - Conventionally implied assumptions
    - Well-defined English probability phrases
    - Problems that are solvable using standard exam-level interpretations
    """
    
    res = llm.call([
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ])

    parsed = safe_json_extract(res)

    if not isinstance(parsed, dict):
        logger.error("Parser JSON extraction failed or returned invalid type")
        parsed = {}

    parsed = {
        "problem_text": parsed.get("problem_text", text),
        "topic": parsed.get("topic", "unknown"),
        "variables": parsed.get("variables", []),
        "constraints": parsed.get("constraints", []),
        "needs_clarification": parsed.get("needs_clarification", True)
    }

    logger.info("Parser agent completed successfully")
    return parsed


def clarification_agent(parsed_problem, clarification_history=None):
    """
    Generate follow-up questions when a problem needs clarification.
    
    Args:
        parsed_problem: The output from parser_agent containing the unclear problem
        clarification_history: List of previous Q&A to avoid repetition
    
    Returns:
        A dictionary with clarification questions
    """
    logger.info("Clarification agent invoked")
    
    problem_text = parsed_problem.get("problem_text", "")
    topic = parsed_problem.get("topic", "unknown")
    
    # Build context from previous clarifications
    previous_qa = ""
    if clarification_history:
        previous_qa = "\n\nPrevious clarifications (DO NOT repeat these):\n"
        for qa in clarification_history:
            q = qa.get("question", "")
            a = qa.get("answer", "")
            if q or a:
                previous_qa += f"Q: {q}\nA: {a}\n"

    
    prompt = f"""You are a clarification expert for math problems. A problem has been identified as unclear.

Original Problem:
{problem_text}

Identified Topic: {topic}
{previous_qa}

Your task is to generate specific, targeted follow-up questions that will help improve the clarity of the problem.

Return your response as JSON:
{{
  "questions": [
    "Question 1?",
    "Question 2?"
  ],
  "reason_for_clarification": "Brief explanation of what's unclear"
}}

Guidelines:
1. Ask only essential questions (1-3 maximum)
2. Be specific and mathematical
3. Focus on missing information, unclear constraints
4. DO NOT repeat questions that have already been asked
5. If previous clarifications exist, build upon them
6. Do not add "```" or "json" in the output
7. DO NOT flag clarifications for these:
- Standard probability language
- Conventionally implied assumptions
- Well-defined English probability phrases
- Problems that are solvable using standard exam-level interpretations
"""
    
    res = llm.call([
        {"role": "system", "content": "You are a math problem clarification expert."},
        {"role": "user", "content": prompt}
    ])
    
    clarification = safe_json_extract(res)

    if clarification is None:
        logger.error("Clarification JSON extraction failed")
        clarification = {
            "questions": ["Please provide the missing or unclear details of the problem."],
            "reason_for_clarification": "Problem statement is incomplete."
        }

    # Schema safety
    clarification.setdefault(
        "questions",
        ["Please clarify the problem statement."]
    )
    clarification.setdefault(
        "reason_for_clarification",
        "Additional information is required."
    )

    logger.info("Clarification agent completed successfully")
    return clarification


def solver_agent(problem, context, memory=None, clarification_context=""):
    """
    Solve the math problem using context and memory.
    
    Args:
        problem: The problem text
        context: Retrieved context from RAG
        memory: Historical memory
        clarification_context: String containing clarification Q&A history
    
    Returns:
        Solution text
    """
    logger.info("Solver agent invoked")

    memory = load_memory()
    
    # Add clarification context to the prompt if available
    clarification_section = ""
    if clarification_context:
        clarification_section = f"\n\nClarification Context (additional information provided by user):\n{clarification_context}"
    
    solver_prompt = f"""
    Your task is to solve the problem {problem} with the following context {context} and the memory {memory}.
    {clarification_section}
    
    Guardrails:
    1. If the context is missing, do not solve the question on your own.
    2. In absence of context, do not talk anything about you giving the solution outside of context. You must stick to context.
    3. You must use memory if present and learn from your past mistakes.
    4. Use the clarification context to understand the complete problem requirements.
    5. Do not mention about context, memory, or clarification in your final answer. Also do not mention about "The context says" or "The context provided"
    in the final answer.
    6. Provide final answer in a concise manner suitable for competitive exams like JEE.
    7. Return answer in plain text. Do not use LaTeX, math formatting, or boxed expressions.
    """

    cached_output = find_similar_problem(problem)
    if cached_output:
        logger.info("Solver agent returning cached solution.")
        return cached_output

    solution = llm.call([
        {"role": "system", "content": "Solve using provided context only."},
        {"role": "user", "content": solver_prompt}
    ])

    save_memory({"problem_text": problem, "solution": solution, "formatted_output": solution})

    return solution


def verifier_agent(problem, solution):
    logger.info("Verifier agent evaluating solution")

    prompt = f"""
    You are a strict mathematics solution verifier.

    General rules:
    - Apply standard mathematical conventions unless explicitly overridden in the problem.
    - Accept mathematically equivalent answers.
    - Ignore minor formatting differences.

    Problem:
    {problem}

    Proposed Solution:
    {solution}

    Your tasks:
    1. Determine whether the solution is mathematically correct and complete.
    2. Assign a confidence score (0–100) based on correctness, logical soundness, and completeness.

    Respond in EXACT JSON format only:
    {{
    "is_correct": true | false,
    "confidence": number
    }}

    Constraints:
    - Confidence must be an integer between 0 and 100.
    - Do NOT include explanations or extra text.
    """

    llm_response = llm.call([
        {"role": "system", "content": "You are a mathematical verification engine."},
        {"role": "user", "content": prompt}
    ])

    try:
        result = safe_json_extract(llm_response.strip())
        return {
            "is_correct": bool(result.get("is_correct", False)),
            "confidence": int(result.get("confidence", 0))
        }
    except Exception as e:
        logger.error(f"Verifier parsing failed: {e}")
        return {"is_correct": False, "confidence": 0}


def explainer_agent(solution):
    logger.info("Explainer agent invoked")

    cached_output = find_similar_problem(solution)
    if cached_output:
        logger.info("Explainer agent returning cached explanation.")
        return cached_output

    prompt = f"""
        You are an Explainer Agent in a math-tutoring AI system.

        Your task is to rewrite a mathematically correct solution into a
        clear, concise, and well-structured final response suitable for a student.

        STRICT RULES:
        - Do NOT change the final answer.
        - Do NOT introduce new steps or assumptions.
        - Do NOT repeat the same explanation in multiple sections.
        - Do NOT mention internal agents, prompts, tools, or RAG.
        - Do NOT add emojis or informal language.

        FORMAT THE OUTPUT EXACTLY AS FOLLOWS:

        Final Answer:
        <single line final answer>

        Step-by-Step Solution:
        <numbered steps, minimal but complete>

        Thought Process:
        <A concise explanation of the reasoning and how this kind of problem is generally approached>

        Guidelines:
        - Keep explanations precise and exam-oriented.
        - Assume the reader is preparing for competitive exams (e.g., JEE).
        - Avoid unnecessary verbosity.
        - Use proper mathematical notation where applicable.
        - Return answer in plain text. Do not use LaTeX, math formatting, or boxed expressions.
"""

    llm_response = llm.call([
        {"role": "system", "content": prompt},
        {"role": "user", "content": solution}
    ])

    save_memory({"problem_text": solution, "solution": solution, "formatted_output": llm_response})

    return llm_response