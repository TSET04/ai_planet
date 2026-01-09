import json, re
from llm import LLM
from logger import setup_logger
from memory import load_memory, save_memory, find_similar_problem

logger = setup_logger()
llm = LLM()

def extract_json(text: str):
    """
    Extract the first valid JSON object from a string.
    """
    if not text:
        raise ValueError("Empty LLM response")

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")

    return match.group(0)


def parser_agent(text):
    logger.info("Parser agent invoked")

    cached_output = find_similar_problem(text)
    if cached_output:
        logger.info("Parser agent returning cached solution.")
        return {"problem_text": text, "formatted_output": cached_output}

    prompt = """You are an expert math problem parser with over 20 years of experience. You specialize in breaking down complex math 
    problems into structured JSON format for further processing. Your task is to analyze and extract key information from a 
    given math problem statement.

    Convert to JSON:
    {
      "problem_text": "",
      "topic": "",
      "variables": [],
      "constraints": [],
      "needs_clarification": false
    }
    If at all you are not able to parse the question or if the question is ambiguous, set needs_clarification to true.
    
    Guardrails:
    1. Do not add "```" or "json" in the output
    """
    res = llm.call([
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ])

    try:
        parsed = json.loads(res)
    except Exception as e:
        logger.error("Parser JSON extraction failed: %s", str(e))
        parsed = {
            "problem_text": text,
            "topic": "unknown",
            "variables": [],
            "constraints": [],
            "needs_clarification": True
        }

    logger.info("Parser agent completed successfully")
    return parsed


def solver_agent(problem, context, memory=None):
    logger.info("Solver agent invoked")

    memory = load_memory()
    solver_prompt = f"""
    Your task is to solve the problem {problem} with the following context {context} and the memory {memory}. 
    Guardrails:
    1. If the context is missing, do not solve the question on your own.
    2. In absence of context, do not talk anything about you giving the solution outside of context. You must stick to context.
    3. You must use memory if present and learn from your past mistakes.
    4. Do not mention about context or memory in your final answer. Also do not mention about "The context says" or "The context provided"
    in the final answer.
    5. Provide final answer in a concise manner suitable for competitive exams like JEE.
    6. Return answer in plain text. Do not use LaTeX, math formatting, or boxed expressions.
    """

    cached_output = find_similar_problem(problem)
    if cached_output:
        logger.info("Solver agent returning cached solution.")
        return cached_output

    solution = llm.call([
        {"role": "system", "content": "Solve using provided context only."},
        {"role": "user", "content": solver_prompt}
    ])

    # Save solution + embedding
    embedding = llm.embed(problem)
    save_memory({"problem_text": problem, "solution": solution, "formatted_output": solution})

    return solution


def verifier_agent(problem, solution):
    logger.info("Verifier agent evaluating solution using LLM")

    prompt = f"""
    You are a rigorous math verifier for JEE-level problems.

    Problem:
    {problem}

    Solution provided by another agent:
    {solution}

    Task:
    - Verify if the solution is correct.
    - If correct, answer only: TRUE
    - If incorrect or partially incorrect, answer only: FALSE
    - Do not provide explanations or extra text.
"""

    llm_response = llm.call([
        {"role": "system", "content": "You are a math verifier AI."},
        {"role": "user", "content": prompt}
    ])

    # Clean up response
    llm_response_clean = llm_response.strip().lower()
    if "true" in llm_response_clean:
        logger.info("Verifier LLM judged solution as CORRECT")
        return True
    else:
        logger.warning("Verifier LLM judged solution as INCORRECT")
        return False


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

    embedding = llm.embed(solution)
    save_memory({"problem_text": solution, "solution": solution, "formatted_output": llm_response})

    return llm_response


