# llm_runner.py
import time
from fonction import ask_llm


def run_llm(question, model, temperature, progress=None):
    start = time.time()

    if progress:
        progress.progress(20)

    result = ask_llm(
        question=question,
        model=model,
        temperature=temperature
    )

    if progress:
        progress.progress(100)

    answer = result["response"] if isinstance(result, dict) else str(result)
    duration = time.time() - start

    return answer, duration
