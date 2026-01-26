# advisor.py
import json
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"

def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def _clean_suggestions(text: str) -> str:
    """
    Garde uniquement les lignes "- ...", max 3, et bloque toute suggestion
    contenant des chiffres ou symboles monétaires.
    """
    if not text:
        return ""

    t = text.strip()

    # hard block: chiffres / monnaie / %
    if any(x in t for x in ["€", "%"]):
        return ""

    # bloque aussi les nombres (simple)
    # (si tu veux autoriser '3 mois' enlève ce bloc)
    import re
    if re.search(r"\d", t):
        return ""

    lines = []
    for line in t.splitlines():
        line = line.strip()
        if line.startswith("- "):
            lines.append(line)

    return "\n".join(lines[:3]).strip()


def make_suggestions(run_llm_func, question: str, answer: str, model: str, temperature: float):
    """
    run_llm_func: ta fonction d'appel LLM (celle qui parle à Cortex ou autre).
    On l'injecte pour ne pas dupliquer ton code.
    """
    base_prompt = load_prompt("advisor_suggestions.txt")

    payload = {
        "question": question,
        "answer": answer
    }

    prompt = f"""{base_prompt}

CONTEXTE (JSON):
{json.dumps(payload, ensure_ascii=False)}
"""

    # Ici on réutilise ton run_llm (ou une fonction LLM bas niveau)
    sugg_text, _, _ = run_llm_func(
        question=prompt,
        model=model,
        temperature=temperature,
        progress=None  # pas de barre de progression pour les suggestions
    )

    sugg_text = _clean_suggestions(sugg_text)

    return sugg_text
