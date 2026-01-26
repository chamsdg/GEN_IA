import json
import re

def load_template(file_path: str) -> str:
    """Charge un template texte et le retourne comme chaîne"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def _clean_suggestions(text: str) -> str:
    """
    Garde uniquement les lignes "- ...", max 3, et bloque toute suggestion
    contenant des chiffres ou symboles monétaires.
    """
    if not text:
        return ""

    t = text.strip()

    # Blocage si présence de symboles monétaires ou pourcentages
    if any(x in t for x in ["€", "%", "$"]):
        return ""

    # Blocage si présence de chiffres
    if re.search(r"\d", t):
        return ""

    lines = []
    for line in t.splitlines():
        line = line.strip()
        if line.startswith("- "):
            lines.append(line)

    return "\n".join(lines[:3]).strip()

def make_suggestions(run_llm_func, question: str, answer: str, model: str, temperature: float):
    # Lecture directe du fichier à la racine
    base_prompt = load_template("advisor_suggestions.txt")

    payload = {
        "question": question,
        "answer": answer
    }

    # Construction du prompt final
    prompt = f"{base_prompt}\n\nCONTEXTE (JSON):\n{json.dumps(payload, ensure_ascii=False)}"

    # Appel de ton LLM (Cortex ou autre)
    sugg_text, _, _ = run_llm_func(
        question=prompt,
        model=model,
        temperature=temperature,
        progress=None
    )

    # Nettoyage selon tes règles strictes
    return _clean_suggestions(sugg_text)
