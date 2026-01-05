def is_evolution_question(question: str) -> bool:
    if not question:
        return False

    q = question.lower()

    keywords = [
        "evolution", "évolution",
        "tendance",
        "historique",
        "par mois",
        "mensuel",
        "mensuelle",
        "mensuelles",
        "courbe",
        "graphique",
        "evoluer",
        "progression"
        "compare"
        "mois par mois"
    ]

    return any(k in q for k in keywords)


def is_comparison_question(question: str) -> bool:
    if not question:
        return False

    q = question.lower()

    comparison_keywords = [
        " et ",
        " vs ",
        " versus ",
        " comparer",
        " comparaison",
        " entre ",
        "comparatif"
    ]

    return any(k in q for k in comparison_keywords)
