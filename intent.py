def is_evolution_question(question: str) -> bool:
    q = question.lower()

    # 1️⃣ Mots liés au temps / évolution
    evolution_keywords = [
        "evolution", "évolution",
        "tendance",
        "historique",
        "par mois",
        "mensuel",
        "mensuelle",
        "mensuelles",
        "courbe",
        "graphique",
        "évoluer",
        "progression",
        "comparaison",
        "compare",
        "mois par mois"
    ]

    # 2️⃣ Mots liés aux ventes / CA
    sales_keywords = [
        "vente", "ventes",
        "chiffre d affaire", "chiffre d'affaires",
        "ca", "c.a",
        "revenue", "revenus",
        "turnover"
    ]

    has_evolution = any(k in q for k in evolution_keywords)
    has_sales = any(k in q for k in sales_keywords)

    return has_evolution and has_sales

"""
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

"""

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
