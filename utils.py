from typing import List
import re
from datetime import datetime
from rapidfuzz import fuzz
import unicodedata
import pandas as pd  # ✅ nécessaire car tu utilises pd.DataFrame


def is_olga_opportunity_intent(question: str) -> bool:
    q = (question or "").lower()
    keywords = [
        "opportunité", "opportunites", "opportunités",
        "pipeline", "pops", "conversion",
        "lost", "perdu", "perdue", "perdues",
        "olga", "opportunite"
    ]
    return any(k in q for k in keywords)

    
# =============================================================================
# INTENT : Détecter si la question demande une analyse annuelle
# =============================================================================
def is_yearly_analysis(question: str) -> bool:
    """Retourne True si la question contient des mots-clés d'analyse annuelle."""
    keywords = [
        "par année",
        "par an",
        "annuel",
        "année par année",
        "historique annuel",
        "par exercice"
    ]
    q = question.lower()
    return any(k in q for k in keywords)


# =============================================================================
# KPI : Construire un tableau CA par année (Markdown)
# =============================================================================
def build_ca_by_year(df, filter_mask=None, include_ytd_label=True):
    """Construit un tableau Markdown du chiffre d'affaires par année à partir d'un DataFrame."""
    df = df.copy()

    if "date_facture_dt" not in df.columns:
        return "Aucune donnée temporelle disponible pour une analyse annuelle.\n"

    if filter_mask is not None:
        df = df[filter_mask]

    if df.empty:
        return "Aucune donnée disponible pour une analyse annuelle.\n"

    df["year"] = df["date_facture_dt"].dt.year

    yearly = (
        df.groupby("year")["GFD_MONTANT_VENTE_EUROS"]
        .sum()
        .sort_index()
    )

    table = "### Chiffre d’affaires par année\n\n"
    table += "| Année | CA (€) |\n|-------|--------|\n"

    current_year = datetime.now().year

    for year, ca in yearly.items():
        label = str(year)
        if include_ytd_label and year == current_year:
            label += " (YTD)"
        table += f"| {label} | {ca:,.2f} € |\n"

    return table


# =============================================================================
# VALIDATION : Vérifier si l'analyse annuelle est possible (années complètes dispo)
# =============================================================================
def yearly_analysis_allowed(df, current_year):
    """True si le dataset contient au moins une année complète (strictement < current_year)."""
    years = df["date_facture_dt"].dt.year.dropna().unique()
    complete_years = [y for y in years if y < current_year]
    return len(complete_years) >= 1


# =============================================================================
# EXTRACTION : Extraire une année (ex: 2025) depuis la question
# =============================================================================
def extract_year_from_question(question: str) -> int | None:
    """Extrait la première année (20xx) détectée dans la question."""
    match = re.search(r"\b(20\d{2})\b", question or "")
    return int(match.group(1)) if match else None


# =============================================================================
# EXTRACTION : Extraire toutes les années mentionnées dans la question
# =============================================================================
def extract_years_from_question(question: str) -> list[int]:
    """Retourne toutes les années (20xx) trouvées dans la question, triées et uniques."""
    years = re.findall(r"\b(20\d{2})\b", question or "")
    return sorted(set(map(int, years)))


# =============================================================================
# KPI : Construire un tableau CA par mois (Markdown) avec évolutions
# =============================================================================
def build_ca_by_month(df: pd.DataFrame, include_ytd_label: bool = False) -> str:
    """Construit un tableau Markdown du CA mensuel + évolutions MoM."""
    if df.empty or "date_facture_dt" not in df.columns:
        return "Aucune donnée temporelle disponible pour une analyse mensuelle.\n"

    df = df.dropna(subset=["date_facture_dt"]).copy()
    if df.empty:
        return "Aucune donnée disponible pour une analyse mensuelle.\n"

    df["ANNEE_MOIS"] = df["date_facture_dt"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("ANNEE_MOIS", as_index=False)
          .agg(CA=("GFD_MONTANT_VENTE_EUROS", "sum"))
          .sort_values("ANNEE_MOIS")
    )

    if monthly.empty:
        return "Aucune donnée disponible pour une analyse mensuelle.\n"

    monthly["Evol (€)"] = monthly["CA"].diff()
    monthly["Evol (%)"] = monthly["CA"].pct_change() * 100

    table = "### Chiffre d’affaires par mois\n\n"
    table += "| Mois | CA (€) | Évolution (€) | Évolution (%) |\n"
    table += "|------|--------|--------------:|--------------:|\n"

    for _, row in monthly.iterrows():
        mois = pd.to_datetime(row["ANNEE_MOIS"]).strftime("%Y-%m")
        ca = row["CA"]
        evol_eur = row["Evol (€)"]
        evol_pct = row["Evol (%)"]

        evol_eur_str = "" if pd.isna(evol_eur) else f"{evol_eur:+,.2f} €"
        evol_pct_str = "" if pd.isna(evol_pct) else f"{evol_pct:+.1f} %"

        table += f"| {mois} | {ca:,.2f} € | {evol_eur_str} | {evol_pct_str} |\n"

    total = monthly["CA"].sum()
    table += f"\n**Total période : {total:,.2f} €**\n"
    return table


# =============================================================================
# RENDER : Afficher une table mensuelle (Markdown) à partir d'un df déjà mensuel
# =============================================================================
def render_monthly_ca_table(monthly: pd.DataFrame) -> str:
    """Rend un tableau Markdown mensuel. Attend 'ANNEE_MOIS' + 'CA' ou 'total_sales'."""
    if monthly is None or monthly.empty or "ANNEE_MOIS" not in monthly.columns:
        return "Aucune donnée disponible pour une analyse mensuelle.\n"

    monthly = monthly.copy().sort_values("ANNEE_MOIS")

    if "CA" not in monthly.columns:
        if "total_sales" in monthly.columns:
            monthly["CA"] = monthly["total_sales"]
        else:
            return "Aucune donnée disponible pour une analyse mensuelle.\n"

    monthly["Evol (€)"] = monthly["CA"].diff()
    monthly["Evol (%)"] = monthly["CA"].pct_change() * 100

    table = "### Chiffre d’affaires par mois\n\n"
    table += "| Mois | CA (€) | Évolution (€) | Évolution (%) |\n"
    table += "|------|--------|--------------:|--------------:|\n"

    for _, row in monthly.iterrows():
        mois = pd.to_datetime(row["ANNEE_MOIS"]).strftime("%Y-%m")
        ca = row["CA"]

        evol_eur = row["Evol (€)"]
        evol_pct = row["Evol (%)"]

        evol_eur_str = "" if pd.isna(evol_eur) else f"{evol_eur:+,.2f} €"
        evol_pct_str = "" if pd.isna(evol_pct) else f"{evol_pct:+.1f} %"

        table += f"| {mois} | {ca:,.2f} € | {evol_eur_str} | {evol_pct_str} |\n"

    table += f"\n**Total période : {monthly['CA'].sum():,.2f} €**\n"
    return table


# =============================================================================
# TRANSFORMATION : Construire un DataFrame de ventes mensuelles (option client)
# =============================================================================
def build_monthly_sales(df: pd.DataFrame, group_by_client: bool = False) -> pd.DataFrame:
    """Retourne un df agrégé au mois (ANNEE_MOIS) avec total_sales, optionnellement par client."""
    if df.empty or "DATE_FACTURE" not in df.columns:
        return pd.DataFrame()

    df = df.copy()

    df["date_facture_dt"] = pd.to_datetime(df["DATE_FACTURE"], errors="coerce")
    df = df.dropna(subset=["date_facture_dt"])

    df["ANNEE_MOIS"] = df["date_facture_dt"].dt.to_period("M")

    group_cols = ["ANNEE_MOIS"]
    if group_by_client and "RAISON_SOCIALE" in df.columns:
        group_cols.append("RAISON_SOCIALE")

    monthly = (
        df.groupby(group_cols, as_index=False)
          .agg(total_sales=("GFD_MONTANT_VENTE_EUROS", "sum"))
    )

    monthly["ANNEE_MOIS"] = monthly["ANNEE_MOIS"].dt.to_timestamp()
    monthly = monthly.sort_values("ANNEE_MOIS")
    return monthly


# =============================================================================
# DICTIONNAIRE : Mois français -> numéro de mois
# =============================================================================
MONTHS_FR = {
    "janvier": 1,
    "février": 2, "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "août": 8, "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "décembre": 12, "decembre": 12
}


# =============================================================================
# EXTRACTION : Extraire un mois depuis la question (ex: "mars" -> 3)
# =============================================================================
def extract_month_from_question(question: str) -> int | None:
    """Retourne le mois (1-12) si un nom de mois FR est détecté, sinon None."""
    q = (question or "").lower()
    for name, m in MONTHS_FR.items():
        if re.search(rf"\b{re.escape(name)}\b", q):
            return m
    return None


# =============================================================================
# INTENT : Détecter une demande sur un mois précis (ex: "en mars", "mois de mai")
# =============================================================================
def is_month_specific_request(question: str) -> bool:
    """True si un mois explicite est présent + formulation 'en/au mois/mois de'."""
    q = (question or "").lower()
    has_month = extract_month_from_question(q) is not None
    return has_month and any(k in q for k in ["au mois", "mois de", "en "])


# =============================================================================
# INTENT : Détecter une demande de CA (et éviter OLGA opportunités)
# =============================================================================
def is_revenue_intent(question: str) -> bool:
    """True si la question vise le CA facturé. False si OLGA/opportunités mentionné."""
    q = (question or "").lower()

    # si comparaison entre 2 clients avec un contexte temps -> on assume CA
    if is_comparison_intent(q) and (re.search(r"\b20\d{2}\b", q) or is_monthly_analysis(q) or is_yearly_analysis(q)):
        return True

    # OLGA = pas CA facturé
    if any(w in q for w in ["opportunité", "opportunites", "opportunités", "pipeline", "pops", "olga", "lost"]):
        return False

    if re.search(r"\bca\b", q):
        return True

    if any(k in q for k in [
        "chiffre d'affaires", "chiffre daffaires", "ca",
        "facture", "facturé", "facturee", "facturation",
        "revenu", "revenue",
        "ventes facturées", "vente facturée"
    ]):
        return True

    has_sales = re.search(r"\bventes?\b", q) is not None
    has_context = (
        any(k in q for k in ["par mois", "mensuel", "mensuelle", "mois", "évolution", "evolution", "historique", "tendance"])
        or re.search(r"\b20\d{2}\b", q) is not None
    )
    return bool(has_sales and has_context)


# =============================================================================
# INTENT : Détecter une demande d'analyse mensuelle
# =============================================================================
def is_monthly_analysis(question: str) -> bool:
    """True si la question exprime une analyse mensuelle / évolution mensuelle."""
    q = (question or "").lower()
    if any(k in q for k in [
        "par mois", "mois par mois",
        "mensuel", "mensuelle", "mensuellement",
        "évolution mensuelle", "evolution mensuelle",
        "historique mensuel",
        "évolution des ventes", "evolution des ventes",
        "vente par mois", "ventes par mois",
        "au mois", "mois de"
    ]):
        return True
    return bool(re.search(r"\bévolution\b", q) and re.search(r"\bmois\b", q))


# =============================================================================
# INTENT : Détecter une demande de comparaison (vs)
# =============================================================================
def is_comparison_intent(question: str) -> bool:
    """True si la question demande une comparaison (compare/vs/versus)."""
    q = (question or "").lower()
    return any(w in q for w in ["compare", "comparaison", "vs", "versus"])


# =============================================================================
# INTENT : Cas 'par mois' présent mais pas vraiment une analyse mensuelle (option)
# =============================================================================
def is_par_mois_only(question: str) -> bool:
    """True si 'par mois' est présent mais pas détecté comme analyse mensuelle."""
    q = (question or "").lower()
    return "par mois" in q and not is_monthly_analysis(q)


# =============================================================================
# POST-PROCESS : Canonicaliser les clients trouvés (ex: 'fekola' -> 'fekola sa')
# =============================================================================
def canonicalize_clients(found: list[str], client_list: list[str]) -> list[str]:
    """
    Convertit les mentions courtes en libellé exact présent en base.
    - exact match -> garder
    - sinon -> prendre le match le plus long qui startswith/contains
    - sinon -> garder tel quel
    """
    client_set = set(client_list)
    out = []

    for c in found:
        c = str(c).lower().strip()
        if c in client_set:
            out.append(c)
            continue

        candidates = [x for x in client_list if x.startswith(c) or (c in x)]
        if candidates:
            best = max(candidates, key=len)
            out.append(best)
        else:
            out.append(c)

    return list(dict.fromkeys(out))


# =============================================================================
# NORMALISATION : Utilitaires texte (accents / tokens)
# =============================================================================
LEGAL_SUFFIXES = {
    "sa", "s.a", "s.a.", "sarl", "s.a.r.l", "s.a.r.l.", "sas", "s.a.s", "s.a.s.",
    "ltd", "limited", "inc", "inc.", "llc", "gmbh", "bv", "spa", "s.p.a", "s.p.a.",
    "plc", "pte", "pte.", "co", "co.", "company"
}

STOPWORDS = {
    "donne", "moi", "le", "la", "les", "de", "des", "du", "d", "un", "une", "et", "ou",
    "avec", "pour", "sur", "dans", "au", "aux",
    "ca", "chiffre", "affaire", "affaires", "total", "montant",
    "compare", "comparaison", "comparer", "vs", "versus", "contre",
    "par", "mois", "mensuel", "mensuelle", "annee", "annuel", "en"
}

def _strip_accents(s: str) -> str:
    """Supprime les accents (NFKD)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )

def _normalize_text(s: str) -> str:
    """Normalise: lower + sans accents + supprime ponctuation + espaces."""
    s = _strip_accents(s.lower())
    s = re.sub(r"[’'`]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str):
    """Tokenise en mots (>=2 chars)."""
    return re.findall(r"\b[a-z0-9]{2,}\b", s)

def _remove_legal_suffixes(tokens):
    """Retire les suffixes légaux (sa/sarl/etc.) des tokens."""
    out = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return out if out else tokens

def _core_name(s: str) -> str:
    """Construit un 'core-name' client sans suffixes légaux."""
    t = _tokens(_normalize_text(s))
    t = _remove_legal_suffixes(t)
    return " ".join(t)


# =============================================================================
# NLP : Détecter les clients dans la question (anti-SA, anti-faux positifs)
# =============================================================================
def find_clients_in_question_v3(question: str, client_list: list, threshold: int = 80, max_clients: int = 3):
    """
    - Ignore suffixes légaux (SA/SARL/SAS...) -> évite faux positifs
    - Exige recouvrement tokens question/client -> évite bruit
    - Si comparaison -> limite à 2 clients
    """
    if not question or not client_list:
        return []

    q_norm = _normalize_text(question)
    q_tokens = [t for t in _tokens(q_norm) if t not in STOPWORDS and t not in LEGAL_SUFFIXES]
    q_token_set = set(q_tokens)

    if not q_tokens:
        return []

    wants_comparison = bool(re.search(r"\b(vs|versus|contre|compare|comparer|comparaison)\b|/|&", q_norm))

    segments = [q_norm]
    if wants_comparison:
        parts = re.split(r"\bvs\b|\bversus\b|\bcontre\b|/|&", q_norm)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            segments = parts

    norm_clients = []
    for c in client_list:
        c_core = _core_name(c)
        c_core_tokens = set(_tokens(c_core))
        norm_clients.append((c, c_core, c_core_tokens))

    scored = []
    for orig_client, core, core_tokens in norm_clients:
        if not core_tokens:
            continue

        if q_token_set.isdisjoint(core_tokens):
            continue

        best = 0
        for seg in segments:
            seg_tokens = [t for t in _tokens(seg) if t not in STOPWORDS and t not in LEGAL_SUFFIXES]
            seg_core = " ".join(seg_tokens)
            if not seg_core:
                continue
            best = max(best, fuzz.token_set_ratio(seg_core, core))

        if best >= threshold:
            bonus = min(4, 1.0 * max(0, len(core_tokens) - 1))
            scored.append((orig_client, best + bonus))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    limit = 2 if wants_comparison else max_clients

    found = []
    for client, _ in scored:
        if client not in found:
            found.append(client)
        if len(found) >= limit:
            break

    return found


# =============================================================================
# INTENT : Pays-first (éviter confusion pays vs clients contenant le pays)
# =============================================================================
def is_country_first_intent(q: str) -> bool:
    """True si la question ressemble à une demande sur un pays (pas un client)."""
    qn = _normalize_text(q)
    if re.search(r"\b(pays|destination|vers|au|aux|en)\b", qn):
        return True
    if re.search(r"\bca\b.*\bde\b", qn):
        return True
    return False


# =============================================================================
# POST-PROCESS : Supprimer les clients 'bruit' quand la question est pays-first
# =============================================================================
def suppress_clients_when_country_query(q: str, clients_mentionnes: List[str], pays_mentionnes: List[str]) -> List[str]:
    """Supprime les clients détectés uniquement car ils contiennent le nom du pays."""
    if not pays_mentionnes:
        return clients_mentionnes
    if not is_country_first_intent(q):
        return clients_mentionnes

    p_tokens = set(_tokens(_normalize_text(pays_mentionnes[0])))

    kept = []
    for c in clients_mentionnes:
        c_tokens = set(_tokens(_normalize_text(c)))
        if not p_tokens.isdisjoint(c_tokens):
            continue
        kept.append(c)

    return kept


# =============================================================================
# NLP : Détecter les pays dans la question (anti-faux positifs + gestion ambiguïtés)
# =============================================================================
COUNTRY_STOPWORDS = {
    "donne", "moi", "le", "la", "les", "de", "des", "du", "d", "un", "une", "et", "ou",
    "avec", "pour", "sur", "dans", "au", "aux", "en", "vers", "chez",
    "ca", "chiffre", "affaire", "affaires", "total", "montant", "compare", "comparaison", "comparer", "vs", "contre",
    "par", "mois", "mensuel", "mensuelle", "annee", "annuel", "annuelle", "jour", "semaine", "trimestre",
    "pays", "destination", "destinations"
}

GENERIC_COUNTRY_TOKENS = {
    "republique", "democratique", "etat", "etats", "unis", "union", "royaume",
    "arabie", "emirats", "saint", "ste"
}

def find_countries_in_question_v3(question: str, country_list: list, threshold: int = 85, max_countries: int = 3):
    """Détecte les pays mentionnés en évitant 'guinee' vs 'guinee bissau' etc."""
    if not question or not country_list:
        return []

    q = _normalize_text(question)
    q_tokens = [t for t in _tokens(q) if t not in COUNTRY_STOPWORDS]
    if not q_tokens:
        return []

    q_token_set = set([t for t in q_tokens if t not in GENERIC_COUNTRY_TOKENS])

    candidates = set()
    L = len(q_tokens)
    for n in range(1, 5):
        for i in range(0, L - n + 1):
            candidates.add(" ".join(q_tokens[i:i + n]))

    norm_countries = []
    for c in country_list:
        cn = _normalize_text(c)
        ctoks = set([t for t in _tokens(cn) if t not in GENERIC_COUNTRY_TOKENS])
        norm_countries.append((c, cn, ctoks))

    exact_matches = []
    for orig, cn, _ctoks in norm_countries:
        if cn in candidates:
            exact_matches.append(orig)
    if exact_matches:
        return exact_matches[:max_countries]

    scored = []
    for orig, cn, ctoks in norm_countries:
        if not cn or not ctoks:
            continue
        if q_token_set and q_token_set.isdisjoint(ctoks):
            continue

        best = 0
        for phrase in candidates:
            best = max(best, fuzz.token_set_ratio(phrase, cn))

        if best >= threshold:
            extra = ctoks - q_token_set
            penalty = 8 * len(extra)
            score_final = best - penalty
            if not extra:
                score_final += min(3, len(ctoks) - 1)
            scored.append((orig, score_final))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)

    found = []
    for c, _ in scored:
        if c not in found:
            found.append(c)
        if len(found) >= max_countries:
            break

    q_has_bissau = "bissau" in q_token_set
    if "guinee" in found and not q_has_bissau:
        found = [x for x in found if x != "guinee bissau"]

    return found


# =============================================================================
# POST-PROCESS : Résoudre les chevauchements pays (court vs long)
# =============================================================================
def resolve_country_overlaps(question: str, countries_found: list) -> list:
    """Résout 'guinee' vs 'guinee bissau' en gardant le plus pertinent selon la question."""
    if not countries_found or len(countries_found) < 2:
        return countries_found

    q = _normalize_text(question)
    q_tokens = set([t for t in _tokens(q) if t not in COUNTRY_STOPWORDS])

    def country_tokens(c):
        return set([t for t in _tokens(_normalize_text(c)) if t not in GENERIC_COUNTRY_TOKENS])

    found = list(dict.fromkeys(countries_found))
    keep = set(found)

    for i in range(len(found)):
        for j in range(len(found)):
            if i == j:
                continue

            short = found[i]
            long = found[j]

            t_short = country_tokens(short)
            t_long = country_tokens(long)

            if t_short and t_short.issubset(t_long) and len(t_long) > len(t_short):
                specific_long = t_long - t_short

                if specific_long and specific_long.issubset(q_tokens):
                    keep.discard(short)
                elif specific_long and not (specific_long & q_tokens):
                    keep.discard(long)

    return [c for c in found if c in keep]




def month_name_from_num(month_num: int) -> str:
    k = [k for k, v in MONTHS_FR.items() if v == month_num][0]
    return (
        k.replace("fevrier", "février")
         .replace("decembre", "décembre")
         .replace("aout", "août")
         .title()
    )

def disp_upper(x: str) -> str:
    return (x or "").upper()


