# ====================================================================================================== #
#                                   SECTION DES LIBRARIES                                                #
#======================================================================================================= #
import streamlit as st
import pandas as pd
import re
import time
import unicodedata
import json
from datetime import datetime
from rapidfuzz import fuzz, process
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark import Session
import json
import re




# Initialize Snowpark session
@st.cache_resource
#def init_session():
 #   return get_active_session()




# ============================================================
# Initialize Snowpark session (Streamlit Cloud compatible)
# ============================================================
@st.cache_resource
def init_session():
    return Session.builder.configs({
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "role": st.secrets["snowflake"]["role"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
    }).create()



#=========================================================================================================================#
#                           CHARGEZ LES DONNEES DEPUIS SNOWFLAKE                                                          #
#=========================================================================================================================#
@st.cache_data(ttl=86400)
def load_data_from_snowflake():
    """
    Charge les données depuis Snowflake et les met en cache pendant 24h.
    Retourne un dictionnaire de DataFrames.
    """
    session = init_session()
    
    fact_query = """
        SELECT 
            *
        FROM NEEMBA.ML.FACTURE
    """
    fact = session.sql(fact_query).to_pandas()

    
    final_query = """
        SELECT 
            RAISON_SOCIALE,TOTAL_VENTES,TOTAL_OPPORTUNITE,
            POPS,LOST_OPPORTUNITE,ANNEE_MOIS
        FROM NEEMBA.ML.OPPORTUNITE_MOIS;
    """
    final = session.sql(final_query).to_pandas()


    opportunite_query = """
        SELECT 
            PAYS,TOTAL_VENTES,TOTAL_OPPORTUNITE,
            POPS,LOST_OPPORTUNITE,ANNEE_MOIS
        FROM NEEMBA.ML.OPPORTUNITE_MOIS_PAYS;
    """
    opportunite_pays = session.sql(opportunite_query).to_pandas()
    

    opportunite_bu_query = """
    SELECT *
    FROM NEEMBA.ML.OPPORTUNITE_BU;
    """
    opportunite_bu = session.sql(opportunite_bu_query).to_pandas()

    equipement_query = """
    SELECT *
    FROM NEEMBA.ML.EQUIPEMENT
    """
    equipement = session.sql(equipement_query).to_pandas()

    

    return {
        "fact": fact,
        "final": final,
        "equipement": equipement,
        "opportunite_pays" : opportunite_pays,
        "opportunite_bu":  opportunite_bu
        
    }







#st.write("Colonnes de la table FACT :")
#st.write(fact.columns.tolist())





# fonction pour changer le template
def load_template(file_path: str) -> str:
    """Charge un template texte et le retourne comme chaîne"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# appel de la fonction
template = load_template("template_long.txt")

# ingestion des données attendues dans le prompt
def format_prompt(question: str, context: str, context_olga: str,context_olga_countries: str,context_equipement_client: str,context_olga_bu: str) -> str:
    """Format le prompt avec les données ventes et opportunités"""
    return template.format(
        question=question,
        context=context,
        context_olga=context_olga,
        context_olga_countries=context_olga_countries,
        context_equipement_client=context_equipement_client,
        context_olga_bu=context_olga_bu
    )
  
# ================================================================================================== #
#                FONCTION POUR TROUVER LES CLIENTS DANS LA QUESTION                                  #
# ================================================================================================== #

def find_clients_in_question(question: str, client_list: list, threshold: int = 80):
    
    #Cherche tous les clients mentionnés dans la question (tolère les noms incomplets).
    
    keywords = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())

    if not keywords:
        return []

    
    candidates = []
    if len(keywords) >= 3:
        candidates.append(' '.join(keywords[-3:]))
    if len(keywords) >= 2:
        candidates.append(' '.join(keywords[-2:]))
    candidates.extend(keywords)  

   
    scored = []
    for phrase in candidates:
        for client in client_list:
            score = fuzz.token_set_ratio(phrase, client.lower())
            if score >= threshold:
                scored.append((client, score))

   
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    found_clients = []
    for client, score in scored:
        if client not in found_clients:
            found_clients.append(client)

    return found_clients


def strip_accents(text: str) -> str:
    """
    Supprime les accents d'une chaîne unicode.
    """
    text = unicodedata.normalize('NFD', text)
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')

# ======================================================================================================
#                           SECTION FONCTION POUR TROUVER LE PAYS                                     #
# =====================================================================================================
def find_countries_in_question(question: str, country_list: list, threshold: int = 75):
    """
    Recherche approximative des pays dans une question.
    Tolère les noms incomplets ou partiels (ex: 'Burkina' ~ 'Burkina Faso').
    """
   
    question_norm = strip_accents(question.lower())

   
    keywords = re.findall(r'\b[a-zA-Z]{3,}\b', question_norm)
    if not keywords:
        return []

   
    candidates = []
    if len(keywords) >= 3:
        candidates.append(' '.join(keywords[-3:]))
    if len(keywords) >= 2:
        candidates.append(' '.join(keywords[-2:]))
    candidates.extend(keywords)

   
    scored = []
    for phrase in candidates:
        for country in country_list:
            country_norm = strip_accents(country.lower())
            score = fuzz.token_set_ratio(phrase, country_norm)
            if score >= threshold:
                scored.append((country, score))

    
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    found_countries = []
    for country, score in scored:
        if country not in found_countries:
            found_countries.append(country)

    return found_countries
    

# ======================================================================================================
#                           SECTION FONCTION POUR CONSTRUCTEUR & SERVICES                                  #
# =====================================================================================================

constructeur_alias_map = {
    "cat": "caterpillar",
    "caterpillar": "caterpillar",
    "manitou": "manitou",
    "epiroc": "epiroc",
    "epiroc mali sarl": "epiroc",
    "epiroc burkina faso sarl": "epiroc",
    "exxon mobile": "exxon mobile",
}


def find_constructeurs_in_question(question, alias_map):
    q = question.lower()
    found = set()

    for alias, canonical in alias_map.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            found.add(canonical)

    return list(found)


def find_services_in_question(question, services_disponibles):
    q = question.lower()
    found = set()

    for service in services_disponibles:
        try:
            re.search(rf"\b{re.escape(service)}\b", q)
        except Exception as e:
            print("❌ REGEX SERVICE CASSÉE →", service)
            raise e

        if re.search(rf"\b{re.escape(service)}\b", q):
            found.add(service)

    return list(found)





# ======================================================================================================== #
#                                   SECTION MAPPING BU                                                     #
# ======================================================================================================== #
bu_alias_map = {
    "mi": "MI",
    "mining": "MI",
    "mines": "MI",
    "miniers": "MI",
    "tp": "TP",
    "travaux publics": "TP",
    "ci": "TP",
    "construction industrielle": "TP",
    "construction industries": "TP",
    "tp/ci": "TP",
    "mo": "MO",
    "mo/e&t": "MO",
    "e & t": "MO",
    "e&t": "MO",
    "energie & transport": "MO",
    "energie et transport": "MO",
}

# ======================================================================================================= #
#                               SECTION FONCTION POUR TROUVER LA DIVISION                                 #
# ======================================================================================================= #
def find_bu_in_question(question: str, bu_alias_map: dict) -> list:
    import re

    question_norm = re.sub(r"[\W_]+", " ", question.lower()).strip()
    bu_found = []

    for alias, canonical in bu_alias_map.items():
        alias_norm = re.sub(r"[\W_]+", " ", alias.lower()).strip()

        
        pattern = rf"\b{re.escape(alias_norm)}\b"
        if re.search(pattern, question_norm):
            bu_found.append(canonical)

        
        pattern_bu = rf"\b(?:bu|secteur|business)\s+{re.escape(alias_norm)}\b"
        if re.search(pattern_bu, question_norm):
            bu_found.append(canonical)

    
    bu_found = list(set(bu_found))
    

    return bu_found



# ======================================================================================================= #
#                                        SECTION COEUR DE L'IA                                            #
# ====================================================================================================== #
# ======================================================================================================= #
#                                        0.EQUIPEMENT & CLIENTS                                           #
# ====================================================================================================== #
def is_ranking_question(question: str) -> bool:
    if not question:
        return False

    q = question.lower()

    patterns = [
        "classement",
        "top",
        "le plus d",
        "le plus d'",
        "plus d equip",
        "plus d'équip",
        "qui a le plus",
        "client qui a le plus",
        "clients qui ont le plus",
        "plus gros parc"
    ]

    return any(p in q for p in patterns)


def is_dominant_family_question(question: str) -> bool:
    q = question.lower()
    triggers = [
        "famille dominante",
        "famille principale",
        "type d'équipement",
        "équipement dominant",
        "famille majoritaire",
        "quel type d'équipement"
    ]
    return any(t in q for t in triggers)
    

def get_dominant_family_for_client(
    equipement: pd.DataFrame,
    client: str
) -> str:
    df = equipement.copy()
    df["client_clean"] = df["RAISON_SOCIALE"].str.lower().str.strip()

    df_client = df[df["client_clean"] == client.lower().strip()]

    if df_client.empty:
        return f"Aucun équipement trouvé pour {client}."

    df_client = df_client.drop_duplicates(subset="EQCAT_SERIALNO")

    top_family = df_client["EQCAT_PRODUCT_FAMILY"].value_counts().idxmax()
    count = df_client["EQCAT_PRODUCT_FAMILY"].value_counts().max()

    return (
        f"La famille d'équipements dominante chez {client} est "
        f"{top_family} avec {count} équipements."
    )


def generate_client_equipment_ranking(
    equipement: pd.DataFrame,
    top_n: int = 10
) -> str:
    """
    Génère un classement des clients par nombre d'équipements uniques.
    
    - Un équipement = 1 EQCAT_SERIALNO unique
    - Classement décroissant
    """

    if equipement.empty:
        return "Aucune donnée d'équipement disponible."

    df = equipement.copy()

    # Normalisation client
    df["client_clean"] = (
        df["RAISON_SOCIALE"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Suppression des doublons équipements
    df_unique = df.drop_duplicates(subset="EQCAT_SERIALNO")

    # Comptage par client
    ranking = (
        df_unique
        .groupby("client_clean")
        .size()
        .reset_index(name="nb_equipements")
        .sort_values("nb_equipements", ascending=False)
        .head(top_n)
    )

    if ranking.empty:
        return "Aucun client trouvé dans les données."

    # Mapping vers nom original
    client_name_map = (
        df_unique[["client_clean", "RAISON_SOCIALE"]]
        .drop_duplicates()
        .set_index("client_clean")["RAISON_SOCIALE"]
        .to_dict()
    )

    summary_lines = []
    summary_lines.append(f"Classement des {top_n} clients par nombre d'équipements")

    for i, row in ranking.iterrows():
        client_name = client_name_map.get(row["client_clean"], row["client_clean"])
        summary_lines.append(
            f"{len(summary_lines)}. {client_name} : {row['nb_equipements']} équipements"
        )

    return "\n".join(summary_lines)


def generate_equipment_summary(equipement: pd.DataFrame, question: str = "") -> str:
    """
    Génère un résumé vertical des équipements pour un ou plusieurs clients
    à partir du DataFrame 'equipement' brut.
    
    - Nombre d'équipements uniques par client
    - Top 5 familles d'équipements par client
    """
     
    if equipement.empty:
        return "Aucune donnée d'équipement disponible."

    equipement = equipement.copy()

    equipement["client_clean"] = (
        equipement["RAISON_SOCIALE"].astype(str).str.lower().str.strip()
    )

    client_list = equipement["client_clean"].dropna().unique().tolist()
    clients_mentionnes = find_clients_in_question(question, client_list)
    clients_mentionnes = [c.lower().strip() for c in clients_mentionnes]

    
    # 🔴 NOUVEAU : famille dominante
    if clients_mentionnes and is_dominant_family_question(question):
        results = []
        for client in clients_mentionnes:
            client_name = (
                equipement[equipement["client_clean"] == client]
                .iloc[0]["RAISON_SOCIALE"]
            )
            results.append(
                get_dominant_family_for_client(
                    equipement,
                    client_name
                )
            )
        return "\n".join(results)


    summary_lines = []

    if not clients_mentionnes:
        summary_lines.append("Résumé global des équipements")

        df_unique = equipement.drop_duplicates(subset="EQCAT_SERIALNO")
        total_unique = df_unique.shape[0]
        summary_lines.append(f"Nombre total d'équipements uniques : {total_unique}")

        top_familles = (
            df_unique["EQCAT_PRODUCT_FAMILY"]
            .value_counts()
            .head(5)
        )

        summary_lines.append("Top 5 familles d'équipements :")
        for famille, count in top_familles.items():
            summary_lines.append(f"- {famille} : {count}")

    else:
        for client in clients_mentionnes:
            df_client = equipement[equipement["client_clean"] == client]

            if df_client.empty:
                summary_lines.append(f"{client.title()} : non trouvé dans les données")
                continue

            df_unique = df_client.drop_duplicates(subset="EQCAT_SERIALNO")
            total_unique = df_unique.shape[0]

            top_familles = (
                df_unique["EQCAT_PRODUCT_FAMILY"]
                .value_counts()
                .head(5)
            )

            client_name = df_unique.iloc[0]["RAISON_SOCIALE"]

            summary_lines.append(f"Résumé équipements pour {client_name}")
            summary_lines.append(f"Nombre d'équipements uniques : {total_unique}")
            summary_lines.append("Top 5 familles d'équipements :")

            for famille, count in top_familles.items():
                summary_lines.append(f"- {famille} : {count}")

    return "\n".join(summary_lines)



# ======================================================================================================= #
#                                        1.OLGA & OPPORTUNITES                                           #
# ====================================================================================================== #
# ------------------------------ FONCTION GENERIQUE -----------------------------------------------
def compute_olga_kpis(
    df: pd.DataFrame,
    dim_col: str,
    dim_values: list,
    month_col: str = "ANNEE_MOIS",
    sales_col: str = "TOTAL_VENTES",
    opp_col: str = "TOTAL_OPPORTUNITE",
    lost_col: str = "LOST_OPPORTUNITE",
    include_monthly: bool = False
) -> dict:
    """
    Moteur générique de calcul OLGA.
    Retourne les KPI du dernier mois + optionnellement l'évolution mensuelle.
    """

    if df.empty or not dim_values:
        return {}

    df = df.copy()

    # Sécurisation numérique
    for col in [sales_col, opp_col, lost_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Filtrage dimension
    df = df[df[dim_col].isin(dim_values)]
    if df.empty:
        return {}

    # Dernier mois disponible
    last_month = df[month_col].max()
    df_last = df[df[month_col] == last_month]

    total_sales = df_last[sales_col].sum()
    total_opp = df_last[opp_col].sum()
    total_lost = df_last[lost_col].sum()
    pops = round((total_sales / total_opp * 100), 2) if total_opp else 0

    result = {
        "last_month": last_month,
        "kpis": {
            "total_sales": total_sales,
            "total_opportunity": total_opp,
            "lost_opportunity": total_lost,
            "pops": pops
        }
    }

    # Évolution mensuelle optionnelle
    if include_monthly:
        monthly = (
            df.groupby(month_col)
            .agg(
                total_sales=(sales_col, "sum"),
                total_opportunity=(opp_col, "sum"),
                lost_opportunity=(lost_col, "sum")
            )
            .reset_index()
            .sort_values(month_col)
        )

        monthly["pops"] = monthly.apply(
            lambda r: (r["total_sales"] / r["total_opportunity"] * 100)
            if r["total_opportunity"] else 0,
            axis=1
        )

        result["monthly"] = monthly.to_dict(orient="records")

    return result

def is_olga_opportunity_intent(question: str) -> bool:
    q = (question or "").lower()
    keywords = [
        "opportunité", "opportunites", "opportunités",
        "pipeline", "pops", "conversion",
        "lost", "perdu", "perdue", "perdues",
        "olga"
    ]
    return any(k in q for k in keywords)

"""
def is_revenue_intent(question: str) -> bool:
    q = (question or "").lower()
    keywords = [
        "ca", "chiffre d'affaires", "chiffre daffaires","ventes","vente",
        "évolution des ventes", "évolution des vente", "ventes", "vente"
        "facture", "facturé", "facturee", "facturation",
        "revenu", "revenue",
        "vente facturée", "ventes facturées"
    ]
    return any(k in q for k in keywords)
"""



# ======================================================================================================= #
#                                        1.1 OLGA & CLIENTS                                              #
# ====================================================================================================== #
def generate_olga(final: pd.DataFrame, question: str = "") -> str:
    """
    Génère un résumé vertical des opportunités OLGA pour un ou plusieurs clients.
    Basé sur le dernier mois disponible.
    Évolution mensuelle affichée uniquement si explicitement demandée.

    IMPORTANT:
    - Si la question porte sur le CA / facturation => OLGA NE DOIT PAS répondre.
    - OLGA répond uniquement si l’intention "opportunités" est détectée.
    """

    # 1) Priorité CA : on coupe OLGA
    if is_revenue_intent(question):
        return ""

    # 2) Si la question ne parle pas d'opportunités -> OLGA ne répond pas
    if not is_olga_opportunity_intent(question):
        return ""

    if final is None or final.empty:
        return "Aucune donnée d'opportunité disponible."

    df = final.copy()

    # Normalisation
    if "RAISON_SOCIALE" not in df.columns:
        return "Aucune donnée d'opportunité disponible (colonne RAISON_SOCIALE manquante)."

    df["client_clean"] = df["RAISON_SOCIALE"].astype(str).str.lower().str.strip()

    client_list = df["client_clean"].dropna().unique().tolist()
    clients_mentionnes = find_clients_in_question(question, client_list)
    clients_mentionnes = [c.lower().strip() for c in clients_mentionnes]

    # 3) Mensuel OLGA UNIQUEMENT si intention opportunité + mots temporels
    q_lower = (question or "").lower()
    show_monthly = (
        is_olga_opportunity_intent(question)
        and any(word in q_lower for word in [
            "mois", "mensuel", "mensuelle",
            "tendance", "dans le temps",
            "par mois", "timeline",
            "progression", "historique"
        ])
    )

    summary_lines = []

    # =========================
    # CAS GLOBAL (aucun client)
    # =========================
    if not clients_mentionnes:
        result = compute_olga_kpis(
            df=df,
            dim_col="client_clean",
            dim_values=df["client_clean"].unique().tolist(),
            include_monthly=False  # global : dernier mois uniquement
        )

        if not result:
            return "Aucune donnée d'opportunité disponible."

        k = result["kpis"]

        summary_lines.append("KPI Opportunités Global (dernier mois)")
        summary_lines.append(f"POPS: {k['pops']:.2f} %")
        summary_lines.append(f"Total Opportunité: {k['total_opportunity']:,.0f} $")
        summary_lines.append(f"Ventes (opportunités OLGA): {k['total_sales']:,.0f} $")
        summary_lines.append(f"Lost Opportunité: {k['lost_opportunity']:,.0f} $")

        return "\n".join(summary_lines)

    # =========================
    # CAS CLIENT(S)
    # =========================
    last_result = None

    for client in clients_mentionnes:
        result = compute_olga_kpis(
            df=df,
            dim_col="client_clean",
            dim_values=[client],
            include_monthly=show_monthly
        )

        if not result:
            summary_lines.append(f"- {client.title()} : non trouvé dans les données")
            continue

        last_result = result  # on garde le dernier pour l'analyse portefeuille

        k = result["kpis"]
        last_month = result.get("last_month")

        summary_lines.append(f"KPI Opportunités pour {client.title()} (dernier mois)")
        summary_lines.append(f"POPS: {k['pops']:.2f} %")
        summary_lines.append(f"Total Opportunité: {k['total_opportunity']:,.0f} $")
        summary_lines.append(f"Ventes (opportunités OLGA): {k['total_sales']:,.0f} $")
        summary_lines.append(f"Lost Opportunité: {k['lost_opportunity']:,.0f} $")

        # Évolution mensuelle (si demandée)
        if show_monthly and "monthly" in result and result["monthly"]:
            summary_lines.append("\nÉvolution mensuelle :")
            for row in result["monthly"]:
                summary_lines.append(
                    f"{row['ANNEE_MOIS']} - "
                    f"POPS: {row['pops']:.2f} %, "
                    f"Total Opportunité: {row['total_opportunity']:,.0f} $, "
                    f"Ventes (opportunités OLGA): {row['total_sales']:,.0f} $, "
                    f"Lost: {row['lost_opportunity']:,.0f} $"
                )

    # Si aucun client n'a retourné de résultat valide, on sort
    if last_result is None:
        return "\n".join(summary_lines) if summary_lines else "Aucune donnée d'opportunité disponible."

    # =========================
    # ANALYSE PORTEFEUILLE CLIENTS (dernier mois)
    # =========================
    if "ANNEE_MOIS" in df.columns and last_result.get("last_month") is not None:
        df_last = df[df["ANNEE_MOIS"] == last_result["last_month"]].copy()

        if not df_last.empty:
            client_stats = (
                df_last
                .groupby("client_clean")
                .agg(
                    total_opportunity=("TOTAL_OPPORTUNITE", "sum"),
                    total_sales=("TOTAL_VENTES", "sum"),
                    lost_opportunity=("LOST_OPPORTUNITE", "sum"),
                    client_name=("RAISON_SOCIALE", "first")
                )
                .reset_index()
            )

            client_stats["pops"] = (
                client_stats["total_sales"] /
                client_stats["total_opportunity"].replace(0, pd.NA)
            ) * 100

            summary_lines.append("")
            summary_lines.append("Analyse du portefeuille d’opportunités")

            # Client avec le plus gros pipeline
            top_pipeline = client_stats.sort_values("total_opportunity", ascending=False).iloc[0]
            summary_lines.append(
                f"Client avec le plus gros pipeline : "
                f"{top_pipeline['client_name']} "
                f"({top_pipeline['total_opportunity']:,.0f} $)"
            )

            # Meilleure conversion
            best_conversion = client_stats.dropna(subset=["pops"]).sort_values("pops", ascending=False).iloc[0]
            summary_lines.append(
                f"Meilleure conversion d’opportunités : "
                f"{best_conversion['client_name']} "
                f"(POPS {best_conversion['pops']:.1f} %)"
            )

            # Client à risque (lost élevé)
            top_lost = client_stats.sort_values("lost_opportunity", ascending=False).iloc[0]
            summary_lines.append(
                f"Client avec le plus d’opportunités perdues : "
                f"{top_lost['client_name']} "
                f"({top_lost['lost_opportunity']:,.0f} $)"
            )

    return "\n".join(summary_lines)


     

# ======================================================================================================= #
#                                        1.2 OLGA & PAYS                                            #
# ====================================================================================================== #

def generate_olga_by_country(opportunite_pays: pd.DataFrame, question: str = "") -> str:
    """
    Génère un résumé OLGA des opportunités par pays.
    Basé sur le dernier mois disponible.
    Évolution mensuelle affichée uniquement si explicitement demandée.
    """

    if opportunite_pays.empty:
        return "Aucune donnée d'opportunité disponible."

    df = opportunite_pays.copy()

    # Normalisation
    df["pays_clean"] = df["PAYS"].astype(str).str.lower().str.strip()

    country_list = df["pays_clean"].dropna().unique().tolist()
    countries_mentionnes = find_countries_in_question(question, country_list)
    countries_mentionnes = [c.lower().strip() for c in countries_mentionnes]

    show_monthly = any(
        word in question.lower()
        for word in ["mois", "mensuel", "mensuelle","évolution","evolution", "tendance",
                     "dans le temps", "par mois", "timeline", "progression", "historique"]
    )

    summary_lines = []

    # =========================
    # CAS GLOBAL (aucun pays)
    # =========================
    if not countries_mentionnes:
        last_month = df["ANNEE_MOIS"].max()
        df_last = df[df["ANNEE_MOIS"] == last_month]

        summary_lines.append("KPI Opportunités par pays (dernier mois)")

        for country in sorted(df_last["pays_clean"].unique()):
            result = compute_olga_kpis(
                df=df,
                dim_col="pays_clean",
                dim_values=[country],
                include_monthly=False
            )

            if not result:
                continue

            k = result["kpis"]

            summary_lines.append(f"\nKPI Opportunites pour {country.title()}")
            summary_lines.append(f"POPS: {k['pops']:.2f} %")
            summary_lines.append(f"Total Opportunite: {k['total_opportunity']:,.0f} $")
            summary_lines.append(f"Total Ventes: {k['total_sales']:,.0f} $")
            summary_lines.append(f"Lost Opportunite: {k['lost_opportunity']:,.0f} $")

        return "\n".join(summary_lines)

    # =========================
    # CAS PAYS MENTIONNÉS
    # =========================
    for country in countries_mentionnes:
        result = compute_olga_kpis(
            df=df,
            dim_col="pays_clean",
            dim_values=[country],
            include_monthly=show_monthly
        )

        if not result:
            summary_lines.append(f"- {country.title()} : non trouvé dans les données")
            continue

        k = result["kpis"]

        summary_lines.append(
            f"KPI Opportunites pour {country.title()} (dernier mois)"
        )
        summary_lines.append(f"POPS: {k['pops']:.2f} %")
        summary_lines.append(f"Total Opportunite: {k['total_opportunity']:,.0f} $")
        summary_lines.append(f"Total Ventes: {k['total_sales']:,.0f} $")
        summary_lines.append(f"Lost Opportunite: {k['lost_opportunity']:,.0f} $")

        # Évolution mensuelle (si demandée)
        if show_monthly and "monthly" in result:
            summary_lines.append("\nÉvolution mensuelle :")
            for row in result["monthly"]:
                summary_lines.append(
                    f"{row['ANNEE_MOIS']} - "
                    f"POPS: {row['pops']:.2f} %, "
                    f"Total Opportunite: {row['total_opportunity']:,.0f} $, "
                    f"Ventes: {row['total_sales']:,.0f} $, "
                    f"Lost: {row['lost_opportunity']:,.0f} $"
                )

    return "\n".join(summary_lines)



# ======================================================================================================= #
#                                        1.2 OLGA & BU                                                    #
# ====================================================================================================== #

def generate_olga_bu(
    opportunite_bu: pd.DataFrame,
    question: str = "",
    bu_alias_map: dict = None
) -> str:
    """
    Génère un résumé OLGA par BU :
    - soit pour UNE BU explicite
    - soit pour TOUTES les BU si la question contient "par BU"
    Basé sur le dernier mois disponible.
    """

    if opportunite_bu.empty:
        return "Aucune donnée d'opportunité disponible."

    df = opportunite_bu.copy()

    # Normalisation
    df["BU_clean"] = df["DBU_CODE"].astype(str).str.lower().str.strip()
    df["pays_clean"] = df["PAY_LIBELLE"].astype(str).str.lower().str.strip()

    question_lower = question.lower()

    # Détection BU
    bu_mentionnes = find_bu_in_question(question, bu_alias_map or {})
    bu_mentionnes = [
        b.lower().strip()
        for b in bu_mentionnes
        if b.lower().strip() in df["BU_clean"].unique()
    ]

    summary_lines = []

    # =========================================================
    # CAS 1 — MODE "PAR BU" (aucune BU explicite)
    # =========================================================
    if not bu_mentionnes and any(k in question_lower for k in ["bu", "division"]):
        summary_lines.append("KPI Opportunités par BU (dernier mois)")

        for bu in sorted(df["BU_clean"].unique()):
            result = compute_olga_kpis(
                df=df,
                dim_col="BU_clean",
                dim_values=[bu],
                month_col="ANNEE_MOIS",
                sales_col="TOTAL_SALES",
                opp_col="TOTAL_OPPORTUNITY",
                lost_col="LOST_OPPORTUNITY",
                include_monthly=False
            )

            if not result:
                continue

            k = result["kpis"]

            summary_lines.append(f"\nBU {bu.upper()}")
            summary_lines.append(f"Total Opportunity : {k['total_opportunity']:,.0f} $")
            summary_lines.append(f"Total Sales       : {k['total_sales']:,.0f} $")
            summary_lines.append(f"Lost Opportunity  : {k['lost_opportunity']:,.0f} $")
            summary_lines.append(f"POPS (%)          : {k['pops']:.2f}")

        return "\n".join(summary_lines)

    # =========================================================
    # CAS 2 — BU EXPLICITE (comportement existant)
    # =========================================================
    if not bu_mentionnes:
        return "Aucune BU détectée dans la question."

    bu = bu_mentionnes[0]
    df_bu = df[df["BU_clean"] == bu]

    if df_bu.empty:
        return f"BU '{bu.upper()}' non trouvée dans les données."

    # Détection pays (optionnel)
    country_list = df_bu["pays_clean"].dropna().unique().tolist()
    countries_mentionnes = find_countries_in_question(question, country_list)
    countries_mentionnes = [c.lower().strip() for c in countries_mentionnes]

    if countries_mentionnes:
        df_bu = df_bu[df_bu["pays_clean"].isin(countries_mentionnes)]
        if df_bu.empty:
            return f"Aucune donnée pour la BU '{bu.upper()}' dans le(s) pays spécifié(s)."

    # Calcul OLGA
    result = compute_olga_kpis(
        df=df_bu,
        dim_col="BU_clean",
        dim_values=[bu],
        month_col="ANNEE_MOIS",
        sales_col="TOTAL_SALES",
        opp_col="TOTAL_OPPORTUNITY",
        lost_col="LOST_OPPORTUNITY",
        include_monthly=False
    )

    if not result:
        return "Aucune donnée d'opportunité disponible."

    k = result["kpis"]
    last_month = result["last_month"]

    summary_lines = [
        f"===== BU {bu.upper()} - MOIS {last_month} =====",
        f"Total Sales       : {k['total_sales']:,.0f}",
        f"Total Opportunity : {k['total_opportunity']:,.0f}",
        f"Lost Opportunity  : {k['lost_opportunity']:,.0f}",
        f"POPS (%)          : {k['pops']:.2f}"
    ]

    return "\n".join(summary_lines)


# ======================================================================================================= #
#                                       2.KPI METIERS                                                    #
# ====================================================================================================== #


# ======================================================================================================= #
#                                       2.1 SECTION  SERVICE                                                  #
# ====================================================================================================== #

# Détection période “X derniers mois”
def extract_period_bounds(question, today, start_of_year):
    match = re.search(r"(\d{1,2})\s*(dernier|derniers)\s*mois", question.lower())
    if match:
        n_months = int(match.group(1))
        first_day = (
            today.replace(day=1)
            - pd.DateOffset(months=n_months - 1)
        ).replace(day=1)
        return first_day, today
    return start_of_year, today

# KPI par dimension (service, constructeur, etc.)
def build_dimension_kpi(
    df_base,
    df_period,
    dim_col,
    value,
    start_of_last_year,
    same_day_last_year
):
    g = df_period[df_period[dim_col] == value]

    ca_ytd = g["GFD_MONTANT_VENTE_EUROS"].sum()

    ca_lytd = df_base.loc[
        (df_base[dim_col] == value) &
        (df_base["date_facture_dt"] >= start_of_last_year) &
        (df_base["date_facture_dt"] <= same_day_last_year),
        "GFD_MONTANT_VENTE_EUROS"
    ].sum()

    evo = ((ca_ytd - ca_lytd) / ca_lytd * 100) if ca_lytd else 0

    return ca_ytd, ca_lytd, evo, g

# Table d’évolution mensuelle
def build_monthly_table(df):
    monthly = (
        df.groupby(df["date_facture_dt"].dt.to_period("M"))
        ["GFD_MONTANT_VENTE_EUROS"]
        .sum()
    )

    if monthly.empty:
        return ""

    table = "\n### Évolution mensuelle\n"
    table += "| Mois | CA (€) |\n|------|--------|\n"

    for m, v in monthly.items():
        table += f"| {m.to_timestamp().strftime('%b %Y')} | {v:,.2f} € |\n"

    return table

# Répartition globale YTD
def build_repartition_table(df, dim_col, title):
    repartition = (
        df.groupby(dim_col)["GFD_MONTANT_VENTE_EUROS"]
        .sum()
        .sort_values(ascending=False)
    )

    total = repartition.sum()

    table = f"### {title} (YTD)\n\n"
    table += f"| {dim_col.replace('_clean','').capitalize()} | CA (€) | Part (%) |\n"
    table += "|---------|--------|----------|\n"

    for v, ca in repartition.items():
        part = (ca / total * 100) if total else 0
        table += f"| {v.capitalize()} | {ca:,.2f} € | {part:.2f} % |\n"

    return table

# Matrice Dimension × BU
def build_dimension_bu_matrix(df, dim_col, title):
    pivot = (
        df.groupby([dim_col, "bu_clean"])["GFD_MONTANT_VENTE_EUROS"]
        .sum()
        .unstack(fill_value=0)
    )

    pivot["TOTAL"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("TOTAL", ascending=False)

    table = f"### {title} (YTD)\n\n"
    table += f"| {dim_col.replace('_clean','').capitalize()} | " + " | ".join(pivot.columns) + " |\n"
    table += "|---------|" + "|".join(["--------"] * len(pivot.columns)) + "|\n"

    for v, row in pivot.iterrows():
        table += (
            f"| {v.capitalize()} | "
            + " | ".join(f"{x:,.2f} €" for x in row)
            + " |\n"
        )

    return table

# ------------------------------------------ SECTION SERVICE FINAL ------------------------------------------
def analyze_service_section(
    fact,
    question,
    services_mentionnes,
    pays_mentionnes,
    bu_mentionnees,
    today,
    start_of_year,
    start_of_last_year,
    same_day_last_year
):
    service_summary = ""

    df_base = fact.copy()

    if pays_mentionnes:
        df_base = df_base[df_base["pays_clean"].isin([p.lower() for p in pays_mentionnes])]

    if bu_mentionnees:
        df_base = df_base[df_base["bu_clean"].isin([b.upper() for b in bu_mentionnees])]

    df_ytd = df_base[
        (df_base["date_facture_dt"] >= start_of_year) &
        (df_base["date_facture_dt"] <= today)
    ]

    # ============================================================
    # CAS 1 — SERVICE(S) MENTIONNÉ(S)
    # ============================================================
    if services_mentionnes:

        first_day, _ = extract_period_bounds(question, today, start_of_year)

        df_period = df_base[
            (df_base["date_facture_dt"] >= first_day) &
            (df_base["date_facture_dt"] <= today) &
            (df_base["service_clean"].isin(services_mentionnes))
        ]

        summaries = []

        for service in services_mentionnes:
            ca_ytd, ca_lytd, evo, g = build_dimension_kpi(
                df_base,
                df_period,
                "service_clean",
                service,
                start_of_last_year,
                same_day_last_year
            )

            monthly_table = build_monthly_table(g)

            summaries.append(
                f"KPI pour le service {service.capitalize()} :\n"
                f"- CA YTD : {ca_ytd:,.2f} €\n"
                f"- CA LYTD : {ca_lytd:,.2f} €\n"
                f"- Évolution : {evo:.2f} %\n"
                f"{monthly_table}"
            )

        # Comparaison si exactement 2 services
        if len(services_mentionnes) == 2:
            a, b = services_mentionnes
            ca_a = df_ytd[df_ytd["service_clean"] == a]["GFD_MONTANT_VENTE_EUROS"].sum()
            ca_b = df_ytd[df_ytd["service_clean"] == b]["GFD_MONTANT_VENTE_EUROS"].sum()
            diff = ca_a - ca_b
            pct = (diff / ca_b * 100) if ca_b else 0

            summaries.append(
                f"\nComparaison {a.capitalize()} vs {b.capitalize()} :\n"
                f"- {a.capitalize()} : {ca_a:,.2f} €\n"
                f"- {b.capitalize()} : {ca_b:,.2f} €\n"
                f"- Écart : {diff:,.2f} € ({pct:.2f} %)"
            )

        service_summary = "\n\n".join(summaries)

    # ============================================================
    # CAS 2 — RÉPARTITION GLOBALE
    # ============================================================
    else:
        service_summary = build_repartition_table(
            df_ytd,
            "service_clean",
            "Répartition des ventes par service"
        )

    # ============================================================
    # MATRICE SERVICE × BU
    # ============================================================
    service_bu_matrix = build_dimension_bu_matrix(
        df_ytd,
        "service_clean",
        "Répartition des ventes par service et par BU"
    )

    return service_summary, service_bu_matrix



# ===================================================================================================== #
#                   2.2     SECTION CONSTRUCTEUR
# ===================================================================================================== #

def analyze_constructeur_section(
    fact,
    question,
    constructeurs_mentionnes,
    pays_mentionnes,
    bu_mentionnees,
    today,
    start_of_year,
    start_of_last_year,
    same_day_last_year
):
    constructeur_summary = ""

    df_base = fact.copy()

    df_base["constructeur_clean"] = df_base["constructeur_clean"].astype(str)
    df_base["bu_clean"] = df_base["BU"].astype(str).str.upper()

    if pays_mentionnes:
        df_base = df_base[df_base["pays_clean"].isin([p.lower() for p in pays_mentionnes])]

    if bu_mentionnees:
        df_base = df_base[df_base["bu_clean"].isin([b.upper() for b in bu_mentionnees])]

    df_ytd = df_base[
        (df_base["date_facture_dt"] >= start_of_year) &
        (df_base["date_facture_dt"] <= today)
    ]

    # ============================================================
    # CAS 1 — CONSTRUCTEUR(S) MENTIONNÉ(S)
    # ============================================================
    if constructeurs_mentionnes:

        first_day, _ = extract_period_bounds(question, today, start_of_year)

        df_period = df_base[
            (df_base["date_facture_dt"] >= first_day) &
            (df_base["date_facture_dt"] <= today) &
            (df_base["constructeur_clean"].isin(constructeurs_mentionnes))
        ]

        summaries = []

        for constructeur in constructeurs_mentionnes:
            ca_ytd, ca_lytd, evo, g = build_dimension_kpi(
                df_base,
                df_period,
                "constructeur_clean",
                constructeur,
                start_of_last_year,
                same_day_last_year
            )

            monthly_table = build_monthly_table(g)

            summaries.append(
                f"KPI pour le constructeur {constructeur.title()} :\n"
                f"- CA YTD : {ca_ytd:,.2f} €\n"
                f"- CA LYTD : {ca_lytd:,.2f} €\n"
                f"- Évolution : {evo:.2f} %\n"
                f"{monthly_table}"
            )

        # ========================================================
        # COMPARAISON SI EXACTEMENT 2 CONSTRUCTEURS
        # ========================================================
        if len(constructeurs_mentionnes) == 2:
            a, b = constructeurs_mentionnes
            ca_a = df_ytd[df_ytd["constructeur_clean"] == a]["GFD_MONTANT_VENTE_EUROS"].sum()
            ca_b = df_ytd[df_ytd["constructeur_clean"] == b]["GFD_MONTANT_VENTE_EUROS"].sum()
            diff = ca_a - ca_b
            pct = (diff / ca_b * 100) if ca_b else 0

            summaries.append(
                f"\nComparaison {a.title()} vs {b.title()} :\n"
                f"- {a.title()} : {ca_a:,.2f} €\n"
                f"- {b.title()} : {ca_b:,.2f} €\n"
                f"- Écart : {diff:,.2f} € ({pct:.2f} %)"
            )

        constructeur_summary = "\n\n".join(summaries)

    # ============================================================
    # CAS 2 — RÉPARTITION GLOBALE
    # ============================================================
    else:
        constructeur_summary = build_repartition_table(
            df_ytd,
            "constructeur_clean",
            "Répartition des ventes par constructeur"
        )

    # ============================================================
    # MATRICE CONSTRUCTEUR × BU
    # ============================================================
    constructeur_bu_matrix = build_dimension_bu_matrix(
        df_ytd,
        "constructeur_clean",
        "Répartition des ventes par constructeur et par BU"
    )

    return constructeur_summary, constructeur_bu_matrix



# ======================================================================================================== #
#                                   2.3 SECTION CLIENT
# ======================================================================================================== #

# KPI Client
def build_client_kpi(
    df_base,
    df_period,
    client,
    start_of_last_year,
    same_day_last_year
):
    return build_dimension_kpi(
        df_base=df_base,
        df_period=df_period,
        dim_col="client_clean",
        value=client,
        start_of_last_year=start_of_last_year,
        same_day_last_year=same_day_last_year
    )


# -------------------------------------------- SECTION CLIENT FINAL --------------------------------------- #
def analyze_client_section(
    fact,
    question,
    clients_mentionnes,
    today,
    start_of_year,
    start_of_last_year,
    same_day_last_year
):
    client_summary = ""

    # Aucun client mentionné → comportement identique
    if not clients_mentionnes:
        return client_summary

    df_base = fact.copy()
    df_base["client_clean"] = df_base["client_clean"].astype(str)

    # ============================================================
    # PÉRIODE D’ANALYSE
    # ============================================================
    first_day, _ = extract_period_bounds(question, today, start_of_year)

    df_period = df_base[
        (df_base["date_facture_dt"] >= first_day) &
        (df_base["date_facture_dt"] <= today) &
        (df_base["client_clean"].isin(clients_mentionnes))
    ]

    summaries = []

    # ============================================================
    # KPI PAR CLIENT
    # ============================================================
    for client in clients_mentionnes:
        ca_ytd, ca_lytd, evo, g = build_client_kpi(
            df_base,
            df_period,
            client,
            start_of_last_year,
            same_day_last_year
        )

        monthly_table = build_monthly_table(g)

        summaries.append(
            f"KPI pour le client {client} :\n"
            f"- CA YTD : {ca_ytd:,.2f} €\n"
            f"- CA LYTD : {ca_lytd:,.2f} €\n"
            f"- Évolution CA : {evo:.2f} %\n"
            f"{monthly_table}"
        )

    # ============================================================
    # COMPARAISON SI EXACTEMENT 2 CLIENTS
    # ============================================================
    if len(clients_mentionnes) == 2:
        a, b = clients_mentionnes

        ca_a = df_base.loc[
            (df_base["client_clean"] == a) &
            (df_base["date_facture_dt"] >= start_of_year) &
            (df_base["date_facture_dt"] <= today),
            "GFD_MONTANT_VENTE_EUROS"
        ].sum()

        ca_b = df_base.loc[
            (df_base["client_clean"] == b) &
            (df_base["date_facture_dt"] >= start_of_year) &
            (df_base["date_facture_dt"] <= today),
            "GFD_MONTANT_VENTE_EUROS"
        ].sum()

        diff = ca_a - ca_b
        pct = (diff / ca_b * 100) if ca_b else 0

        summaries.append(
            f"\nComparaison entre {a} et {b} :\n"
            f"- {a} : {ca_a:,.2f} €\n"
            f"- {b} : {ca_b:,.2f} €\n"
            f"- Différence : {diff:,.2f} € ({pct:.2f} %)"
        )

    client_summary = "\n\n".join(summaries)

    return client_summary



# ======================================================================================================== #
#                       2.4 SECTION PAYS                                                                   #
# ======================================================================================================== #

#Mini-wrapper lisibilité
def build_country_kpi(
    df_base,
    df_period,
    pays,
    start_of_last_year,
    same_day_last_year
):
    return build_dimension_kpi(
        df_base=df_base,
        df_period=df_period,
        dim_col="pays_clean",
        value=pays,
        start_of_last_year=start_of_last_year,
        same_day_last_year=same_day_last_year
    )


# -------------------------------- SECTION PAYS FINAL ------------------------------------------------

def analyze_pays_section(
    fact,
    question,
    pays_mentionnes,
    today,
    start_of_year,
    start_of_last_year,
    same_day_last_year
):
    pays_summary = ""

    # Aucun pays mentionné → comportement identique
    if not pays_mentionnes:
        return pays_summary

    df_base = fact.copy()
    df_base["pays_clean"] = df_base["pays_clean"].astype(str)

    # ============================================================
    # PÉRIODE D’ANALYSE
    # ============================================================
    first_day, _ = extract_period_bounds(question, today, start_of_year)

    df_period = df_base[
        (df_base["date_facture_dt"] >= first_day) &
        (df_base["date_facture_dt"] <= today) &
        (df_base["pays_clean"].isin(pays_mentionnes))
    ]

    summaries = []

    # ============================================================
    # KPI PAR PAYS
    # ============================================================
    for pays in pays_mentionnes:
        ca_ytd, ca_lytd, evo, g = build_country_kpi(
            df_base,
            df_period,
            pays,
            start_of_last_year,
            same_day_last_year
        )

        monthly_table = build_monthly_table(g)

        summaries.append(
            f"KPI pour le pays {pays} :\n"
            f"- CA YTD : {ca_ytd:,.2f} €\n"
            f"- CA LYTD : {ca_lytd:,.2f} €\n"
            f"- Évolution CA : {evo:.2f} %\n"
            f"{monthly_table}"
        )

    # ============================================================
    # COMPARAISON SI EXACTEMENT 2 PAYS
    # ============================================================
    if len(pays_mentionnes) == 2:
        a, b = pays_mentionnes

        ca_a = df_base.loc[
            (df_base["pays_clean"] == a) &
            (df_base["date_facture_dt"] >= start_of_year) &
            (df_base["date_facture_dt"] <= today),
            "GFD_MONTANT_VENTE_EUROS"
        ].sum()

        ca_b = df_base.loc[
            (df_base["pays_clean"] == b) &
            (df_base["date_facture_dt"] >= start_of_year) &
            (df_base["date_facture_dt"] <= today),
            "GFD_MONTANT_VENTE_EUROS"
        ].sum()

        diff = ca_a - ca_b
        pct = (diff / ca_b * 100) if ca_b else 0

        summaries.append(
            f"\nComparaison entre {a} et {b} :\n"
            f"- {a} : {ca_a:,.2f} €\n"
            f"- {b} : {ca_b:,.2f} €\n"
            f"- Différence : {diff:,.2f} € ({pct:.2f} %)"
        )

    pays_summary = "\n\n".join(summaries)

    return pays_summary



# ==================================================================================================== #
#                               2.5 SECTION BU
# ==================================================================================================== #


# KPI BU (avec ou sans pays)

def build_bu_kpi(
    df_base,
    df_period,
    bu,
    pays=None,
    start_of_last_year=None,
    same_day_last_year=None
):
    if pays:
        mask = (
            (df_base["BU"] == bu) &
            (df_base["pays_clean"] == pays) &
            (df_base["date_facture_dt"] >= start_of_last_year) &
            (df_base["date_facture_dt"] <= same_day_last_year)
        )
    else:
        mask = (
            (df_base["BU"] == bu) &
            (df_base["date_facture_dt"] >= start_of_last_year) &
            (df_base["date_facture_dt"] <= same_day_last_year)
        )

    ca_lytd = df_base.loc[mask, "GFD_MONTANT_VENTE_EUROS"].sum()

    ca_ytd = df_period.loc[
        (df_period["BU"] == bu) &
        ((df_period["pays_clean"] == pays) if pays else True),
        "GFD_MONTANT_VENTE_EUROS"
    ].sum()

    evo = ((ca_ytd - ca_lytd) / ca_lytd * 100) if ca_lytd else 0

    return ca_ytd, ca_lytd, evo

# Répartition BU (simple et claire)
def build_bu_repartition(df, title):
    repartition = (
        df.groupby("BU")["GFD_MONTANT_VENTE_EUROS"]
        .sum()
        .sort_values(ascending=False)
    )

    total = repartition.sum()

    table = f"### {title}\n\n"
    table += "| BU | CA (€) | Part (%) |\n|----|--------|----------|\n"

    for bu, ca in repartition.items():
        pct = (ca / total * 100) if total else 0
        table += f"| {bu} | {ca:,.2f} € | {pct:.2f} % |\n"

    return table


# --------------------------------- SECTION BU FINAL — VERSION REFACTORISÉE ---------------------------- ""

def analyze_bu_section(
    fact,
    question,
    bu_mentionnees,
    pays_mentionnes,
    today,
    start_of_year,
    start_of_last_year,
    same_day_last_year
):
    bu_summary = ""

    df_base = fact.copy()
    df_base["BU"] = df_base["BU"].astype(str).str.upper()
    df_base["pays_clean"] = df_base["pays_clean"].astype(str).str.lower()

    # ============================================================
    # PÉRIODE D’ANALYSE
    # ============================================================
    first_day, _ = extract_period_bounds(question, today, start_of_year)

    df_period = df_base[
        (df_base["date_facture_dt"] >= first_day) &
        (df_base["date_facture_dt"] <= today)
    ]

    summaries = []

    # ============================================================
    # CAS 1 — PAYS + BU
    # ============================================================
    if pays_mentionnes and bu_mentionnees:
        for pays in pays_mentionnes:
            df_filtered = df_period[df_period["pays_clean"] == pays.lower()]

            for bu in bu_mentionnees:
                ca_ytd, ca_lytd, evo = build_bu_kpi(
                    df_base,
                    df_filtered,
                    bu.upper(),
                    pays.lower(),
                    start_of_last_year,
                    same_day_last_year
                )

                summaries.append(
                    f"KPI pour la BU {bu.upper()} au {pays.title()} :\n"
                    f"- CA YTD : {ca_ytd:,.2f} €\n"
                    f"- CA LYTD : {ca_lytd:,.2f} €\n"
                    f"- Évolution CA : {evo:.2f} %"
                )

        # Comparaison si exactement 2 BU
        if len(bu_mentionnees) == 2:
            bu_a, bu_b = bu_mentionnees
            for pays in pays_mentionnes:
                ca_a = df_period.loc[
                    (df_period["BU"] == bu_a.upper()) &
                    (df_period["pays_clean"] == pays.lower()),
                    "GFD_MONTANT_VENTE_EUROS"
                ].sum()

                ca_b = df_period.loc[
                    (df_period["BU"] == bu_b.upper()) &
                    (df_period["pays_clean"] == pays.lower()),
                    "GFD_MONTANT_VENTE_EUROS"
                ].sum()

                diff = ca_a - ca_b
                pct = (diff / ca_b * 100) if ca_b else 0

                summaries.append(
                    f"\nComparaison au {pays.title()} entre BU {bu_a.upper()} et {bu_b.upper()} :\n"
                    f"- {bu_a.upper()} : {ca_a:,.2f} €\n"
                    f"- {bu_b.upper()} : {ca_b:,.2f} €\n"
                    f"- Différence : {diff:,.2f} € ({pct:.2f} %)"
                )

    # ============================================================
    # CAS 2 — BU SEULEMENT
    # ============================================================
    elif bu_mentionnees:
        for bu in bu_mentionnees:
            ca_ytd, ca_lytd, evo = build_bu_kpi(
                df_base,
                df_period,
                bu.upper(),
                None,
                start_of_last_year,
                same_day_last_year
            )

            summaries.append(
                f"KPI pour la BU {bu.upper()} (tous pays confondus) :\n"
                f"- CA YTD : {ca_ytd:,.2f} €\n"
                f"- CA LYTD : {ca_lytd:,.2f} €\n"
                f"- Évolution CA : {evo:.2f} %"
            )

        if len(bu_mentionnees) == 2:
            bu_a, bu_b = bu_mentionnees
            ca_a = df_period.loc[df_period["BU"] == bu_a.upper(), "GFD_MONTANT_VENTE_EUROS"].sum()
            ca_b = df_period.loc[df_period["BU"] == bu_b.upper(), "GFD_MONTANT_VENTE_EUROS"].sum()

            diff = ca_a - ca_b
            pct = (diff / ca_b * 100) if ca_b else 0

            summaries.append(
                f"\nComparaison entre BU {bu_a.upper()} et {bu_b.upper()} (tous pays) :\n"
                f"- {bu_a.upper()} : {ca_a:,.2f} €\n"
                f"- {bu_b.upper()} : {ca_b:,.2f} €\n"
                f"- Différence : {diff:,.2f} € ({pct:.2f} %)"
            )

    # ============================================================
    # CAS 3 — PAYS SEULEMENT
    # ============================================================
    elif pays_mentionnes:
        for pays in pays_mentionnes:
            df_pays = df_period[df_period["pays_clean"] == pays.lower()]
            summaries.append(
                build_bu_repartition(
                    df_pays,
                    f"Répartition des ventes par BU au {pays.title()} (YTD)"
                )
            )

    # ============================================================
    # CAS 4 — GLOBAL
    # ============================================================
    else:
        summaries.append(
            build_bu_repartition(
                df_period,
                "Répartition des ventes par BU (YTD)"
            )
        )

    bu_summary = "\n\n".join(summaries)
    return bu_summary


# ==================================================================================================== #
#                           2.6 TOP CLIENTS
# ==================================================================================================== #

#Fonction universelle Top Clients    
def build_top_clients_block(df, title, n=5):
    if df.empty:
        return f"### {title}\nℹ️ Aucune donnée disponible.\n"

    top_clients = (
        df.groupby("RAISON_SOCIALE")["GFD_MONTANT_VENTE_EUROS"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )

    lines = [f"### {title} (Top {len(top_clients)})"]

    for i, (client, ca) in enumerate(top_clients.items(), start=1):
        lines.append(f"{i}. {client} : {ca:,.2f} €")

    return "\n".join(lines) + "\n"



# ---------------------------------- SECTION TOP CLIENTS FINAL ----------------------------------------- #

def analyze_top_clients_section(
    fact,
    pays_mentionnes,
    bu_mentionnees,
    today,
    start_of_year
):
    top_clients_summary = ""

    # ============================================================
    # DATASET YTD
    # ============================================================
    fact_ytd = fact[
        (fact["date_facture_dt"] >= start_of_year) &
        (fact["date_facture_dt"] <= today)
    ].copy()

    fact_ytd["BU"] = fact_ytd["BU"].astype(str).str.upper()
    fact_ytd["pays_clean"] = fact_ytd["pays_clean"].astype(str).str.lower()
    fact_ytd["RAISON_SOCIALE"] = fact_ytd["RAISON_SOCIALE"].astype(str).str.strip()

    # Normalisation listes
    pays_mentionnes = [p.lower() for p in pays_mentionnes] if pays_mentionnes else []
    bu_mentionnees = [b.upper() for b in bu_mentionnees] if bu_mentionnees else []

    # ============================================================
    # CAS 1 — PAYS + BU
    # ============================================================
    if pays_mentionnes and bu_mentionnees:
        for pays in pays_mentionnes:
            for bu in bu_mentionnees:
                df_filtered = fact_ytd[
                    (fact_ytd["pays_clean"] == pays) &
                    (fact_ytd["BU"] == bu)
                ]

                title = f"Pays : {pays.title()} | BU : {bu}"
                top_clients_summary += build_top_clients_block(df_filtered, title)

    # ============================================================
    # CAS 2 — PAYS SEULEMENT
    # ============================================================
    elif pays_mentionnes:
        for pays in pays_mentionnes:
            df_filtered = fact_ytd[fact_ytd["pays_clean"] == pays]
            title = f"Pays : {pays.title()}"
            top_clients_summary += build_top_clients_block(df_filtered, title)

    # ============================================================
    # CAS 3 — BU SEULEMENT
    # ============================================================
    elif bu_mentionnees:
        for bu in bu_mentionnees:
            df_filtered = fact_ytd[fact_ytd["BU"] == bu]
            title = f"BU : {bu}"
            top_clients_summary += build_top_clients_block(df_filtered, title)

    # ============================================================
    # CAS 4 — GLOBAL
    # ============================================================
    else:
        # Top par BU
        for bu in sorted(fact_ytd["BU"].dropna().unique()):
            df_filtered = fact_ytd[fact_ytd["BU"] == bu]
            title = f"BU : {bu}"
            top_clients_summary += build_top_clients_block(df_filtered, title)

        # Top global
        top_clients_summary += build_top_clients_block(
            fact_ytd,
            "Top clients globaux"
        )

    return top_clients_summary


# ===================================================================================================== #
#                                   2.7 MAJOR CLASS
# ===================================================================================================== #

#Top Major Classe (générique)
def build_top_major_classe(df, title, labels_map, n=5):
    if df.empty:
        return f"### {title}\nℹ️ Aucune donnée disponible.\n"

    grouped = (
        df.groupby("MAJOR_CLASSE")["GFD_MONTANT_VENTE_EUROS"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
    )

    total = df["GFD_MONTANT_VENTE_EUROS"].sum()

    lines = [f"### {title} (Top {len(grouped)})"]
    for i, (major_id, ca) in enumerate(grouped.items(), start=1):
        label = labels_map.get(major_id, f"Unknown ({major_id})")
        pct = (ca / total * 100) if total else 0
        lines.append(f"{i}. {label} : {ca:,.2f} € ({pct:.1f}%)")

    return "\n".join(lines) + "\n"



#Répartition Major Classe × Pays
def build_major_classe_by_country(df, labels_map):
    if df.empty:
        return "ℹ️ Aucune donnée disponible.\n"

    pivot = (
        df.groupby(["pays_clean", "MAJOR_CLASSE"])["GFD_MONTANT_VENTE_EUROS"]
        .sum()
        .reset_index()
    )

    total_global = df["GFD_MONTANT_VENTE_EUROS"].sum()
    lines = ["### Répartition du CA par Major Classe et par Pays (YTD)"]

    for pays in sorted(pivot["pays_clean"].unique()):
        df_pays = pivot[pivot["pays_clean"] == pays]
        total_pays = df_pays["GFD_MONTANT_VENTE_EUROS"].sum()

        lines.append(
            f"\n**{pays.title()}** — Total : {total_pays:,.2f} € "
            f"({total_pays / total_global * 100:.1f}%)"
        )

        for _, row in df_pays.sort_values("GFD_MONTANT_VENTE_EUROS", ascending=False).iterrows():
            label = labels_map.get(row["MAJOR_CLASSE"], f"Unknown ({row['MAJOR_CLASSE']})")
            pct = (row["GFD_MONTANT_VENTE_EUROS"] / total_pays * 100) if total_pays else 0
            lines.append(
                f"  - {label} : {row['GFD_MONTANT_VENTE_EUROS']:,.2f} € ({pct:.1f}%)"
            )

    return "\n".join(lines) + "\n"


#Analyse Major Classe par Client (structurée)
def build_major_classe_by_client(
    df,
    client_name,
    labels_map,
    n_major=3,
    n_products=5
):
    df_client = df[
        df["RAISON_SOCIALE"].str.contains(client_name, case=False, na=False, regex=False)
    ]

    if df_client.empty:
        return f"### {client_name.title()} — Aucune donnée disponible\n"

    total_client = df_client["GFD_MONTANT_VENTE_EUROS"].sum()
    output = ""

    # Top produits globaux
    products = (
        df_client.groupby("DESCRIPTION_PRODUIT")["GFD_MONTANT_VENTE_EUROS"]
        .sum()
        .sort_values(ascending=False)
        .head(n_products)
    )

    output += f"### Top produits – **{client_name.title()}**\n"
    for i, (prod, ca) in enumerate(products.items(), start=1):
        pct = (ca / total_client * 100) if total_client else 0
        output += f"{i}. {prod} : {ca:,.2f} € ({pct:.1f}%)\n"

    # Top Major Classes
    majors = (
        df_client.groupby("MAJOR_CLASSE")["GFD_MONTANT_VENTE_EUROS"]
        .sum()
        .sort_values(ascending=False)
        .head(n_major)
    )

    output += f"\n### Top Major Classes – **{client_name.title()}**\n"
    for major_id, ca in majors.items():
        label = labels_map.get(major_id, f"Unknown ({major_id})")
        pct = (ca / total_client * 100) if total_client else 0
        output += f"- {label} : {ca:,.2f} € ({pct:.1f}%)\n"

        df_major = df_client[df_client["MAJOR_CLASSE"] == major_id]
        products_major = (
            df_major.groupby("DESCRIPTION_PRODUIT")["GFD_MONTANT_VENTE_EUROS"]
            .sum()
            .sort_values(ascending=False)
            .head(n_products)
        )

        for prod, ca_prod in products_major.items():
            pct_prod = (ca_prod / ca * 100) if ca else 0
            output += f"    • {prod} : {ca_prod:,.2f} € ({pct_prod:.1f}%)\n"

    return output + "\n"



# ---------------------------------- MAJOR CLASSE FINALE -------------------------------------------- #

def analyze_major_classe_section(
    fact,
    pays_mentionnes,
    bu_mentionnees,
    clients_mentionnes,
    today,
    start_of_year
):
    major_classe_labels = {
        1: '1-UNDERCARRIAGE',
        2: '2-ENGINE',
        3: '3-GROUND ENGAGING TOOLS',
        5: '5-DRIVETRAIN',
        6: '6-HYDRAULICS',
        7: '7-FILTERS & FLUIDS',
        8: '8-ELECTRONICS & ELECTRICAL COMP.',
        9: '9-STRUCTURAL'
    }

    # ============================================================
    # DATASET YTD
    # ============================================================
    fact_ytd = fact[
        (fact["date_facture_dt"] >= start_of_year) &
        (fact["date_facture_dt"] <= today)
    ].copy()

    major_classe_summary = ""

    # ============================================================
    # GLOBAL
    # ============================================================
    major_classe_summary += build_top_major_classe(
        fact_ytd,
        "Top Major Classe globaux",
        major_classe_labels
    )

    # ============================================================
    # PAR PAYS
    # ============================================================
    if pays_mentionnes:
        for pays in pays_mentionnes:
            df_pays = fact_ytd[fact_ytd["pays_clean"] == pays.lower()]
            major_classe_summary += build_top_major_classe(
                df_pays,
                f"Top Major Classe — {pays.title()}",
                major_classe_labels
            )

    # ============================================================
    # PAR BU
    # ============================================================
    if bu_mentionnees:
        for bu in bu_mentionnees:
            df_bu = fact_ytd[fact_ytd["BU"].str.upper() == bu.upper()]
            major_classe_summary += build_top_major_classe(
                df_bu,
                f"Top Major Classe — BU {bu.upper()}",
                major_classe_labels
            )

    # ============================================================
    # DISTRIBUTION PAR PAYS
    # ============================================================
    major_classe_summary += build_major_classe_by_country(
        fact_ytd,
        major_classe_labels
    )

    # ============================================================
    # ANALYSE PAR CLIENT
    # ============================================================
    major_classe_summary_client = ""
    if clients_mentionnes:
        for client in clients_mentionnes:
            major_classe_summary_client += build_major_classe_by_client(
                fact_ytd,
                client,
                major_classe_labels
            )

    return major_classe_summary, major_classe_summary_client





# ===================================================================================================== #
#                                   2.8 SECTION NOMBRE DE CLIENTS
# ===================================================================================================== #

#Nombre total de clients
def count_total_clients(df, title):
    if df.empty:
        return f"{title} : Aucune donnée disponible.\n"

    n_clients = df["RAISON_SOCIALE"].nunique()
    total_ca = df["GFD_MONTANT_VENTE_EUROS"].sum()

    return (
        f"{title} : **{n_clients:,}** clients actifs "
        f"({total_ca:,.2f} € de CA total)\n"
    )


#Nombre de clients par dimension
def count_clients_by_group(df, group_col, title):
    if df.empty:
        return f"{title} : Aucune donnée disponible.\n"

    grouped = (
        df.groupby(group_col)["RAISON_SOCIALE"]
        .nunique()
        .sort_values(ascending=False)
    )

    total_clients = df["RAISON_SOCIALE"].nunique()

    lines = [f"### {title}"]
    for group, n in grouped.items():
        pct = (n / total_clients * 100) if total_clients else 0
        label = group.title() if isinstance(group, str) else str(group)
        lines.append(f"- **{label}** : {n} clients ({pct:.1f}%)")

    return "\n".join(lines) + "\n"


#Insight automatique (BU dominante)
def build_clients_insight_by_bu(df):
    try:
        total_clients = df["RAISON_SOCIALE"].nunique()

        clients_by_bu = (
            df.groupby("BU")["RAISON_SOCIALE"]
            .nunique()
            .sort_values(ascending=False)
        )

        top_bu, top_count = clients_by_bu.index[0], clients_by_bu.iloc[0]
        pct = (top_count / total_clients * 100) if total_clients else 0

        return (
            f"\n**Analyse :** La BU **{top_bu}** concentre le plus grand "
            f"nombre de clients actifs ({top_count} sur {total_clients}, "
            f"soit {pct:.1f} % du portefeuille). "
            f"Les autres BU présentent une base plus restreinte, "
            f"mais souvent à plus forte valeur moyenne par client.\n"
        )
    except Exception:
        return ""



# -------------------------- SECTION NOMBRE DE CLIENTS —---------------------------------------------- #
def analyze_client_count_section(
    fact,
    today,
    start_of_year
):
    client_count_summary = ""

    # ============================================================
    # DATASET YTD
    # ============================================================
    fact_ytd = fact[
        (fact["date_facture_dt"] >= start_of_year) &
        (fact["date_facture_dt"] <= today)
    ].copy()

    fact_ytd["BU"] = fact_ytd["BU"].astype(str).str.upper()
    fact_ytd["pays_clean"] = fact_ytd["pays_clean"].astype(str).str.lower()
    fact_ytd["RAISON_SOCIALE"] = fact_ytd["RAISON_SOCIALE"].astype(str).str.strip()

    # ============================================================
    # CONSTRUCTION DU RÉSUMÉ
    # ============================================================
    client_count_summary += "## Nombre de Clients (YTD)\n\n"

    # Total clients
    client_count_summary += count_total_clients(
        fact_ytd,
        "Nombre total de clients (groupe)"
    )

    # Par pays
    client_count_summary += "\n" + count_clients_by_group(
        fact_ytd,
        "pays_clean",
        "Nombre de clients par pays"
    )

    # Par BU
    client_count_summary += "\n" + count_clients_by_group(
        fact_ytd,
        "BU",
        "Nombre de clients par BU"
    )

    # Insight automatique
    client_count_summary += build_clients_insight_by_bu(fact_ytd)

    return client_count_summary




# ====================================================================================================== #
#                           ASSEMBLAGE VERSION FINAL
# ====================================================================================================== #
# -------------------------------------- CAS SPECIFIQUE CA PAR ANNEE ----------------------------
def is_yearly_analysis(question: str) -> bool:
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

def build_ca_by_year(
    df,
    filter_mask=None,
    include_ytd_label=True
):
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

    

def yearly_analysis_allowed(df, current_year):
    years = df["date_facture_dt"].dt.year.dropna().unique()
    complete_years = [y for y in years if y < current_year]
    return len(complete_years) >= 1




def extract_year_from_question(question: str):
    match = re.search(r"(20\d{2})", question)
    return int(match.group(1)) if match else None


def extract_years_from_question(question: str) -> list[int]:
    import re
    years = re.findall(r"\b(20\d{2})\b", question)
    return sorted(set(map(int, years)))


def build_ca_by_month(
    df: pd.DataFrame,
    include_ytd_label: bool = False,   # optionnel si tu veux marquer l'année en cours
) -> str:
    if df.empty or "date_facture_dt" not in df.columns:
        return "Aucune donnée temporelle disponible pour une analyse mensuelle.\n"

    # On garde seulement les dates valides
    df = df.dropna(subset=["date_facture_dt"]).copy()
    if df.empty:
        return "Aucune donnée disponible pour une analyse mensuelle.\n"

    # Période mensuelle
    df["ANNEE_MOIS"] = df["date_facture_dt"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("ANNEE_MOIS", as_index=False)
          .agg(CA=("GFD_MONTANT_VENTE_EUROS", "sum"))
          .sort_values("ANNEE_MOIS")
    )

    if monthly.empty:
        return "Aucune donnée disponible pour une analyse mensuelle.\n"

    # Evolution vs mois précédent
    monthly["Evol (€)"] = monthly["CA"].diff()
    monthly["Evol (%)"] = monthly["CA"].pct_change() * 100

    # Formatage table markdown
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

    # (Option) rappel total
    total = monthly["CA"].sum()
    table += f"\n**Total période : {total:,.2f} €**\n"

    return table



def render_monthly_ca_table(monthly: pd.DataFrame) -> str:
    if monthly is None or monthly.empty or "ANNEE_MOIS" not in monthly.columns:
        return "Aucune donnée disponible pour une analyse mensuelle.\n"

    monthly = monthly.copy().sort_values("ANNEE_MOIS")

    # standardise colonne CA
    if "CA" not in monthly.columns:
        if "total_sales" in monthly.columns:
            monthly["CA"] = monthly["total_sales"]
        else:
            return "Aucune donnée disponible pour une analyse mensuelle.\n"

    # Evolution MoM
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

    
def build_monthly_sales(
    df: pd.DataFrame,
    group_by_client: bool = False
) -> pd.DataFrame:

    if df.empty or "DATE_FACTURE" not in df.columns:
        return pd.DataFrame()

    df = df.copy()

    # 🔹 Conversion UNIQUE et fiable de la date
    df["date_facture_dt"] = pd.to_datetime(
        df["DATE_FACTURE"],
        errors="coerce"
    )

    df = df.dropna(subset=["date_facture_dt"])

    # 🔹 Groupement mensuel (Period)
    df["ANNEE_MOIS"] = df["date_facture_dt"].dt.to_period("M")

    group_cols = ["ANNEE_MOIS"]
    if group_by_client and "RAISON_SOCIALE" in df.columns:
        group_cols.append("RAISON_SOCIALE")

    monthly = (
        df
        .groupby(group_cols, as_index=False)
        .agg(
            total_sales=("GFD_MONTANT_VENTE_EUROS", "sum")
        )
    )

    # 🔹 CONVERSION CRUCIALE POUR PLOTLY
    monthly["ANNEE_MOIS"] = monthly["ANNEE_MOIS"].dt.to_timestamp()
    monthly = monthly.sort_values("ANNEE_MOIS")

    return monthly

import re
from datetime import datetime

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

def extract_month_from_question(question: str) -> int | None:
    q = (question or "").lower()
    # match mot entier (évite "mars" dans autre chose)
    for name, m in MONTHS_FR.items():
        if re.search(rf"\b{re.escape(name)}\b", q):
            return m
    return None

def extract_year_from_question(question: str) -> int | None:
    match = re.search(r"\b(20\d{2})\b", question or "")
    return int(match.group(1)) if match else None

def is_month_specific_request(question: str) -> bool:
    q = (question or "").lower()
    # On veut un mois explicite + une formulation de type "au mois de", "en mars", etc.
    has_month = extract_month_from_question(q) is not None
    return has_month and any(k in q for k in ["au mois", "mois de", "en "])


import re



def is_revenue_intent(question: str) -> bool:
    q = (question or "").lower()

    # si OLGA est mentionné -> pas du CA facturé
    if any(w in q for w in ["opportunité", "opportunites", "opportunités", "pipeline", "pops", "olga", "lost"]):
        return False

    # CA explicite
    if re.search(r"\bca\b", q):
        return True

    if any(k in q for k in [
        "chiffre d'affaires", "chiffre daffaires",
        "facture", "facturé", "facturee", "facturation",
        "revenu", "revenue",
        "ventes facturées", "vente facturée"
    ]):
        return True

    # ventes => CA seulement si contexte temps/analyse
    has_sales = re.search(r"\bventes?\b", q) is not None
    has_context = (
        any(k in q for k in ["par mois", "mensuel", "mensuelle", "mois", "évolution", "evolution", "historique", "tendance"])
        or re.search(r"\b20\d{2}\b", q) is not None
    )
    return bool(has_sales and has_context)



def is_monthly_analysis(question: str) -> bool:
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

def is_comparison_intent(question: str) -> bool:
    q = (question or "").lower()
    return any(w in q for w in ["compare", "comparaison", "vs", "versus"])

    
def generate_summary(fact: pd.DataFrame, question: str = "") -> str:
    # =================================================================================
    # NORMALISATION DE BASE
    # =================================================================================
    fact = fact.copy()

    fact["client_clean"] = fact["RAISON_SOCIALE"].astype(str).str.lower().str.strip()
    fact["pays_clean"] = fact["PAYS_DESTINATION"].astype(str).str.lower().str.strip()
    fact["bu_clean"] = fact["BU"].astype(str).str.upper().str.strip()
    fact["constructeur_clean"] = (
        fact["LIBELLE_CONSTRUCTEUR"]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(lambda x: constructeur_alias_map.get(x, x))
    )
    fact["service_clean"] = (
        fact["LIBELLE_SERVICE_FR"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
    ) # dayfirst=True

    fact["date_facture_dt"] = pd.to_datetime(
        fact["DATE_FACTURE"], errors="coerce"
    )

    

    # =================================================================================
    # DATES DE RÉFÉRENCE
    # =================================================================================
    today = datetime.now()

    current_year = today.year
    last_year = current_year - 1

    start_of_year = datetime(current_year, 1, 1)
    start_of_last_year = datetime(last_year, 1, 1)

    same_day_last_year = datetime(last_year, today.month, today.day)

    # =================================================================================
    # DÉTECTION DES ENTITÉS DANS LA QUESTION
    # =================================================================================
    question_lower = question.lower()

    pays_mentionnes = find_countries_in_question(
        question_lower,
        fact["pays_clean"].dropna().unique().tolist()
    )

    clients_mentionnes = find_clients_in_question(
        question_lower,
        fact["client_clean"].dropna().unique().tolist()
    )

    bu_mentionnees = find_bu_in_question(
        question_lower,
        bu_alias_map
    )

    constructeurs_mentionnes = find_constructeurs_in_question(
        question_lower,
        constructeur_alias_map
    )

    services_mentionnes = find_services_in_question(
        question_lower,
        fact["service_clean"].dropna().unique().tolist()
    )


    # =================================================================================
    # CAS COMPARAISON CA MENSUEL ENTRE DEUX CLIENTS
    # ex: "compare le CA de la SNIM vs Fekola SA par mois en 2025"
    # =================================================================================
    # =================================================================================
    # CAS COMPARAISON CA ANNUELLE / YTD ENTRE DEUX CLIENTS
    # ex: "compare le CA de la SNIM vs Fekola en 2025"
    # ex: "compare le CA de la SNIM vs Fekola" (=> YTD année courante)
    # =================================================================================
    if (
        is_comparison_intent(question)
        and is_revenue_intent(question)
        and len(clients_mentionnes) == 2
        and not is_monthly_analysis(question)   # IMPORTANT: pas "par mois"
        and any(w in question_lower for w in ["compare", "comparaison", "vs", "versus"])
    ):
        client_a, client_b = clients_mentionnes[0], clients_mentionnes[1]
    
        years_requested = extract_years_from_question(question)
        year_target = years_requested[0] if len(years_requested) == 1 else None
    
        df_target = fact.copy()
    
        # filtre clients
        df_target = df_target[df_target["client_clean"].isin([client_a, client_b])]
    
        # ✅ si année non précisée => YTD année courante
        if year_target is None:
            df_target = df_target[
                (df_target["date_facture_dt"] >= start_of_year) &
                (df_target["date_facture_dt"] <= today)
            ]
            period_label = (
                f"YTD {current_year} (du {start_of_year.strftime('%Y-%m-%d')} "
                f"au {today.strftime('%Y-%m-%d')})"
            )
        else:
            df_target = df_target[df_target["date_facture_dt"].dt.year == year_target]
            period_label = f"{year_target}"
    
        if df_target.empty:
            return (
                f"Aucune donnée de chiffre d’affaires disponible pour comparer "
                f"{client_a.upper()} et {client_b.upper()} sur la période : {period_label}."
            )
    
        ca = (
            df_target
            .groupby("client_clean")["GFD_MONTANT_VENTE_EUROS"]
            .sum()
            .to_dict()
        )
    
        ca_a = float(ca.get(client_a, 0.0))
        ca_b = float(ca.get(client_b, 0.0))
        diff = ca_a - ca_b
        pct = (diff / ca_b * 100) if ca_b != 0 else None
    
        out = (
            f"## Comparaison du chiffre d’affaires – {client_a.upper()} vs {client_b.upper()} – {period_label}\n\n"
            f"| Client | CA (€) |\n"
            f"|--------|-------:|\n"
            f"| {client_a.upper()} | {ca_a:,.2f} € |\n"
            f"| {client_b.upper()} | {ca_b:,.2f} € |\n"
            f"\n**Écart : {diff:+,.2f} €**"
        )
    
        if pct is not None:
            out += f" (**{pct:+.1f} %**)"
        else:
            out += " (pourcentage non calculable – CA de référence nul)"
    
        return out


    
   


    # =================================================================================
    # CAS CA MENSUEL (FACTURES) : ex "donne le CA de la SNIM par mois en 2025"
    # =================================================================================
    if is_monthly_analysis(question) and is_revenue_intent(question):

        # --- année demandée ---
        years_requested = extract_years_from_question(question)
        year_target = years_requested[0] if len(years_requested) == 1 else None
    
        # --- construction du TARGET ---
        df_target = fact.copy()
    
        # Filtre client (SNIM)
        if clients_mentionnes:
            df_target = df_target[df_target["client_clean"].isin(clients_mentionnes)]
    
        # Filtre année (2025)
        if year_target is not None:
            df_target = df_target[
                df_target["date_facture_dt"].dt.year == year_target
            ]
    
        # --- sécurité ---
        if df_target.empty:
            return (
                f"Aucune donnée de chiffre d’affaires n’est disponible "
                f"pour {clients_mentionnes[0].upper()} en {year_target}."
            )
    
        # --- APPEL CORRECT ---
        monthly = build_monthly_sales(
            df_target,
            group_by_client=False
        )
    
        # --- garde-fou anti-hallucination ---
        if monthly.empty:
            total = df_target["GFD_MONTANT_VENTE_EUROS"].sum()
            return (
                f"Le chiffre d’affaires total de {clients_mentionnes[0].upper()} "
                f"en {year_target} est de {total:,.2f} €.\n\n"
                "⚠️ La ventilation mensuelle n’est pas disponible à partir "
                "des données de facturation actuelles."
            )
    
        # --- RENDU ---
        return (
            f"## Chiffre d’affaires mensuel – {clients_mentionnes[0].upper()} – {year_target}\n\n"
            + render_monthly_ca_table(monthly)
        )


    # =================================================================================
    # CAS CA SUR UN MOIS PRECIS (ex: "CA SNIM en mars 2024")
    # =================================================================================
    month_target = extract_month_from_question(question)
    year_target = extract_year_from_question(question)
    
    # On déclenche si mois + année + intent CA
    if month_target and year_target and is_revenue_intent(question):
        df_target = fact.copy()
    
        # filtre client
        if clients_mentionnes:
            df_target = df_target[df_target["client_clean"].isin(clients_mentionnes)]
    
        # filtre année + mois
        df_target = df_target[
            (df_target["date_facture_dt"].dt.year == year_target) &
            (df_target["date_facture_dt"].dt.month == month_target)
        ]
    
        if df_target.empty:
            if clients_mentionnes:
                return (
                    f"Aucune donnée de chiffre d’affaires n’est disponible pour "
                    f"{clients_mentionnes[0].upper()} en {list(MONTHS_FR.keys())[month_target-1].title()} {year_target}."
                )
            return f"Aucune donnée de chiffre d’affaires n’est disponible pour {year_target}-{month_target:02d}."
    
        ca_month = df_target["GFD_MONTANT_VENTE_EUROS"].sum()
    
        # libellé mois FR “propre”
        month_name = [k for k,v in MONTHS_FR.items() if v == month_target and len(k) > 3][0]
        month_name = month_name.replace("fevrier", "février").replace("decembre", "décembre").replace("aout","août")
    
        if clients_mentionnes:
            return (
                f"En **{month_name.title()} {year_target}**, le chiffre d’affaires réalisé avec "
                f"**{clients_mentionnes[0].upper()}** est de **{ca_month:,.2f} €**."
            )
        else:
            return (
                f"En **{month_name.title()} {year_target}**, le chiffre d’affaires total est de **{ca_month:,.2f} €**."
            )

    



    # =================================================================================
    # RÈGLES MÉTIER SÉMANTIQUES (SERVICE)
    # =================================================================================
    is_service_dimension = any(
        kw in question_lower
        for kw in [
            "par service",
            "ventes par service",
            "répartition par service",
            "vue globale par service",
        ]
    )

    if is_service_dimension:
        services_mentionnes = []

    if len(services_mentionnes) > 1 and "service" in services_mentionnes:
        services_mentionnes.remove("service")

    # =================================================================================
    # APPEL DES SECTIONS ANALYTIQUES
    # =================================================================================
    client_summary = analyze_client_section(
        fact,
        question,
        clients_mentionnes,
        today,
        start_of_year,
        start_of_last_year,
        same_day_last_year
    )

    pays_summary = analyze_pays_section(
        fact,
        question,
        pays_mentionnes,
        today,
        start_of_year,
        start_of_last_year,
        same_day_last_year
    )

    bu_summary = analyze_bu_section(
        fact,
        question,
        bu_mentionnees,
        pays_mentionnes,
        today,
        start_of_year,
        start_of_last_year,
        same_day_last_year
    )

    service_summary, service_bu_matrix = analyze_service_section(
        fact,
        question,
        services_mentionnes,
        pays_mentionnes,
        bu_mentionnees,
        today,
        start_of_year,
        start_of_last_year,
        same_day_last_year
    )

    constructeur_summary, constructeur_bu_matrix = analyze_constructeur_section(
        fact,
        question,
        constructeurs_mentionnes,
        pays_mentionnes,
        bu_mentionnees,
        today,
        start_of_year,
        start_of_last_year,
        same_day_last_year
    )

    top_clients_summary = analyze_top_clients_section(
        fact,
        pays_mentionnes,
        bu_mentionnees,
        today,
        start_of_year
    )

    major_classe_summary, major_classe_summary_client = analyze_major_classe_section(
        fact,
        pays_mentionnes,
        bu_mentionnees,
        clients_mentionnes,
        today,
        start_of_year
    )

    client_count_summary = analyze_client_count_section(
        fact,
        today,
        start_of_year
    )

    # =================================================================================
    # KPI GLOBAUX (SYNTHÈSE)
    # =================================================================================
    ca_ytd = fact.loc[
        (fact["date_facture_dt"] >= start_of_year) &
        (fact["date_facture_dt"] <= today),
        "GFD_MONTANT_VENTE_EUROS"
    ].sum()

    ca_lytd = fact.loc[
        (fact["date_facture_dt"] >= start_of_last_year) &
        (fact["date_facture_dt"] <= same_day_last_year),
        "GFD_MONTANT_VENTE_EUROS"
    ].sum()


    # =================================================================================
    # INTERPRÉTATION MÉTIER DE L'ÉVOLUTION (LECTURE ANALYSTE SENIOR)
    # =================================================================================
    evolution_label = ""
    
    if ca_lytd > 0 and ca_ytd == 0:
        evolution_label = (
            f"Aucune facturation n’a été enregistrée à ce stade de l’année en cours, "
            f"contre {ca_lytd:,.2f} € sur la même période l’an dernier. "
            "L’activité n’ayant pas encore démarré cette année, "
            "la comparaison d’évolution n’est pas pertinente à ce stade."
        )
    
    elif ca_lytd == 0 and ca_ytd > 0:
        evolution_label = (
            f"Une activité est observée à date cette année, avec un chiffre d’affaires "
            f"de {ca_ytd:,.2f} €. "
            "Aucune facturation n’avait été enregistrée sur la même période l’an dernier, "
            "ce qui ne permet pas une comparaison directe."
        )
    
    else:
        evolution_label = (
            "Aucune activité commerciale n’a été enregistrée sur les périodes comparées, "
            "ni cette année ni sur la même période l’an dernier."
        )


    # =================================================================================
    # ASSEMBLAGE FINAL
    # =================================================================================
    summary = f"""
## KPI Globaux
- CA YTD : {ca_ytd:,.2f} €
- CA LYTD : {ca_lytd:,.2f} €
- Évolution : {evolution_label}

{client_summary}
{pays_summary}
{bu_summary}
{top_clients_summary}
{major_classe_summary}
{major_classe_summary_client}
{client_count_summary}
{constructeur_summary}
{service_summary}
{service_bu_matrix}
{constructeur_bu_matrix}
"""

    return summary.strip()




    
# ==================================================================================================== #
#                           FONCTION NETTOYAGE DU RESULTAT RENVOIYE PAR l'IA                           #
# ==================================================================================================== #

def clean_llm_text(text: str) -> str:
    """
    Nettoie le texte renvoyé par Cortex AI :
    - Transforme les séquences \n en vrais retours à la ligne
    - Supprime les guillemets inutiles
    - Supprime les tabulations \t
    - Normalise les espaces
    """
    if not text:
        return ""

   
    text = text.replace("\\n", "\n")

  
    text = text.strip('"')

    
    text = text.replace("\\t", " ")  
    text = text.replace("\t", " ")  

   
    text = re.sub(r' +', ' ', text)

   
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()






# ==================================================================================================== #
#                           UTILISATION DU LLM DEPUIS SNOWFLAKE                                        #
# ==================================================================================================== #

def ask_llm(
    question: str,
    model: str = "llama3.1-8b",
    temperature: float = 0.1
):
    """
    Utilisation Snowflake Cortex AI avec contexte ventes + opportunités + équipements
    """

    # --- Chargement des données depuis la session --- #
    fact = st.session_state.get("fact")
    final = st.session_state.get("final")
    equipement = st.session_state.get("equipement")
    opportunite_pays = st.session_state.get("opportunite_pays")
    opportunite_bu = st.session_state.get("opportunite_bu")

    if fact is None or fact.empty:
        return {
            "source": "error",
            "response": "⚠️ Données non chargées. Veuillez rafraîchir l’application."
        }

    # --- Contexte Ventes --- #
    context = generate_summary(fact, question=question)

    # --- Contexte Opportunités par clients --- #
    try:
        context_olga = generate_olga(final, question=question)
    except Exception as e:
        print(f"⚠️ Erreur dans generate_olga: {e}")
        context_olga = "Aucune donnée d'opportunité disponible."

    # --- Contexte Opportunités par pays --- #
    try:
        context_olga_countries = generate_olga_by_country(
            opportunite_pays, question=question
        )
    except Exception as e:
        print(f"⚠️ Erreur dans generate_olga_by_country: {e}")
        context_olga_countries = "Aucune donnée d'opportunité par pays disponible."

    # --- Contexte Equipement Client --- #
    try:
        if is_ranking_question(question):
            context_equipement_client = generate_client_equipment_ranking(equipement)
        else:
            context_equipement_client = generate_equipment_summary(
                equipement, question=question
            )
    except Exception as e:
        print(f"⚠️ Erreur dans le contexte équipement: {e}")
        context_equipement_client = "Aucune donnée d'équipement disponible."

    # --- Contexte Opportunités par BU --- #
    try:
        context_olga_bu = generate_olga_bu(
            opportunite_bu,
            question=question,
            bu_alias_map=bu_alias_map
        )
    except Exception as e:
        print(f"⚠️ Erreur dans generate_olga_bu: {e}")
        context_olga_bu = "Aucune donnée d'opportunité par BU disponible."


    # --- Création du prompt final
    formatted_prompt = format_prompt(question, context, context_olga, context_olga_countries,context_equipement_client,context_olga_bu)



    # --- Initialisation session Cortex AI --- #
    session = init_session()

    try:
        prompt_array = [
            {
                'role': 'system',
                'content': "Tu es un assistant expert en analyse client B2B pour l'entreprise Neemba. Réponds uniquement basé sur les données fournies, sans extrapolation."
            },
            {
                'role': 'user',
                'content': formatted_prompt
            }
        ]

        options = {
            'temperature': temperature,
            'max_tokens': 8192,
            'guardrails': True
        }

        sql_query = "SELECT AI_COMPLETE(?, ?, ?) as response"
        result = session.sql(sql_query, params=[model, prompt_array, options]).collect()

        if not result or len(result) == 0:
            return {"source": "error", "response": "Aucune réponse de Cortex AI"}

        response_json = json.loads(result[0]['RESPONSE'])

        # --- Extraire le texte --- #
        llm_text = ""
        if 'choices' in response_json and len(response_json['choices']) > 0:
            messages = response_json['choices'][0]['messages']
            for msg in messages:
                if 'content' in msg:
                    llm_text += msg['content'].strip() + "\n\n"

        llm_text = clean_llm_text(llm_text)

        return {
            "source": "cortex_ai",
            "response": llm_text,
            "model_used": response_json.get('model', model),
            "tokens_used": response_json.get('usage', {})
        }

    except Exception as e:
        # --- Fallback simple --- #
        try:
            sql_query_simple = "SELECT AI_COMPLETE(?, ?) as response"
            result_simple = session.sql(sql_query_simple, params=[model, formatted_prompt]).collect()
            if result_simple and len(result_simple) > 0:
                clean_text = clean_llm_text(result_simple[0]['RESPONSE'])
                return {"source": "cortex_ai_simple", "response": clean_text}
            else:
                return {"source": "error", "response": f"Erreur Cortex AI: {str(e)}"}
        except Exception as fallback_error:
            return {"source": "error", "response": f"Erreur Cortex AI: {str(fallback_error)}"}








#================================================================================================================#
#                                       CSS FONT-END                                                             #
#================================================================================================================#

import re

def clean_html_tags(text):
    """Supprime toutes les balises HTML d'une chaîne."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

import markdown

def markdown_to_html(md_text):
    """Convertit une chaîne Markdown en HTML"""
    return markdown.markdown(md_text)


