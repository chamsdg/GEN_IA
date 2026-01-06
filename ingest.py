import snowflake.connector
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas
import requests
import sys
import os
try:
    # =====================================================
    # 1️⃣ LECTURE DU CSV
    # =====================================================
    df = pd.read_csv("fact_24_mois.csv", sep=";")
    assert not df.empty, "Dataset vide"

    # =====================================================
    # 2️⃣ NETTOYAGE DES VALEURS
    # =====================================================
    df = df.replace(["NaN", "nan", ""], None)

    # =====================================================
    # 3️⃣ GESTION DES DATES
    # =====================================================
    if "date_facture" in df.columns:
        df["date_facture"] = pd.to_datetime(
            df["date_facture"],
            errors="coerce",
            dayfirst=True
        ).dt.strftime("%Y-%m-%d")

    df = df.where(pd.notna(df), None)

    # =====================================================
    # 4️⃣ ALIGNEMENT COLONNES SNOWFLAKE
    # =====================================================
    df.columns = [c.upper() for c in df.columns]

    # =====================================================
    # 5️⃣ CONNEXION SNOWFLAKE
    # =====================================================

    conn = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    role=os.environ.get("SNOWFLAKE_ROLE"),
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    database=os.environ["SNOWFLAKE_DATABASE"],
    schema=os.environ["SNOWFLAKE_SCHEMA"]
)


    # =====================================================
    # 6️⃣ INGESTION ULTRA RAPIDE
    # =====================================================
    success, nchunks, nrows, _ = write_pandas(
        conn,
        df,
        table_name="FACTURE",
        database="NEEMBA",
        schema="ML",
        auto_create_table=False
    )

    conn.close()

    if not success:
        raise Exception("write_pandas a échoué")

    print(f"✅ Ingestion Snowflake réussie : {nrows} lignes ({nchunks} batchs)")

except Exception as e:
    print("❌ Erreur ingestion :", e)
    sys.exit(1)

# =====================================================
# 7️⃣ RÉVEIL STREAMLIT (SI INGESTION OK)
# =====================================================
try:
    response = requests.get("https://genianeemba.streamlit.app", timeout=10)
    print(f"🚀 Streamlit réveillé (status {response.status_code})")
except Exception as e:
    print("⚠️ Impossible de réveiller Streamlit :", e)
