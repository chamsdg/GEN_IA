# ventes.py
import pandas as pd

def build_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date_facture_dt" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df = df.dropna(subset=["date_facture_dt"])

    df["ANNEE_MOIS"] = (
        df["date_facture_dt"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        df.groupby("ANNEE_MOIS", as_index=False)
          .agg(total_sales=("GFD_MONTANT_VENTE_EUROS", "sum"))
          .sort_values("ANNEE_MOIS")
    )

    return monthly


def normalize_client(x: str) -> str:
    return (
        x.lower()
        .replace(" sa", "")
        .replace(" sarl", "")
        .replace(" ltd", "")
        .replace(".", "")
        .strip()
    )



def build_monthly_sales_by_client(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    if "ANNEE_MOIS" not in df.columns:
        df["ANNEE_MOIS"] = (
            pd.to_datetime(df["date_facture_dt"], errors="coerce")
            .dt.to_period("M")
            .astype(str)
        )

    df["client_norm"] = (
        df["RAISON_SOCIALE"]
        .astype(str)
        .str.lower()
        .str.replace(" sa", "", regex=False)
        .str.replace(" sarl", "", regex=False)
        .str.replace(" ltd", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.strip()
    )

    monthly = (
        df.groupby(
            ["ANNEE_MOIS", "client_norm"],
            as_index=False
        )
        .agg(
            total_sales=("GFD_MONTANT_VENTE_EUROS", "sum")
        )
        .sort_values("ANNEE_MOIS")
    )

    return monthly

