# charts.py

import plotly.express as px

def build_line_chart(
    df,
    x_col,
    y_col,
    title,
    y_label="Valeur"
):
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        markers=True
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        title_font_size=16,
        xaxis_title="Période",
        yaxis_title=y_label,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig




def build_multi_line_chart(
    df,
    x_col,
    y_col,
    color_col,
    title,
    y_label
):
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        markers=True,
        title=title,
        text=y_col
    )

    # Format des valeurs affichées sur les points
    fig.update_traces(
        texttemplate="%{y:.2s} €",  # ex: 8.9M €
        textposition="top center"
    )

    # Mise en forme du graphique
    fig.update_layout(
        yaxis=dict(
            title=y_label,
            tickformat="~s"  # ex: 1k, 1M, 10M
        ),
        xaxis_title="Mois",
        hovermode="x unified",
        legend_title_text="Client",
        template="plotly_white"
    )

    return fig



def build_evolution_title(clients: list) -> str:
    if not clients:
        return "📈 Évolution mensuelle des ventes (global)"

    if len(clients) == 1:
        return f"📈 Évolution mensuelle des ventes – {clients[0]}"

    # Cas comparaison
    clients_str = " vs ".join(clients)
    return f"📈 Évolution mensuelle des ventes – {clients_str}"



"""
def build_multi_line_chart(
    df,
    x_col,
    y_col,
    color_col,
    title,
    y_label
):
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        markers=True,
        title=title
    )

    fig.update_layout(
        yaxis_title=y_label,
        xaxis_title="Mois",
        hovermode="x unified",
        legend_title_text="Client",
        template="plotly_white"
    )

    return fig
"""
