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
