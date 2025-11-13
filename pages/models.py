import pandas as pd
import json
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash import html, dcc, register_page
import dash_bootstrap_components as dbc

# ===========================================================
# PAGE REGISTRATION
# ===========================================================
register_page(
    __name__,
    path="/models",
    name="Models",
    title="ChurnApp - Models",
    description="Model performance and comparison"
)

# ===========================================================
# LOAD DATA
# ===========================================================
# Decision Tree
tree_metrics = pd.read_csv("models/DecisionTreeClassifier_metrics.csv")
with open("models/confusion_matrix_DecisionTreeClassifier.json", "r") as f:
    tree_conf_matrix = json.load(f)["DecisionTreeClassifier"]

# Random Forest
rf_metrics = pd.read_csv("models/RandomForestClassifier_metrics.csv")
with open("models/confusion_matrix_RandomForestClassifier.json", "r") as f:
    rf_conf_matrix = json.load(f)["RandomForestClassifier"]

# MLP Classifier
mlp_metrics = pd.read_csv("models/MLPClassifier_metrics.csv")
with open("models/confusion_matrix_MLPClassifier.json", "r") as f:
    mlp_conf_matrix = json.load(f)["MLPClassifier"]

# Feature importances (solo RF por ahora)
try:
    rf_importances = pd.read_csv("models/RandomForestClassifier_importances.csv")
except FileNotFoundError:
    rf_importances = None


# ===========================================================
# FUNCTIONS
# ===========================================================
def make_confusion_heatmap(matrix, title):
    """
    Confusion matrix with color intensity controlled by alpha (transparency).
    Green diagonal for TP/TN, red off-diagonal for FP/FN.
    """
    import numpy as np
    import plotly.graph_objects as go

    z = np.array(matrix, dtype=float)
    z_text = [[str(int(val)) for val in row] for row in z]

    # Normalizamos los valores para controlar la transparencia (0 → más claro, 1 → más opaco)
    z_norm = z / z.max()

    fig = go.Figure()

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            alpha = 0.2 + 0.8 * z_norm[i, j]  # base 0.2 → mínimo visible, hasta 1 → máximo
            if i == j:
                color = f"rgba(76,175,80,{alpha})"   # verde
            else:
                color = f"rgba(244,67,54,{alpha})"   # rojo

            # Dibujamos cada celda como rectángulo coloreado
            fig.add_shape(
                type="rect",
                x0=j - 0.5, x1=j + 0.5,
                y0=i - 0.5, y1=i + 0.5,
                fillcolor=color,
                line=dict(color="white", width=2),
            )
            # Añadimos el texto centrado
            fig.add_annotation(
                x=j, y=i, text=z_text[i][j],
                showarrow=False,
                font=dict(color="white", size=16, family="Segoe UI"),
                xanchor="center", yanchor="middle"
            )

    # Ejes y layout igual que antes
    fig.update_xaxes(
        tickvals=[0, 1],
        ticktext=["No", "Yes"],
        title_text="Predicted",
        side="bottom",
        range=[-0.5, 1.5],
        showgrid=False,
        zeroline=False
    )
    fig.update_yaxes(
        tickvals=[0, 1],
        ticktext=["No", "Yes"],
        title_text="True",
        autorange="reversed",
        range=[-0.5, 1.5],
        showgrid=False,
        zeroline=False
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=450, height=400,
        margin=dict(l=60, r=60, t=60, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI", size=14),
        showlegend=False
    )

    return fig

# === CREAR GRÁFICO DE COMPARACIÓN ===
def make_model_comparison_chart(tree_metrics, rf_metrics, mlp_metrics):
    # Obtener métricas (fila 0 de cada CSV)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    models = ["Decision Tree", "Random Forest", "Neural Network (MLP)"]

    tree_values = [tree_metrics[m].iloc[0] for m in metrics]
    rf_values = [rf_metrics[m].iloc[0] for m in metrics]
    mlp_values = [mlp_metrics[m].iloc[0] for m in metrics]

    # Crear gráfico de barras agrupadas
    fig = go.Figure(data=[
        go.Bar(name="Decision Tree 🌳", x=metrics, y=tree_values, marker_color="rgba(76,175,80,0.8)"),
        go.Bar(name="Random Forest 🌲", x=metrics, y=rf_values, marker_color="rgba(33,150,243,0.8)"),
        go.Bar(name="Neural Network (MLP) 🧠", x=metrics, y=mlp_values, marker_color="rgba(233,30,99,0.8)")
    ])

    fig.update_layout(
        title="Based on metrics",
        title_x=0.5,
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1]),
        xaxis=dict(title="Metric"),
        font=dict(family="Segoe UI"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=60, t=60, b=60),
        height=450,
        legend=dict(
        orientation="h",
        y=-0.25,
        x=0.5,
        xanchor="center",
        font=dict(size=13),
        bgcolor="rgba(255,255,255,0.3)",
        bordercolor="rgba(0,0,0,0)",
    ))

    return fig

def make_feature_importance_chart(df, title):
    """Gráfico horizontal de importancia de variables."""
    if df is None or "Feature" not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=f"{title} (no data available)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
        )
        return fig

    df = df.sort_values("Importance", ascending=True).tail(10)
    fig = go.Figure(
        go.Bar(
            x=df["Importance"],
            y=df["Feature"],
            orientation="h",
            marker=dict(
                color=df["Importance"],
                colorscale="Blues",
                line=dict(color="rgba(0,0,0,0)")
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>"
        )
    )
    fig.update_layout(
        title=title,
        height=300,
        margin=dict(l=60, r=20, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI"),
        showlegend=False
    )
    return fig


def make_metrics_card(model_name, metrics, icon):
    """Tarjeta con métricas clave de cada modelo."""
    row = metrics.iloc[0]
    return dbc.Card([
        dbc.CardHeader(
            html.H5(f"{icon} {model_name}", className="text-center fw-bold text-secondary")
        ),
        dbc.CardBody([
            html.P(f"Accuracy: {row['Accuracy']:.2%}", className="mb-1"),
            html.P(f"Precision: {row['Precision']:.2%}", className="mb-1"),
            html.P(f"Recall: {row['Recall']:.2%}", className="mb-1"),
            html.P(f"F1-Score: {row['F1']:.2%}", className="mb-1"),
            html.P(f"AUC: {row['AUC']:.2%}", className="mb-1"),
        ])
    ], className="shadow-sm", style={
        "borderRadius": "15px",
        "backgroundColor": "#f8f9fa",
        "minHeight": "210px"
    })


# ===========================================================
# LAYOUT
# ===========================================================
layout = dbc.Container([
    html.Br(),
    html.H2("Model Performance", className="text-center fw-bold mb-4 text-secondary"),

    # MÉTRICAS COMPARATIVAS (3 modelos)
    dbc.Row([
        dbc.Col(make_metrics_card("Decision Tree", tree_metrics, "🌳"), width=3),
        dbc.Col(make_metrics_card("Random Forest", rf_metrics, "🌲"), width=3),
        dbc.Col(make_metrics_card("Neural Network (MLP)", mlp_metrics, "🧠"), width=3),
    ], justify="center"),

    html.Br(), html.Br(),

    # === NUEVO BLOQUE EN EL LAYOUT ===
    html.H4("Model Performance Comparison", className="text-center text-secondary mt-5"),
    dcc.Graph(
        figure=make_model_comparison_chart(tree_metrics, rf_metrics, mlp_metrics),
        config={"displayModeBar": False}
    ),
    html.Br(),

    html.Br(), html.Br(),

    # MATRICES DE CONFUSIÓN
    html.H4("Confusion Matrices", className="text-center text-secondary mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(
            figure=make_confusion_heatmap(tree_conf_matrix, "Decision Tree - Confusion Matrix"),
            config={"displayModeBar": False}
        ), width=4),
        dbc.Col(dcc.Graph(
            figure=make_confusion_heatmap(rf_conf_matrix, "Random Forest - Confusion Matrix"),
            config={"displayModeBar": False}
        ), width=4),
        dbc.Col(dcc.Graph(
            figure=make_confusion_heatmap(mlp_conf_matrix, "Neural Network (MLP) - Confusion Matrix"),
            config={"displayModeBar": False}
        ), width=4)
    ], justify="center"),

    html.Br(), html.H4("Feature Importances", className="text-center text-secondary mb-4"),

    # FEATURE IMPORTANCE – solo RF por ahora
    dbc.Row([
        dbc.Col(dcc.Graph(
            figure=make_feature_importance_chart(
                rf_importances,
                "Top 10 Feature Importances – Random Forest"
            ),
            config={"displayModeBar": False}
        ), width=8)
    ], justify="center")
], fluid=True)
