import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from dash import html, dcc, register_page, callback, Input, Output
import dash_bootstrap_components as dbc

# ==============================
#  REGISTRO DE LA PÁGINA
# ==============================
register_page(
    __name__,
    path="/eda",
    name="EDA",
    title="ChurnApp - EDA",
    description="Explore customer churn data interactively"
)

# ==============================
#  CARGA DE DATOS
# ==============================
df = pd.read_csv("pages/Telco-Customer-Churn.csv")

# Limpieza y tipo de datos
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
df["SeniorCitizen"] = df["SeniorCitizen"].replace({1: "Yes", 0: "No"})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Variables disponibles
variables = [col for col in df.columns if col not in ["customerID", "Churn"]]
numeric_vars = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ==============================
#  LAYOUT
# ==============================
layout = dbc.Container([
    html.Br(),
    html.H2("Exploratory Data Analysis", className="text-center mb-4"),

    # Dropdowns
    dbc.Row([
        dbc.Col([
            html.Label("Select variable to analyze:", className="fw-bold"),
            dcc.Dropdown(
                id="eda-variable",
                options=[{"label": var, "value": var} for var in variables],
                value="InternetService",
                clearable=False,
                className="mb-3"
            ),
        ], width=4),

        dbc.Col([
            html.Label("Graph type (for numeric vars):", className="fw-bold"),
            dcc.Dropdown(
                id="graph-type",
                options=[
                    {"label": "Boxplot", "value": "box"},
                    {"label": "Histogram", "value": "hist"}
                ],
                value="box",
                clearable=False,
                className="mb-3"
            ),
        ], width=3)
    ], justify="center"),

    # Main chart
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="eda-graph", config={"displayModeBar": False})
        ], width=10)
    ], justify="center"),

    html.Br(),

    # KPIs
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Total Customers", className="card-title text-center"),
                html.H3(id="total-customers", className="text-center text-primary fw-bold")
            ])
        ], color="light", className="shadow-sm"), width=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Churn Rate", className="card-title text-center"),
                html.H3(id="churn-rate", className="text-center text-danger fw-bold")
            ])
        ], color="light", className="shadow-sm"), width=3),
    ], justify="center", className="mt-4"),

    html.Br(), html.Br(),

    # Correlation matrix
    html.H4("Correlation Matrix (Numerical Variables)", className="text-center mt-5 mb-4"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="corr-matrix")
        ], width=10)
    ], justify="center"),
], fluid=True)

# ==============================
#  CALLBACKS
# ==============================

@callback(
    Output("eda-graph", "figure"),
    Input("eda-variable", "value"),
    Input("graph-type", "value")
)
def update_graph(selected_var, graph_type):
    churn_labels = {0: "No", 1: "Yes"}

    # Variable categórica
    if df[selected_var].dtype == "object" or selected_var == "SeniorCitizen":
        fig = px.histogram(
            df, x=selected_var, color=df["Churn"].map(churn_labels),
            barmode="group", 
            title=f"Distribution of {selected_var} by Churn",
            color_discrete_map={"No": "#00BFFF", "Yes": "#FF6B6B"}
        )
        fig.update_layout(
            xaxis_title=selected_var,
            yaxis_title="Count",
            legend_title_text="Churn"
        )

    # Variable numérica
    else:
        if graph_type == "box":
            fig = px.box(
                df, x=df["Churn"].map(churn_labels), y=selected_var,
                color=df["Churn"].map(churn_labels),
                color_discrete_map={"No": "#00BFFF", "Yes": "#FF6B6B"},
                title=f"Boxplot of {selected_var} by Churn"
            )
            fig.update_layout(
                xaxis_title="Churn",
                yaxis_title=selected_var,
                legend_title_text="Churn"
            )

        else:
            fig = px.histogram(
                df, x=selected_var, color=df["Churn"].map(churn_labels),
                barmode="overlay", nbins=40,
                color_discrete_map={"No": "#00BFFF", "Yes": "#FF6B6B"},
                title=f"Histogram of {selected_var} by Churn"
            )
            fig.update_layout(
                xaxis_title=selected_var,
                yaxis_title="Count",
                legend_title_text="Churn"
            )

    # Estilo general
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI"),
        title_x=0.5
    )
    return fig


@callback(
    Output("total-customers", "children"),
    Output("churn-rate", "children"),
    Input("eda-variable", "value")
)
def update_metrics(_):
    total = len(df)
    churn_rate = df["Churn"].mean() * 100
    return f"{total:,}", f"{churn_rate:.1f}%"


@callback(
    Output("corr-matrix", "figure"),
    Input("eda-variable", "value")
)
def update_corr_matrix(_):
    num_vars = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    corr = df[num_vars].corr()

    z = corr.values
    x = corr.columns.tolist()
    y = corr.columns.tolist()

    fig = ff.create_annotated_heatmap(
        z=z,
        x=x, y=y,
        annotation_text=np.round(z, 2),
        colorscale="Blues",
        showscale=True
    )

    fig.update_layout(
        title="Correlation Matrix (Churn + Key Numeric Variables)",
        title_x=0.5,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI"),
        margin=dict(l=80, r=80, t=80, b=80)
    )

    return fig
