import pandas as pd
import joblib
from dash import html, dcc, dash_table, register_page, Input, Output, callback
import dash_bootstrap_components as dbc

# ==============================
#  REGISTRO DE LA PÁGINA
# ==============================
register_page(
    __name__,
    path="/data",
    name="Data",
    title="ChurnApp - Data",
    description="Explore customer data and model predictions"
)

# ==============================
#  CARGA DE DATOS Y MODELOS
# ==============================
df = pd.read_csv("pages/Telco-Customer-Churn.csv")

# Procesamiento igual que antes
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
df["SeniorCitizen"] = df["SeniorCitizen"].replace({1: "Yes", 0: "No"})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# --- Modelos ---
MODELS = {
    "Decision Tree": {
        "model": joblib.load("models/churn_DecisionTreeClassifier.pkl"),
        "columns": joblib.load("models/churn_DecisionTreeClassifier_columns.pkl"),
    },
    "Random Forest": {
        "model": joblib.load("models/churn_RandomForestClassifier.pkl"),
        "columns": joblib.load("models/churn_RandomForestClassifier_columns.pkl"),
    },
    "Neural Network (MLP)": {
        "model": joblib.load("models/churn_MLPClassifier.pkl"),
        "columns": joblib.load("models/churn_RandomForestClassifier_columns.pkl"),  # usa mismas columnas
    }
}

# ==============================
#  LAYOUT
# ==============================
layout = dbc.Container([
    html.Br(),
    html.H2("Customer Data & Predictions", className="text-center mb-4 fw-bold text-secondary"),

    # --- Tarjetas arriba ---
    dbc.Row(id="prediction-cards", justify="center", className="mb-4"),

    # --- Tabla de datos ---
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(
                id="data-table",
                data=df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in df.columns],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "center",
                    "padding": "8px",
                    "fontFamily": "Segoe UI",
                    "fontSize": "13px"
                },
                style_header={
                    "backgroundColor": "#0074D9",
                    "color": "white",
                    "fontWeight": "bold"
                },
                row_selectable="single",
                selected_rows=[],
                filter_action="native",
                sort_action="native"
            )
        ], width=12)
    ]),
], fluid=True)

# ==============================
#  CALLBACK
# ==============================
@callback(
    Output("prediction-cards", "children"),
    Input("data-table", "selected_rows")
)
def update_predictions(selected_rows):
    if not selected_rows:
        return dbc.Alert("Select a row to view predictions.", color="secondary", className="text-center")

    row_idx = selected_rows[0]
    selected_data = df.iloc[[row_idx]].copy()

    # Valor real
    real_churn = "Yes" if selected_data["Churn"].values[0] == 1 else "No"
    color_real = "info" if real_churn == "Yes" else "secondary"

    # --- Tarjetas de predicciones ---
    cards = [
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("Real Churn", className="text-center text-muted"),
                html.H3(real_churn, className=f"text-center text-{color_real} fw-bold"),
            ])
        ], className="shadow-sm border-0 rounded-4"), width=3)
    ]

    # --- Para cada modelo ---
    for model_name, data in MODELS.items():
        model = data["model"]
        feature_columns = data["columns"]

        encoded = pd.get_dummies(selected_data)
        for col in feature_columns:
            if col not in encoded.columns:
                encoded[col] = 0
        encoded = encoded[feature_columns]

        pred = model.predict(encoded)[0]
        pred_label = "Yes" if pred == "Yes" else "No"
        color_pred = "danger" if pred_label == "Yes" else "success"

        cards.append(
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6(model_name, className="text-center text-muted"),
                    html.H3(pred_label, className=f"text-center text-{color_pred} fw-bold"),
                ])
            ], className="shadow-sm border-0 rounded-4"), width=3)
        )

    return dbc.Row(cards, justify="center", className="mb-3")
