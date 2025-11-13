import pandas as pd
import joblib
from dash import html, dcc, register_page, callback, Input, Output
import dash_bootstrap_components as dbc

register_page(
    __name__,
    path="/predict",
    name="Predict",
    title="ChurnApp - Predict",
    description="Predict churn for a custom customer profile"
)

# Cargar modelo
dt_model = joblib.load("models/churn_DecisionTreeClassifier.pkl")
dt_columns = joblib.load("models/churn_DecisionTreeClassifier_columns.pkl")
rf_model = joblib.load("models/churn_RandomForestClassifier.pkl")
rf_columns = joblib.load("models/churn_RandomForestClassifier_columns.pkl")
mlp_model = joblib.load("models/churn_MLPClassifier.pkl")
mlp_preprocessor = joblib.load("models/churn_MLPClassifier_preprocessor.pkl")

# ==========================
#  CAMPOS DEL FORMULARIO
# ==========================
def make_dropdown(id, label, options):
    return dbc.Col([
        html.Label(label, className="fw-bold mt-2"),
        dcc.Dropdown(id=id, options=[{"label": opt, "value": opt} for opt in options],
                     value=options[0], className="mb-2")
    ], width=4)

def make_input(id, label, value=0):
    return dbc.Col([
        html.Label(label, className="fw-bold mt-2"),
        dbc.Input(id=id, type="number", value=value, min=0, step=1, className="mb-2")
    ], width=4)

# Opciones
yes_no = ["Yes", "No"]
internet_opts = ["DSL", "Fiber optic", "No"]
service_opts = ["Yes", "No", "No internet service"]
phone_opts = ["Yes", "No", "No phone service"]
contract_opts = ["Month-to-month", "One year", "Two year"]
payment_opts = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
]

# ==========================
#  LAYOUT
# ==========================
layout = dbc.Container([
    html.Br(),
    html.H2("Churn Prediction Form", className="text-left mb-4 fw-bold text-secondary"),

    dbc.Row([
        # Columna izquierda: formulario
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("👤 Customer Information", className="text-primary fw-bold mb-3"),
                    dbc.Row([
                        make_dropdown("gender", "Gender", ["Female", "Male"]),
                        make_dropdown("SeniorCitizen", "Senior Citizen", yes_no),
                        make_dropdown("Partner", "Partner", yes_no),
                        make_dropdown("Dependents", "Dependents", yes_no)
                    ]),
                    html.Hr(),

                    html.H5("🌐 Services", className="text-primary fw-bold mb-3"),
                    dbc.Row([
                        make_dropdown("PhoneService", "Phone Service", phone_opts),
                        make_dropdown("MultipleLines", "Multiple Lines", phone_opts),
                        make_dropdown("InternetService", "Internet Service", internet_opts),
                        make_dropdown("OnlineSecurity", "Online Security", service_opts),
                        make_dropdown("OnlineBackup", "Online Backup", service_opts),
                        make_dropdown("DeviceProtection", "Device Protection", service_opts),
                        make_dropdown("TechSupport", "Tech Support", service_opts),
                        make_dropdown("StreamingTV", "Streaming TV", service_opts),
                        make_dropdown("StreamingMovies", "Streaming Movies", service_opts),
                    ]),
                    html.Hr(),

                    html.H5("💼 Account & Payment", className="text-primary fw-bold mb-3"),
                    dbc.Row([
                        make_dropdown("Contract", "Contract", contract_opts),
                        make_dropdown("PaperlessBilling", "Paperless Billing", yes_no),
                        make_dropdown("PaymentMethod", "Payment Method", payment_opts)
                    ]),
                    html.Hr(),

                    html.H5("💰 Charges", className="text-primary fw-bold mb-3"),
                    dbc.Row([
                        make_input("tenure", "Tenure (months)", 12),
                        make_input("MonthlyCharges", "Monthly Charges", 70),
                        make_input("TotalCharges", "Total Charges", 830)
                    ])
                ])
            ], className="shadow-sm p-4 mb-4 rounded-4 border-0 bg-white")
        ], width=8),
        # Columna derecha: predicción
        dbc.Col([
            html.Div(
                id="prediction-card",
                className="shadow-lg border-0 rounded-4 p-4 bg-light text-center",
                style={
                    "position": "sticky",
                    "top": "100px",
                    "backgroundColor": "white",
                    "boxShadow": "0px 4px 10px rgba(0,0,0,0.1)",
                    "borderRadius": "20px",
                    "minHeight": "220px"
                }
            )
        ], width=4),
    ])
], fluid=True)

# ==========================
#  CALLBACK
# ==========================
@callback(
    Output("prediction-card", "children"),
    [Input("gender", "value"),
     Input("SeniorCitizen", "value"),
     Input("Partner", "value"),
     Input("Dependents", "value"),
     Input("PhoneService", "value"),
     Input("MultipleLines", "value"),
     Input("InternetService", "value"),
     Input("OnlineSecurity", "value"),
     Input("OnlineBackup", "value"),
     Input("DeviceProtection", "value"),
     Input("TechSupport", "value"),
     Input("StreamingTV", "value"),
     Input("StreamingMovies", "value"),
     Input("Contract", "value"),
     Input("PaperlessBilling", "value"),
     Input("PaymentMethod", "value"),
     Input("tenure", "value"),
     Input("MonthlyCharges", "value"),
     Input("TotalCharges", "value")]
)
def update_prediction(*inputs):
    feature_names = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]

     # --- Decision Tree ---
    input_df = pd.DataFrame([inputs], columns=feature_names)
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=dt_columns, fill_value=0)

    y_prob_churn = dt_model.predict_proba(input_encoded)[0][1]

    if y_prob_churn > 0.5:
        churn_label = "CHURN"
        color = "danger"
    else:
        churn_label = "NO CHURN"
        color = "success"

    tree_card = dbc.Card([
        dbc.CardBody([
            html.H4(" 🌳 Decision Tree Prediction", className="text-muted mb-2"),
            html.H2(churn_label, className=f"text-{color} fw-bold"),
            dbc.Progress(
                value=y_prob_churn * 100,
                color=color,
                striped=True,
                animated=True,
                style={"height": "20px", "borderRadius": "10px"},
                className="my-3"
            ),
            html.P(f"Predicted Churn Probability: {y_prob_churn:.2%}",
                   className="text-secondary mt-2")
        ])
    ], className="shadow-lg border-0 rounded-4 p-4 bg-light mb-4")

    # --- Random Forest ---
    X_rf = pd.get_dummies(input_df)
    X_rf = X_rf.reindex(columns=rf_columns, fill_value=0)
    y_prob_rf = rf_model.predict_proba(X_rf)[0][1]

    if y_prob_rf > 0.5:
        rf_label = "CHURN"
        rf_color = "danger"
    else:
        rf_label = "NO CHURN"
        rf_color = "success"

    rf_card = dbc.Card([
        dbc.CardBody([
            html.H4("🌲 Random Forest Prediction", className="text-muted mb-2"),
            html.H2(rf_label, className=f"text-{rf_color} fw-bold"),
            dbc.Progress(
                value=y_prob_rf * 100,
                color=rf_color,
                striped=True,
                animated=True,
                style={"height": "20px", "borderRadius": "10px"},
                className="my-3"
            ),
            html.P(f"Predicted Churn Probability: {y_prob_rf:.2%}",
                   className="text-secondary mt-2")
        ])
    ], className="shadow-lg border-0 rounded-4 p-4 bg-light mb-4")

    # --- Neural Network (MLP) ---
    X_mlp = mlp_preprocessor.transform(input_df)
    y_prob_mlp = mlp_model.predict_proba(X_mlp)[0][1]

    if y_prob_mlp > 0.5:
        mlp_label = "CHURN"
        mlp_color = "danger"
    else:
        mlp_label = "NO CHURN"
        mlp_color = "success"

    mlp_card = dbc.Card([
        dbc.CardBody([
            html.H4("🧠 Neural Network (MLP) Prediction", className="text-muted mb-2"),
            html.H2(mlp_label, className=f"text-{mlp_color} fw-bold"),
            dbc.Progress(
                value=y_prob_mlp * 100,
                color=mlp_color,
                striped=True,
                animated=True,
                style={"height": "20px", "borderRadius": "10px"},
                className="my-3"
            ),
            html.P(f"Predicted Churn Probability: {y_prob_mlp:.2%}",
                   className="text-secondary mt-2")
        ])
    ], className="shadow-lg border-0 rounded-4 p-4 bg-light mb-4")


    # --- Devolver ambas tarjetas ---
    return html.Div([
        tree_card,
        rf_card,
        mlp_card
    ])