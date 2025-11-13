from dash import html, dcc, register_page
import dash_bootstrap_components as dbc

register_page(__name__, path="/")

layout = dbc.Container(
    [
        html.Div(
            [
                html.H1("Welcome to ChurnApp", className="text-center text-light mt-5"),
                html.P(
                    "An intelligent dashboard to explore customer churn, visualize model performance, "
                    "and predict churn probability for new clients.",
                    className="lead text-center text-secondary"
                ),
                html.Br(),
                dbc.Row(
                    dbc.Col(
                        dbc.Button(
                            "Get Started",
                            href="/eda",
                            color="primary",
                            size="lg",
                            className="d-block mx-auto"
                        ),
                        width=12
                    )
                )
            ],
            className="home-container",  # 👈 añade esto
            style={
                "height": "80vh",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center",
            },
        )
    ],
    fluid=True
)
