### ACTIVAR ENTORNO VIRTUAL: venv\Scripts\activate

# app.py
import dash
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

# Crea la app
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUX],  # tema oscuro moderno
    use_pages=True  # permite multipáginas
)

# 👇 NUEVO: esto es lo que usará gunicorn
server = app.server

app.title = "ChurnApp"

# Layout principal (navbar + contenido dinámico)
app.layout = dbc.Container([
    dcc.Location(id="url"),  # controla la URL/página activa

    # Encabezado fijo (Navbar)
    dbc.NavbarSimple(
        brand="ChurnApp",
        brand_href="/",
        color="primary",
        dark=True,
        fluid=True,
        children=[
            dbc.NavItem(dbc.NavLink("🏠 Home", href="/")),
            dbc.NavItem(dbc.NavLink("📊 EDA", href="/eda")),
            dbc.NavItem(dbc.NavLink("🤖 Models", href="/models")),
            dbc.NavItem(dbc.NavLink("📦 Data", href="/data")),
            dbc.NavItem(dbc.NavLink("🔮 Predict", href="/predict"))
        ]
    ),

    html.Br(),
    html.Div(dcc.Loading(dash.page_container, type="cube")),
], fluid=True)


#if __name__ == "__main__":
#    app.run_server(debug=True)  # 🔥 modo auto-reload activado

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
