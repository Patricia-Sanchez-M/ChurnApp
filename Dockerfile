# Imagen base con Python 3.10 (compatible con pandas y scikit-learn)
FROM python:3.10-slim

# Instalamos compiladores básicos por si alguna librería lo necesita
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# Copiamos primero requirements.txt para aprovechar la cache de Docker
COPY requirements.txt .

# Instalamos dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos TODO el código de la app
COPY . .

# Render te da la variable de entorno PORT; la usamos en gunicorn
CMD gunicorn app:server --workers=1 --threads=4 --timeout=180 --bind 0.0.0.0:${PORT:-8050}
