# Utiliser une image Python officielle
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY app ./app
COPY docs ./docs

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger les données
RUN mkdir -p data/derived && \
    curl -L https://minio.lab.sspcloud.fr/fabienhos/titanic/data/raw/data.csv -o data/derived/raw_data.csv

# Variables d'environnement
ENV JETON_API=$JETON_API
ENV PYTHONPATH=/app

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["bash", "-c", "./app/run.sh"]