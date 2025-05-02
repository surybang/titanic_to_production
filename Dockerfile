# Étape de construction
FROM ubuntu:22.04 as builder

# Installer les dépendances système
RUN apt-get update && \
    apt-get install -y curl python3-pip && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt uv.lock ./
COPY app ./app
COPY docs ./docs
COPY data/derived ./data/derived

# Configurer uv et installer les dépendances
RUN . ~/.cargo/env && \
    uv pip install -r requirements.txt --python /usr/bin/python3

# Étape d'exécution finale
FROM ubuntu:22.04

# Installer les dépendances runtime minimales
RUN apt-get update && \
    apt-get install -y python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier depuis le builder
COPY --from=builder /app /app
COPY --from=builder /root/.cache /root/.cache

# Créer les répertoires nécessaires
RUN mkdir -p models logs

# Configurer les variables d'environnement
ENV JETON_API=$JETON_API
ENV data_path=/app/data/derived
ENV PYTHONPATH=/app
ENV PATH="/root/.local/bin:${PATH}"

# Exposer le port
EXPOSE 8000

# Commande d'exécution
CMD ["bash", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 && ./app/run.sh"]