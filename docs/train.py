"""Prédiction de la survie d'un individu sur le titanic"""

import os
import sys
import argparse
from dotenv import load_dotenv

import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger
from pathlib import Path

from titanicml import create_pipeline, evaluate_model

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> \
    <cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

logger.add(
    "logs/titanic.log",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {module}:{function}:{line} - {message}",
    level="DEBUG",
)

# =============================================================================
# ENVIRONNEMENT CONFIG
# =============================================================================
load_dotenv()


@logger.catch  # Auto-log exceptions
def main():
    parser = argparse.ArgumentParser(description="Combien d'arbres ?")
    parser.add_argument(
        "--n_trees", type=int, default=20, help="Nombre d'arbres à utiliser"
    )
    args = parser.parse_args()
    logger.info(f"Nombre d'arbres utilisés pour la classification : {args.n_trees}")

    n_trees = args.n_trees

    URL_RAW = "https://minio.lab.sspcloud.fr/fabienhos/titanic/data/raw/data.csv"
    jeton_api = os.environ.get("JETON_API", "")
    data_path = os.environ.get("data_path", URL_RAW)
    data_train_path = os.environ.get("train_path", "data/derived/train.parquet")
    data_test_path = os.environ.get("test_path", "data/derived/test.parquet")

    if not data_path:
        logger.error("DATA_PATH non configuré dans l'environnement !")
        sys.exit(1)

    if jeton_api.startswith("$"):
        logger.success("API token configuré correctement")
    else:
        logger.warning("API token non configuré - mode limité activé")

    # =========================================================================
    # IMPORT DATA
    # =========================================================================
    p = Path("data/derived/")
    p.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Chargement des données depuis {}", data_path)
        TrainingData = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.critical("Fichier de données introuvable : {}", data_path)
        raise

    y = TrainingData["Survived"]
    X = TrainingData.drop("Survived", axis="columns")

    # =========================================================================
    # DATA PROCESSING
    # =========================================================================
    logger.debug("Séparation train/test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    pd.concat([X_train, y_train], axis=1).to_parquet(data_train_path)
    pd.concat([X_test, y_test], axis=1).to_parquet(data_test_path)
    logger.info(f"Données sauvegardées dans {data_train_path} et {data_test_path}")

    # =========================================================================
    # PIPELINE EXECUTION
    # =========================================================================
    logger.info("Création de la pipeline avec {} arbres", n_trees)
    pipe = create_pipeline(n_trees)

    logger.info("Entraînement du modèle...")
    pipe.fit(X_train, y_train)

    logger.info("Évaluation du modèle...")
    score, matrix = evaluate_model(pipe, X_test, y_test)

    logger.success("Performance du modèle : {:.1%}", score)
    logger.debug("Matrice de confusion :\n{}", matrix)


if __name__ == "__main__":
    main()
