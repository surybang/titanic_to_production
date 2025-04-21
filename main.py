"""
Prédiction de la survie d'un individu sur le titanic
"""

import os
import argparse
from dotenv import load_dotenv

import pandas as pd
from sklearn.model_selection import train_test_split

from build_pipeline import create_pipeline
from train_evaluate import evaluate_model

# ENVIRONNEMENT CONFIG ---------------------------
load_dotenv()

parser = argparse.ArgumentParser(description="Combien d'arbres ?")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Nombre d'arbres à utiliser"
)
args = parser.parse_args()
print(f"Nombre d'arbres utilisés pour la classification : {args.n_trees}")

n_trees = args.n_trees
MAX_DEPTH = None
MAX_FEATURES = "sqrt"

jeton_api = os.environ.get("JETON_API", "")
data_path = os.environ.get("DATA_PATH")

if jeton_api.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")

# IMPORT DATA --------------------------
TrainingData = pd.read_csv(data_path)

y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([X_train, y_train], axis=1).to_csv("train.csv")
pd.concat([X_test, y_test], axis=1).to_csv("test.csv")

# PIPELINE ----------------------------
pipe = create_pipeline(n_trees)

# RUNNING
pipe.fit(X_train, y_train)
score, matrix = evaluate_model(pipe, X_test, y_test)
print(f"{score:.1%} de bonnes réponses sur les données de test pour validation")
print(20 * "-")
print("matrice de confusion")
print(matrix)
