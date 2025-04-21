"""
Prédiction de la survie d'un individu sur le titanic
"""

import os
import argparse
from dotenv import load_dotenv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix


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

# FUNCTIONS -------------------------------------
def create_pipeline(
    n_trees: int,
    numeric_features: list = ["Age", "Fare"],
    categorical_features: list = ["Embarked", "Sex"],
    max_depth: int = None,
    max_features: str = "sqrt",
) -> Pipeline :
    """
    Create a pipeline for preprocessing and model definition

    Params:
        n_trees : The number of trees in the random forest
        numeric_features: The numeric features used in the pipeline
        categorical_features: The categorical features used in the pipeline
        max_depth : The maximum depth to the forest. Default is "None"
        max_features: the maximum number of features used when looking for \
            the best split. Default is "sqrt"

    Returns:
        pipe : the Pipeline object

    """
    # Variables quantitatives
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    # Variables qualitatives
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder()),
        ]
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("Preprocessing numerical", numeric_transformer, numeric_features),
            (
                "Preprocessing categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    # Pipeline
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=20)),
        ]
    )
    return pipe


def evaluate_model(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> tuple:
    """
    Evaluate the model by calculating the score and confusion matrix

    Params:
        pipe: the trained pipeline object
        X_test: the test data
        y_test: the true labels from test data

    Returns:
        tuple: A tuple containing the score and confusion matrix
    """
    score = pipe.score(X_test, y_test)
    matrix = confusion_matrix(y_test, pipe.predict(X_test))
    return score, matrix


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
