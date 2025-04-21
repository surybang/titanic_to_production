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
if jeton_api.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")

# Import data
os.chdir("/home/onyxia/work/titanic_to_production")
TrainingData = pd.read_csv("data.csv")
TrainingData.head()
TrainingData["Ticket"].str.split("/").str.len()
TrainingData["Name"].str.split(",").str.len()
TrainingData.isnull().sum()

## Un peu d'exploration et de feature engineering
### Statut socioéconomique
fig, axes = plt.subplots(
    1, 2, figsize=(12, 6)
)  # layout matplotlib 1 ligne 2 colonnes taile 16*8
fig1_pclass = sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(
    "fréquence des Pclass"
)
fig2_pclass = sns.barplot(
    data=TrainingData, x="Pclass", y="Survived", ax=axes[1]
).set_title("survie des Pclass")


### Age
sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(
    "Distribution de l'âge"
)
plt.show()

## Encoder les données imputées ou transformées.
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

# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

# On _split_ notre _dataset_ d'apprentisage pour faire de la validation croisée \
# une partie pour apprendre une partie pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([X_train, y_train], axis=1).to_csv("train.csv")
pd.concat([X_test, y_test], axis=1).to_csv("test.csv")

# Random Forest
# Ici demandons d'avoir 20 arbres
pipe.fit(X_train, y_train)


# calculons le score sur le dataset d'apprentissage et sur le dataset de test \
# (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction
rdmf_score = pipe.score(X_test, y_test)
rdmf_score_tr = pipe.score(X_train, y_train)
print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")
print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_test)))
