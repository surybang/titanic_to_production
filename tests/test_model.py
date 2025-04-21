import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.models.train_evaluate import evaluate_model
from src.pipeline.build_pipeline import create_pipeline


@pytest.fixture
def sample_data():
    # Génération de données synthétiques réalistes
    np.random.seed(42)

    # 1. Création des features numériques
    age = np.random.normal(loc=30, scale=15, size=100).clip(0.1, 80)  # Âge entre 0.1 et 80
    fare = np.random.gamma(shape=2, scale=30, size=100).clip(7.25, 512)  # Tarifs réalistes

    # 2. Features catégorielles
    embarked = np.random.choice(['S', 'C', 'Q'], size=100, p=[0.7, 0.2, 0.1])
    sex = np.random.choice(['male', 'female'], size=100, p=[0.65, 0.35])

    # 3. Création du DataFrame
    X = pd.DataFrame({
        'Age': age,
        'Fare': fare,
        'Embarked': embarked,
        'Sex': sex
    })

    # 4. Génération de la cible Survived de façon réaliste
    survival_prob = (
        3 * (sex == 'female') + 
        -0.05 * age + 
        0.02 * fare + 
        0.5 * (embarked == 'C') + 
        np.random.normal(0, 1, 100)
    )
    y = pd.Series((survival_prob > 0).astype(int), name='Survived')

    # Split en conservant les DataFrames
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Fixture pour un pipeline entraîné
@pytest.fixture
def trained_pipeline(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    pipe = create_pipeline(n_trees=10)
    pipe.fit(X_train, y_train)
    return pipe


# Tests pour evaluate_model
def test_evaluate_model_returns_tuple(trained_pipeline, sample_data):
    _, X_test, _, y_test = sample_data
    result = evaluate_model(trained_pipeline, X_test, y_test)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], np.ndarray)


def test_evaluate_model_confusion_matrix_shape(trained_pipeline, sample_data):
    _, X_test, _, y_test = sample_data
    _, matrix = evaluate_model(trained_pipeline, X_test, y_test)

    assert matrix.shape == (2, 2)


# Tests pour create_pipeline
def test_create_pipeline_returns_pipeline():
    pipe = create_pipeline(n_trees=5)
    assert isinstance(pipe, Pipeline)


def test_create_pipeline_structure():
    pipe = create_pipeline(n_trees=10)

    # Vérifie les étapes du pipeline
    assert 'preprocessor' in pipe.named_steps
    assert 'classifier' in pipe.named_steps

    # Vérifie le type du classifieur
    assert isinstance(pipe.named_steps['classifier'], RandomForestClassifier)


@pytest.mark.parametrize("n_trees, expected", [(5, 5), (20, 20)])
def test_create_pipeline_n_trees(n_trees, expected):
    pipe = create_pipeline(n_trees=n_trees)
    assert pipe.named_steps['classifier'].n_estimators == expected


def test_create_pipeline_max_depth_parameter():
    pipe = create_pipeline(n_trees=10, max_depth=5)
    assert pipe.named_steps['classifier'].max_depth == 5


# Test d'entraînement basique
def test_pipeline_can_train(trained_pipeline):
    assert hasattr(trained_pipeline.named_steps['classifier'], 'classes_')


# Test des prédictions
def test_pipeline_predictions(trained_pipeline, sample_data):
    _, X_test, _, y_test = sample_data
    predictions = trained_pipeline.predict(X_test)
    assert len(predictions) == len(y_test)
