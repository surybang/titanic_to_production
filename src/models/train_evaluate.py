import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline


def evaluate_model(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple:
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
