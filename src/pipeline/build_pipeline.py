from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def create_pipeline(
    n_trees: int,
    numeric_features: list = ["Age", "Fare"],
    categorical_features: list = ["Embarked", "Sex"],
    max_depth: int = None,
    max_features: str = "sqrt",
) -> Pipeline:
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
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_trees, max_depth=max_depth, max_features=max_features
                ),
            ),
        ]
    )
    return pipe
