import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import mlflow
from mlflow import log_metric, log_param, log_artifacts


def main():

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("test experiment")

    np.random.seed(41)

    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.15)

    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA()), ("dt", DecisionTreeClassifier())])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)


    with mlflow.start_run():
        mlflow.log_param("test_size", 1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("balanced_accuracy", balanced_accuracy)
        mlflow.sklearn.log_model(pipe, "sk_models")


if __name__ == "__main__":
    main()


