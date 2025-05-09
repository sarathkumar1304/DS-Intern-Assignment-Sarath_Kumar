import logging
from typing import Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

from xgboost import XGBRegressor
import optuna

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RegressionModelBuilder:
    def __init__(self):
        self.study_trials = 20  # You can parameterize this too

    def linear_regression(self, X_train, y_train, tune: bool = False)-> Pipeline:
        """Return pipeline: StandardScaler + Linear Regression (no tuning needed)."""
        logging.info("Building pipeline for Linear Regression (no tuning).")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def random_forest(self, X_train, y_train, tune: bool = False) -> Pipeline:
        """Return pipeline: StandardScaler + Random Forest, with Optuna tuning if enabled."""
        if not tune:
            logging.info("Building pipeline with default Random Forest Regressor.")
            model = RandomForestRegressor()
        else:
            logging.info("Running Optuna hyperparameter tuning for Random Forest...")

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }
                model = RandomForestRegressor(**params)
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", model)
                ])
                pipeline.fit(X_train, y_train)
                score = pipeline.score(X_train, y_train)  # You can change this to cross_val_score
                return score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.study_trials)
            logging.info(f"Best hyperparameters: {study.best_params}")

            model = RandomForestRegressor(**study.best_params)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", model)
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def decision_tree(self, X_train, y_train, tune: bool = False) -> Pipeline:
        """Return pipeline: StandardScaler + Decision Tree, with Optuna tuning if enabled."""
        if not tune:
            logging.info("Building pipeline with default Decision Tree Regressor.")
            model = DecisionTreeRegressor()
        else:
            logging.info("Running Optuna hyperparameter tuning for Decision Tree...")

            def objective(trial):
                params = {
                    "max_depth": trial.suggest_int("max_depth", 2, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }
                model = DecisionTreeRegressor(**params)
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", model)
                ])
                pipeline.fit(X_train, y_train)
                score = pipeline.score(X_train, y_train)
                return score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.study_trials)
            logging.info(f"Best hyperparameters: {study.best_params}")

            model = DecisionTreeRegressor(**study.best_params)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", model)
        ])
        pipeline.fit(X_train, y_train)
        return pipeline
    def xgboost(self, X_train, y_train, tune: bool = False) -> Pipeline:
        """Return pipeline: StandardScaler + XGBoost, with Optuna tuning if enabled."""
        if not tune:
            logging.info("Building pipeline with default XGBoost Regressor.")
            model = XGBRegressor()
        else:
            logging.info("Running Optuna hyperparameter tuning for XGBoost...")

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                }
                model = XGBRegressor(**params)
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", model)
                ])
                pipeline.fit(X_train, y_train)
                score = pipeline.score(X_train, y_train)
                return score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.study_trials)
            logging.info(f"Best hyperparameters: {study.best_params}")

            model = XGBRegressor(**study.best_params)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", model)
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def get_model(self, model_name: str, X_train, y_train, tune: bool = False) -> Pipeline:
        """Build model pipeline with optional Optuna tuning."""
        if model_name == "linear_regression":
            return self.linear_regression(X_train, y_train, tune)
        elif model_name == "random_forest":
            return self.random_forest(X_train, y_train, tune)
        elif model_name == "decision_tree":
            return self.decision_tree(X_train, y_train, tune)
        elif model_name == "xgboost":
            return self.xgboost(X_train, y_train, tune)

        else:
            logging.error(f"Model '{model_name}' not recognized.")
            raise ValueError(f"Model '{model_name}' not recognized.")
    def save_model(self, pipeline: Pipeline, model_name: str, output_dir: str = "models") -> None:
        """Save the trained pipeline to a pickle file."""
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{model_name}_pipeline.pkl")
        joblib.dump(pipeline, file_path)
        logging.info(f"Model pipeline saved to {file_path}")
