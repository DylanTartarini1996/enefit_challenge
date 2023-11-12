import numpy as np
import pandas as pd
import mlflow
import mlflow.catboost
from catboost import CatBoostRegressor
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

import optuna
from optuna.integration.mlflow import MLflowCallback
import joblib

from typing import Optional, Dict, Tuple
from pathlib import Path
from enefit_challenge.utils.dataset import load_enefit_training_data

from enefit_challenge.models.forecaster import Forecaster
import warnings

warnings.filterwarnings("ignore")


class CatBoostForecaster(Forecaster):
    def __init__(self)-> None:
        pass

    def fit_model(
        self,  
        X: pd.DataFrame,
        y: pd.Series,
        categorical_features: list=[],
        params:Optional[Dict]=None,
    ) -> CatBoostRegressor:
        """
        Trains a `CatBoostRegressor` with a L1 loss

        -------     
        params:
        -------
        `X`:`pd.DataFrame`
            Features to use for fitting
        `y`:`pd.Series`
            Target variable
        `categorical_features`: `list`
            list of categorical features in the dataset
        `params`: `Optional[Dict]`
            optional dictionary of parameters to use
        -------     
        returns:
        -------
        fitted `CatBoostRegressor`

        """
        model = CatBoostRegressor(
            n_estimators=100, 
            objective='MAE',
            thread_count=1,
            bootstrap_type =  "Bernoulli",
            sampling_frequency= 'PerTree',
            verbose=0,
            cat_features=categorical_features,
            leaf_estimation_iterations=1
        )
        if params:
            model.set_params(**params)

        model.fit(X, y)
    
        return model
    
    def fit_and_test_fold(
        self, 
        params:Dict,
        X: pd.DataFrame, 
        y: pd.Series, 
        year_month_train, 
        year_month_test,
        categorical_features: list=[],
        experiment_name: str="catboost",
        artifact_path: str="catboost_model",
        metrics: list=["mae"]
    ) -> float:
        """
        Used for cross validation on different time splits
        """
        
        first_dates_month = pd.to_datetime(X[['year', 'month']].assign(day=1))
        train_index = first_dates_month.isin(year_month_train)
        test_index = first_dates_month.isin(year_month_test)
        X_train = X[train_index];X_test = X[test_index]
        y_train = y[train_index]; y_test = y[test_index]
        # fit model on training data
        model = self.fit_model(
            X_train, 
            y_train, 
            categorical_features,
            params
        )
        # generate predictions
        y_test_pred = model.predict(X_test)
        self.signature = infer_signature(X_train, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)

        mlflow.catboost.log_model(
            model, 
            artifact_path=artifact_path,
            signature=self.signature
        )
        mlflow.log_params(params)

        return mae

    def train_model(
        self, 
        train_df: pd.DataFrame, 
        target_col: str,
        model_name: str,
        exclude_cols: list=[],
        categorical_features: list=[],
        experiment_name: str="catboost",
        artifact_path: str="catboost_model",
        params: Optional[Dict]=None,
        metrics: list=["MAE"]
    ) -> Dict:
        """ 
        Takes an instance of `CatBoostRegressor` model and tracks the hyperparameter tuning
        experiment on training set using `mlflow` and `optuna`.  
        Registers the best version of the model according to a specified metric
        
        -------     
        params:
        -------
        `experiment_name`: `str`
            the name of the experiment used to store runs in mlflow, 
            as well as the name of the optuna study
        `model_name`: `str`
            the name the final model will have in the registry
        `train_df`: `pd.DataFrame`
            the training data for the model.
        `target_col`: `str`
            the time-series target column
        `exclude_cols`: `list`  
            columns in dataset that should not be used
        `categorical_features`: `list`
            list of categorical features in the dataset
        `artifact_path`: `str`
            the path pointing to the mlflow artifact
        `metrics`: `list`
            list of the metrics to track in the mlflow experiment run.
        `params`: `Optional[Dict]`
            optional dictionary of parameters to use
        """
        self.model_name = model_name
        X = train_df.drop([target_col] + exclude_cols, axis=1)
        y = train_df[target_col]
        # unique year-month combinations -> to be used in cross-validation
        timesteps = np.sort(np.array(
            pd.to_datetime(X[['year', 'month']].assign(day=1)).unique().tolist()
        ))

        # define mlflow callback Handler for optuna 
        mlflc = MLflowCallback(
            metric_name="MAE",
        )
    
        @mlflc.track_in_mlflow() # decorator to allow mlflow logging
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.95,log=True),
                'depth': trial.suggest_int('depth', 3, 10, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg',1e-8,100,log=True),
                'model_size_reg': trial.suggest_float('model_size_reg',1e-8,100,log=True),
                'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.1, 1),
                'subsample': trial.suggest_float("subsample", 0.5, 1)
            }
            cv = TimeSeriesSplit(n_splits=3) # cross validation
            cv_mae = [None]*3
            for i, (train_index, test_index) in enumerate(cv.split(timesteps)):
                cv_mae[i] = self.fit_and_test_fold(
                    params,
                    X, 
                    y, 
                    timesteps[train_index], 
                    timesteps[test_index],
                    categorical_features
                )
            trial.set_user_attr('split_mae', cv_mae)
            return np.mean(cv_mae)

        
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10, 
            seed=42
        )

        self.study = optuna.create_study(
            directions=['minimize'],
            sampler=sampler,
            study_name=experiment_name
        )

        self.study.optimize(objective, n_trials=2, timeout= 3600, callbacks=[mlflc]) 
        
        # search for the best run at the end of the experiment
        # best_run = mlflow.search_runs(max_results=1,order_by=["metrics.MAE"]).run_id
        # # register new model version in mlflow
        # self.result = mlflow.register_model(
        #     model_uri=f"runs:/{best_run}/{artifact_path}",
        #     name=self.model_name
        # )

    def forecast(
        self, 
        input_data: pd.DataFrame,
        ) -> pd.DataFrame:
        """ 
        Fetches the latest version of the model from the mlflow backend and uses it
        to perform prediction on new input data.
        """
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{self.model_name}/{self.result.version}"
        )
        y_pred = model.predict(input_data)

        return y_pred
        