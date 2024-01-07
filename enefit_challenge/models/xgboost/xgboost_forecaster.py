import ast
import numpy as np 
import pandas as pd 
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import TimeSeriesSplit
from sktime.performance_metrics.forecasting import (MeanAbsoluteScaledError, 
    MeanAbsolutePercentageError, MeanAbsoluteError, MeanSquaredError)
from xgboost import XGBRegressor

import optuna
from optuna.integration.mlflow import MLflowCallback
import joblib

from typing import Optional, Dict, Tuple, Literal
from enefit_challenge.models.forecaster import Forecaster

import warnings
warnings.filterwarnings('ignore')


TRACKING_URI = "http://127.0.0.1:5000/" # local tracking URI -> launch mlflow before training 


class XGBoostForecaster(Forecaster):
    """
        Implementation of a Forecaster using `XGBRegressor` as base model, 
        `optuna` for hyperparameters optimization and `mlflow` as backend to track experiments
        and register best-in-class model for time series prediction.
    """
    def __init__(
        self,
        experiment_name: str="xgboost",
        artifact_path: str="xgboost_model",
        model_name: str="enefit_xgboost"
    )-> None:
        """
        Initializes the `XGBForecaster`
        -------
        params:
        -------
        `experiment_name`: `str`
            the name the of the experiment under which mlflow's runs (or Optuna's trials) 
            will be collected
        `model_name`: `str`
            the name the final model will have in the registry
        `artifact_path`: `str`
            the path pointing to the mlflow artifact
        """
        self.tracking_uri = mlflow.set_tracking_uri(TRACKING_URI)
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.artifact_path = artifact_path
        pass


    def fit_model(
        self,  
        X:pd.DataFrame,
        y:pd.Series,
        params:Optional[Dict]=None,
    ) -> XGBRegressor:
        """
        Trains a `XGBRegressor`
        -------     
        params:
        -------
        `X`:`pd.DataFrame`
            Features to use for fitting
        `y`:`pd.Series`
            Target variable
        `params`: `Optional[Dict]`
            optional dictionary of parameters to use
        -------     
        returns:
        -------
        fitted `XGBRegressor`
        """
        model = XGBRegressor(
            n_estimators=100, 
            objective='reg:squarederror',
            eval_metric='mae'
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
        metrics: list=["mae"]
    ) -> float:
        """
        Used for cross validation on different time splits; 
        also in charge of logging every experiment run / study trial into the backend.
        """
        
        first_dates_month = pd.to_datetime(X[['year', 'month']].assign(day=1))
        train_index = first_dates_month.isin(year_month_train)
        test_index = first_dates_month.isin(year_month_test)

        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        # fit model on training data
        mlflow.xgboost.autolog(log_datasets=False, log_models=False)
        model = self.fit_model(
            X_train, 
            y_train, 
            params
        )
        
        # generate predictions
        y_test_pred = model.predict(X_test)
        self.signature = infer_signature(X_train, y_test_pred)
        MAE = MeanAbsoluteError()
        mae = MAE(y_test, y_test_pred)
        MASE = MeanAbsoluteScaledError()
        mase = MASE(y_test, y_test_pred, y_train=y_train)
        MAPE = MeanAbsolutePercentageError()
        mape = MAPE(y_test, y_test_pred)
        MSE = MeanSquaredError()
        mse = MSE(y_test, y_test_pred)
        RMSE = MeanSquaredError(square_root=True)
        rmse = RMSE(y_test, y_test_pred)

        mlflow.xgboost.log_model(
            model, 
            artifact_path=self.artifact_path,
            signature=self.signature
        )
        mlflow.log_params(params)
        
        return mae, mase, mse, rmse, mape


    def train_model(
        self, 
        train_df: pd.DataFrame, 
        target_col: str,
        exclude_cols: list=[],
        categorical_features: list=[],
        params: Optional[Dict]=None,
        n_trials: int=100
    ) -> None:
        """ 
        Takes an instance of `XGBRegressor` model and tracks the hyperparameter tuning
        experiment on training set using `mlflow` and `optuna`.  
        Registers the best version of the model according to a specified metric (to be implemented).
        -------     
        params:
        -------
        `train_df`: `pd.DataFrame`
            the training data for the model.
        `target_col`: `str`
            the time-series target column
        `exclude_cols`: `list`  
            columns in dataset that should not be used
        `categorical_features`: `list`
            list of categorical features in the dataset
        `params`: `Optional[Dict]`
            optional dictionary of parameters to use
        `n_trials`: `int=100`
            number of Optuna trials to conduct for hyperparameters tuning
        """
        self.categorical_features = categorical_features
        if len(self.categorical_features) > 0: 
           train_df = pd.get_dummies(train_df, columns=self.categorical_features)

        X = train_df.drop([target_col] + exclude_cols, axis=1)
        y = train_df[target_col]
        # unique year-month combinations -> to be used in cross-validation
        timesteps = np.sort(np.array(
            pd.to_datetime(X[['year', 'month']].assign(day=1)).unique().tolist()
        ))
        
        # define mlflow callback Handler for optuna 
        mlflc = MLflowCallback(
            metric_name=["MAE"]
        )
    
        @mlflc.track_in_mlflow() # decorator to allow mlflow logging
        def objective(trial):
            params = {
                'eval_metric': 'mae',
                'n_estimators': trial.suggest_int('n_estimators', 50, 250, log=True),
                'eta': trial.suggest_float('eta', 0.01, 0.95,log=True),
                'max_depth': trial.suggest_int('max_depth', 1, 10, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 25, log=True),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1, log=True),
                'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.1, 1, log=True),
                'colsample_bynode': trial.suggest_float("colsample_bynode", 0.1, 1, log=True),
                'subsample': trial.suggest_float("subsample", 0.5, 1, log=True),
                'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True)
            }
            cv = TimeSeriesSplit(n_splits=3) # cross validation
            cv_mae = [None]*3
            cv_mase = [None]*3
            cv_mse = [None]*3
            cv_rmse = [None]*3
            cv_mape = [None]*3
            for i, (train_index, test_index) in enumerate(cv.split(timesteps)):
                cv_mae[i], cv_mase[i], cv_mse[i], cv_rmse[i], cv_mape[i] = self.fit_and_test_fold(
                    params,
                    X, 
                    y, 
                    timesteps[train_index], 
                    timesteps[test_index]
                )
            trial.set_user_attr('split_mae', cv_mae)
            trial.set_user_attr('split_mase', cv_mase)
            trial.set_user_attr('split_mse', cv_mse)
            trial.set_user_attr('split_rmse', cv_rmse)
            trial.set_user_attr('split_mape', cv_mape)

            mlflow.log_metrics(
                {
                    "MAE":np.mean(cv_mae),
                    "MASE": np.mean(cv_mase),
                    "MSE": np.mean(cv_mse),
                    "RMSE":np.mean(cv_rmse),
                    "MAPE":np.mean(cv_mape)
                }
            )
            mlflow.log_dict(
                dictionary={
                    "categorical_features": self.categorical_features
                },
                artifact_file="categorical_features.json"
            )

            return np.mean(cv_mae) 

        
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10, 
            seed=0
        )

        self.study = optuna.create_study(
            directions=['minimize'],
            sampler=sampler,
            study_name=self.experiment_name
        )

        self.study.optimize(
            objective, 
            n_trials=n_trials, 
            timeout= 7200, 
            callbacks=[mlflc]
        ) 


    def predict(
        self, 
        input_data: pd.DataFrame,
        use_best_from_run: bool=True,
        use_env_model: Literal["Staging", "Production", None]=None,
        use_version: int=None,
        ) -> pd.DataFrame:
        """ 
        Fetches a version of the model from the mlflow backend and uses it
        to perform prediction on new input data.  
        What version is used depends on params settings, 
        defaults to using the best version from the last experiment run. 
        -------     
        params:
        -------
        `input_data`: `pd.DataFrame`
            the input data for prediction,
            must have the same schema as what's in the model's signature.
        `use_best_from_run`: `bool=True`      
            use the best model from the current series of iterations, defaults to True
        `use_env_model`: `Literal["Staging", "Production", None]=None`
            use model from a given mlflow environment, defaults to None.  
            Said model might come from past iterations, depending on what you decide in the UI
        `use_version`: `int=None`
            use a previously trained version of the model. 
            Said version must have been registered from a previous iteration,  
            either by the UI or with mlflow's API
        """
        client = MlflowClient(tracking_uri=TRACKING_URI)

        if (use_best_from_run) & (use_env_model is None) & (use_version is None):
            experiment = mlflow.search_experiments(
                filter_string=f"name='{self.experiment_name}'"
            )
            experiment_id = experiment[0]._experiment_id
            best_run = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="",
                max_results=1,
                order_by=["metrics.MAE ASC"], # best run according to MAE
            )[0]
            model = mlflow.xgboost.load_model(
                model_uri=f"runs:/{best_run.info.run_id}/{self.artifact_path}"
            )
            model_info = mlflow.models.get_model_info(
                f"runs:/{best_run.info.run_id}/{self.artifact_path}"
            )

        elif (not use_best_from_run) & (use_env_model in ["Staging", "Production"]) & (use_version is None):
            model_metadata = client.get_latest_versions(
                name=self.model_name, 
                stages=[use_env_model]
            )
            run_id = model_metadata[0].run_id
            model = mlflow.xgboost.load_model(
                model_uri=f"runs:/{run_id}/{self.artifact_path}"
            )
            model_info = mlflow.models.get_model_info(
                f"runs:/{run_id}/{self.artifact_path}"
            )
            
        elif (not use_best_from_run) & (use_env_model is None) & (use_version is not None):
            model = mlflow.xgboost.load_model(
                f"models:/{self.model_name}/{use_version}"
            )
            model_info = mlflow.models.get_model_info(
                f"models:/{self.model_name}/{use_version}"
            )
            
        if (not use_best_from_run) & (use_env_model is None) & (use_version is None):
            return ValueError(
                    "You must specify which kind of XGBoostForecaster you intend to use for prediction"
            )
        
        inputs = ast.literal_eval(model_info.signature_dict["inputs"])
        input_features = [d['name'] for d in inputs]
        artifact_uri = mlflow.search_runs(
            experiment_names=[self.experiment_name], 
            filter_string=f"run_id='{model_info.run_id}'"
        )["artifact_uri"][0]
        # access categorical features used in the training iterations
        categorical_features = mlflow.artifacts.load_dict(
            artifact_uri=artifact_uri+"/categorical_features.json"
        )["categorical_features"]
        
        if len(categorical_features) > 0: 
           input_data = pd.get_dummies(input_data, columns=categorical_features)
        
        X = input_data[input_features]

        y_pred = model.predict(X)

        return y_pred
        