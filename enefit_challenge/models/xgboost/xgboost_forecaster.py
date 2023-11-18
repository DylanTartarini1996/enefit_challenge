import numpy as np 
import pandas as pd 
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
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
        Implementaiton of a Forecaster using `XGBRegressor` as base model, 
        `optuna` for hyperparameters optimization and `mlflow` as backend to track experiments
        and register best-in-class model for time series prediction.
    """
    def __init__(self)-> None:
        self.tracking_uri = mlflow.set_tracking_uri(TRACKING_URI)
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
            objective='reg:squarederror'
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
        experiment_name: str="xgboost",
        artifact_path: str="xgboost_model",
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
        model = self.fit_model(
            X_train, 
            y_train, 
            params
        )
        # generate predictions
        y_test_pred = model.predict(X_test)
        self.signature = infer_signature(X_train, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)

        mlflow.xgboost.log_model(
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
        experiment_name: str="xgboost",
        artifact_path: str="xgboost_model",
        params: Optional[Dict]=None,
        metrics: list=["MAE"]
    ) -> None:
        """ 
        Takes an instance of `XGBRegressor` model and tracks the hyperparameter tuning
        experiment on training set using `mlflow` and `optuna`.  
        Registers the best version of the model according to a specified metric (to be implemented).
        
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
                'n_estimators': trial.suggest_int('n_estimators', 50, 200, log=True),
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
            for i, (train_index, test_index) in enumerate(cv.split(timesteps)):
                cv_mae[i] = self.fit_and_test_fold(
                    params,
                    X, 
                    y, 
                    timesteps[train_index], 
                    timesteps[test_index]
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

        self.study.optimize(objective, n_trials=100, timeout= 7200, callbacks=[mlflc]) 
        
        # # search for the best run at the end of the experiment # not implemented now bc of callback bug
        # best_run = mlflow.search_runs(max_results=1,order_by=["metrics.MAE"]).run_id
        # # register new model version in mlflow
        # self.result = mlflow.register_model(
        #     model_uri=f"runs:/{best_run}/{artifact_path}",
        #     name=self.model_name
        # )

    def forecast(
        self, 
        input_data: pd.DataFrame,
        use_best_from_run: bool=True,
        use_env_model: Literal["Staging", "Production", None]=None,
        use_version: int=None
        ) -> pd.DataFrame:
        """ 
        Fetches a version of the model from the mlflow backend and uses it
        to perform prediction on new input data.  
        What version is used depends on params settings, 
        defaults to using the best version from the last experiment run (currently not implemented). 
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
        if use_best_from_run:
            # not implemented now bc of callback bug
            use_prod_model=None
            use_version=None
        
            # model = mlflow.pyfunc.load_model(
            #     model_uri=f"models:/{self.model_name}/{self.result.version}"
            # )
            # y_pred = model.predict(input_data)
            # return y_pred
        
        if use_env_model is not None:
            use_version = None

            model = mlflow.pyfunc.load_model(
                # get registered model in given environment
                model_uri=f"models:/{self.model_name}/{use_env_model}"
            )
            y_pred = model.predict(input_data)
            return y_pred

        if use_version is not None:
            # get specific registered version of model
            model = mlflow.pyfunc.load_model(
                model_uri=f"models:/{self.model_name}/{use_version}"
            )
            y_pred = model.predict(input_data)
            return y_pred

        
        if (not use_best_from_run) & (use_env_model is None) & (use_version is None):
            return ValueError(
                    "You must specify which kind of XGBoostForecaster you intend to use for prediction"
                    )
        