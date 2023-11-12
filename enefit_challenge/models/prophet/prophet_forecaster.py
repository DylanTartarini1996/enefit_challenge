import json

import mlflow.prophet
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics

from enefit_challenge.models.forecaster import Forecaster

class ProphetForecaster(Forecaster):

    def __init__(self) -> None:
        pass

    def extract_params(self, pr_model):
        return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

    def train_model(
            self,
            experiment_name: str, 
            train_df: pd.DataFrame, 
            target_col: str,
            date_col: str = "index", 
            horizon: str="28 days",
            period: str="14 days",
            initial: str="1680 days", 
            artifact_path: str="prophet", 
            metrics: list=["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"], 
            time_series_params: dict={
                'changepoint_range': 0.7,
                'changepoint_prior_scale': 0.5, 
                'seasonality_prior_scale': 0.1, 
                'seasonality_mode': 'multiplicative'
                }
        ) -> str:
            """
            Trains an instance of the Prophet model on a given DataFrame and tracks 
            the experiment with Mlflow. Prophet model accepts only a column `y` with
            the target variable and a column `ds` with the dates, so we preprocess
            the training df before fitting it. 
            
            Args
            ---------
            `train_df`: `pd.DataFrame`
                the training data for the model.
            `target_col`: `str`
                the time-series target column
            `date_col`: `str`
                the column holding the time informations on the series
            `horizon`: `str`
                it's the horizon interval of each prediction performed during training 
            `period`: `str`
                it's the frequency of predictions performed during training
            `initial`: `str`
                the initial training set
            `artifact_path`: `str`
                the path pointing to the MLflow artifact
            `metrics`: `list`:
                list of the metrics to track in the mlflow experiment run.
            `time_series_params`: `dict`
                dictionary of parameters that can be used to configure the Prophet model.
            
            Returns
            --------
            `model_uri`: `str`
                the model uri to load the prophet model directly from MLflow
            """
            self.train_df = train_df
            self.date_col = date_col
            self.target_col = target_col
            
            if self.date_col == "index":
                self.train_df.reset_index(inplace=True)
                self.train_df = train_df.rename(columns={self.date_col:"ds", self.target_col:"y"})

            else: 
                self.train_df = self.train_df.rename(columns={date_col:"ds", target_col:"y"})

            mlflow.set_experiment(experiment_name=experiment_name)
            with mlflow.start_run():
                
                model = Prophet(
                    changepoint_range=time_series_params["changepoint_range"],
                    changepoint_prior_scale=time_series_params["changepoint_prior_scale"],
                    seasonality_prior_scale=time_series_params["seasonality_prior_scale"],
                    seasonality_mode=time_series_params["seasonality_mode"]
                    ).fit(self.train_df)

                params = self.extract_params(model)

                metrics_raw = cross_validation(
                    model=model,
                    horizon=horizon,
                    period=period,
                    initial=initial,
                    disable_tqdm=True,
                )
                cv_metrics = performance_metrics(metrics_raw)
                metrics_dict = {k: cv_metrics[k].mean() for k in metrics}

                print(f"Logged Metrics: \n{json.dumps(metrics_dict, indent=2)}")
                print(f"Logged Params: \n{json.dumps(params, indent=2)}")

                train = model.history
                future = model.make_future_dataframe(
                    periods=10,
                    freq=pd.infer_freq(self.train_df["ds"])
                )
                predictions = model.predict(future)
                signature = infer_signature(train, predictions)

                mlflow.prophet.log_model(
                    model, 
                    artifact_path=artifact_path, 
                    signature=signature
                )
                mlflow.log_params(params)
                mlflow.log_metrics(metrics_dict)
                self.model_uri = mlflow.get_artifact_uri(artifact_path)
            
            self.model = model # to use outside of mlflow

            return self.model_uri
    
    def forecast(
            self, 
            n_steps: int=0, 
            keep_in_sample_forecast: bool=True,
            model_uri: str = None
        ) -> pd.DataFrame:
        """
        Use the trained model to predict into the future. \n
        The user can predict only on the training data horizon (in-sample forecast) 
        and also forecast into the future, specifying how many steps ahead with 
        the param `n_steps` (out-of-sample forecast).

        Args:
        ----------
        n_steps: int
            the number of steps (1 step is one skip in the frequency of the training set)
            into the future you want to obtain a forecast for. \n
            When set to 0, we are only predicting in-sample.
        keep_in_sample_forecast: bool
            wether or not to keep the predictions made on the test set 
        """
        
        if not model_uri: # use the model nested in the forecaster class
        
            future = self.model.make_future_dataframe(
                periods=n_steps,
                freq=pd.infer_freq(self.train_df["ds"]),
                include_history=True
            )
            predictions = self.model.predict(future)

        else: # use the model logged into mlflow
            loaded_model = mlflow.prophet.load_model(model_uri)

            future = loaded_model.make_future_dataframe(
                periods=n_steps,
                freq=pd.infer_freq(self.train_df["ds"]),
                include_history=True
            )
            predictions = loaded_model.predict(future)

        if not keep_in_sample_forecast:
            predictions = predictions.tail(n_steps)

        return predictions