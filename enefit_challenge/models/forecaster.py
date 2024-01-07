import mlflow
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


class Forecaster(ABC):
    """
    An abstract Forecaster Class with a train and forecast method
    """
    def __init__(self) -> None:
        pass    

    @abstractmethod
    def train_model(self):
        pass
            
    @abstractmethod
    def predict(self):
        pass
            