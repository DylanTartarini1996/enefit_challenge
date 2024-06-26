# enefit_challenge
Repository created for the [Enefit Kaggle Challenge](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/code)

## Challenge Description
The goal of the competition is to create an energy prediction model of prosumers to reduce energy imbalance costs.  
This competition aims to tackle the issue of energy imbalance, a situation where the energy expected to be used doesn't line up with the actual energy used or produced. Prosumers, who both consume and generate energy, contribute a large part of the energy imbalance. Despite being only a small part of all consumers, their unpredictable energy use causes logistical and financial problems for the energy companies.  

The number of prosumers is rapidly increasing, and solving the problems of energy imbalance and their rising costs is vital. If left unaddressed, this could lead to increased operational costs, potential grid instability, and inefficient use of energy resources. If this problem were effectively solved, it would significantly reduce the imbalance costs, improve the reliability of the grid, and make the integration of prosumers into the energy system more efficient and sustainable. Moreover, it could potentially incentivize more consumers to become prosumers, knowing that their energy behavior can be adequately managed, thus promoting renewable energy production and use.  

## Environment Setup
```
    # using pip
    pip install -r requirements.txt

    # using Conda
    conda create --name <env_name> --file requirements.txt
```

## Download Data & Assets
After having obtained your [Kaggle API key](https://github.com/Kaggle/kaggle-api) go on your terminal and launch
```
    kaggle competitions download -c predict-energy-behavior-of-prosumers
```

## Install Package for the competition
Code, utilities and other classes (like model classes) can be properly written and integrated in the `enefit_challenge` python package created here.  
To actually install the python package, you should be able to clone this repository and then, on your active environment, run 
```
    python3 -m pip install --upgrade build
    python setup.py install
```
You should now be able to import modules and methods, for example
```
    from enefit_challenge.dataset.dataset import EnefitDataset
```  
or 
```
    from enefit_challenge.models.lightgbm.lightgbm_forecaster import LightGBMForecaster
```

If you want to ***modify the package***, add or remove code from it, cd into your folder and, 
after removal from environment, run 
```
    python setup.py sdist
    python3 -m pip install --upgrade build
```

## Commiting Notebooks
To commit a notebook, use the various `notebooks/` subfolders, possibly following naming conventions: 
name of notebook: `{your_initials}_{number}_{notebook_title}.ipynb`

Folder structure:
```
    .
    ├── enefit -> to be downloaded from kaggle (git-ignored)
    ├── enefit_challenge -> python package for the challenge
        ├── dataset
        ├── models
            ├── catboost
            ├── lightgbm
            ├── xgboost
            ├── forecaster.py -> abstract class for the challenge
        ├── tests
    ├── input -> data from the challenge, download from kaggle (git-ignored)
    ├── notebooks -> notebook folder for various phases of the project
        ├── EDA
        ├── feature_eng
        ├── modelling
        ├── prediction
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    ├── setup.py -> enefit_challenge package setup installation
```

## Tracking with MLflow
Once you launch an experiment (i.e. training a model) you can run 
```
    mlflow server
```  

to visualize the mlflow UI and to compare all the experiment runs for all models, their metrics and so on like in the example below   

![](assets/lgbm_par_coordinates.png)

You can also use the UI to register the best-performing models

![](assets/model_registry.png)
