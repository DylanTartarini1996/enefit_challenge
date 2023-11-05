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