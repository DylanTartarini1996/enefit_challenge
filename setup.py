import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="enefit_challenge", # Replace with your username
    version="0.1.1",
    author="Dylan Tartarini",
    author_email="<tartarinidylan@gmail.com>",
    description="<Package developed while partecipating to Enefit Challenge>",
    long_description=long_description,
    url="<https://https://github.com/DylanTartarini1996/enefit_challenge>",
    packages=setuptools.find_packages(),
    install_requires=[
        "catboost", 
        "matplotlib",
        "mlflow",
        "numpy",
        "pandas",
        "pmdarima",
        "prophet",
        "seaborn",
        "shap",
        "statsmodels",
        "xgboost",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],

    python_requires='>=3.9',
)