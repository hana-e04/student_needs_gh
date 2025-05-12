# student_needs

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

The model contains a regression and grouping of different student financial needs based on student and consumer spending data in the US. The regression and cluster models aim to predict how student spending on books, hobbies, and school supplies changes in order to ensure schools are able to respond by providing the resources students may need.

## Impact
The K Modes and Decision Tree models generated in this project can be used to help universities and organizations proactively plan programs and budgets to meet student needs and promote equitable learning environments.

## Data Pipeline
The raw datasets are included in this repo, but the Student Monthly Spending dataset from Kaggle can be found here (https://www.kaggle.com/datasets/shariful07/nice-work-thanks-for-share) and the latest US Census Bureau data can be found here (https://www.census.gov/retail/marts/historic_releases.html).

The Kaggle data can just be downloaded as a csv. However, the categorical data must be encoded as numerical values. Missing data was coded as 0, "No" as 1, and "Yes" as 2. 

The US Census Bureau data must be downloaded as a xlsx file, converted to a csv file, and cleaned. Data from each year from 2024 to the 1980s are held in separate Excel sheet tabs. Combine them into a single dataset to view aggregate market trends and to compare market trends by year. The data pipeline for the Census Bureau data is in the market_data_cleaned jupyter notebook file. 

## Models

A K Modes model was used to predict the Student Monthly Spending dataset using the Kmodes package from the kmodes.kmodes module. Various k values were assessed for this model, and the mean values of each variable within the clusters were used to assess the clusters.

A Decision Tree was used to model the US Census Bureau data. Then, the individual influence of each variable was determined by assessing the impact of their removal on the overall mean absolute error and mean standard error of the model. Using this analysis, a final predictive decision tree model was created to predict consumer spending on books and hobbies. 

## Project Organization

```
├── LICENSE            <- There is currently no license determined for the code. All people are free to use.
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`, 
					contains all needed set up information, including pip and flake8
├── README.md          <- The top-level README for developers using this project. Current file, explains how repo is set up
├── data			<- contains all data used for project
│   ├── external       <- Data from third party sources. Contains all datasets used.
│   ├── interim        <- Intermediate data that has been transformed. 
│   ├── processed      <- The final, canonical data sets for modeling. Contains all datasets formatted for analysis.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         get_sn and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── get_sn   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes get_sn a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

