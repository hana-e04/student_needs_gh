# student_needs

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A regression and grouping of different student financial needs.

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

