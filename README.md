# GBK Marketing Insights Suite

A web app that lets non-technical consultants run key driver analysis 
on survey data without needing Python or stats knowledge.

## What it does

Upload an Excel file, pick your outcome variable, select your 
predictors, and the app tells you which variables are most strongly 
associated with the outcome.

## Features

- Four-step pipeline: select Y, select X, optional subgroup loop, 
  choose method
- Three analysis methods: Random Forest, Correlation, Regression
- Subgroup analysis: run the model separately for each level of a 
  categorical variable
- Auto data cleaning: excludes ID columns, fills missing values, 
  drops constant columns
- Live app: https://pww9k9yv8na2dnlcr6pdfs.streamlit.app/

## How to run locally

pip install -r requirements.txt
streamlit run GBK_app.py

## Built with

Python, Streamlit, scikit-learn, pandas
