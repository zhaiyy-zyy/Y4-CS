House Price Prediction Using Stacking Ensembles
COMP4131 Data Modelling and Analysis
Author: Yuyang Zhang (20514470)

--------------------------------------------------

Overview
This project implements a full data modelling pipeline for predicting house prices using stacking ensembles on the King County housing dataset. 
--------------------------------------------------

Requirements
Python 3.9+

Required libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm

Install dependencies with:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm

--------------------------------------------------

Dataset
Place the dataset file:

data.csv

in the same directory as the notebook.

--------------------------------------------------

How to Run
1. Open the Jupyter Notebook file:
   YuyangZhang_20514470.ipynb

2. Ensure data.csv is in the same folder

3. Run all cells from top to bottom

--------------------------------------------------

Reproducibility
- A fixed random seed (42) is used where applicable
- The notebook is fully self-contained and can be executed from start to finish
- Results are reproducible up to minor variations due to stochastic elements in model training

--------------------------------------------------

Project Structure
1. Data loading
2. Exploratory data analysis (EDA)
3. Preprocessing and feature engineering
4. Model training and evaluation
5. Stacking ensemble construction
6. Ablation study
7. Diagnostics and visualisation