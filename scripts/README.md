This directory contains scripts that are organized according to the main stages of the data science pipeline to improve code readability, maintainability, and reusability.
Script Overview
 * data_preprocessing.py
      * Contains functions for preparing the raw dataset before analysis.
      * Functions include: Removing duplicate records, Handling missing values, Converting date and time columns to datetime format, Performing basic data cleaning
        
 * feature_engineering.py
      * Contains functions for creating additional variables used during exploratory analysis and predictive modelling.
      * Engineered features include: Hour of day, Day of week, Month, Weekend indicator, Rush-hour indicator, Season, Binary On-Time Performance (OTP)
        
 *  model_training.py
     * Contains reusable functions for machine learning model development.
     * This module includes: Data preprocessing pipeline, Feature encoding, Feature scaling
     * Baseline regression models: Linear Regression, Random Forest Regression and Gradient Boosting
     * Model evaluation using MAE, RMSE, and R²
       
 * model_classification.py
   Implements classification models used to predict train on-time performance (otp_binary).
   * The target variable is: 1 = On Time (delay ≤ 5 minutes), 0 = Delayed (delay > 5 minutes)
   * The module includes: Data preprocessing pipeline, Feature encoding, Model training
   * Model evaluation using: Accuracy, Precision, Recall, F1-Score
   * This script evaluates the feasibility of using historical operational data to classify whether a train is likely to arrive on time or be delayed.
