The notebook follows the standard data science workflow.

1. Data Quality Assessment
   *  Missing value analysis
   *  Duplicate record detection
   *  Data consistency checks
   *  Data type verification
   *  Datetime conversion
     
2. Data Cleaning
  *  Missing value treatment
  *  Duplicate removal
  *  Datetime parsing
  *  Categorical value inspection
    
3. Exploratory Data Analysis: This section explores:
  * Delay distribution
  * Correlation analysis
  * Station reliability
  * Route reliability
  * Historical On-Time Performance (OTP)
  * Peak-hour congestion
  * Day-of-week trends
  * Monthly delay patterns
  * Weekend analysis
  * Operator comparison

4. Feature Engineering : Additional variables created include:
  * Hour
  * Day of Week
  * Month
  * Weekend
  * Rush Hour
  * Season
  * Binary On-Time Performance (OTP)
  * Average Station Delay
  * Route Delay Frequency
  * Historical Route Reliability
    
These engineered features provide additional operational information for predictive modelling.

5. Machine Learning : Historical data is divided chronologically:
 * Training: January–April 2020
 * Testing: May 2020
   
This time-based split simulates a real-world forecasting scenario where historical data is used to predict future train delays.

The notebook develops and compares:
 * Linear Regression
 * Random Forest Regression
 * Gradient Boosting
   
Model performance is evaluated using:
 * Mean Absolute Error (MAE)
 * Root Mean Squared Error (RMSE)
 * R² Score
   
Expected Outcome: 
By the end of this notebook, the analysis will:
 * Identify the operational factors contributing to train delays.
 * Compare delay behaviour between NJ Transit and Amtrak.
 * Evaluate whether machine learning models can predict future train delays using historical operational data.
 * Establish baseline predictive models that can support future improvements through additional operational and external data.

   
Software Requirements
 * Python 3.x
 * Jupyter Notebook
   
Required libraries include:
 * pandas
 * numpy
 * matplotlib
 * seaborn
 * scikit-learn
   
Note: The notebook is organized sequentially. Each section builds upon the previous one; therefore, it is recommended to execute the notebook from top to bottom to ensure all preprocessing, feature engineering, and modelling steps are completed correctly.
