#MACHINE LEARNING FEATURES

# ================================
# 1. Import required libraries
# ================================

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ================================
# 2. Sort data by date
# ================================
# Since we are predicting future delays, the dataset should be sorted by date.
# This makes sure older data is used for training and future data is used for testing.

df = OTP_Copy.sort_values("date").copy()


# ================================
# 3. Define ML features and target
# ================================
# These are the input variables used to predict delay_minutes.
#Station-level variables were excluded from some model runs 
#because one-hot encoding created a high-dimensional feature space, making tree-based models computationally expensive.

features = [
    #"from",
    #"to",
    "line",
    "type",
    "status",
    "hour",
    "day_of_week",
    "month",
    "weekend",
    "rush_hour"
]

target = "delay_minutes"


# ================================
# 4. Time-based train-test split
# ================================
# January to April data is used for training.
# May data is used for testing.
# This supports the research question: can historical data predict future delays?


train = df[df["month"] < 5]
test = df[df["month"] >= 5]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# Remove rows where target is missing
train_valid = y_train.notna()
test_valid = y_test.notna()

X_train = X_train[train_valid]
y_train = y_train[train_valid]

X_test = X_test[test_valid]
y_test = y_test[test_valid]

# ================================
# 5. Identify numeric and categorical columns
# ================================

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns


# ================================
# 6. Preprocessing for numeric variables
# ================================
# Missing numeric values are filled with median.
# StandardScaler is used because Linear Regression and SVR are sensitive to scale.

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ================================
# 7. Preprocessing for categorical variables
# ================================
# Missing categorical values are filled with the most frequent value.
# OneHotEncoder converts text categories into numeric format for ML models.
# handle_unknown="ignore" prevents errors if May has categories not seen in Jan-Apr.

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# ================================
# 8. Combine preprocessing steps
# ================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ================================
# 9. Create models
# ================================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
}

# ================================
# 10. Train and evaluate models
# ================================

results = []

for model_name, model in models.items():
    
    print(f"Training {model_name}...")
    
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )
    
    pipeline.fit(X_train, y_train)
    
    print(f"Predicting with {model_name}...")
    
    predictions = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

# ================================
# 11. Display model comparison
# ================================

results_df = pd.DataFrame(results)
results_df

#GRADIENT BOOST FOR DELAY MINUTE PREDICTION
from sklearn.ensemble import GradientBoostingRegressor

# ================================
# Gradient Boosting Model
# ================================

gradient_boost_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ]
)

# Train the model
print("Training Gradient Boosting model...")
gradient_boost_model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
gb_predictions = gradient_boost_model.predict(X_test)

# Evaluate model
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
gb_r2 = r2_score(y_test, gb_predictions)

print("Gradient Boosting Results")
print("MAE:", gb_mae)
print("RMSE:", gb_rmse)
print("R² Score:", gb_r2)
