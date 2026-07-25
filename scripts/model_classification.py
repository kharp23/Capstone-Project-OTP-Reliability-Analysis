# Check result
print(OTP_Copy["otp_binary"].value_counts())

# Sort by date
df = OTP_Copy.sort_values("date").copy()

features_Class = [
    "from",
    "to",
    "line",
    "type",
    "status",
    "hour",
    "day_of_week",
    "month",
    "weekend",
    "rush_hour"
]

target_Class = "otp_binary"

# Time-based split
train = df[df["month"] < 5]
test = df[df["month"] >= 5]

X_train = train[features_Class]
y_train = train[target_Class]

X_test = test[features_Class]
y_test = test[target_Class]

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

logistic_model.fit(X_train, y_train)

log_predictions = logistic_model.predict(X_test)

print("Logistic Accuracy :", accuracy_score(y_test, log_predictions))
print("Logistic Precision:", precision_score(y_test, log_predictions))
print("Logistic Recall   :", recall_score(y_test, log_predictions))
print("Logistic F1 Score :", f1_score(y_test, log_predictions))

#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
print("Random Forest Accuracy :", accuracy_score(y_test, rf_predictions))
print("Random Forest Precision:", precision_score(y_test, rf_predictions))
print("Random Forest Recall   :", recall_score(y_test, rf_predictions))
print("Random Forest F1 Score :", f1_score(y_test, rf_predictions))

#GRADIENT BOOSTING CLASSIFIER
from sklearn.ensemble import GradientBoostingClassifier

gb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ]
)

gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_test)

print("Gradient Boost Accuracy :", accuracy_score(y_test, gb_predictions))
print("Gradient Boost Precision:", precision_score(y_test, gb_predictions))
print("Gradient Boost Recall   :", recall_score(y_test, gb_predictions))
print("Gradient Boost F1 Score :", f1_score(y_test, gb_predictions))


# ==========================================================
# CLASSIFICATION COMPUTATIONAL COST
# ==========================================================

classification_compute_results = []

classification_timing_models = {
    "Logistic Regression": clone(logistic_model),
    "Random Forest Classifier": clone(rf_model),
    "Gradient Boosting Classifier": clone(gb_model)
}

for model_name, timing_pipeline in classification_timing_models.items():

    print(f"Measuring {model_name}...")

    # Measure model training time
    training_start = time.perf_counter()

    timing_pipeline.fit(
        X_train,
        y_train
    )

    training_end = time.perf_counter()

    training_time = (
        training_end - training_start
    )

    # Measure prediction time
    prediction_start = time.perf_counter()

    timing_predictions = timing_pipeline.predict(
        X_test
    )

    prediction_end = time.perf_counter()

    prediction_time = (
        prediction_end - prediction_start
    )

    # Number of predictions generated per second
    predictions_per_second = (
        len(X_test) / prediction_time
        if prediction_time > 0
        else 0
    )

    # Approximate fitted pipeline size
    model_size_mb = (
        len(pickle.dumps(timing_pipeline))
        / (1024 ** 2)
    )

    classification_compute_results.append(
        {
            "Task": "Classification",
            "Model": model_name,
            "Training Rows": len(X_train),
            "Testing Rows": len(X_test),
            "Training Time (Seconds)": training_time,
            "Prediction Time (Seconds)": prediction_time,
            "Predictions Per Second": predictions_per_second,
            "Model Size (MB)": model_size_mb
        }
    )

classification_compute_table = pd.DataFrame(
    classification_compute_results
)

classification_compute_table = (
    classification_compute_table.round(
        {
            "Training Time (Seconds)": 4,
            "Prediction Time (Seconds)": 4,
            "Predictions Per Second": 2,
            "Model Size (MB)": 4
        }
    )
)

print("\nClassification Computational Cost")
display(classification_compute_table)
