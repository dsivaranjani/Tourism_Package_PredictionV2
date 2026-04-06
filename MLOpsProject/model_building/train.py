# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file://" + os.path.abspath("./mlruns"))
mlflow.set_tracking_uri(tracking_uri)
# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# api = HfApi(token=os.getenv("HF_TOKEN"))

Xtrain_path = "hf://datasets/RanjaniD/Tourism-Package-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/RanjaniD/Tourism-Package-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/RanjaniD/Tourism-Package-Prediction/ytrain.csv"
ytest_path = "hf://datasets/RanjaniD/Tourism-Package-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# One-hot encode 'Type' and scale numeric features
numeric_features = [
    'Age',
    'CityTier',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups',
    'DurationOfPitch'
]
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']


# Class weight to handle imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.8, 1.0],
    'xgbclassifier__colsample_bytree': [0.7, 0.8, 1.0],
    'xgbclassifier__reg_lambda': [0.1, 1, 10]
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
  # Grid search with cross-validation
  grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1)
  grid_search.fit(Xtrain, ytrain)


  # Log parameter sets
  results = grid_search.cv_results_
  for i in range(len(results['params'])):
      param_set = results['params'][i]
      mean_score = results['mean_test_score'][i]

      with mlflow.start_run(nested=True):
          mlflow.log_params(param_set)
          mlflow.log_metric("mean_neg_mse", mean_score)

  # Best model
  mlflow.log_params(grid_search.best_params_)
  best_model = grid_search.best_estimator_
  print("Best Params:\n", grid_search.best_params_)

  # Predictions
  y_pred_train = best_model.predict(Xtrain)
  y_pred_test = best_model.predict(Xtest)

  # Metrics 
  train_acc = accuracy_score(ytrain, y_pred_train)
  test_acc = accuracy_score(ytest, y_pred_test)

  train_recall = recall_score(ytrain, y_pred_train)
  test_recall = recall_score(ytest, y_pred_test)

  # Log metrics
  mlflow.log_metrics({
      "train_Accuracy": train_acc,
      "test_Accuracy": test_acc,
      "train_Recall": train_recall,
      "test_Recall": test_recall
  })

  # Evaluation
  print("\nTraining Classification Report:")
  print(classification_report(ytrain, y_pred_train))

  print("\nTest Classification Report:")
  print(classification_report(ytest, y_pred_test))

  # Save best model
  model_path = "best_Tourism_Package_Prediction_model_v1.joblib"
  joblib.dump(best_model, model_path)

  # Log the model artifact
  mlflow.log_artifact(model_path, artifact_path="model")
  print(f"Model saved as artifact at: {model_path}")

  # Upload to Hugging Face
  repo_id = "RanjaniD/Tourism-Package-Prediction"
  repo_type = "model"

  api = HfApi(token=os.getenv("HF_TOKEN"))

  # Step 1: Check if the space exists
  try:
      api.repo_info(repo_id=repo_id, repo_type=repo_type)
      print(f"Model Space '{repo_id}' already exists. Using it.")
  except RepositoryNotFoundError:
      print(f"Model Space '{repo_id}' not found. Creating new space...")
      create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
      print(f"Model Space '{repo_id}' created.")

  # create_repo("best_machine_failure_model", repo_type="model", private=False)
  api.upload_file(
      path_or_fileobj=model_path,
      path_in_repo=model_path,
      repo_id=repo_id,
      repo_type=repo_type,
  )
