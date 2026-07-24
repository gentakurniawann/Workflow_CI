import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

data_path = os.path.join("preprocessing", "dataset_preprocessing", "breast_cancer_processed.csv")
if not os.path.exists(data_path):
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
else:
    df = pd.read_csv(data_path)

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_example = X_train[0:5]

mlflow.autolog()
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
mlflow.log_metric("accuracy", accuracy)

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=input_example
)

print("Basic modelling and autologging completed successfully!")
