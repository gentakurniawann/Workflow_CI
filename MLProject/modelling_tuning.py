import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import dagshub

# Initialize DagsHub for MLflow tracking (Advance Requirement)
dagshub.init(repo_owner='gentastudyacc', repo_name='proyek-sistem-machine-learning', mlflow=True)

def main():
    try:
        df = pd.read_csv('../Eksperimen_SML_muhammad_genta_kmw35/iris_preprocessing/iris_processed.csv')
    except FileNotFoundError:
        print("Data not found. Please run automate_muhammad_genta_kmw35.py first.")
        return
        
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [10, 50],
        'max_depth': [None, 5]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    
    with mlflow.start_run():
        grid.fit(X, y)
        best_model = grid.best_estimator_
        
        preds = best_model.predict(X)
        acc = accuracy_score(y, preds)
        
        # Manual Logging
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", acc)
        
        # Artifact 1: Confusion Matrix
        cm = confusion_matrix(y, preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()
        
        # Artifact 2: Feature Importance
        plt.figure(figsize=(6,5))
        sns.barplot(x=best_model.feature_importances_, y=X.columns)
        plt.title('Feature Importances')
        plt.savefig('feature_importances.png')
        mlflow.log_artifact('feature_importances.png')
        plt.close()
        
        mlflow.sklearn.log_model(best_model, "model")

if __name__ == "__main__":
    main()
