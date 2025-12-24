import setuptools
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_model_with_mlflow(data_path, model_output_path):
    print("Memuat dataset...")
    df = pd.read_csv(data_path)
    
    target = 'mental_state'
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "Mental_Health_Classification_Exp"
    mlflow.set_experiment(experiment_name)    
    mlflow.sklearn.autolog()

    print(f"Memulai training dengan MLflow Tracking (Experiment: {experiment_name})...")

    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_score, multi_class='ovr', average='weighted')
        except:
            roc_auc = 0.0

        print("Mencatat metrik test set ke MLflow...")
        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)

        joblib.dump(model, model_output_path)
        print(f"Model .joblib berhasil disimpan di: {model_output_path}")

        print("Mencatat model ke MLflow...")
        mlflow.sklearn.log_model(sk_model=model, artifact_path="final_model_artifact", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        print("Model berhasil dicatat di MLflow.")

if __name__ == "__main__":
    train_model_with_mlflow(
        'mh_sosmed_dataset.csv', 
        'model_random_forest.joblib', 
    )