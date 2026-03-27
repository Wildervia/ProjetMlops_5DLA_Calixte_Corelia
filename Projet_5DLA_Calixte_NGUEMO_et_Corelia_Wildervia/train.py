"""
train.py — Script d'entraînement MLOps Fraude Bancaire
Usage : python train.py --config config.yaml
"""
import argparse
import yaml
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from imblearn.over_sampling import SMOTE


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(path):
    df = pd.read_csv(path)
    missing_rate = df.isnull().mean().max()
    if missing_rate > 0.05:
        raise ValueError(f"Data Quality Gate : valeurs manquantes = {missing_rate:.2%} > 5%")
    return df


def preprocess(df):
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    df = df.drop(columns=['Time', 'Amount'])
    return df


def train(config):
    df  = load_data(config['data']['path'])
    df  = preprocess(df)
    X   = df.drop(columns=['Class'])
    y   = df['Class']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y,
        random_state=config['training']['seed'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        stratify=y_temp, random_state=config['training']['seed'])

    smote = SMOTE(random_state=config['training']['seed'])
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(config['mlflow']['experiment'])
    with mlflow.start_run(run_name=config['mlflow']['run_name']):
        model = RandomForestClassifier(
            n_estimators=config['model']['n_estimators'],
            class_weight='balanced',
            random_state=config['training']['seed'],
            n_jobs=-1)
        model.fit(X_train_res, y_train_res)

        y_proba = model.predict_proba(X_test)[:, 1]
        auc     = roc_auc_score(y_test, y_proba)
        prauc   = average_precision_score(y_test, y_proba)

        mlflow.log_params(config['model'])
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("pr_auc",  prauc)
        mlflow.sklearn.log_model(model, "model")

        print(f"ROC-AUC : {auc:.4f} | PR-AUC : {prauc:.4f}")
        print(classification_report(y_test, model.predict(X_test),
                                    target_names=['Normal', 'Fraude']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    cfg  = load_config(args.config)
    train(cfg)
