# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

def load_data(path="data/iris.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    X = df.drop("species", axis=1)
    y = df["species"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

def save_artifacts(model, scaler, model_dir="models"):
    joblib.dump(model, "artifacts/model.joblib")
   

def main():
    df = load_data()
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = train_model(X_train, y_train)
    save_artifacts(model, scaler)
# 
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
