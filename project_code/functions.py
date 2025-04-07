from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import json
from sklearn.metrics import f1_score, precision_score
import joblib

def read_dataset():
    df = pd.read_csv('./dataset/bank.csv')
    return df

def preprocess_data(df):
    df = df.copy()

    df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

    selected_features = ['contact', 'housing', 'duration', 'pdays', 'previous', 'deposit']
    df = df[selected_features]

    cat_cols = ['contact', 'housing']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df

def preprocess_user_data(df):
    df = df.copy()

    selected_features = ['contact', 'housing', 'duration', 'pdays', 'previous']
    df = df[selected_features]

    cat_cols = ['contact', 'housing']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def split_data(df):
    X = df.drop('deposit',axis = 1)
    y = df['deposit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = round(f1_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred), 4)
    return f1, precision

def save_model(model, name):
    joblib.dump(model, f'./checkpoints/{name}.pkl')

def load_model(name):
    return joblib.load(f'./checkpoints/{name}.pkl')

