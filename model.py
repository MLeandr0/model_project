from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os

class TextMerger(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.iloc[:, 0] + " " + X.iloc[:, 1]

def build_pipeline():
    text_preprocessor = Pipeline([
        ('merge', TextMerger()),
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english'))
    ])

    preprocessor = ColumnTransformer([
        ('text', text_preprocessor, ['CDESCR', 'COMPDESC']),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), 
         ['MAKETXT', 'MODELTXT', 'CRASH', 'FIRE', 'DEATHS', 'FAILDATE']),
        ('numerical', MinMaxScaler(), ['YEARTXT'])
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    return pipeline

def train_and_save_model(data_path='data\complaints_with_category.csv', save_dir="saved_file/"):

    df = pd.read_csv(data_path, low_memory=False)
    df = df.dropna()
    
    X = df.drop('CATEGORY', axis=1)
    y = df['CATEGORY']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(save_dir, 'complaint_classification_pipeline.pkl'))
    print(f"Pipeline saved to {save_dir}")

if __name__ == "__main__":
    train_and_save_model()