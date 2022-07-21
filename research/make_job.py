import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('https://raw.githubusercontent.com/alslaos3/dataset/main/LungCancer_TRAIN.csv')
x_cols = [c for c in df.columns if c != 'Target']
X = df[x_cols]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# 데이터 전처리
df = df.fillna(0)

rf = RandomForestClassifier(n_estimators=100)
rf = rf.fit(X_train, y_train)

joblib.dump(rf, "./random_forest.joblib", compress=True)
