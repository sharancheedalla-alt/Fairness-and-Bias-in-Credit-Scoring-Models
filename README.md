
# Fairness and Bias in Credit Scoring Models

**Project:** Fairness and Bias in Credit Scoring Models  
**Team:** Ch. Sharan, K. Abhiram, L. Akash  
**Guide:** Mr. V. Shashivanth

### Objective
Analyze and mitigate bias in credit scoring models, and **compare performance and fairness before and after mitigation**.

# Import libraries and install dependencies if missing
!pip install --quiet numpy pandas scikit-learn matplotlib seaborn xgboost fairlearn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_openml

print("Fetching German Credit dataset...")
german = fetch_openml(name='credit-g', version=1, as_frame=True)
df = german.frame.copy()

# Handle target
if 'class' in df.columns:
    df.rename(columns={'class': 'target'}, inplace=True)
elif 'Creditability' in df.columns:
    df.rename(columns={'Creditability': 'target'}, inplace=True)

df['target'] = df['target'].astype(str).map({'good': 1, 'bad': 0})

# Define protected attribute
if 'sex' in df.columns:
    df['gender'] = df['sex'].map({'male': 'male', 'female': 'female'})
elif 'personal_status' in df.columns:
    df['gender'] = df['personal_status'].apply(lambda x: 'female' if 'female' in x.lower() else 'male')
else:
    df['gender'] = np.where(np.random.rand(len(df)) > 0.5, 'male', 'female')

print("Dataset shape:", df.shape)
df.head()
Fetching German Credit dataset...
Dataset shape: (1000, 22)
