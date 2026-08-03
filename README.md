
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

# Preprocessing
numeric_features = df.select_dtypes(include=['int64','float64']).columns.tolist()
numeric_features = [c for c in numeric_features if c not in ['target']]
categorical_features = df.select_dtypes(include=['object','category']).columns.tolist()
categorical_features = [c for c in categorical_features if c not in ['gender']]

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = df.drop(columns=['target'])
y = df['target']
A = df['gender']

X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(X, y, A, test_size=0.25, random_state=42, stratify=y)

def make_pipeline(model):
    return Pipeline(steps=[('preproc', preprocessor), ('model', model)])

pip_lr = make_pipeline(LogisticRegression(max_iter=1000))
pip_rf = make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42))
pip_xgb = make_pipeline(xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=200))

models = {'LogisticRegression': pip_lr, 'RandomForest': pip_rf, 'XGBoost': pip_xgb}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
print("Training complete.")
Training LogisticRegression...
Training RandomForest...
Training XGBoost...
Training complete.

print("Applying fairness mitigation...")
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

est = LogisticRegression(max_iter=1000)
mitigator = ExponentiatedGradient(est, constraints=DemographicParity())
mitigator.fit(X_train_trans, y_train, sensitive_features=A_train)

y_pred_mit = mitigator.predict(X_test_trans)

mf_mit = MetricFrame(metrics={'selection_rate': selection_rate, 'tpr': true_positive_rate},
                     y_true=y_test, y_pred=y_pred_mit, sensitive_features=A_test)

sel = mf_mit.by_group['selection_rate']
tpr = mf_mit.by_group['tpr']

mit_res = {
    'accuracy': accuracy_score(y_test, y_pred_mit),
    'precision': precision_score(y_test, y_pred_mit, zero_division=0),
    'recall': recall_score(y_test, y_pred_mit, zero_division=0),
    'f1': f1_score(y_test, y_pred_mit, zero_division=0),
    'statistical_parity_diff': sel.max() - sel.min(),
    'equal_opportunity_diff': tpr.max() - tpr.min(),
    'disparate_impact': sel.min() / sel.max() if sel.max() > 0 else np.nan
}

pd.DataFrame([mit_res], index=['After Mitigation']).T
Applying fairness mitigation...

# --- Comparison: Before vs After Mitigation ---
baseline_metrics = results['LogisticRegression']

comparison_df = pd.DataFrame([baseline_metrics, mit_res],
                             index=['Before Mitigation', 'After Mitigation']).T

display(comparison_df)

# Visualize key metrics
metrics_to_plot = ['accuracy', 'statistical_parity_diff', 'equal_opportunity_diff', 'disparate_impact']

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    comparison_df.loc[metric].plot(kind='bar', ax=axes[i], color=['#007acc', '#009e73'])
    axes[i].set_title(metric.replace('_', ' ').title())
    axes[i].set_ylabel('Score')
    axes[i].set_xticklabels(['Before', 'After'], rotation=0)

plt.suptitle('Model Performance & Fairness: Before vs After Mitigation', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("Interpretation:")
print("- ↓ SPD & EOD after mitigation => improved fairness.")
print("- DI closer to 1 => improved fairness balance.")
print("- Slight accuracy drop acceptable for ethical AI.")

# --- Comparison: Before vs After Mitigation ---
baseline_metrics = results['LogisticRegression']

comparison_df = pd.DataFrame([baseline_metrics, mit_res],
                             index=['Before Mitigation', 'After Mitigation']).T

display(comparison_df)

# Visualize key metrics
metrics_to_plot = ['accuracy', 'statistical_parity_diff', 'equal_opportunity_diff', 'disparate_impact']

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    comparison_df.loc[metric].plot(kind='bar', ax=axes[i], color=['#007acc', '#009e73'])
    axes[i].set_title(metric.replace('_', ' ').title())
    axes[i].set_ylabel('Score')
    axes[i].set_xticklabels(['Before', 'After'], rotation=0)

plt.suptitle('Model Performance & Fairness: Before vs After Mitigation', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("Interpretation:")
print("- ↓ SPD & EOD after mitigation => improved fairness.")
print("- DI closer to 1 => improved fairness balance.")
print("- Slight accuracy drop acceptable for ethical AI.")
<Figure size 1000x800 with 4 Axes><img width="989" height="789" alt="image" src="https://github.com/user-attachments/assets/120d725f-36b4-4309-86be-10940de13f75" />
Interpretation:
- ↓ SPD & EOD after mitigation => improved fairness.
- DI closer to 1 => improved fairness balance.
- Slight accuracy drop acceptable for ethical AI.

## Conclusion

- The **baseline Logistic Regression model** achieved good accuracy but showed bias across gender groups.
- After applying **Exponentiated Gradient fairness constraint**, fairness metrics (SPD, EOD, DI) improved notably.
- The **accuracy dropped slightly**, but fairness increased significantly — showing that responsible AI can balance both.
- Results support **ethical and fair credit scoring** for financial inclusion.

**Next Steps:**
- Try EqualizedOdds constraint for deeper fairness trade-offs.
- Compare with AIF360 Reweighing preprocessing method.
- Perform k-fold validation and interpretability analysis (SHAP/LIME).
import joblib
joblib.dump(model, "full_fair_credit_model.pkl")
joblib.dump(preprocessor, "full_preprocessor.pkl")
['full_preprocessor.pkl']

