import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

try:
    df = pd.read_csv('data\\raw\personalized_learning_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: personalized_learning_dataset.csv not found.")
    exit()

df = df.drop('Student_ID', axis=1)

if 'Final_Exam_Score' in df.columns:
    df = df.drop('Final_Exam_Score', axis=1)
    print("Dropped 'Final_Exam_Score' to prevent data leakage.")
else:
    print("'Final_Exam_Score' column not found.")

df['Dropout_Likelihood'] = df['Dropout_Likelihood'].map({'No': 0, 'Yes': 1})

X = df.drop('Dropout_Likelihood', axis=1)
y = df['Dropout_Likelihood']

catFeatures = X.select_dtypes(include=['object', 'category']).columns.tolist()
numFeatures = X.select_dtypes(include=np.number).columns.tolist()

print(f"\nNumerical features: {numFeatures}")
print(f"Categorical features: {catFeatures}")

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"\nData split: {len(XTrain)} training samples, {len(XTest)} testing samples.")
print(f"Dropout rate in training set: {yTrain.mean():.2%}")
print(f"Dropout rate in testing set: {yTest.mean():.2%}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numFeatures),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), catFeatures)
    ],
    remainder='passthrough'
)

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = {}

print("\n--- Training and Evaluating Models ---")

for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(XTrain, yTrain)
    yPred = pipeline.predict(XTest)
    yProba = pipeline.predict_proba(XTest)[:, 1]

    acc = accuracy_score(yTest, yPred)
    prec = precision_score(yTest, yPred)
    rec = recall_score(yTest, yPred)
    f1 = f1_score(yTest, yPred)
    auc = roc_auc_score(yTest, yProba)

    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc
    }

    print(f"{name} Evaluation:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

print("\n--- Model Comparison ---")
resultsDf = pd.DataFrame(results).T
print(resultsDf)

bestAccModel = resultsDf['Accuracy'].idxmax()
bestF1Model = resultsDf['F1-Score'].idxmax()
bestAucModel = resultsDf['AUC-ROC'].idxmax()

print(f"\nBest Model based on Accuracy:  {bestAccModel} (Accuracy: {resultsDf.loc[bestAccModel, 'Accuracy']:.4f})")
print(f"Best Model based on F1-Score:  {bestF1Model} (F1-Score: {resultsDf.loc[bestF1Model, 'F1-Score']:.4f})")
print(f"Best Model based on AUC-ROC:   {bestAucModel} (AUC-ROC: {resultsDf.loc[bestAucModel, 'AUC-ROC']:.4f})")

print("\nRecommendation:")
if bestF1Model == bestAucModel:
    print(f"{bestF1Model} appears to be the best overall.")
elif bestAccModel in [bestF1Model, bestAucModel]:
    print(f"{bestAccModel} performed best on accuracy, but consider {bestF1Model} or {bestAucModel} for balanced performance.")
else:
    print(f"{bestAccModel} leads in accuracy, {bestF1Model} in F1, and {bestAucModel} in AUC. Prefer {bestF1Model} or {bestAucModel} for balanced results.")
