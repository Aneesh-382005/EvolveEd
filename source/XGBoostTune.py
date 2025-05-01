import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import warnings

warnings.filterwarnings('ignore')

try:
    dataFrame = pd.read_csv('data\\raw\\personalized_learning_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: personalized_learning_dataset.csv not found.")
    exit()

dataFrame = dataFrame.drop('Student_ID', axis=1)

if 'Final_Exam_Score' in dataFrame.columns:
    dataFrame = dataFrame.drop('Final_Exam_Score', axis=1)
    print("Dropped 'Final_Exam_Score' to prevent data leakage.")
else:
    print("'Final_Exam_Score' column not found.")

dataFrame['Dropout_Likelihood'] = dataFrame['Dropout_Likelihood'].map({'No': 0, 'Yes': 1})

X = dataFrame.drop('Dropout_Likelihood', axis=1)
y = dataFrame['Dropout_Likelihood']

catFeatures = X.select_dtypes(include=['object', 'category']).columns.tolist()
numFeatures = X.select_dtypes(include=np.number).columns.tolist()

print(f"\nNumerical features: {numFeatures}")
print(f"Categorical features: {catFeatures}")

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"\nData split: {len(xTrain)} training samples, {len(xTest)} testing samples.")
print(f"Dropout rate in training set: {yTrain.mean():.2%}")
print(f"Dropout rate in testing set: {yTest.mean():.2%}")

preProcessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numFeatures),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), catFeatures)
    ],
    remainder='passthrough'
)

xgbClassifier = XGBClassifier(random_state=42, eval_metric='logloss')

paramGrid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 7, 9],
    'classifier__learning_rate': [0.05, 0.1, 0.15],
    'classifier__subsample': [0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0],
    'classifier__gamma': [0, 0.1, 0.2]
}

pipeline = Pipeline(steps=[('preProcessor', preProcessor), ('classifier', xgbClassifier)])

gridSearch = GridSearchCV(pipeline, paramGrid, cv=3, scoring='f1', verbose=1, n_jobs=-1)
gridSearch.fit(xTrain, yTrain)

bestModel = gridSearch.best_estimator_
bestModel.fit(xTrain, yTrain)

bestParams = gridSearch.best_params_
print("\nBest Parameters:", bestParams)

yPred = bestModel.predict(xTest)
yProba = bestModel.predict_proba(xTest)[:, 1]

accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
aucRoc = roc_auc_score(yTest, yProba)

print("\nEvaluation Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  AUC-ROC:   {aucRoc:.4f}")

joblib.dump(bestModel, 'models\\xgboostBESTpipeline.pkl')
print("\nPipeline (including scaler) saved as 'xgboostBESTpipeline.pkl'.")

X_full = pd.concat([xTrain, xTest])
y_full = pd.concat([yTrain, yTest])

bestModel = gridSearch.best_estimator_
bestModel.fit(X_full, y_full)

joblib.dump(bestModel, 'models\\xgboostBESTpipeline_full.pkl')
print("\nFinal pipeline (trained on full dataset) saved as 'xgboostBESTpipeline_full.pkl'.")