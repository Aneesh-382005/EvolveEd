import pandas as pd
import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../"))) #Adds the parent dir to the path

df = pd.read_csv('data/raw/StudentInteractions.csv')

X = df.drop(columns=['correct', 'studentID'])
Y = df['correct']


preprocessor = ColumnTransformer([
    ('topic', OneHotEncoder(), ['topic']),
    ('num', StandardScaler(), ['attempts', 'timeTaken', 'PreviousScoreAverage'])
])

KnowledgeTracingPipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver = 'liblinear'))
])

XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.2, random_state=42)


KnowledgeTracingPipeline.fit(XTrain, YTrain)

yPredictions = KnowledgeTracingPipeline.predict(XTest)
yProbabilities = KnowledgeTracingPipeline.predict_proba(XTest)[:, 1]

print(classification_report(YTest, yPredictions))
print("ROC-AUC:", roc_auc_score(YTest, yProbabilities))
