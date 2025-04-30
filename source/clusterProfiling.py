import pandas as pd
# Removed unused import
import joblib
import os

from sklearn.preprocessing import StandardScaler
# Removed unused import
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering

from preprocessing import DataPreprocessor

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../")))

df = pd.read_csv('data/preprocessed/personalized_learning_dataset.csv')
preprocessor = DataPreprocessor(df)
df = preprocessor.preprocess()
if 'Student_ID' in df.columns:
    df = df.drop(columns=['Student_ID'])


df['avgQuizScorePerAttempt'] = df['Quiz_Scores'] / df['Quiz_Attempts'].replace(0, 1)
df['engagementScore'] = 0.4 * df['Time_Spent_on_Videos'] + \
                        0.3 * df['Forum_Participation'] + \
                        0.3 * df['Assignment_Completion_Rate']
df['videoEfficiency'] = df['Quiz_Scores'] / df['Time_Spent_on_Videos'].replace(0, 1)
df['consistencyIndex'] = abs(df['Quiz_Scores'] - df['Final_Exam_Score'])


clusterConfigs = {
    "Engagement": [
        'Time_Spent_on_Videos', 'Forum_Participation',
        'Assignment_Completion_Rate', 'engagementScore', 'Engagement_Level'
    ],
    "Performance": [
        'Quiz_Scores', 'Quiz_Attempts', 'Final_Exam_Score',
        'avgQuizScorePerAttempt', 'consistencyIndex'
    ],
    "LearningStyleBehavior": [
        'Learning_Style', 'Time_Spent_on_Videos',
        'Dropout_Likelihood', 'Feedback_Score', 'videoEfficiency'
    ],
    "DropoutRisk": [
        'Dropout_Likelihood', 'Assignment_Completion_Rate',
        'Quiz_Attempts', 'Feedback_Score', 'consistencyIndex', 'engagementScore'
    ]
}

os.makedirs("models", exist_ok=True)
os.makedirs("clusterProfiles", exist_ok=True)

def clusterAndProfile(df, features, clusterName, nClusters=4):
    print(f"\n🔹 Clustering: {clusterName}")
    X = df[features]
    scaler = StandardScaler()
    XScaled = scaler.fit_transform(X)

    kmModel = KMeans(n_clusters=nClusters, random_state=42)
    agModel = AgglomerativeClustering(n_clusters=nClusters)

    kmLabels = kmModel.fit_predict(XScaled)
    agLabels = agModel.fit_predict(XScaled)

    joblib.dump(kmModel, f"models/{clusterName}_KMeans.pkl")
    joblib.dump(scaler, f"models/{clusterName}_Scaler.pkl")

    df[f"{clusterName}_KM"] = kmLabels
    df[f"{clusterName}_AG"] = agLabels

    silKm = silhouette_score(XScaled, kmLabels)
    silAg = silhouette_score(XScaled, agLabels)
    print(f"Silhouette Scores -> KMeans: {silKm:.3f}, Agglomerative: {silAg:.3f}")

    profile = df.groupby(f"{clusterName}_KM")[features].mean().round(2)
    profile.to_csv(f"clusterProfiles/{clusterName}_profile.csv")
    print(f"Saved profile to: clusterProfiles/{clusterName}_profile.csv")

    return df, profile


profiles = {}
for name, feats in clusterConfigs.items():
    df, profile = clusterAndProfile(df, feats, name)
    profiles[name] = profile


def generateRecommendations(df, cluster_column):
    print(f"\nRecommendations for: {cluster_column}")
    recs = {}

    grouped = df.groupby(cluster_column)
    for cluster_id, group in grouped:
        recommendations = []

        # Engagement cluster
        if "Engagement" in cluster_column:
            if group['engagementScore'].mean() < 0:
                recommendations.append("Increase time on videos and forum activity.")
            if group['Assignment_Completion_Rate'].mean() < 0:
                recommendations.append("Focus on completing more assignments.")

        # Performance cluster
        if "Performance" in cluster_column:
            if group['avgQuizScorePerAttempt'].mean() < 0:
                recommendations.append("Review quiz material; low efficiency detected.")
            if group['consistencyIndex'].mean() > 1.5:
                recommendations.append("Focus on consistent performance across exams.")

        # Dropout cluster
        if "DropoutRisk" in cluster_column:
            if group['Dropout_Likelihood'].mean() > 0.5:
                recommendations.append("High dropout risk – engage with mentors.")
            if group['Feedback_Score'].mean() < 0:
                recommendations.append("Submit more feedback for personalized help.")

        recs[f"Cluster {cluster_id}"] = recommendations or ["Performing well! Keep going!"]
    
    return recs


for clusterName in clusterConfigs.keys():
    if f"{clusterName}_KM" in df.columns:
        df[f"{clusterName}_Recs"] = generateRecommendations(df, f"{clusterName}_KM")
    else:
        print(f"Warning: {clusterName}_KM column not found in DataFrame.")


df.to_csv("data/finalWithClustersAndRecommendations.csv", index=False)
print("\n All clustering, profiling, and recommendations saved.")
