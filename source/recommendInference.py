import streamlit as st
import pandas as pd
# Removed unused numpy import
import joblib
import os

# -----------------------
# Encoding Maps
# -----------------------
encoders = {
    "Gender": {"Male": 0, "Female": 1},
    "Education_Level": {"High School": 0, "Undergraduate": 1, "Postgraduate": 2},
    "Engagement_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Learning_Style": {"Visual": 0, "Reading/Writing": 1, "Kinesthetic": 2, "Auditory": 3},
}

def encode_categorical(df):
    for col, mapping in encoders.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

# -----------------------
# Cluster Configurations
# -----------------------
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

# -----------------------
# Recommendation Logic
# -----------------------
def generate_recommendations(row, cluster_name):
    recommendations = []
    if "Engagement" in cluster_name:
        if row['engagementScore'] < 0:
            recommendations.append("Increase time on videos and forum activity.")
        if row['Assignment_Completion_Rate'] < 0:
            recommendations.append("Focus on completing more assignments.")

    if "Performance" in cluster_name:
        if row['avgQuizScorePerAttempt'] < 0:
            recommendations.append("Review quiz material; low efficiency detected.")
        if row['consistencyIndex'] > 1.5:
            recommendations.append("Focus on consistent performance across exams.")

    if "DropoutRisk" in cluster_name:
        if row['Dropout_Likelihood'] > 0.5:
            recommendations.append("High dropout risk – engage with mentors.")
        if row['Feedback_Score'] < 0:
            recommendations.append("Submit more feedback for personalized help.")

    return recommendations or ["Performing well! Keep going!"]

# -----------------------
# Feature Engineering
# -----------------------
def engineer_features(df):
    df['avgQuizScorePerAttempt'] = df['Quiz_Scores'] / df['Quiz_Attempts'].replace(0, 1)
    df['engagementScore'] = 0.4 * df['Time_Spent_on_Videos'] + \
                            0.3 * df['Forum_Participation'] + \
                            0.3 * df['Assignment_Completion_Rate']
    df['videoEfficiency'] = df['Quiz_Scores'] / df['Time_Spent_on_Videos'].replace(0, 1)
    df['consistencyIndex'] = abs(df['Quiz_Scores'] - df['Final_Exam_Score'])
    return df

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Adaptive Learning Clustering", layout="wide")
st.title("📊 Adaptive Learning Path Clustering & Recommendations")

mode = st.radio("Choose input mode:", ["Upload CSV", "Manual Entry"])

if mode == "Manual Entry":
    st.subheader("🎓 Enter Student Details")

    with st.form("student_form"):
        student_data = {
            "Student_ID": st.text_input("Student ID", "S00008"),
            "Age": st.number_input("Age", 10, 100, 20),
            "Gender": st.selectbox("Gender", list(encoders["Gender"].keys())),
            "Education_Level": st.selectbox("Education Level", list(encoders["Education_Level"].keys())),
            "Course_Name": st.text_input("Course Name", "Python Basics"),  # Not used in clustering
            "Time_Spent_on_Videos": st.number_input("Time Spent on Videos", 0.0, 1000.0, 200.0),
            "Quiz_Attempts": st.number_input("Quiz Attempts", 0, 20, 3),
            "Quiz_Scores": st.number_input("Quiz Scores", 0.0, 100.0, 65.0),
            "Forum_Participation": st.number_input("Forum Participation", 0.0, 100.0, 10.0),
            "Assignment_Completion_Rate": st.number_input("Assignment Completion Rate", 0.0, 100.0, 85.0),
            "Engagement_Level": st.selectbox("Engagement Level", list(encoders["Engagement_Level"].keys())),
            "Final_Exam_Score": st.number_input("Final Exam Score", 0.0, 100.0, 70.0),
            "Learning_Style": st.selectbox("Learning Style", list(encoders["Learning_Style"].keys())),
            "Feedback_Score": st.number_input("Feedback Score", 0.0, 5.0, 3.0),
            "Dropout_Likelihood": st.number_input("Dropout Likelihood", 0.0, 1.0, 0.2),
        }

        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        input_df = pd.DataFrame([student_data])
        input_df = encode_categorical(input_df)
        input_df = engineer_features(input_df)

        for cluster_name, features in clusterConfigs.items():
            model_path = f"models/{cluster_name}_KMeans.pkl"
            scaler_path = f"models/{cluster_name}_Scaler.pkl"

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                st.warning(f"⚠️ Missing model or scaler for {cluster_name}. Skipping.")
                continue

            km_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            X_scaled = scaler.transform(input_df[features])
            cluster_label = km_model.predict(X_scaled)[0]
            input_df[f"{cluster_name}_Cluster"] = cluster_label
            input_df[f"{cluster_name}_Recommendations"] = generate_recommendations(input_df.iloc[0], cluster_name)

        st.success("✅ Prediction & Recommendations Ready")
        # Ensure consistent data types for Arrow compatibility
        input_df = input_df.astype(str)
        st.write(input_df.T)

else:
    uploaded_file = st.file_uploader("📁 Upload student CSV file", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df = encode_categorical(df)
        df = engineer_features(df)

        for cluster_name, features in clusterConfigs.items():
            model_path = f"models/{cluster_name}_KMeans.pkl"
            scaler_path = f"models/{cluster_name}_Scaler.pkl"

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                st.warning(f"⚠️ Missing model or scaler for {cluster_name}. Skipping.")
                continue

            km_model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            X_scaled = scaler.transform(df[features])
            df[f"{cluster_name}_Cluster"] = km_model.predict(X_scaled)
            df[f"{cluster_name}_Recommendations"] = df.apply(
                lambda row: generate_recommendations(row, cluster_name), axis=1
            )
        # Ensure consistent data types for Arrow compatibility
        df = df.astype(str)
        st.dataframe(df)
        st.success("✅ All Students Clustered and Profiled")
        st.dataframe(df)
        st.download_button("📥 Download Result CSV", df.to_csv(index=False), file_name="clustered_output.csv")
    else:
        st.info("👈 Upload a student dataset CSV to get started.")
