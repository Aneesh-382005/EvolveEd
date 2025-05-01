import streamlit as st
import pandas as pd
import joblib

model = joblib.load(r"models\xgboostBESTpipeline_full.pkl")


st.set_page_config(page_title="Student Dropout Predictor", layout="centered")
st.title("🎓 Student Dropout Prediction App")
st.markdown("Predict the likelihood of student dropout based on personalized learning data.")

st.markdown("---")
st.subheader("📥 Enter Student Information")

age = st.number_input("Age", min_value=10, max_value=100, step=1)
gender = st.selectbox("Gender", options=["Male", "Female"])
education_level = st.selectbox("Education Level", options=["High School", "Undergraduate", "Postgraduate"])
course_name = st.selectbox("Course Name", options=[
    "Machine Learning", "Python Basics", "Data Science", "Web Development", "Cybersecurity"
])
time_spent_on_videos = st.number_input("Time Spent on Videos (minutes)", min_value=0, step=1)
quiz_attempts = st.number_input("Quiz Attempts", min_value=0, step=1)
quiz_scores = st.number_input("Quiz Scores (0–100)", min_value=0, max_value=100, step=1)
forum_participation = st.number_input("Forum Participation (posts/comments)", min_value=0, step=1)
assignment_completion_rate = st.number_input("Assignment Completion Rate (%)", min_value=0, max_value=100, step=1)
engagement_level = st.selectbox("Engagement Level", options=["Low", "Medium", "High"])
learning_style = st.selectbox("Learning Style", options=["Visual", "Reading/Writing", "Kinesthetic", "Auditory"])
feedback_score = st.slider("Feedback Score (1–5)", min_value=1, max_value=5)

input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Education_Level": education_level,
    "Course_Name": course_name,
    "Time_Spent_on_Videos": time_spent_on_videos,
    "Quiz_Attempts": quiz_attempts,
    "Quiz_Scores": quiz_scores,
    "Forum_Participation": forum_participation,
    "Assignment_Completion_Rate": assignment_completion_rate,
    "Engagement_Level": engagement_level,
    "Learning_Style": learning_style,
    "Feedback_Score": feedback_score
}])

st.markdown("---")
if st.button("🔍 Predict Dropout Likelihood"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")
    st.write(f"**Dropout Probability:** `{1 - probability:.2%}`")
    if prediction == 1:
        st.success("✅ Low Dropout Risk")
    else:
        st.error("⚠️ High Dropout Risk")
