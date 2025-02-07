import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

st.set_page_config(
    page_title="Anxiety Attack Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MentalHealthApp:
    def __init__(self):
        self.model = self.load_model()
        self.setup_page()
        
    def load_model(self):
        try:
            if os.path.exists('mental_health_model.pkl'):
                with open('mental_health_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                    st.success("Model loaded successfully!")
                    return model
            else:
                st.error("Model file not found. Please ensure 'mental_health_model.pkl' is in the same directory.")
                return None
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    def setup_page(self):
        st.title("🧠 Anxiety Attack Assessment Tool")
        st.write("This tool assesses anxiety attack severity based on various factors.")
        
    def create_input_form(self):
        with st.form("assessment_form"):
            st.subheader("Personal Information")
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=100, value=30)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
                physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0, max_value=168, value=3)
                occupation = st.selectbox("Occupation", ["Professional", "Student", "Self-employed", "Unemployed", "Other"])
            
            with col2:
                caffeine_intake = st.number_input("Caffeine Intake (mg/day)", min_value=0, max_value=1000, value=200)
                alcohol_consumption = st.number_input("Alcohol Consumption (drinks/week)", min_value=0, max_value=50, value=2)
                smoking = st.selectbox("Smoking", ["No", "Yes"])
                family_history = st.selectbox("Family History of Anxiety", ["No", "Yes"])
                stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

            st.subheader("Symptoms")
            col3, col4 = st.columns(2)
            
            with col3:
                heart_rate = st.number_input("Heart Rate during attack (bpm)", min_value=40, max_value=200, value=80)
                breathing_rate = st.number_input("Breathing Rate (breaths/min)", min_value=8, max_value=40, value=16)
                sweating_level = st.slider("Sweating Level (1-5)", 1, 5, 3)
                dizziness = st.selectbox("Dizziness", ["No", "Yes"])
            
            with col4:
                medication = st.selectbox("Currently on Medication", ["No", "Yes"])
                therapy_sessions = st.number_input("Therapy Sessions (per month)", min_value=0, max_value=31, value=0)
                recent_event = st.selectbox("Recent Major Life Event", ["No", "Yes"])
                diet_quality = st.slider("Diet Quality (1-10)", 1, 10, 7)

            submitted = st.form_submit_button("Analyze")
            
            if submitted:
                data = {
                    'Age': age,
                    'Gender': gender,
                    'Occupation': occupation,
                    'Sleep Hours': sleep_hours,
                    'Physical Activity (hrs/week)': physical_activity,
                    'Caffeine Intake (mg/day)': caffeine_intake,
                    'Alcohol Consumption (drinks/week)': alcohol_consumption,
                    'Smoking': smoking,
                    'Family History of Anxiety': family_history,
                    'Stress Level (1-10)': stress_level,
                    'Heart Rate (bpm during attack)': heart_rate,
                    'Breathing Rate (breaths/min)': breathing_rate,
                    'Sweating Level (1-5)': sweating_level,
                    'Dizziness': dizziness,
                    'Medication': medication,
                    'Therapy Sessions (per month)': therapy_sessions,
                    'Recent Major Life Event': recent_event,
                    'Diet Quality (1-10)': diet_quality
                }
                return data
            return None

    def encode_categorical(self, df):
        categorical_columns = ['Gender', 'Occupation', 'Smoking', 'Family History of Anxiety', 
                             'Dizziness', 'Medication', 'Recent Major Life Event']
        
        for column in categorical_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
        return df

    def predict(self, inputs):
        try:
            df = pd.DataFrame([inputs])
            df = self.encode_categorical(df)
            
            prediction = self.model.predict(df)[0]
            probabilities = self.model.predict_proba(df)[0]
            confidence = float(np.max(probabilities) * 100)
            
            return {
                'severity': int(prediction),
                'confidence': confidence,
                'probabilities': probabilities
            }
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

    def display_results(self, results):
        if not results:
            return

        st.success("Analysis complete!")
        
        severity_texts = {
            1: "Very Low", 2: "Low", 3: "Mild", 4: "Moderate", 5: "Moderate-High",
            6: "High", 7: "Very High", 8: "Severe", 9: "Extreme", 10: "Critical"
        }
        
        col1, col2 = st.columns(2)
        with col1:
            severity = results['severity']
            severity_text = severity_texts.get(severity, "Unknown")
            st.metric("Severity Level", f"{severity_text} ({severity}/10)")
        
        with col2:
            st.metric("Confidence", f"{results['confidence']:.1f}%")
        
        st.subheader("Probability Distribution")
        probs_df = pd.DataFrame({
            'Severity Level': range(1, 11),
            'Probability': results['probabilities']
        })
        st.bar_chart(probs_df.set_index('Severity Level'))
        
        self.show_recommendations(results['severity'])
        
        st.warning("""
        **Disclaimer**: This tool provides an initial assessment only and should not be considered 
        as a substitute for professional medical advice. Please consult with a qualified mental 
        health professional for proper diagnosis and treatment.
        """)

    def show_recommendations(self, severity):
        st.subheader("Recommendations")
        
        if severity <= 3:
            recommendations = [
                "Continue maintaining healthy lifestyle habits",
                "Practice regular relaxation techniques",
                "Monitor triggers and symptoms",
                "Maintain regular exercise and sleep schedule"
            ]
        elif severity <= 6:
            recommendations = [
                "Consider consulting a mental health professional",
                "Learn and practice anxiety management techniques",
                "Establish a regular self-care routine",
                "Consider joining a support group",
                "Review and possibly reduce caffeine intake"
            ]
        else:
            recommendations = [
                "Strongly recommend immediate professional consultation",
                "Develop an anxiety management plan with professional guidance",
                "Create a support system with trusted individuals",
                "Consider medication options with healthcare provider",
                "Learn and practice emergency coping strategies"
            ]
        
        for rec in recommendations:
            st.markdown(f"• {rec}")

def main():
    app = MentalHealthApp()
    if app.model is not None:
        inputs = app.create_input_form()
        if inputs:
            with st.spinner("Analyzing your responses..."):
                results = app.predict(inputs)
                app.display_results(results)

if __name__ == "__main__":
    main()