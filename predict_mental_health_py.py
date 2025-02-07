import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg') 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import shap
import pickle

"""Data Preprocessing"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class MentalHealthDataPreprocessor:
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)

    def clean_data(self):
        # Remove duplicate rows
        self.data.drop_duplicates(inplace=True)
        # Handle missing values
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        # Numeric imputation with median
        numeric_imputer = SimpleImputer(strategy='median')
        self.data[numeric_columns] = numeric_imputer.fit_transform(self.data[numeric_columns])

        # Categorical imputation with mode
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.data[categorical_columns] = categorical_imputer.fit_transform(self.data[categorical_columns])
        return self

    def encode_categorical_features(self):
        # Encode categorical variables
        le = LabelEncoder()
        categorical_columns = ['Gender', 'Occupation', 'Medication', 'Family History of Anxiety', 'Smoking', 'Dizziness','Recent Major Life Event' ]
        for col in categorical_columns:
            self.data[col] = le.fit_transform(self.data[col].astype(str))
        return self

    def scale_features(self, features):
        # Scale numeric features
        scaler = StandardScaler()
        self.data[features] = scaler.fit_transform(self.data[features])
        return self, scaler

    def split_data(self, target_column, test_size=0.2, random_state=42):
        # Features and target split
        X = self.data.drop(columns=[target_column, 'ID'])
        y = self.data[target_column]
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
preprocessor = MentalHealthDataPreprocessor('anxiety_attack_dataset.csv')
preprocessor.clean_data().encode_categorical_features()
print(preprocessor.data.head())
X_train, X_test, y_train, y_test = preprocessor.split_data('Severity of Anxiety Attack (1-10)')
print(X_train.head())
print(y_train.head())
print(X_test.head())
print(y_test.head())
"""EDA"""

def perform_eda(data):
    plt.figure(figsize=(20, 20))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    # Distribution of Target Variable
    plt.figure(figsize=(15, 10))
    data['Severity of Anxiety Attack (1-10)'].hist(bins=10)
    plt.title('Distribution of Anxiety Attack Severity')
    plt.xlabel('Severity Level')
    plt.ylabel('Frequency')
    plt.show()

    key_features = [
        'Age', 'Sleep Hours', 'Physical Activity (hrs/week)',
        'Caffeine Intake (mg/day)', 'Stress Level (1-10)'
    ]

    plt.figure(figsize=(20, 15))
    data[key_features].boxplot()
    plt.title('Box Plots of Key Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Feature importance using correlation with target
    feature_importance = correlation_matrix['Severity of Anxiety Attack (1-10)'].abs().sort_values(ascending=False)
    print("Feature Importance Relative to Anxiety Severity:")
    print(feature_importance)

perform_eda(preprocessor.data)

"""Model Development and deployment"""

class MentalHealthPredictor:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1, max_depth=10,  # Prevent overfitting
                verbose=1 ),
            'LogisticRegression': LogisticRegression(multi_class='ovr', max_iter=1000,  n_jobs=-1,  # Parallel processing
                verbose=1 )
        }
        self.best_model = None

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        results = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }

        # Select best model based on F1 score
        self.best_model = max(results, key=lambda x: results[x]['f1_score'])
        print(f"Best Model: {self.best_model}")
        print(classification_report(y_test, self.models[self.best_model].predict(X_test)))
        return results

    def explain_predictions(self, X_test):
        # SHAP explanation for model interpretability
        explainer = shap.TreeExplainer(self.models[self.best_model])
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test)

    def save_model(self, filename='mental_health_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.models[self.best_model], f)

    @classmethod
    def load_model(cls, filename='mental_health_model.pkl'):
        with open(filename, 'rb') as f:
            return pickle.load(f)

predictor = MentalHealthPredictor()
results = predictor.train_and_evaluate(X_train, X_test, y_train, y_test)
predictor.explain_predictions(X_test)
predictor.save_model()

"""LLM"""

class MentalHealthLLMExplainer:
    def __init__(self, model_name="facebook/opt-350m"):
        """Initialize LLM for mental health explanations"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.generator = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=500,
                do_sample=True,  # Enable sampling
                temperature=0.7,
                truncation=True,  # Enable truncation
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            self.generator = None

    def generate_explanation(self, severity_level, symptoms):
        """Generate detailed explanation for mental health condition"""
        if not self.generator:
            return self._fallback_explanation(severity_level)

        prompt = self._construct_prompt(severity_level, symptoms)

        try:
            response = self.generator(
                prompt,
                max_new_tokens=300,  # Control the length of generated text
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2
            )[0]['generated_text']

            explanation = response.split("Explanation:")[-1].strip()

            # If explanation is empty or too short, use fallback
            if not explanation or len(explanation.split()) < 10:
                return self._fallback_explanation(severity_level)

            return {
                'full_explanation': explanation,
                'key_points': self._extract_key_points(explanation)
            }
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._fallback_explanation(severity_level)

    def _construct_prompt(self, severity_level, symptoms):
        """Construct prompt for the LLM"""
        symptom_text = "\n".join([f"- {k}: {v}" for k, v in symptoms.items()])
        return f"""As a mental health professional, provide a detailed assessment based on the following:

Severity Level: {severity_level}

Patient Symptoms:
{symptom_text}

Please provide a comprehensive explanation including:
1. An assessment of the current mental health state
2. The significance of the observed symptoms
3. Potential impact on daily life
4. Suggested coping strategies

Explanation:"""

    def _extract_key_points(self, explanation):
        """Extract key points from the explanation"""
        if not explanation:
            return ["Please seek professional mental health assessment"]

        # Split into sentences and clean them
        sentences = [s.strip() for s in explanation.split('.') if s.strip()]

        # If we have less than 3 sentences, add standard points
        while len(sentences) < 3:
            sentences.append("Consider consulting a mental health professional for a complete assessment")

        return sentences[:3]  # Return first 3 non-empty sentences

    def _fallback_explanation(self, severity_level):
        """Enhanced fallback explanations"""
        severity_explanations = {
            'Very Low': {
                'explanation': """Based on the assessment, your current stress and anxiety levels appear to be within a manageable range. Your reported symptoms suggest you have generally effective coping mechanisms in place. While your current state shows good mental health management, it's important to maintain these positive practices and stay aware of any changes in your mental well-being. Your lifestyle choices, including sleep patterns and physical activity, are contributing positively to your mental health stability.""",
                'key_points': [
                    "Current stress levels are well-managed with effective coping strategies",
                    "Lifestyle choices are supporting good mental health",
                    "Continue monitoring and maintaining healthy practices"
                ]
            },
            'Moderate': {
                'explanation': """The assessment indicates moderate levels of mental health challenges that could benefit from professional support. Your reported symptoms suggest some impact on daily functioning, though you're maintaining stability. While you have some coping mechanisms in place, additional support and strategies could help improve your mental well-being. Consider this an opportunity to strengthen your mental health toolkit and develop more robust coping strategies.""",
                'key_points': [
                    "Current symptoms indicate moderate stress affecting daily life",
                    "Existing coping strategies may need enhancement",
                    "Professional support could provide additional beneficial tools"
                ]
            },
            'High': {
                'explanation': """The assessment suggests significant mental health challenges that require immediate professional attention. Your reported symptoms indicate substantial impact on daily functioning and well-being. The combination and intensity of symptoms suggest a need for comprehensive professional support and potentially a structured treatment plan. It's important to prioritize seeking professional help to develop effective management strategies.""",
                'key_points': [
                    "Symptoms indicate significant impact on daily functioning",
                    "Immediate professional support is strongly recommended",
                    "Comprehensive treatment plan may be beneficial"
                ]
            }
        }

        # Get the closest severity level if exact match not found
        if severity_level not in severity_explanations:
            if 'Low' in severity_level:
                severity_level = 'Very Low'
            elif 'High' in severity_level:
                severity_level = 'High'
            else:
                severity_level = 'Moderate'

        return severity_explanations.get(
            severity_level,
            {
                'explanation': """Based on the assessment results, a professional mental health evaluation is recommended for a more comprehensive understanding of your current mental health state. Your reported symptoms suggest the importance of consulting with a mental health professional who can provide personalized guidance and support strategies.""",
                'key_points': [
                    "Professional mental health evaluation is recommended",
                    "Personalized assessment would be beneficial",
                    "Consider consulting with a mental health professional"
                ]
            }
        )
"""Inference Script"""

from sklearn.impute import SimpleImputer
import pickle
import os

class MentalHealthInference:
    def __init__(self, model_path='mental_health_model.pkl'):
        """Initialize the inference class with a trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.label_encoder = LabelEncoder()
        self.llm_explainer = MentalHealthLLMExplainer()

        self.expected_columns = [
            'Age', 'Gender', 'Occupation', 'Sleep Hours',
            'Physical Activity (hrs/week)', 'Caffeine Intake (mg/day)',
            'Alcohol Consumption (drinks/week)', 'Smoking',
            'Family History of Anxiety', 'Stress Level (1-10)',
            'Heart Rate (bpm during attack)', 'Breathing Rate (breaths/min)',
            'Sweating Level (1-5)', 'Dizziness', 'Medication',
            'Therapy Sessions (per month)', 'Recent Major Life Event',
            'Diet Quality (1-10)'
        ]

        self.categorical_columns = [
            'Gender', 'Occupation', 'Medication',
            'Family History of Anxiety', 'Smoking',
            'Dizziness', 'Recent Major Life Event'
        ]

    def validate_input(self, input_data):
        """Validate input data format and values"""
        if isinstance(input_data, dict):
            missing_cols = set(self.expected_columns) - set(input_data.keys())
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

        df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else input_data

        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().all():
                raise ValueError(f"Column {col} contains all null values")

        return df

    def preprocess_input(self, input_data):
        """Preprocess input data using the same steps as training"""
        try:
            input_df = self.validate_input(input_data)

            # Handle numeric columns
            numeric_columns = input_df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in input_df.columns:
                    input_df[[col]] = self.numeric_imputer.fit_transform(input_df[[col]])

            # Handle categorical columns
            for col in self.categorical_columns:
                if col in input_df.columns:
                    input_df[col] = input_df[col].astype(str)
                    input_df[[col]] = self.categorical_imputer.fit_transform(input_df[[col]])
                    input_df[col] = self.label_encoder.fit_transform(input_df[col])

            return input_df

        except Exception as e:
            raise Exception(f"Error in preprocessing: {str(e)}")

    def predict_severity(self, symptoms):
        """Predict mental health severity based on input symptoms"""
        try:
            processed_input = self.preprocess_input(symptoms)
            severity_prediction = self.model.predict(processed_input)
            prediction_proba = self.model.predict_proba(processed_input)

            severity_levels = {
                1: "Very Low",
                2: "Low",
                3: "Mild",
                4: "Moderate",
                5: "Moderate-High",
                6: "High",
                7: "Very High",
                8: "Severe",
                9: "Extreme",
                10: "Critical"
            }

            result = {
                'predicted_severity': int(severity_prediction[0]),
                'severity_description': severity_levels.get(int(severity_prediction[0]), "Unknown"),
                'confidence': float(max(prediction_proba[0]) * 100),
                'probability_distribution': {
                    f'Level {i+1}': float(prob * 100)
                    for i, prob in enumerate(prediction_proba[0])
                }
            }

            return result

        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")

    def generate_recommendations(self, severity_level):
        """Generate personalized recommendations based on severity level"""
        base_recommendations = {
            'Very Low': [
                "Continue maintaining your current healthy practices",
                "Practice regular mindfulness or meditation",
                "Maintain a consistent sleep schedule",
                "Stay physically active"
            ],
            'Low': [
                "Consider starting a stress diary",
                "Practice deep breathing exercises",
                "Maintain regular exercise routine",
                "Establish a regular sleep schedule"
            ],
            'Mild': [
                "Consider talking to a counselor or therapist",
                "Learn and practice stress management techniques",
                "Maintain social connections",
                "Establish a regular exercise routine"
            ],
            'Moderate': [
                "Schedule an appointment with a mental health professional",
                "Learn and practice anxiety management techniques",
                "Consider joining a support group",
                "Maintain regular communication with loved ones"
            ],
            'Moderate-High': [
                "Seek professional mental health support",
                "Consider therapy or counseling sessions",
                "Learn and practice grounding techniques",
                "Establish a strong support system"
            ],
            'High': [
                "Consult with a mental health professional immediately",
                "Consider therapy and possibly medication options",
                "Practice daily anxiety management techniques",
                "Stay connected with support system"
            ],
            'Very High': [
                "Seek immediate professional mental health support",
                "Consider comprehensive treatment options",
                "Maintain close contact with healthcare providers",
                "Ensure daily support from family or friends"
            ],
            'Severe': [
                "Seek immediate professional help",
                "Consider emergency mental health services if needed",
                "Maintain constant communication with support system",
                "Follow crisis management plan if available"
            ],
            'Extreme': [
                "Contact emergency mental health services",
                "Seek immediate professional intervention",
                "Do not remain alone - contact trusted support person",
                "Consider emergency room visit if symptoms are overwhelming"
            ],
            'Critical': [
                "Seek emergency medical attention immediately",
                "Contact crisis hotline or emergency services",
                "Do not remain alone - seek immediate support",
                "Consider immediate psychiatric evaluation"
            ]
        }

        return base_recommendations.get(severity_level, [
            "Seek professional mental health evaluation",
            "Contact a mental health professional",
            "Consider discussing symptoms with your healthcare provider"
        ])

    def predict_and_explain(self, symptoms):
        """Complete mental health assessment with ML prediction and LLM explanation"""
        try:
            # Get prediction
            prediction = self.predict_severity(symptoms)

            # Generate explanation
            explanation = self.llm_explainer.generate_explanation(
                prediction['severity_description'],
                symptoms
            )

            # Combine results
            result = {
                **prediction,
                'explanation': explanation['full_explanation'],
                'key_points': explanation['key_points'],
                'recommendations': self.generate_recommendations(prediction['severity_description'])
            }

            # Print comprehensive results
            print("\n=== Mental Health Assessment Results ===")
            print(f"\nPredicted Severity Level: {result['severity_description']}")
            print(f"Confidence: {result['confidence']:.2f}%")

            print("\nProbability Distribution:")
            for level, prob in result['probability_distribution'].items():
                print(f"{level}: {prob:.2f}%")

            print("\nExplanation:")
            print(result['explanation'])

            print("\nKey Points:")
            for point in result['key_points']:
                print(f"- {point}")

            print("\nRecommended Actions:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"{i}. {rec}")

            return result

        except Exception as e:
            raise Exception(f"Error in assessment: {str(e)}")

# Example usage
if __name__ == "__main__":
    test_input = {
        'Age': 30,
        'Gender': 'Male',
        'Occupation': 'Professional',
        'Sleep Hours': 7,
        'Physical Activity (hrs/week)': 3,
        'Caffeine Intake (mg/day)': 200,
        'Alcohol Consumption (drinks/week)': 2,
        'Smoking': 'No',
        'Family History of Anxiety': 'Yes',
        'Stress Level (1-10)': 6,
        'Heart Rate (bpm during attack)': 100,
        'Breathing Rate (breaths/min)': 20,
        'Sweating Level (1-5)': 3,
        'Dizziness': 'No',
        'Medication': 'No',
        'Therapy Sessions (per month)': 0,
        'Recent Major Life Event': 'No',
        'Diet Quality (1-10)': 7
    }

    try:
        inference = MentalHealthInference('mental_health_model.pkl')
        result = inference.predict_and_explain(test_input)
    except Exception as e:
        print(f"Error: {str(e)}")