# Anxiety Attack Assessment Tool

This project is a **Machine Learning-based Web Application** that assesses the severity of anxiety attacks based on various lifestyle and health factors. The application is built using **Streamlit** and **Scikit-learn**, and utilizes a **Random Forest Classifier** to predict the severity of anxiety attacks.

---

## Installation Guide

Follow these steps to install the required dependencies and set up the project.

### 1. Clone the Repository
```sh
$ git clone <(https://github.com/22BCS16084/Arogo-AI-Mnetal-Health-Project)>
$ cd <repository_folder>
```

### 2. Create a Virtual Environment (Optional but Recommended)
```sh
$ python -m venv venv
$ source venv/bin/activate  # On macOS/Linux
$ venv\Scripts\activate     # On Windows
```

### 3. Install Required Libraries
```sh
$ pip install pandas numpy matplotlib seaborn scikit-learn torch transformers shap pickle5 streamlit
```

---

## Model Training

To train the **mental health prediction model**, run:
```sh
$ python predict_mental_health_py.py
```
This script will:
- Preprocess the dataset (`anxiety_attack_dataset.csv`)
- Train a **Random Forest Classifier**
- Save the trained model as `mental_health_model.pkl`

---

## Running the Web Application

Once the model is trained, start the **Streamlit** application:
```sh
$ streamlit run app.py
```
This will open a web interface where users can input their symptoms and receive an anxiety severity assessment.

---

## Project Structure
```
📂 Project Root
│── app.py                         # Streamlit web application
│── predict_mental_health_py.py    # Model training script
│── mental_health_model.pkl        # Trained ML model
|── anxiety_attack_dataset         # Dataset
│── requirements.txt               # Python dependencies
│── README.md                      # Documentation
```

---

## Troubleshooting

1. **Model File Not Found?**
   - Ensure that `mental_health_model.pkl` exists. If not, re-run `python predict_mental_health_py.py`.

2. **Streamlit App Not Loading?**
   - Make sure all dependencies are installed (`pip install -r requirements.txt`).

3. **Permission Issues?**
   - Run the script with administrator privileges or try using a virtual environment.

---

## Contributing
Feel free to contribute by submitting pull requests or reporting issues.

---

## License
This project is open-source under the MIT License.

