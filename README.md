# 🎓 Student Exam Performance & Academic Risk Intelligence Suite

A complete, production-ready Machine Learning portfolio project built from scratch with Python, Scikit-Learn, Pandas, Matplotlib, Seaborn, and Streamlit.

---

## 📌 Project Summary
This project covers **both foundational branches of Supervised Machine Learning**:
1. **Regression (Continuous Prediction):** Predicts exact final exam marks (0 to 100) based on student habits and past performance.
2. **Classification (Categorical Prediction):** Evaluates if a student qualifies for **Distinction**, a **Pass**, or is at **Academic Risk (<50%)** for early educational intervention.

---

## 🗂️ Project Structure

```text
student_performance_predictor/
│
├── data/
│   ├── generate_data.py          # Generates realistic student dataset (2,000 records)
│   └── student_data.csv          # Feature matrix and target variables
│
├── models/                       # Saved model artifacts for production inference
│   ├── best_model.joblib         # Champion Regression Model (Ridge Regression, R² = 0.898)
│   ├── scaler.joblib             # Regression StandardScaler
│   ├── metadata.joblib           # Feature importance & regression metrics
│   ├── best_classifier.joblib    # Champion Classifier (Logistic Regression, 89% Accuracy)
│   ├── classifier_scaler.joblib  # Classification StandardScaler
│   └── classifier_metadata.joblib# Classification classes & accuracy
│
├── plots/                        # Generated visual analytics & diagnostics
│   ├── correlation_heatmap.png   # Correlation matrix across features
│   ├── feature_importance.png    # Ranking of what features drive performance most
│   ├── model_comparison.png      # 4-model benchmark comparison (R² vs RMSE)
│   ├── confusion_matrix.png      # Classification accuracy confusion matrix
│   ├── actual_vs_predicted.png   # Fit plot of test scores
│   ├── study_hours_vs_score.png  # Study hours vs exam score regression trend
│   └── feature_distributions.png # Distribution histograms
│
├── eda.py                        # Exploratory Data Analysis & statistical summaries
├── train.py                      # Regression pipeline (Linear, Ridge, RF, Gradient Boosting)
├── train_classifier.py           # Classification pipeline (Logistic Regression, RF Classifier)
├── predict.py                    # Unified CLI script (predicts Score + Outcome Tier + Probabilities)
├── app.py                        # Interactive Streamlit Web Suite (4 tabs with live simulation)
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # Complete project documentation
```

---

## 🏆 Model Benchmarks & Results

### 1. Regression Pipeline (Predicting Exact Marks)
Trained on 1,600 samples (80%), evaluated on 400 unseen test samples (20%):

| Model | MAE (Marks Error) | RMSE (Root Mean Sq) | $R^2$ Score (Accuracy) |
| :--- | :---: | :---: | :---: |
| **Ridge Regression (L2)** | **3.103** | **3.894** | **0.8978 (89.8%)** 🏆 |
| **Linear Regression** | 3.103 | 3.894 | 0.8978 (89.8%) |
| **Gradient Boosting** | 3.195 | 3.928 | 0.8961 (89.6%) |
| **Random Forest Regressor** | 3.468 | 4.281 | 0.8765 (87.7%) |

### 2. Feature Importance Breakdown
What drives a student's exam score the most?
1. **Study Hours:** **42.9%** contribution
2. **Previous Exam Score:** **26.6%** contribution
3. **Practice Questions Solved:** **10.9%** contribution
4. **Class Attendance:** **10.4%** contribution
5. **Sleep Schedule:** **4.7%** contribution
6. **Extracurricular Activities:** **4.5%** contribution

### 3. Classification Pipeline (Academic Risk Detection)
* **Best Classifier:** Logistic Regression
* **Test Accuracy:** **89.00%**
* **Categories:** `Distinction`, `Pass`, `Academic Risk`

---

## 🚀 How to Run

### Activate Environment
```powershell
cd C:\Users\shatr\student_performance_predictor
.\venv\Scripts\activate
```

### Run CLI Prediction
```powershell
python predict.py
```

### Launch Interactive Web Dashboard
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 💼 Interview Talking Point
> *"I designed an end-to-end Machine Learning pipeline that tackles both regression and classification on student academic data. In the regression pipeline, Ridge Regression achieved an $R^2$ of 0.898 with an MAE of 3.1 marks, outperforming Gradient Boosting and Random Forest. In classification, Logistic Regression achieved 89% accuracy with full probability calibration to detect at-risk students for early intervention. I packaged the solution with feature scaling, model persistence, CLI inference, and a 4-tab Streamlit dashboard."*
