"""
Classification Training Pipeline: Student Performance Category & Risk Detector
Predicts outcome tiers:
  - 'Distinction' (Score >= 80)
  - 'Pass' (Score between 50 and 79)
  - 'Academic Risk' (Score < 50)
Trains Logistic Regression & Random Forest Classifier, evaluates accuracy,
plots a Confusion Matrix, and saves the trained classification model.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Segoe UI', 'font.size': 10})

def get_performance_tier(score):
    if score >= 80.0:
        return 'Distinction'
    elif score >= 50.0:
        return 'Pass'
    else:
        return 'Academic Risk'

def train_classifier():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'student_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'plots')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    print("=" * 70)
    print("      CLASSIFICATION PIPELINE: STUDENT RISK & TIER DETECTOR")
    print("=" * 70)

    df = pd.read_csv(data_path)
    df_processed = df.copy()
    df_processed['extracurricular'] = df_processed['extracurricular'].map({'Yes': 1, 'No': 0})
    
    # Create categorical target variable from exam_score
    df_processed['outcome_tier'] = df_processed['exam_score'].apply(get_performance_tier)
    
    feature_cols = ['study_hours', 'previous_score', 'sleep_hours', 'attendance_percent', 'practice_questions', 'extracurricular']
    X = df_processed[feature_cols]
    y = df_processed['outcome_tier']

    print("\n[1/4] Class Distribution:")
    class_counts = y.value_counts()
    for tier, count in class_counts.items():
        print(f"   * {tier:<16}: {count} students ({count/len(y)*100:.1f}%)")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train 2 Classifiers
    print("\n[2/4] Training Classification Models...")
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }

    best_acc = 0.0
    best_clf_name = None
    best_clf = None
    best_preds = None

    print("-" * 70)
    print(f"{'Classifier':<30} | {'Test Accuracy':>15}")
    print("-" * 70)

    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name:<30} | {acc*100:>14.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_clf_name = name
            best_clf = clf
            best_preds = y_pred

    print("-" * 70)
    print(f"\n[*] Best Classifier Selected: {best_clf_name} (Accuracy = {best_acc*100:.2f}%)")

    # Detailed Classification Report
    labels = ['Academic Risk', 'Pass', 'Distinction']
    print("\n[3/4] Detailed Classification Report:")
    print(classification_report(y_test, best_preds, target_names=labels, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, best_preds, labels=labels)

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f"Confusion Matrix ({best_clf_name})", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Outcome", fontsize=11)
    plt.ylabel("Actual Outcome", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrix plot to: {cm_path}")

    # Save artifacts
    clf_save_path = os.path.join(models_dir, 'best_classifier.joblib')
    scaler_clf_path = os.path.join(models_dir, 'classifier_scaler.joblib')
    meta_clf_path = os.path.join(models_dir, 'classifier_metadata.joblib')

    joblib.dump(best_clf, clf_save_path)
    joblib.dump(scaler, scaler_clf_path)
    joblib.dump({
        'features': feature_cols,
        'model_name': best_clf_name,
        'accuracy': float(best_acc),
        'classes': labels
    }, meta_clf_path)

    print(f"[+] Saved Classifier model to: {clf_save_path}")
    print("\nClassification Training Pipeline Complete!")

if __name__ == '__main__':
    train_classifier()
