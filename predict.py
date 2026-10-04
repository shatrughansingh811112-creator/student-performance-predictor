"""
Unified CLI Inference Script: Score Predictor & Risk Classifier
Loads both Regression and Classification models for comprehensive prediction.
"""

import os
import joblib
import pandas as pd
import numpy as np

def load_all_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    # Regression artifacts
    reg_model_path = os.path.join(models_dir, 'best_model.joblib')
    reg_scaler_path = os.path.join(models_dir, 'scaler.joblib')
    reg_meta_path = os.path.join(models_dir, 'metadata.joblib')
    
    # Classification artifacts
    clf_model_path = os.path.join(models_dir, 'best_classifier.joblib')
    clf_scaler_path = os.path.join(models_dir, 'classifier_scaler.joblib')
    clf_meta_path = os.path.join(models_dir, 'classifier_metadata.joblib')
    
    reg_model = joblib.load(reg_model_path) if os.path.exists(reg_model_path) else None
    reg_scaler = joblib.load(reg_scaler_path) if os.path.exists(reg_scaler_path) else None
    reg_meta = joblib.load(reg_meta_path) if os.path.exists(reg_meta_path) else {}
    
    clf_model = joblib.load(clf_model_path) if os.path.exists(clf_model_path) else None
    clf_scaler = joblib.load(clf_scaler_path) if os.path.exists(clf_scaler_path) else None
    clf_meta = joblib.load(clf_meta_path) if os.path.exists(clf_meta_path) else {}
    
    return {
        'reg_model': reg_model,
        'reg_scaler': reg_scaler,
        'reg_meta': reg_meta,
        'clf_model': clf_model,
        'clf_scaler': clf_scaler,
        'clf_meta': clf_meta
    }

def run_predictions(study_hours, previous_score, sleep_hours, attendance_percent, practice_questions, extracurricular):
    artifacts = load_all_artifacts()
    extra_val = 1 if str(extracurricular).strip().lower() in ['yes', 'y', '1', 'true'] else 0
    
    input_df = pd.DataFrame([{
        'study_hours': float(study_hours),
        'previous_score': float(previous_score),
        'sleep_hours': float(sleep_hours),
        'attendance_percent': float(attendance_percent),
        'practice_questions': int(practice_questions),
        'extracurricular': extra_val
    }])
    
    # 1. Regression Score Prediction
    reg_model = artifacts['reg_model']
    reg_scaler = artifacts['reg_scaler']
    if reg_model and reg_scaler:
        scaled_reg = reg_scaler.transform(input_df)
        predicted_score = round(float(np.clip(reg_model.predict(scaled_reg)[0], 0.0, 100.0)), 1)
    else:
        predicted_score = None
        
    # 2. Classification Tier Prediction
    clf_model = artifacts['clf_model']
    clf_scaler = artifacts['clf_scaler']
    tier = None
    probs = {}
    if clf_model and clf_scaler:
        scaled_clf = clf_scaler.transform(input_df)
        tier = clf_model.predict(scaled_clf)[0]
        if hasattr(clf_model, 'predict_proba'):
            raw_probs = clf_model.predict_proba(scaled_clf)[0]
            for cls_name, p in zip(clf_model.classes_, raw_probs):
                probs[cls_name] = round(float(p) * 100, 1)

    return predicted_score, tier, probs

def main():
    print("=" * 65)
    print("      STUDENT PERFORMANCE & RISK PREDICTOR (CLI)")
    print("=" * 65)
    
    artifacts = load_all_artifacts()
    if not artifacts['reg_model']:
        print("[!] Regression model not found. Run 'python train.py' first.")
        return

    print("\nEnter student details below (or press Enter to use default):")
    try:
        study_hours = float(input("1. Daily study hours (e.g. 5.5)        : ") or "5.5")
        previous_score = float(input("2. Previous exam score (0 - 100)       : ") or "75.0")
        sleep_hours = float(input("3. Daily sleep hours (e.g. 7.0)        : ") or "7.0")
        attendance = float(input("4. Attendance percentage (e.g. 85.0)   : ") or "85.0")
        practice = int(input("5. Practice questions solved (e.g. 40) : ") or "40")
        extracurricular = input("6. Extracurricular activities? (Yes/No): ") or "Yes"

        score, tier, probs = run_predictions(
            study_hours, previous_score, sleep_hours, attendance, practice, extracurricular
        )

        print("\n" + "=" * 65)
        print("                      PREDICTION RESULTS")
        print("=" * 65)
        print(f"  * Continuous Score (Regression) : {score} / 100")
        if tier:
            print(f"  * Performance Tier (Classifier) : {tier}")
        if probs:
            print("  * Probability Breakdown         :")
            for cls_name, pct in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                print(f"     - {cls_name:<16}: {pct:>5.1f}%")
        
        # Insights
        if score >= 85:
            advice = "[+] Outstanding trajectory! Maintain this balanced study and sleep routine."
        elif score >= 70:
            advice = "[*] Strong foundation. Increasing daily practice questions can push you above 85+."
        elif score >= 50:
            advice = "[!] Passing grade, but at risk of dropping. Increase study hours and attendance."
        else:
            advice = "[!] High academic risk. Immediate intervention needed: prioritize attendance and study."
        print(f"\n  * Actionable Recommendation     : {advice}")
        print("=" * 65)

    except ValueError as e:
        print(f"[!] Invalid input: {e}")
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == '__main__':
    main()
