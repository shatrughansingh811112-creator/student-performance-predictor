"""
Step 1: Dataset Generator for Student Performance Prediction
This script creates a realistic dataset of 2,000 students.
Features:
- study_hours: Hours spent studying per day (1 - 12)
- previous_score: Score in the previous exam (30 - 100)
- sleep_hours: Average sleep hours per day (4 - 10)
- attendance_percent: Class attendance percentage (50 - 100)
- practice_questions: Number of sample test questions practiced (0 - 150)
- extracurricular: Participates in sports/arts/clubs ('Yes' or 'No')
Target:
- exam_score: Final exam score (0 - 100)
"""

import numpy as np
import pandas as pd
import os

def generate_student_dataset(n_samples=2000, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Feature Generation with realistic distributions
    study_hours = np.round(np.random.uniform(1.0, 11.5, n_samples), 1)
    previous_score = np.round(np.random.normal(loc=68, scale=14, size=n_samples), 1)
    previous_score = np.clip(previous_score, 30.0, 100.0)
    
    sleep_hours = np.round(np.random.normal(loc=7.0, scale=1.2, size=n_samples), 1)
    sleep_hours = np.clip(sleep_hours, 4.0, 10.0)
    
    attendance_percent = np.round(np.random.uniform(55.0, 100.0, n_samples), 1)
    practice_questions = np.random.randint(5, 150, size=n_samples)
    extracurricular = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.45, 0.55])
    
    # 2. Formula for Exam Score with realistic weights and subtle non-linear factors
    # Base baseline score
    extra_bonus = np.where(extracurricular == 'Yes', 2.0, 0.0)
    
    # Moderate sleep (6.5 to 8.5) gives optimal cognitive rest
    sleep_effect = -0.8 * ((sleep_hours - 7.5) ** 2) + 3.0
    
    score_formula = (
        0.42 * previous_score +
        3.10 * study_hours +
        0.18 * attendance_percent +
        0.06 * practice_questions +
        extra_bonus +
        sleep_effect
    )
    
    # Adding Gaussian noise to represent real-world randomness (luck, exam-day mood, tricky questions)
    noise = np.random.normal(loc=0.0, scale=3.5, size=n_samples)
    raw_exam_score = score_formula + noise
    
    # Normalize/clip to realistic 0-100 range
    exam_score = np.round(np.clip(raw_exam_score, 10.0, 100.0), 1)
    
    # 3. Create DataFrame
    df = pd.DataFrame({
        'study_hours': study_hours,
        'previous_score': previous_score,
        'sleep_hours': sleep_hours,
        'attendance_percent': attendance_percent,
        'practice_questions': practice_questions,
        'extracurricular': extracurricular,
        'exam_score': exam_score
    })
    
    return df

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(data_dir, 'student_data.csv')
    
    print("Generating student performance dataset...")
    df = generate_student_dataset(n_samples=2000)
    df.to_csv(output_path, index=False)
    
    print(f" Dataset created successfully at: {output_path}")
    print(f"Total rows: {len(df)}")
    print("\nFirst 5 rows of data:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())
