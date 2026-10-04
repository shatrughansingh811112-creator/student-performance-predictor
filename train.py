"""
Machine Learning Training Pipeline (Regression)
Trains and benchmarks:
1. Linear Regression (Baseline)
2. Ridge Regression (L2 Regularization)
3. Random Forest Regressor (Ensemble Trees)
4. Gradient Boosting Regressor (Sequential Boosting)
Evaluates on unseen test data, calculates Feature Importance, and saves artifacts.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.sans-serif': 'Segoe UI', 'font.size': 10})

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'student_data.csv')
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'plots')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Running data/generate_data.py first...")
        from data.generate_data import generate_student_dataset
        df = generate_student_dataset(n_samples=2000)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    print("=" * 70)
    print("      REGRESSION MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 70)

    # 1. Feature Preprocessing
    print("\n[1/5] Preprocessing Features...")
    df_processed = df.copy()
    df_processed['extracurricular'] = df_processed['extracurricular'].map({'Yes': 1, 'No': 0})
    
    feature_cols = ['study_hours', 'previous_score', 'sleep_hours', 'attendance_percent', 'practice_questions', 'extracurricular']
    X = df_processed[feature_cols]
    y = df_processed['exam_score']
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target: exam_score")

    # 2. Train-Test Split (80% Train, 20% Test)
    print("\n[2/5] Splitting data into 80% Train and 20% Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples : {X_test.shape[0]}")

    # 3. Feature Scaling (StandardScaler)
    print("\n[3/5] Scaling numeric features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Models
    print("\n[4/5] Training 4 Candidate Models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    }

    results = []
    trained_models = {}
    test_predictions = {}

    for name, model in models.items():
        print(f"   - Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2 Score': r2
        })
        trained_models[name] = model
        test_predictions[name] = y_pred

    results_df = pd.DataFrame(results)

    # 5. Display Evaluation Results
    print("\n[5/5] Model Performance Summary on Test Set:")
    print("-" * 70)
    print(f"{'Model':<22} | {'MAE':>8} | {'RMSE':>8} | {'R^2 Score':>10}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<22} | {row['MAE']:>8.3f} | {row['RMSE']:>8.3f} | {row['R2 Score']:>10.4f}")
    print("-" * 70)

    # Best Model Selection
    best_row = results_df.sort_values(by='R2 Score', ascending=False).iloc[0]
    best_model_name = best_row['Model']
    best_model = trained_models[best_model_name]
    best_preds = test_predictions[best_model_name]

    print(f"\n[*] Best Champion Model Selected: {best_model_name} (R^2 = {best_row['R2 Score']:.4f})")

    # Feature Importance / Coefficients calculation
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_)
        importances = importances / np.sum(importances)
    else:
        importances = np.ones(len(feature_cols)) / len(feature_cols)

    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Contribution Ranking:")
    for _, imp_row in importance_df.iterrows():
        print(f"   * {imp_row['Feature']:<20}: {imp_row['Importance']*100:>5.1f}%")

    # Save artifacts
    model_save_path = os.path.join(models_dir, 'best_model.joblib')
    scaler_save_path = os.path.join(models_dir, 'scaler.joblib')
    meta_save_path = os.path.join(models_dir, 'metadata.joblib')

    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump({
        'features': feature_cols,
        'model_name': best_model_name,
        'metrics': best_row.to_dict(),
        'feature_importance': importance_df.to_dict(orient='records')
    }, meta_save_path)

    print(f"\n[+] Saved best model to: {model_save_path}")
    print(f"[+] Saved scaler to    : {scaler_save_path}")

    # ================= PLOTS ================= #
    # Plot 1: Model Comparison Bar Chart
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.barplot(data=results_df, x='Model', y='R2 Score', hue='Model', ax=ax[0], palette='Blues_d', legend=False)
    ax[0].set_title("R2 Score Comparison (Higher is better)", fontweight="bold")
    ax[0].set_ylim(0.70, 1.0)
    ax[0].tick_params(axis='x', rotation=15)
    for p in ax[0].patches:
        ax[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontweight='bold')

    sns.barplot(data=results_df, x='Model', y='RMSE', hue='Model', ax=ax[1], palette='Reds_d', legend=False)
    ax[1].set_title("RMSE Error Comparison (Lower is better)", fontweight="bold")
    ax[1].tick_params(axis='x', rotation=15)
    for p in ax[1].patches:
        ax[1].annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontweight='bold')

    plt.tight_layout()
    comp_plot_path = os.path.join(plots_dir, 'model_comparison.png')
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"[+] Comparison chart saved to: {comp_plot_path}")

    # Plot 2: Actual vs Predicted Scores
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, best_preds, alpha=0.45, color='#2b5c8f', edgecolors='none')
    min_val = min(y_test.min(), best_preds.min())
    max_val = max(y_test.max(), best_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='#d9381e', linestyle='--', linewidth=2, label="Perfect 1:1 Fit")
    
    plt.title(f"Actual vs. Predicted Scores ({best_model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Exam Score (Marks)", fontsize=11)
    plt.ylabel("Predicted Exam Score (Marks)", fontsize=11)
    plt.legend()
    plt.tight_layout()
    pred_plot_path = os.path.join(plots_dir, 'actual_vs_predicted.png')
    plt.savefig(pred_plot_path, dpi=300)
    plt.close()
    print(f"[+] Actual vs Predicted chart saved to: {pred_plot_path}")

    # Plot 3: Feature Importance Bar Chart
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title(f"Feature Importance ({best_model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Relative Contribution Factor", fontsize=11)
    plt.ylabel("Student Feature", fontsize=11)
    plt.tight_layout()
    feat_plot_path = os.path.join(plots_dir, 'feature_importance.png')
    plt.savefig(feat_plot_path, dpi=300)
    plt.close()
    print(f"[+] Feature Importance chart saved to: {feat_plot_path}")

    print("\nRegression Training Pipeline Complete!")

if __name__ == '__main__':
    train_and_evaluate()
