"""
Step 2: Exploratory Data Analysis (EDA)
This script analyzes the dataset and saves visual charts to the 'plots/' folder.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Segoe UI', 'font.size': 11})

def run_eda():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'student_data.csv')
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please run data/generate_data.py first!")
        return

    print("=" * 60)
    print("      EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    df = pd.read_csv(data_path)
    
    # 1. Dataset Dimensions & Types
    print(f"\n1. Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n2. Data Types & Non-Null Values:")
    print(df.info())
    
    # 2. Check for missing values
    missing = df.isnull().sum()
    print("\n3. Missing Values Check:")
    print(missing)
    if missing.sum() == 0:
        print("-> Clean dataset! No missing/null values detected.")

    # 3. Five-number summary / Descriptive Statistics
    print("\n4. Descriptive Statistics:")
    print(df.describe().round(2))

    # 4. Correlation Analysis
    # Convert categorical 'extracurricular' to numeric 1/0 for correlation calculation
    df_numeric = df.copy()
    df_numeric['extracurricular'] = df_numeric['extracurricular'].map({'Yes': 1, 'No': 0})
    
    corr = df_numeric.corr()
    print("\n5. Feature Correlation with Target ('exam_score'):")
    corr_target = corr['exam_score'].sort_values(ascending=False)
    for col, val in corr_target.items():
        print(f"   - {col:<22}: {val:>6.3f}")

    # ================= PLOTS ================= #
    print("\nGenerating visual charts...")

    # Plot 1: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, cbar_kws={'shrink': 0.8}, linewidths=0.5)
    plt.title("Correlation Matrix of Features", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plot_path1 = os.path.join(plots_dir, 'correlation_heatmap.png')
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f" -> Saved: {plot_path1}")

    # Plot 2: Study Hours vs Exam Score (Scatter + Regression line)
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df, 
        x='study_hours', 
        y='exam_score', 
        scatter_kws={'alpha': 0.35, 'color': '#2b5c8f'}, 
        line_kws={'color': '#d9381e', 'linewidth': 2}
    )
    plt.title("Study Hours vs. Final Exam Score", fontsize=14, fontweight="bold")
    plt.xlabel("Study Hours per Day", fontsize=12)
    plt.ylabel("Final Exam Score (0 - 100)", fontsize=12)
    plt.tight_layout()
    plot_path2 = os.path.join(plots_dir, 'study_hours_vs_score.png')
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f" -> Saved: {plot_path2}")

    # Plot 3: Distribution of Key Features
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    sns.histplot(df['exam_score'], kde=True, ax=axes[0, 0], color='#1f77b4', bins=25)
    axes[0, 0].set_title('Target Distribution: Exam Score', fontweight="bold")
    
    sns.histplot(df['study_hours'], kde=True, ax=axes[0, 1], color='#2ca02c', bins=20)
    axes[0, 1].set_title('Study Hours Distribution', fontweight="bold")
    
    sns.histplot(df['previous_score'], kde=True, ax=axes[1, 0], color='#ff7f0e', bins=25)
    axes[1, 0].set_title('Previous Exam Score Distribution', fontweight="bold")
    
    sns.boxplot(data=df, x='extracurricular', y='exam_score', hue='extracurricular', ax=axes[1, 1], palette='Set2', legend=False)
    axes[1, 1].set_title('Exam Score by Extracurricular Activity', fontweight="bold")
    
    plt.tight_layout()
    plot_path3 = os.path.join(plots_dir, 'feature_distributions.png')
    plt.savefig(plot_path3, dpi=300)
    plt.close()
    print(f" -> Saved: {plot_path3}")

    print("\nEDA Completed successfully! Visualizations are stored in the 'plots/' folder.")

if __name__ == '__main__':
    run_eda()
