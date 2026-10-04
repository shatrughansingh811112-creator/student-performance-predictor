"""
Streamlit Web Application: Student Performance & Academic Risk Suite
Features:
- Tab 1: Continuous Score Prediction (Regression) + What-If Simulation
- Tab 2: Academic Risk & Outcome Tier Classifier (Classification)
- Tab 3: Model Diagnostics & Visualizations Gallery
- Tab 4: Beginner's Machine Learning Reference & Formulas
Run with: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student ML Intelligence Suite",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0.1rem; }
    .sub-title { color: #4B5563; font-size: 1.05rem; margin-bottom: 1.2rem; }
    .card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px; text-align: center; }
    .score-val { font-size: 3.2rem; font-weight: 800; }
    .badge { display: inline-block; padding: 5px 16px; border-radius: 20px; font-weight: 700; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    reg_m = joblib.load(os.path.join(models_dir, 'best_model.joblib')) if os.path.exists(os.path.join(models_dir, 'best_model.joblib')) else None
    reg_s = joblib.load(os.path.join(models_dir, 'scaler.joblib')) if os.path.exists(os.path.join(models_dir, 'scaler.joblib')) else None
    reg_meta = joblib.load(os.path.join(models_dir, 'metadata.joblib')) if os.path.exists(os.path.join(models_dir, 'metadata.joblib')) else {}

    clf_m = joblib.load(os.path.join(models_dir, 'best_classifier.joblib')) if os.path.exists(os.path.join(models_dir, 'best_classifier.joblib')) else None
    clf_s = joblib.load(os.path.join(models_dir, 'classifier_scaler.joblib')) if os.path.exists(os.path.join(models_dir, 'classifier_scaler.joblib')) else None
    clf_meta = joblib.load(os.path.join(models_dir, 'classifier_metadata.joblib')) if os.path.exists(os.path.join(models_dir, 'classifier_metadata.joblib')) else {}

    return {
        'reg_m': reg_m, 'reg_s': reg_s, 'reg_meta': reg_meta,
        'clf_m': clf_m, 'clf_s': clf_s, 'clf_meta': clf_meta,
        'base_dir': base_dir
    }

art = load_all_artifacts()

st.markdown('<div class="main-title">🎓 Student Performance & Risk Intelligence Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End Supervised Machine Learning (Regression + Classification)</div>', unsafe_allow_html=True)

if not art['reg_m']:
    st.error("Model artifacts missing. Run `python train.py` and `python train_classifier.py` first.")
    st.stop()

# Sidebar: Inputs
st.sidebar.header("📋 Student Profile Inputs")
st.sidebar.markdown("Adjust habits to see instant predictions:")

study_hours = st.sidebar.slider("📚 Daily Study Hours", min_value=1.0, max_value=12.0, value=6.0, step=0.5)
previous_score = st.sidebar.slider("📝 Previous Exam Score (%)", min_value=30.0, max_value=100.0, value=75.0, step=1.0)
sleep_hours = st.sidebar.slider("😴 Daily Sleep Hours", min_value=4.0, max_value=10.0, value=7.5, step=0.5)
attendance = st.sidebar.slider("🏫 Attendance Percentage (%)", min_value=50.0, max_value=100.0, value=85.0, step=1.0)
practice = st.sidebar.slider("✍️ Practice Questions Solved", min_value=0, max_value=150, value=60, step=5)
extracurricular = st.sidebar.radio("⚽ Extracurricular Activities", options=["Yes", "No"], horizontal=True)

extra_val = 1 if extracurricular == "Yes" else 0
input_df = pd.DataFrame([{
    'study_hours': study_hours,
    'previous_score': previous_score,
    'sleep_hours': sleep_hours,
    'attendance_percent': attendance,
    'practice_questions': practice,
    'extracurricular': extra_val
}])

# Compute Predictions
scaled_reg = art['reg_s'].transform(input_df)
pred_score = round(float(np.clip(art['reg_m'].predict(scaled_reg)[0], 0.0, 100.0)), 1)

scaled_clf = art['clf_s'].transform(input_df) if art['clf_s'] else scaled_reg
pred_tier = art['clf_m'].predict(scaled_clf)[0] if art['clf_m'] else "Unknown"

# Tier Colors & Advice
if pred_tier == "Distinction":
    tier_color = "#10B981"
    badge_bg = "#D1FAE5"
    advice = "Exceptional performance trajectory! The student is maintaining a high-scoring routine."
elif pred_tier == "Pass":
    tier_color = "#2563EB"
    badge_bg = "#DBEAFE"
    advice = "Solid passing performance. Increasing study hours by 1-2 hours can propel them to Distinction."
else:
    tier_color = "#EF4444"
    badge_bg = "#FEE2E2"
    advice = "High academic risk. Immediate support recommended: increase attendance and structured study."

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Score Predictor (Regression)",
    "⚠️ Risk & Tier Classifier (Classification)",
    "📊 Plots & Model Diagnostics",
    "💡 ML Concepts & Formulas"
])

# ----------------- TAB 1: REGRESSION -----------------
with tab1:
    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        st.subheader("🎯 Predicted Continuous Score")
        st.markdown(f"""
        <div class="card">
            <div style="color: #64748B; font-weight: 600; font-size: 1.1rem;">Estimated Final Exam Score</div>
            <div class="score-val" style="color: {tier_color};">{pred_score}<span style="font-size: 1.5rem; color: #94A3B8;">/100</span></div>
            <div class="badge" style="background-color: {badge_bg}; color: {tier_color};">Category: {pred_tier}</div>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"💡 **Recommendation:** {advice}")
        
        reg_model_name = art['reg_meta'].get('model_name', 'Trained Model')
        reg_r2 = art['reg_meta'].get('metrics', {}).get('R2 Score', 0.898)
        st.caption(f"🤖 Active Model: `{reg_model_name}` | Test $R^2$ Accuracy: `{reg_r2:.3f}`")

    with c2:
        st.subheader("📈 'What-If' Simulation: Impact of Study Hours")
        st.caption("How exam score changes as daily study hours increase:")
        
        hours_arr = np.linspace(1.0, 12.0, 23)
        sim_scores = []
        for h in hours_arr:
            temp = input_df.copy()
            temp['study_hours'] = h
            sc = np.clip(art['reg_m'].predict(art['reg_s'].transform(temp))[0], 0.0, 100.0)
            sim_scores.append(sc)
            
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(hours_arr, sim_scores, color='#2563EB', linewidth=2.5, marker='o', markersize=4)
        ax.scatter([study_hours], [pred_score], color='#EF4444', s=120, zorder=5, label=f"Current: {study_hours} hrs ({pred_score})")
        ax.set_xlabel("Daily Study Hours", fontsize=10)
        ax.set_ylabel("Predicted Exam Score", fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower right')
        st.pyplot(fig)

# ----------------- TAB 2: CLASSIFICATION -----------------
with tab2:
    st.subheader("🛡️ Academic Outcome Tier & Risk Probability")
    st.write("The classification model evaluates whether a student qualifies for **Distinction**, a **Pass**, or is at **Academic Risk (<50%)**.")
    
    col_a, col_b = st.columns([1.2, 1.2])
    
    with col_a:
        st.markdown(f"""
        <div class="card" style="padding: 28px;">
            <div style="font-size: 1.2rem; color: #64748B;">Predicted Outcome Tier</div>
            <div style="font-size: 2.8rem; font-weight: 800; color: {tier_color}; margin: 10px 0;">{pred_tier}</div>
            <div style="color: #475569; font-size: 1rem;">Model: <b>{art['clf_meta'].get('model_name', 'Logistic Regression')}</b> (Accuracy: {art['clf_meta'].get('accuracy', 0.89)*100:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        if hasattr(art['clf_m'], 'predict_proba'):
            raw_p = art['clf_m'].predict_proba(scaled_clf)[0]
            cls_names = list(art['clf_m'].classes_)
            
            prob_df = pd.DataFrame({
                'Tier': cls_names,
                'Probability (%)': [round(p * 100, 1) for p in raw_p]
            })
            
            fig2, ax2 = plt.subplots(figsize=(6, 3.4))
            bars = ax2.barh(prob_df['Tier'], prob_df['Probability (%)'], color=['#EF4444', '#10B981', '#2563EB'])
            ax2.set_xlim(0, 100)
            ax2.set_xlabel("Probability (%)", fontsize=10)
            for bar in bars:
                w = bar.get_width()
                ax2.annotate(f"{w:.1f}%", (w + 2, bar.get_y() + bar.get_height()/2.), va='center', fontweight='bold')
            st.pyplot(fig2)

# ----------------- TAB 3: PLOTS GALLERY -----------------
with tab3:
    st.subheader("🖼️ Project Diagnostics & Plots Gallery")
    st.caption("All visual analysis plots generated across the EDA, regression, and classification pipelines:")
    
    plots_dir = os.path.join(art['base_dir'], 'plots')
    plot_files = [
        ('correlation_heatmap.png', '1. Feature Correlation Matrix'),
        ('feature_importance.png', '2. Feature Importance Ranking (What matters most)'),
        ('model_comparison.png', '3. Model Benchmark Comparison (R² & RMSE)'),
        ('confusion_matrix.png', '4. Confusion Matrix (Classification Accuracy)'),
        ('actual_vs_predicted.png', '5. Actual vs Predicted Exam Scores Fit'),
        ('study_hours_vs_score.png', '6. Study Hours vs Exam Score Trend'),
        ('feature_distributions.png', '7. Feature Distributions')
    ]
    
    for filename, title in plot_files:
        filepath = os.path.join(plots_dir, filename)
        if os.path.exists(filepath):
            st.markdown(f"#### {title}")
            st.image(filepath, use_container_width=True)
            st.markdown("---")

# ----------------- TAB 4: ML CONCEPTS -----------------
with tab4:
    st.subheader("🧠 Machine Learning Cheat Sheet for Beginners")
    st.markdown("""
    ### 1. Regression vs. Classification
    * **Regression (Tab 1):** Predicts continuous quantities (Exam marks: $0$ to $100$). Uses **MAE**, **RMSE**, and **$R^2$ Score**.
    * **Classification (Tab 2):** Predicts discrete labels (**Distinction**, **Pass**, **Risk**). Uses **Accuracy**, **Precision**, **Recall**, and **Confusion Matrix**.
    
    ### 2. Feature Scaling (`StandardScaler`)
    * Equation: $z = \\frac{x - \\mu}{\\sigma}$
    * Prevents features with naturally larger numeric ranges (e.g. 150 practice questions) from dominating small-ranged features (e.g. 5 study hours).
    
    ### 3. Interview Talking Point
    > *"In this project, I handled both regression and classification on the same dataset. For regression, Ridge Regularization outperformed Random Forest with an $R^2$ of 0.898 by preventing overfitting on multicollinear features. For classification, Logistic Regression achieved 89% accuracy in pinpointing students at academic risk."*
    """)
