"""
BANK CUSTOMER CHURN PREDICTION - STREAMLIT DASHBOARD
====================================================
Complete interactive web application for churn risk assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size:2.5rem; font-weight:bold; color:#1f77b4; text-align:center; padding:1rem 0; }
    .stButton>button { width:100%; background-color:#1f77b4; color:white; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ── Loaders ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    for name in ['final_optimized_model.pkl', 'best_model.pkl']:
        try:
            with open(name, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    st.error("❌ Model file not found!")
    return None

@st.cache_data
def load_data():
    try:
        X_train = pd.read_csv('X_train.csv')
        X_test  = pd.read_csv('X_test.csv')
        y_train = pd.read_csv('y_train.csv').squeeze()
        y_test  = pd.read_csv('y_test.csv').squeeze()
        return X_train, X_test, y_train, y_test
    except:
        return None, None, None, None

@st.cache_data
def load_raw():
    for name in ['European_Bank.csv', 'engineered_dataset.csv']:
        try:
            return pd.read_csv(name)
        except:
            pass
    return None

def make_input(credit_score, geography, gender, age, tenure,
               balance, num_products, has_cr_card, is_active_member, estimated_salary):
    geo_france  = 1 if geography == "France"  else 0
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain   = 1 if geography == "Spain"   else 0
    gender_val  = 1 if gender == "Male" else 0
    card_val    = 1 if has_cr_card == "Yes" else 0
    active_val  = 1 if is_active_member == "Yes" else 0
    bal_sal     = balance / (estimated_salary + 1)
    age_ten     = age * tenure
    prod_div    = num_products / 4

    df = pd.DataFrame([[
        credit_score, geo_france, geo_germany, geo_spain,
        gender_val, age, tenure, balance, num_products,
        card_val, active_val, estimated_salary,
        bal_sal, age_ten, prod_div
    ]], columns=[
        'CreditScore','Geography_France','Geography_Germany','Geography_Spain',
        'Gender','Age','Tenure','Balance','NumOfProducts',
        'HasCrCard','IsActiveMember','EstimatedSalary',
        'Balance_Salary_Ratio','Age_Tenure_Interaction','Product_Diversity'
    ])
    return df

model = load_model()
X_train, X_test, y_train, y_test = load_data()
raw_df = load_raw()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("# 🏦 Navigation")
page = st.sidebar.radio("Select Page:", [
    "🏠 Home", "🔮 Churn Predictor", "📊 Model Performance",
    "💡 Business Insights", "🎯 What-If Simulator", "📈 Data Explorer"
])

# ── PAGE 1: HOME ──────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<div class="main-header">🏦 Bank Customer Churn Prediction System</div>', unsafe_allow_html=True)
    st.markdown("### Welcome to the Predictive Churn Intelligence Dashboard")
    st.markdown("This app uses **Machine Learning** to predict which customers are likely to leave the bank.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Accuracy", "84.5%", "Best in class")
    c2.metric("Precision",      "61.1%", "High reliability")
    c3.metric("Recall",         "65.1%", "Catches 65% churners")
    c4.metric("ROC-AUC",        "0.86",  "Excellent discrimination")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📋 Project Overview")
        st.markdown("""
- **Objective**: Predict customer churn & assign risk scores  
- **Dataset**: 10,000 bank customers with 24 features  
- **Models tested**: Logistic Regression, Decision Tree, Random Forest ✅, XGBoost  
- **Best Model**: Random Forest (84.5% accuracy)
        """)
    with c2:
        st.markdown("### 📊 Quick Statistics")
        st.markdown("""
**Churn Rate**
- Overall: **20.4%** | High-risk: **71.5%** | Low-risk: **5.5%**

**Top Churn Predictors**
1. Age (21.7%) &nbsp; 2. Product Diversity (9.8%) &nbsp; 3. Num Products (9.5%)

**Risk Distribution**
- 🔴 High: 15.1% &nbsp; 🟡 Medium: 26.4% &nbsp; 🟢 Low: 58.6%
        """)
    st.markdown("---")
    st.info("👈 Navigate using the sidebar to explore all features!")

# ── PAGE 2: CHURN PREDICTOR ───────────────────────────────────────────────────
elif page == "🔮 Churn Predictor":
    st.markdown('<div class="main-header">🔮 Customer Churn Risk Calculator</div>', unsafe_allow_html=True)
    st.markdown("Enter customer details to get an instant churn probability and risk category.")

    if model is None:
        st.error("❌ Model not loaded.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Demographics")
            credit_score   = st.slider("Credit Score", 300, 850, 650)
            age            = st.slider("Age", 18, 100, 35)
            gender         = st.selectbox("Gender", ["Male", "Female"])
            geography      = st.selectbox("Geography", ["France", "Germany", "Spain"])
        with c2:
            st.markdown("#### Account Details")
            tenure           = st.slider("Tenure (Years)", 0, 10, 5)
            balance          = st.number_input("Account Balance (€)", 0.0, 250000.0, 75000.0, step=1000.0)
            num_products     = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
            estimated_salary = st.number_input("Estimated Salary (€)", 0.0, 200000.0, 50000.0, step=1000.0)
        with c3:
            st.markdown("#### Engagement")
            has_cr_card      = st.selectbox("Has Credit Card?", ["Yes", "No"])
            is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])

        st.markdown("---")
        if st.button("🔍 Calculate Churn Risk", type="primary"):
            inp = make_input(credit_score, geography, gender, age, tenure,
                             balance, num_products, has_cr_card, is_active_member, estimated_salary)
            try:
                # Align columns with model if possible
                if hasattr(model, 'feature_names_in_'):
                    inp = inp.reindex(columns=model.feature_names_in_, fill_value=0)

                proba = model.predict_proba(inp)[0][1]

                if proba >= 0.6:
                    risk, color, emoji = "HIGH RISK",   "#e74c3c", "🔴"
                elif proba >= 0.3:
                    risk, color, emoji = "MEDIUM RISK", "#f39c12", "🟡"
                else:
                    risk, color, emoji = "LOW RISK",    "#27ae60", "🟢"

                r1, r2, r3 = st.columns(3)
                r1.metric("Churn Probability", f"{proba*100:.1f}%")
                r2.metric("Risk Category", f"{emoji} {risk}")
                r3.metric("Retention Probability", f"{(1-proba)*100:.1f}%")

                st.progress(float(proba))

                st.markdown(f"""
<div style='background:#f8f9fa; padding:1rem; border-left:5px solid {color}; border-radius:4px; margin-top:1rem;'>
<b style='color:{color};'>{emoji} {risk}</b><br>
This customer has a <b>{proba*100:.1f}%</b> probability of churning.
{"Immediate retention action recommended." if proba>=0.6 else "Monitor and engage proactively." if proba>=0.3 else "Customer is likely to stay. Maintain engagement."}
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Tip: The feature columns in your model may differ. Check feature_names.txt in your repo.")

# ── PAGE 3: MODEL PERFORMANCE ─────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.markdown('<div class="main-header">📊 Model Performance</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  "84.5%")
    c2.metric("Precision", "61.1%")
    c3.metric("Recall",    "65.1%")
    c4.metric("ROC-AUC",   "0.86")

    st.markdown("---")

    if model is not None and X_test is not None and y_test is not None:
        try:
            X_test_aligned = X_test
            if hasattr(model, 'feature_names_in_'):
                X_test_aligned = X_test.reindex(columns=model.feature_names_in_, fill_value=0)

            y_pred  = model.predict(X_test_aligned)
            y_proba = model.predict_proba(X_test_aligned)[:, 1]

            c1, c2 = st.columns(2)

            # Confusion matrix
            with c1:
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(
                    cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Predicted: Stay", "Predicted: Churn"],
                    y=["Actual: Stay",    "Actual: Churn"],
                    title="Confusion Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)

            # ROC curve
            with c2:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                          name=f'ROC (AUC={roc_auc:.3f})', line=dict(color='#1f77b4', width=2)))
                fig2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                          name='Random', line=dict(dash='dash', color='gray')))
                fig2.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                                   yaxis_title="True Positive Rate")
                st.plotly_chart(fig2, use_container_width=True)

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feat_cols = X_test_aligned.columns.tolist()
                imp_df = pd.DataFrame({
                    'Feature': feat_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True).tail(15)

                fig3 = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                              title="Feature Importances (Top 15)",
                              color='Importance', color_continuous_scale='Blues')
                st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.error(f"Could not generate charts: {e}")
    else:
        st.warning("⚠️ Model or test data not loaded. Upload y_test.csv and y_train.csv to GitHub repo.")

        # Static charts as fallback
        st.markdown("### 📊 Reported Model Metrics")
        metrics = pd.DataFrame({
            'Model':    ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
            'Accuracy': [0.811, 0.792, 0.845, 0.838],
            'ROC-AUC':  [0.839, 0.721, 0.860, 0.855]
        })
        fig = px.bar(metrics, x='Model', y=['Accuracy','ROC-AUC'],
                     barmode='group', title="Model Comparison")
        st.plotly_chart(fig, use_container_width=True)

        feat_imp = pd.DataFrame({
            'Feature':    ['Age','Product_Diversity','NumOfProducts','Balance',
                           'IsActiveMember','CreditScore','Tenure','EstimatedSalary'],
            'Importance': [0.217, 0.098, 0.095, 0.079, 0.071, 0.065, 0.058, 0.052]
        }).sort_values('Importance', ascending=True)
        fig2 = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                      title="Feature Importances", color='Importance',
                      color_continuous_scale='Blues')
        st.plotly_chart(fig2, use_container_width=True)

# ── PAGE 4: BUSINESS INSIGHTS ─────────────────────────────────────────────────
elif page == "💡 Business Insights":
    st.markdown('<div class="main-header">💡 Business Insights</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔍 Key Findings")
        st.markdown("""
- **Age** is the #1 predictor — customers aged **45–60 churn most**
- **Inactive members** are **2× more likely** to churn
- **Germany** has significantly higher churn than France & Spain
- Customers with **only 1 product** are at high risk
- **High balance + inactive** = highest risk combination
        """)
    with c2:
        st.markdown("### ✅ Recommendations")
        st.markdown("""
- 🎯 Target retention campaigns at **age 45–60 segment**
- 📞 Re-engage inactive members with **personalised offers**
- 🇩🇪 Investigate **Germany-specific** service issues
- 📦 **Cross-sell products** to single-product customers
- 💰 Create **loyalty rewards** for high-balance inactive customers
        """)

    st.markdown("---")

    # Charts from raw data if available
    if raw_df is not None:
        st.markdown("### 📊 Data Insights")
        churn_col = None
        for c in ['Exited', 'Churn', 'churn', 'exited']:
            if c in raw_df.columns:
                churn_col = c
                break

        if churn_col:
            c1, c2 = st.columns(2)
            with c1:
                if 'Age' in raw_df.columns:
                    fig = px.histogram(raw_df, x='Age', color=churn_col,
                                       barmode='overlay', title="Age vs Churn",
                                       color_discrete_map={0:'#2ecc71', 1:'#e74c3c'})
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'Geography' in raw_df.columns:
                    geo_churn = raw_df.groupby('Geography')[churn_col].mean().reset_index()
                    geo_churn.columns = ['Geography', 'Churn Rate']
                    fig2 = px.bar(geo_churn, x='Geography', y='Churn Rate',
                                  title="Churn Rate by Geography",
                                  color='Churn Rate', color_continuous_scale='Reds')
                    st.plotly_chart(fig2, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                if 'NumOfProducts' in raw_df.columns:
                    prod_churn = raw_df.groupby('NumOfProducts')[churn_col].mean().reset_index()
                    prod_churn.columns = ['NumOfProducts', 'Churn Rate']
                    fig3 = px.bar(prod_churn, x='NumOfProducts', y='Churn Rate',
                                  title="Churn Rate by Number of Products",
                                  color='Churn Rate', color_continuous_scale='Oranges')
                    st.plotly_chart(fig3, use_container_width=True)
            with c4:
                if 'IsActiveMember' in raw_df.columns:
                    act_churn = raw_df.groupby('IsActiveMember')[churn_col].mean().reset_index()
                    act_churn['IsActiveMember'] = act_churn['IsActiveMember'].map({0:'Inactive', 1:'Active'})
                    act_churn.columns = ['Status', 'Churn Rate']
                    fig4 = px.bar(act_churn, x='Status', y='Churn Rate',
                                  title="Churn Rate: Active vs Inactive",
                                  color='Churn Rate', color_continuous_scale='Blues')
                    st.plotly_chart(fig4, use_container_width=True)
    else:
        # Static fallback chart
        seg = pd.DataFrame({
            'Segment':    ['Young Active', 'Middle Inactive', 'Senior Germany', 'Multi-Product', 'Single Product'],
            'Churn Rate': [0.055, 0.285, 0.420, 0.072, 0.278]
        })
        fig = px.bar(seg, x='Segment', y='Churn Rate',
                     title="Churn Rate by Customer Segment",
                     color='Churn Rate', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💰 Estimated Business Impact")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers at High Risk", "~302", "Need immediate action")
    c2.metric("Avg Revenue per Customer", "€1,200/yr", "Estimated")
    c3.metric("Potential Revenue Saved", "€362,400", "If 100% retained")

# ── PAGE 5: WHAT-IF SIMULATOR ─────────────────────────────────────────────────
elif page == "🎯 What-If Simulator":
    st.markdown('<div class="main-header">🎯 What-If Simulator</div>', unsafe_allow_html=True)
    st.markdown("Adjust the sliders to see how each factor affects churn probability in real time.")

    if model is None:
        st.error("❌ Model not loaded.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            age_sim      = st.slider("Age",              18, 100,    40)
            balance_sim  = st.slider("Balance (€)",       0, 250000, 80000, step=1000)
            products_sim = st.slider("Number of Products", 1, 4,      2)
            credit_sim   = st.slider("Credit Score",     300, 850,   650)
        with c2:
            tenure_sim   = st.slider("Tenure (Years)",   0, 10, 5)
            salary_sim   = st.slider("Salary (€)",       0, 200000, 50000, step=1000)
            active_sim   = st.selectbox("Active Member?",  ["Yes", "No"])
            geo_sim      = st.selectbox("Geography",       ["France", "Germany", "Spain"])
            gender_sim   = st.selectbox("Gender",          ["Male", "Female"])

        inp = make_input(credit_sim, geo_sim, gender_sim, age_sim, tenure_sim,
                         balance_sim, products_sim, "Yes", active_sim, salary_sim)
        try:
            if hasattr(model, 'feature_names_in_'):
                inp = inp.reindex(columns=model.feature_names_in_, fill_value=0)

            prob = model.predict_proba(inp)[0][1]

            if prob >= 0.6:
                color, label = "#e74c3c", "🔴 HIGH RISK"
            elif prob >= 0.3:
                color, label = "#f39c12", "🟡 MEDIUM RISK"
            else:
                color, label = "#27ae60", "🟢 LOW RISK"

            st.markdown("---")
            r1, r2 = st.columns(2)
            r1.metric("Live Churn Probability", f"{prob*100:.1f}%")
            r2.metric("Risk Level", label)
            st.progress(float(prob))

            # Sensitivity chart
            st.markdown("### 📊 Sensitivity Analysis — How each factor contributes")
            factors = {
                'Age':             age_sim / 100,
                'Balance':         balance_sim / 250000,
                'Num Products':    1 - (products_sim / 4),
                'Active Member':   0 if active_sim == "Yes" else 0.5,
                'Credit Score':    1 - (credit_sim / 850),
                'Germany Flag':    0.3 if geo_sim == "Germany" else 0.0,
            }
            sens_df = pd.DataFrame({
                'Factor':       list(factors.keys()),
                'Risk Weight':  list(factors.values())
            }).sort_values('Risk Weight', ascending=True)
            fig = px.bar(sens_df, x='Risk Weight', y='Factor', orientation='h',
                         title="Relative Risk Contribution per Factor",
                         color='Risk Weight', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Simulation error: {e}")

# ── PAGE 6: DATA EXPLORER ─────────────────────────────────────────────────────
elif page == "📈 Data Explorer":
    st.markdown('<div class="main-header">📈 Data Explorer</div>', unsafe_allow_html=True)

    df = raw_df if raw_df is not None else (X_train if X_train is not None else None)

    if df is None:
        st.error("❌ No data file found. Upload European_Bank.csv to your GitHub repo.")
    else:
        st.markdown(f"**Dataset shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows",    f"{df.shape[0]:,}")
        c2.metric("Total Columns", df.shape[1])
        c3.metric("Missing Values", df.isnull().sum().sum())

        st.markdown("---")
        st.markdown("### 🔍 Preview (first 50 rows)")
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Distribution Plot")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            col = st.selectbox("Select column", num_cols)
            fig = px.histogram(df, x=col, nbins=40, title=f"Distribution of {col}",
                               color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📈 Correlation Heatmap")
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig2 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                             title="Feature Correlation Matrix", aspect='auto')
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#7f8c8d; padding:2rem 0;'>
    <p><strong>Bank Customer Churn Prediction System</strong></p>
    <p>Predictive Modeling and Risk Scoring | Machine Learning Project</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
