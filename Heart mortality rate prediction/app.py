import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Failure Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #e63946;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #e63946;
    }
    .stAlert > div {
        border-radius: 8px;
    }
    .sidebar .sidebar-content {
        background: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">❤️ Heart Failure Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict mortality risk using clinical records with ML models</div>', unsafe_allow_html=True)
st.divider()

# ─── Sidebar Navigation ─────────────────────────────────────────────────────────
st.sidebar.title("🗂️ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📂 Data Overview", "📊 EDA", "🤖 Model Training & Evaluation", "🔮 Predict Patient"],
    index=0,
)

st.sidebar.divider()
st.sidebar.info("Upload the **heart_failure_clinical_records_dataset.csv** to get started.")

# ─── File Upload ────────────────────────────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])


@st.cache_data
def load_data(file):
    return pd.read_csv(file)


def preprocess(df):
    """Full preprocessing pipeline matching the notebook."""
    df = df.copy()

    # Outlier capping (IQR method)
    def cap_outliers(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower, upper)
        return data

    for col in ['serum_creatinine', 'creatinine_phosphokinase', 'platelets']:
        df = cap_outliers(df, col)

    # Time period feature
    df['time_period'] = pd.cut(
        df['time'],
        bins=[0, 50, 100, 150, 200, 300],
        labels=['0-50 days', '51-100 days', '101-150 days', '151-200 days', '200+ days']
    )

    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for split in [X_train, X_test, y_train, y_test]:
        split.reset_index(drop=True, inplace=True)

    # Encode time_period
    le = LabelEncoder()
    X_train['time_period'] = le.fit_transform(X_train['time_period'].astype(str))
    X_test['time_period'] = le.transform(X_test['time_period'].astype(str))

    # SMOTE
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # Select top 3 features
    top_features = ['time', 'serum_creatinine', 'ejection_fraction']
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=top_features)
    X_test = pd.DataFrame(X_test_scaled, columns=top_features)

    return X_train, X_test, y_train, y_test, scaler


# ─── Model Definitions (matching the notebook) ──────────────────────────────────
def get_models():
    return {
        'Logistic Regression': LogisticRegression(),
        'Gaussian NB': GaussianNB(),
        'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'Decision Tree': DecisionTreeClassifier(max_depth=13, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=21, max_depth=10, random_state=42),
        'SVC': SVC(C=10, gamma=0.0002, kernel='rbf', probability=True),
        'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=55, learning_rate=0.05, n_iter_no_change=5,
            validation_fraction=0.20, random_state=42
        ),
        'XGB (tuned)': xgb.XGBClassifier(n_estimators=55, random_state=42, verbosity=0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📂 Data Overview":
    st.header("📂 Data Overview")

    if uploaded_file is None:
        st.warning("⬅️ Please upload the dataset using the sidebar.")
        st.stop()

    df = load_data(uploaded_file)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Deaths (1)", int(df['DEATH_EVENT'].sum()))
    col4.metric("Survived (0)", int((df['DEATH_EVENT'] == 0).sum()))

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📈 Statistics", "🔢 Data Quality"])

    with tab1:
        st.dataframe(df.head(35), use_container_width=True)

    with tab2:
        st.dataframe(df.describe().T.style.background_gradient(cmap='RdYlGn'), use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Missing Values")
            missing = df.isnull().sum().rename("Missing Count").to_frame()
            missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
            st.dataframe(missing, use_container_width=True)
        with c2:
            st.subheader("Duplicate Rows")
            dupes = df.duplicated().sum()
            if dupes == 0:
                st.success(f"✅ No duplicate rows found.")
            else:
                st.warning(f"⚠️ {dupes} duplicate rows found.")

            st.subheader("Data Types")
            st.dataframe(df.dtypes.rename("dtype").to_frame(), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.header("📊 Exploratory Data Analysis")

    if uploaded_file is None:
        st.warning("⬅️ Please upload the dataset using the sidebar.")
        st.stop()

    df = load_data(uploaded_file)

    # Target distribution
    st.subheader("🎯 Target Variable Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 5))
        death_counts = df['DEATH_EVENT'].value_counts()
        ax.pie(
            death_counts,
            labels=['Survived', 'Died'],
            autopct='%1.1f%%',
            startangle=140,
            colors=['#66b3ff', '#ff9999'],
            explode=(0, 0.1),
        )
        ax.set_title("Mortality Rate During Follow-Up")
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 5))
        death_counts.plot(kind='bar', color=['green', 'red'], ax=ax)
        ax.set_title("DEATH_EVENT Class Counts")
        ax.set_xlabel("0 = Survived, 1 = Died")
        ax.set_ylabel("Count")
        ax.set_xticklabels(['Survived', 'Died'], rotation=0)
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Age distributions
    st.subheader("👤 Age Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x="age", hue="DEATH_EVENT", multiple="stack", bins=20,
                     palette="Set1", ax=ax)
        ax.set_title("Age Distribution by Death Event")
        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Patients")
        ax.legend(["Survived", "Died"])
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='DEATH_EVENT', y='age', data=df, hue='DEATH_EVENT',
                    palette='coolwarm', legend=False, ax=ax)
        ax.set_title("Age Distribution by Death Event")
        ax.set_xlabel("Death Event (0 = Survived, 1 = Died)")
        ax.set_ylabel("Age")
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Follow-up time analysis
    st.subheader("⏱️ Follow-up Time Analysis")
    df_tmp = df.copy()
    df_tmp['time_period'] = pd.cut(
        df_tmp['time'],
        bins=[0, 50, 100, 150, 200, 300],
        labels=['0-50 days', '51-100 days', '101-150 days', '151-200 days', '200+ days']
    )
    mortality_by_period = df_tmp.groupby('time_period', observed=True)['DEATH_EVENT'].agg(
        ['count', 'sum', 'mean']
    )
    mortality_by_period.columns = ['Total_Patients', 'Deaths', 'Mortality_Rate']

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        mortality_by_period['Mortality_Rate'].plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Mortality Rate by Follow-up Period')
        ax.set_ylabel('Mortality Rate')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        time_outcome = df_tmp.groupby(
            ['time_period', 'DEATH_EVENT'], observed=True
        ).size().unstack(fill_value=0)
        time_outcome.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'red'])
        ax.set_title('Patient Distribution by Follow-up Period')
        ax.set_ylabel('Number of Patients')
        ax.legend(['Survived', 'Died'])
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        plt.close()

    st.divider()

    # Categorical features
    st.subheader("🔢 Categorical Features vs Death Event")
    cat_features = ["anaemia", "diabetes", "high_blood_pressure", "smoking"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for i, feature in enumerate(cat_features):
        ax = axes[i // 2, i % 2]
        sns.countplot(data=df, x=feature, hue="DEATH_EVENT", palette="Set2", ax=ax)
        ax.set_title(f"{feature.capitalize()} vs Death Event")
        ax.set_xlabel(feature.capitalize())
        ax.set_ylabel("Number of Patients")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No', 'Yes'])
        ax.legend(["Survived", "Died"])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Continuous features boxplots
    st.subheader("📦 Continuous Features by Death Event")
    cont_features = ["ejection_fraction", "serum_creatinine", "platelets",
                     "age", "serum_sodium", "creatinine_phosphokinase", "time"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for i, feature in enumerate(cont_features):
        ax = axes[i // 3, i % 3]
        sns.boxplot(data=df, x="DEATH_EVENT", y=feature, hue="DEATH_EVENT",
                    palette="Set3", legend=False, ax=ax)
        ax.set_title(f"{feature.replace('_', ' ').capitalize()} by Death Event")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Survived', 'Died'])
    # Turn off unused subplot
    axes[2, 1].set_visible(False)
    axes[2, 2].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Correlation heatmap
    st.subheader("🔥 Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix of Numerical Features")
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training & Evaluation":
    st.header("🤖 Model Training & Evaluation")

    if uploaded_file is None:
        st.warning("⬅️ Please upload the dataset using the sidebar.")
        st.stop()

    df = load_data(uploaded_file)

    st.info("Training uses the top 3 features **(time, serum_creatinine, ejection_fraction)** after SMOTE balancing, matching the notebook pipeline.")

    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Preprocessing data and training models…"):
            X_train, X_test, y_train, y_test, scaler = preprocess(df)
            models = get_models()
            results = []
            trained_models = {}

            prog = st.progress(0)
            for i, (name, model) in enumerate(models.items()):
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                results.append({
                    "Model": name,
                    "Train Acc": round(accuracy_score(y_train, y_train_pred), 4),
                    "Test Acc": round(accuracy_score(y_test, y_test_pred), 4),
                    "Precision": round(precision_score(y_test, y_test_pred, average='weighted', zero_division=0), 4),
                    "Recall": round(recall_score(y_test, y_test_pred, average='weighted'), 4),
                    "F1": round(f1_score(y_test, y_test_pred, average='weighted'), 4),
                })
                trained_models[name] = model
                prog.progress((i + 1) / len(models))

        df_results = pd.DataFrame(results).sort_values("Recall", ascending=False).reset_index(drop=True)
        best_name = df_results.iloc[0]["Model"]
        best_model = trained_models[best_name]

        st.success(f"✅ Training complete! Best model by Recall: **{best_name}**")

        st.subheader("📊 Model Comparison")
        st.dataframe(
            df_results.style.highlight_max(
                subset=["Test Acc", "Precision", "Recall", "F1"], color="#c8f7c5"
            ).highlight_min(subset=["Test Acc"], color="#f7c5c5"),
            use_container_width=True,
        )

        # Bar chart
        st.subheader("📈 Performance Metrics Chart")
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(df_results))
        width = 0.18
        for j, metric in enumerate(["Train Acc", "Test Acc", "Precision", "Recall", "F1"]):
            ax.bar(x + j * width, df_results[metric], width, label=metric)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Metrics")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(df_results["Model"], rotation=45, ha="right")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Confusion matrices
        st.subheader("🔲 Confusion Matrices")
        cols = st.columns(3)
        for idx, (name, model) in enumerate(trained_models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Survived", "Died"],
                        yticklabels=["Survived", "Died"], ax=ax)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            cols[idx % 3].pyplot(fig)
            plt.close()

        # ROC Curves for models with predict_proba
        st.subheader("📉 ROC Curves")
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, model in trained_models.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = roc_auc_score(y_test, y_prob)
                ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves – All Models")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

        # Feature importance (ExtraTrees)
        st.subheader("🌟 Feature Importance (ExtraTrees)")
        et = ExtraTreesClassifier(random_state=42)
        et.fit(X_train, y_train)
        feat_imp = pd.Series(et.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 3))
        feat_imp.plot(kind='barh', color='red', ax=ax)
        ax.set_title("Feature Importances (ExtraTreesClassifier)")
        ax.set_xlabel("Importance Score")
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

        # Save best model & scaler
        joblib.dump(best_model, "/tmp/best_model.pkl")
        joblib.dump(scaler, "/tmp/scaler.pkl")
        st.session_state["model_trained"] = True
        st.session_state["best_model_name"] = best_name


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT PATIENT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Patient":
    st.header("🔮 Predict Heart Failure Risk for a Patient")

    model_ready = os.path.exists("/tmp/best_model.pkl") and os.path.exists("/tmp/scaler.pkl")

    if not model_ready:
        st.warning("⚠️ No trained model found. Please go to **🤖 Model Training & Evaluation** and train the models first.")
        st.stop()

    best_model = joblib.load("/tmp/best_model.pkl")
    scaler = joblib.load("/tmp/scaler.pkl")
    model_name = st.session_state.get("best_model_name", "Best Model")

    st.info(f"Using model: **{model_name}** (highest Recall on test set)")

    st.subheader("🧾 Enter Patient Clinical Values")
    st.caption("Only the top 3 features (identified via feature importance) are used for prediction.")

    c1, c2, c3 = st.columns(3)
    with c1:
        time = st.number_input(
            "Follow-up Period (days)",
            min_value=1, max_value=300, value=100,
            help="Follow-up period in days after heart failure event."
        )
    with c2:
        serum_creatinine = st.number_input(
            "Serum Creatinine (mg/dL)",
            min_value=0.1, max_value=10.0, value=1.1, step=0.1,
            help="Level of serum creatinine in the blood. Normal: 0.6–1.2 mg/dL."
        )
    with c3:
        ejection_fraction = st.number_input(
            "Ejection Fraction (%)",
            min_value=10, max_value=80, value=38,
            help="Percentage of blood pumped out per heartbeat. Normal: 55–70%."
        )

    if st.button("🔍 Predict", type="primary"):
        input_data = np.array([[time, serum_creatinine, ejection_fraction]])
        input_scaled = scaler.transform(input_data)
        prediction = best_model.predict(input_scaled)[0]

        prob = None
        if hasattr(best_model, "predict_proba"):
            prob = best_model.predict_proba(input_scaled)[0][1]

        st.divider()
        if prediction == 1:
            st.error("⚠️ **Prediction: HIGH RISK — Patient likely to experience a death event.**")
        else:
            st.success("✅ **Prediction: LOW RISK — Patient likely to survive.**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Follow-up Time", f"{time} days")
        c2.metric("Serum Creatinine", f"{serum_creatinine} mg/dL")
        c3.metric("Ejection Fraction", f"{ejection_fraction}%")

        if prob is not None:
            st.divider()
            st.subheader("📊 Death Probability")
            st.progress(float(prob))
            col_a, col_b = st.columns(2)
            col_a.metric("Death Probability", f"{prob * 100:.1f}%")
            col_b.metric("Survival Probability", f"{(1 - prob) * 100:.1f}%")

            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.barh(
                ["Died", "Survived"],
                [prob, 1 - prob],
                color=["#e63946", "#2a9d8f"],
                height=0.4,
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title("Prediction Probabilities")
            for bar, val in zip(bars, [prob, 1 - prob]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1%}", va='center')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()
        st.caption("⚕️ **Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.")
