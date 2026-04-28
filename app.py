import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="BiasGuard AI", layout="wide")

st.title("BiasGuard AI - Fairness Firewall")
st.markdown("Detecting and Correcting Bias in AI Decisions")

# ---------------- SESSION STATE ----------------
if "data" not in st.session_state:
    st.session_state.data = None

if "corrected" not in st.session_state:
    st.session_state.corrected = False

# ---------------- DATA LOADING ----------------
st.subheader("Load Dataset")

col1, col2 = st.columns(2)

with col1:
    if st.button("Load Default Dataset"):
        try:
            st.session_state.data = pd.read_csv("loan_data.csv")
        except:
            st.error("Default dataset not found")

with col2:
    if st.button("Generate Sample Data"):
        np.random.seed(42)
        st.session_state.data = pd.DataFrame({
            "experience": np.random.randint(0, 10, 200),
            "test_score": np.random.randint(40, 100, 200),
            "gender": np.random.choice([0, 1], 200),
            "selected": np.random.choice([0, 1], 100)
        })

if st.session_state.data is None:
    st.warning("Please load or generate a dataset to continue")
    st.stop()

data = st.session_state.data

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["Data", "Analysis", "Mitigation"])

# ================= TAB 1 =================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

# ================= TAB 2 =================
with tab2:

    st.subheader("Configuration")

    columns = data.columns
    target = st.selectbox("Select Target Column", columns)
    sensitive = st.selectbox("Select Sensitive Column", columns)

    if target == sensitive:
        st.error("Target and Sensitive column must be different")
        st.stop()

    if data[sensitive].nunique() != 2:
        st.error("Sensitive column must have exactly 2 groups")
        st.stop()

    original_data = data.copy()

    try:
        drop_cols = ['First Name', 'Last Name', 'Email', 'Address',
                     'Phone Number', 'Applicant ID', 'Application Date']

        X = data.drop(columns=[target, sensitive])
        X = X.drop(columns=[col for col in drop_cols if col in X.columns])

        y = data[target]

        for col in X.columns:
            if X[col].dtype == "object":
                if X[col].nunique() > 20:
                    X = X.drop(columns=[col])
                else:
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y.astype(str))

        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        y = pd.Series(y).fillna(0).astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000, solver="liblinear")
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    # ---------------- BIAS DETECTION ----------------
    def detect_bias(df, preds, sensitive_col):
        df = df.copy()
        df["pred"] = preds

        g1_data = df[df[sensitive_col] == 1]["pred"]
        g0_data = df[df[sensitive_col] == 0]["pred"]

        g1 = g1_data.mean() if len(g1_data) > 0 else 0
        g0 = g0_data.mean() if len(g0_data) > 0 else 0

        return g1, g0, g1 - g0

    bias_df = original_data.loc[X_test.index]

    g1, g0, bias = detect_bias(bias_df, preds, sensitive)

    st.subheader("Model Performance")
    st.metric("Accuracy", f"{accuracy_score(y_test, preds):.2f}")

    st.subheader("Bias Before BiasGuard")
    st.write(f"Group 1 Rate: {g1:.2f}")
    st.write(f"Group 0 Rate: {g0:.2f}")
    st.write(f"Bias Score: {bias:.2f}")

    st.session_state.model = model
    st.session_state.bias_df = bias_df
    st.session_state.preds = preds
    st.session_state.probs = probs
    st.session_state.bias = bias
    st.session_state.g1 = g1
    st.session_state.g0 = g0
    st.session_state.sensitive = sensitive

# ================= TAB 3 =================
with tab3:

    if "model" not in st.session_state:
        st.warning("Please complete analysis first")
        st.stop()

    # 🔥 FIXED BIAS CORRECTION
    def correct_bias(df, preds, probs, sensitive_col):
        df = df.copy()
        df['prediction'] = preds
        df['prob'] = probs

        group1 = df[df[sensitive_col] == 1]
        group0 = df[df[sensitive_col] == 0]

        g1 = group1['prediction'].mean() if len(group1) > 0 else 0
        g0 = group0['prediction'].mean() if len(group0) > 0 else 0

        g1 = np.nan_to_num(g1)
        g0 = np.nan_to_num(g0)

        diff = int(abs(g1 - g0) * len(df))

        corrected = pd.Series(preds, index=df.index)

        if diff == 0:
            return corrected.values

        if g1 > g0:
            candidates = group1[group1['prediction'] == 1].sort_values(by='prob')
        else:
            candidates = group0[group0['prediction'] == 1].sort_values(by='prob')

        flip_idx = candidates.index[:diff]
        corrected.loc[flip_idx] = 0

        return corrected.values

    if st.button("Run BiasGuard"):
        st.session_state.corrected = True

    if st.session_state.corrected:

        bias_df = st.session_state.bias_df
        preds = st.session_state.preds
        probs = st.session_state.probs
        sensitive = st.session_state.sensitive

        corrected_preds = correct_bias(bias_df, preds, probs, sensitive)

        g1_c, g0_c, bias_c = detect_bias(bias_df, corrected_preds, sensitive)

        st.subheader("Bias After BiasGuard")
        st.write(f"Group 1 Rate: {g1_c:.2f}")
        st.write(f"Group 0 Rate: {g0_c:.2f}")
        st.write(f"Bias Score: {bias_c:.2f}")

        improvement = abs(st.session_state.bias) - abs(bias_c)
        st.write(f"Bias Reduced By: {improvement:.2f}")

        # 🔥 GRAPH FIXED
        labels = [f"{sensitive}=1", f"{sensitive}=0"]
        before = [st.session_state.g1, st.session_state.g0]
        after = [g1_c, g0_c]

        x = np.arange(len(labels))

        fig, ax = plt.subplots()

        ax.bar(x - 0.2, before, 0.4, label="Before")
        ax.bar(x + 0.2, after, 0.4, label="After")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Prediction Rate")
        ax.set_title("Bias Comparison")
        ax.legend()

        st.pyplot(fig, clear_figure=True)
