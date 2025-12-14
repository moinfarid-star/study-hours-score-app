import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------
# Page config + CSS (School vibe)
# ----------------------------
st.set_page_config(page_title="Student Score Predictor", page_icon="📚", layout="wide")

st.markdown("""
<style>
/* Main background */
.main {background-color: #fbfcff;}
/* Sidebar */
section[data-testid="stSidebar"] {background-color: #f3f6ff;}
/* Big title */
.big-title {font-size: 44px; font-weight: 800; margin-bottom: 0px;}
.sub-title {font-size: 16px; opacity: 0.8; margin-top: 6px;}
/* Card style */
.card {
    padding: 18px 18px 10px 18px;
    border-radius: 18px;
    background: white;
    border: 1px solid #eef1ff;
    box-shadow: 0 8px 24px rgba(14, 22, 56, 0.06);
}
.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: #eef2ff;
    font-size: 13px;
    font-weight: 600;
}
hr {border: none; border-top: 1px solid #eef1ff; margin: 18px 0;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("score.csv")  # <-- keeping your filename
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_data()
except Exception as e:
    st.error("❌ 'score.csv' file not found in the same folder as app.py. Please upload it to the repo and redeploy.")
    st.stop()

# Validate columns
required_cols = {"Hours", "Scores"}
if not required_cols.issubset(set(df.columns)):
    st.error(f"❌ Dataset must contain columns: {required_cols}. Your columns are: {list(df.columns)}")
    st.stop()

# ----------------------------
# Sidebar (School panel)
# ----------------------------
st.sidebar.markdown("## 🎓 Student Lab Panel")
st.sidebar.caption("A mini ML app for predicting marks from study hours.")
st.sidebar.markdown("---")

show_data = st.sidebar.checkbox("📄 Show dataset preview", value=True)
show_plot = st.sidebar.checkbox("📈 Show regression chart", value=True)
use_example = st.sidebar.button("✨ Use example values")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 About")
st.sidebar.info(
    "This app uses **Simple Linear Regression** to predict student scores based on study hours."
)

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="big-title">📚 Student Score Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict student marks based on study hours • Simple Linear Regression</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ----------------------------
# Train model on full dataset (as per assignment)
# ----------------------------
X = df[["Hours"]]
y = df["Scores"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# ----------------------------
# KPI cards row
# ----------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown('<div class="card"><span class="badge">📌 Records</span><h2>'
                f'{len(df)}</h2><p style="opacity:0.8">Total students</p></div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card"><span class="badge">⏱️ Avg Hours</span><h2>'
                f'{df["Hours"].mean():.2f}</h2><p style="opacity:0.8">Average study time</p></div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card"><span class="badge">🎯 R² Score</span><h2>'
                f'{r2:.3f}</h2><p style="opacity:0.8">Model fit on dataset</p></div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="card"><span class="badge">📉 MAE</span><h2>'
                f'{mae:.2f}</h2><p style="opacity:0.8">Average prediction error</p></div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ----------------------------
# Main layout
# ----------------------------
left, right = st.columns([1.1, 0.9])

# Default values
default_hours = 9.25
if use_example:
    default_hours = 7.5

with left:
    st.markdown("### 📝 Enter Student Details")
    st.caption("Tip: Try different hours to see how marks change.")

    hours = st.slider("📌 Study Hours", min_value=0.0, max_value=float(max(10.0, df["Hours"].max())),
                      value=float(default_hours), step=0.25)

    predict_btn = st.button("🚀 Predict Score", type="primary")

    if show_data:
        st.markdown("### 📄 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

with right:
    st.markdown("### ✅ Prediction Output")

    if predict_btn:
        pred = model.predict(np.array([[hours]]))[0]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"#### 📌 Predicted Score for **{hours} hours**")
        st.markdown(f"## 🎯 **{pred:.2f} marks**")
        st.caption("This prediction is based on a simple linear relationship in the dataset.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👈 Enter study hours and click **Predict Score** to see the result.")

    if show_plot:
        st.markdown("### 📈 Regression Chart")
        fig, ax = plt.subplots()
        ax.scatter(df["Hours"], df["Scores"])
        ax.plot(df["Hours"], y_pred)
        ax.set_xlabel("Hours Studied")
        ax.set_ylabel("Scores")
        ax.set_title("Study Hours vs Scores (Regression Line)")
        st.pyplot(fig)

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built with ❤️ using Streamlit + Scikit-learn • Simple Linear Regression")
