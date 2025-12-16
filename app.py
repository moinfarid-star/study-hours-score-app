import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------
# PAGE CONFIG (MUST BE TOP)
# ---------------------------
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# FORCE LIGHT THEME + UI STYLES
# ---------------------------
st.markdown(
    """
    <style>
    .stApp { background: #ffffff; color: #0f172a; }

    /* Make the main container a bit nicer */
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    /* Headings */
    h1, h2, h3 { color: #0f172a !important; }

    /* Soft cards */
    .card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    }

    .badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        color: #0f172a;
        margin-right: 8px;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 12px;
        padding: 10px 14px;
        font-weight: 700;
    }

    /* Inputs */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 12px !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }

    /* Hide Streamlit default menu/footer if you want */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# HELPERS
# ---------------------------
def load_data():
    # Support both filenames
    for fname in ["score.csv", "scores.csv"]:
        try:
            df = pd.read_csv(fname)
            return df, fname
        except Exception:
            pass
    return None, None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Try to find Hours column
    cols = {c.lower().strip(): c for c in df.columns}

    # Expected: Hours, Scores
    hours_col = None
    score_col = None

    for key in cols:
        if key in ["hours", "hour", "studyhours", "study_hours", "study hours"]:
            hours_col = cols[key]
        if key in ["scores", "score", "marks", "mark"]:
            score_col = cols[key]

    # If not found, fallback to first two columns
    if hours_col is None or score_col is None:
        if df.shape[1] >= 2:
            hours_col = df.columns[0]
            score_col = df.columns[1]
        else:
            raise ValueError("CSV must have at least two columns (Hours and Scores).")

    df = df.rename(columns={hours_col: "Hours", score_col: "Scores"})
    df = df[["Hours", "Scores"]].copy()
    df["Hours"] = pd.to_numeric(df["Hours"], errors="coerce")
    df["Scores"] = pd.to_numeric(df["Scores"], errors="coerce")
    df = df.dropna()
    return df

def train_model(df, use_split=True, test_size=0.2, random_state=42):
    X = df[["Hours"]].values
    y = df["Scores"].values

    if use_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "r2": r2_score(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            "split": True,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        return model, metrics
    else:
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        metrics = {
            "r2": r2_score(y, preds),
            "mae": mean_absolute_error(y, preds),
            "split": False,
            "n_train": len(X),
            "n_test": 0,
        }
        return model, metrics

def plot_fit(df, model):
    X = df[["Hours"]].values
    y = df["Scores"].values
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    fig = plt.figure()
    plt.scatter(X, y)
    plt.plot(x_line, y_line)
    plt.xlabel("Study Hours")
    plt.ylabel("Score")
    plt.title("Study Hours vs Score (Model Fit)")
    return fig

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.markdown("## ⚙️ Settings")
use_split = st.sidebar.checkbox("Use train-test split (recommended)", value=True)
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
show_data = st.sidebar.checkbox("Show dataset preview", value=False)
show_model = st.sidebar.checkbox("Show model details", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎓 Tips")
st.sidebar.write("Try different study hours and see how the prediction changes.")

# ---------------------------
# HEADER
# ---------------------------
st.markdown(
    """
    <div class="card">
        <span class="badge">📚 Education</span>
        <span class="badge">📈 Regression</span>
        <span class="badge">🐍 Python</span>
        <span class="badge">⚡ Streamlit</span>
        <h1 style="margin-top:10px;">Student Score Predictor</h1>
        <p style="margin-top:-6px; font-size:16px;">
            Predict student scores based on study hours using <b>Simple Linear Regression</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ---------------------------
# LOAD + TRAIN
# ---------------------------
df_raw, used_file = load_data()
if df_raw is None:
    st.error("❌ Could not find `score.csv` or `scores.csv` in the same folder as `app.py`.")
    st.stop()

try:
    df = normalize_columns(df_raw)
except Exception as e:
    st.error(f"❌ Data format issue: {e}")
    st.stop()

model, metrics = train_model(df, use_split=use_split, test_size=test_size, random_state=random_state)

# ---------------------------
# LAYOUT
# ---------------------------
left, right = st.columns([1.05, 1])

with left:
    st.markdown(
        """
        <div class="card">
            <h2>📝 Enter Student Details</h2>
            <p style="margin-top:-6px;">Move the slider and predict the expected score.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    # Input
    min_h = float(df["Hours"].min())
    max_h = float(df["Hours"].max())
    default_h = float(np.clip(5.0, min_h, max_h))

    hours = st.slider("📘 Study Hours", min_value=min_h, max_value=max_h, value=default_h, step=0.25)

    # Predict
    pred = float(model.predict(np.array([[hours]])).ravel()[0])

    st.write("")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="card">
                <h3>⏱️ Hours</h3>
                <h2 style="margin-top:-6px;">{hours:.2f}</h2>
                <p style="color:#334155;">Study time input</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="card">
                <h3>🎯 Predicted</h3>
                <h2 style="margin-top:-6px;">{pred:.2f}</h2>
                <p style="color:#334155;">Expected score</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        r2 = metrics["r2"]
        st.markdown(
            f"""
            <div class="card">
                <h3>📏 R²</h3>
                <h2 style="margin-top:-6px;">{r2:.3f}</h2>
                <p style="color:#334155;">Model fit score</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")
    st.markdown(
        """
        <div class="card">
            <h3>🎒 Quick Note</h3>
            <p style="color:#334155;">
                This is a learning project. Real-world performance depends on data quality and model assumptions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:
    st.markdown(
        """
        <div class="card">
            <h2>📊 Model & Data</h2>
            <p style="margin-top:-6px;">Visualize the dataset and model fit.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    # Chart
    fig = plot_fit(df, model)
    st.pyplot(fig, clear_figure=True)

    # Dataset preview
    if show_data:
        st.markdown("### 🗂 Dataset Preview")
        st.caption(f"Loaded from: `{used_file}` | Rows: {len(df)}")
        st.dataframe(df, use_container_width=True)

    # Model details
    if show_model:
        st.markdown("### 🧠 Model Details")
        st.caption("Simple Linear Regression: Score = (coef × Hours) + intercept")

        coef = float(model.coef_.ravel()[0])
        intercept = float(model.intercept_.ravel()[0])

        st.write(f"**Coefficient (Hours):** {coef:.4f}")
        st.write(f"**Intercept:** {intercept:.4f}")

        if metrics["split"]:
            st.write(f"**Train size:** {metrics['n_train']} | **Test size:** {metrics['n_test']}")
        st.write(f"**MAE:** {metrics['mae']:.3f}")

st.write("")
st.markdown(
    """
    <div class="card">
        <h3>🚀 Next Improvements</h3>
        <ul style="color:#334155; margin-top:-6px;">
            <li>Add grade bands (A/B/C) based on predicted score</li>
            <li>Add confidence note (based on residuals / error)</li>
            <li>Try Polynomial Regression and compare results</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
