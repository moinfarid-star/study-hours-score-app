import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Page configuration
st.set_page_config(
    page_title="Student Study Hours vs Scores Predictor",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 Student Study Hours vs Scores Predictor")
st.markdown("""
This app uses **Simple Linear Regression** to predict student scores based on the number of hours they studied.
""")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("score.csv")
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["📊 Data Overview", "🤖 Model Training & Evaluation", "🔮 Predict Scores", "📈 Visualizations"]
)

# Page 1: Data Overview
if page == "📊 Data Overview":
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Dataset Information")
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Features:** {list(df.columns)}")
        
        info_dict = {
            "Column": df.columns.tolist(),
            "Non-Null Count": [df[col].notna().sum() for col in df.columns],
            "Data Type": [str(df[col].dtype) for col in df.columns]
        }
        st.dataframe(pd.DataFrame(info_dict), use_container_width=True)
    
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Full Dataset")
    st.dataframe(df, use_container_width=True)

# Page 2: Model Training & Evaluation
elif page == "🤖 Model Training & Evaluation":
    st.header("Model Training & Evaluation")
    
    # Prepare data
    X = df[['Hours']]
    y = df['Scores']
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Display model information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    
    with col2:
        st.metric("Slope (Coefficient)", f"{slope:.2f}")
    
    with col3:
        st.metric("Intercept", f"{intercept:.2f}")
    
    st.markdown("---")
    
    # Model equation
    st.subheader("Model Equation")
    st.latex(f"Score = {intercept:.2f} + {slope:.2f} \\times Hours")
    
    st.markdown("---")
    
    # Actual vs Predicted comparison
    st.subheader("Actual vs Predicted Scores")
    comparison_df = pd.DataFrame({
        'Hours Studied': X['Hours'],
        'Actual Score': y,
        'Predicted Score': y_pred.round(2),
        'Difference': (y - y_pred).round(2)
    })
    st.dataframe(comparison_df, use_container_width=True)
    
    # Download button
    csv = comparison_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Comparison Table",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

# Page 3: Predict Scores
elif page == "🔮 Predict Scores":
    st.header("Predict Student Scores")
    
    # Prepare and train model
    X = df[['Hours']]
    y = df['Scores']
    model = LinearRegression()
    model.fit(X, y)
    
    # Input section
    st.subheader("Enter Study Hours")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        hours_input = st.number_input(
            "Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=9.25,
            step=0.25,
            help="Enter the number of hours a student studied"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        predict_button = st.button("🔮 Predict Score", type="primary", use_container_width=True)
    
    if predict_button or hours_input:
        # Make prediction
        prediction = model.predict(np.array([[hours_input]]))[0]
        
        # Display result
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Study Hours", f"{hours_input} hours")
        
        with col2:
            st.metric("Predicted Score", f"{prediction:.2f} marks", delta=None)
        
        # Show prediction on a gauge
        st.markdown("---")
        st.subheader("Score Visualization")
        
        # Create a simple progress bar visualization
        max_score = max(y.max(), prediction) * 1.1
        score_percentage = (prediction / max_score) * 100
        
        st.progress(score_percentage / 100)
        st.caption(f"Predicted Score: {prediction:.2f} out of ~{max_score:.0f} (estimated max)")
        
        # Show model info
        with st.expander("📊 Model Information"):
            r2 = r2_score(y, model.predict(X))
            st.write(f"**R² Score:** {r2:.4f}")
            st.write(f"**Model Equation:** Score = {model.intercept_:.2f} + {model.coef_[0]:.2f} × Hours")
    
    # Batch prediction
    st.markdown("---")
    st.subheader("Batch Prediction")
    
    hours_list = st.text_input(
        "Enter multiple hours (comma-separated)",
        placeholder="e.g., 2.5, 5.1, 8.5, 9.25",
        help="Enter study hours separated by commas"
    )
    
    if hours_list:
        try:
            hours_array = [float(h.strip()) for h in hours_list.split(",")]
            predictions = model.predict(np.array(hours_array).reshape(-1, 1))
            
            batch_df = pd.DataFrame({
                'Hours Studied': hours_array,
                'Predicted Score': predictions.round(2)
            })
            
            st.dataframe(batch_df, use_container_width=True)
            
            # Download batch predictions
            csv = batch_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Batch Predictions",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
        except ValueError:
            st.error("Please enter valid numbers separated by commas.")

# Page 4: Visualizations
elif page == "📈 Visualizations":
    st.header("Data Visualizations")
    
    # Prepare data and train model
    X = df[['Hours']]
    y = df['Scores']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Scatter plot with regression line
    st.subheader("Study Hours vs Student Scores")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='blue', alpha=0.6, s=100, label='Actual Data', edgecolors='black', linewidth=1)
    ax.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
    ax.set_xlabel("Hours Studied", fontsize=12)
    ax.set_ylabel("Scores", fontsize=12)
    ax.set_title("Study Hours vs Student Scores", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Residual plot
    st.subheader("Residual Plot")
    
    residuals = y - y_pred
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(y_pred, residuals, color='green', alpha=0.6, s=100, edgecolors='black', linewidth=1)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel("Predicted Scores", fontsize=12)
    ax2.set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
    ax2.set_title("Residual Plot", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(fig2)
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Study Hours")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.hist(df['Hours'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.set_xlabel("Hours Studied", fontsize=12)
        ax3.set_ylabel("Frequency", fontsize=12)
        ax3.set_title("Distribution of Study Hours", fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
    
    with col2:
        st.subheader("Distribution of Scores")
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.hist(df['Scores'], bins=10, color='lightcoral', edgecolor='black', alpha=0.7)
        ax4.set_xlabel("Scores", fontsize=12)
        ax4.set_ylabel("Frequency", fontsize=12)
        ax4.set_title("Distribution of Scores", fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Simple Linear Regression Model | Student Study Hours vs Scores Predictor</p>
</div>
""", unsafe_allow_html=True)

