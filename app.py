"""
Streamlit web application for wildfire prediction.
Interactive demo for making predictions and viewing model insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

import config
from src.models import WildfireModel
from src.preprocessing import preprocess_features


# Page configuration
st.set_page_config(
    page_title="Wildfire Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name, dataset_name="synthetic"):
    """Load trained model from disk."""
    model_path = config.MODELS_DIR / f"{model_name}_{dataset_name}.pkl"
    
    if not model_path.exists():
        return None
    
    return WildfireModel.load(model_path)


@st.cache_data
def load_evaluation_metrics(dataset_name="synthetic"):
    """Load evaluation metrics from CSV."""
    metrics_path = config.RESULTS_DIR / f"evaluation_{dataset_name}" / "evaluation_metrics.csv"
    
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return None


def get_risk_level(probability):
    """Determine risk level based on probability."""
    if probability >= 0.7:
        return "High Risk", "🔴", "risk-high"
    elif probability >= 0.4:
        return "Medium Risk", "🟡", "risk-medium"
    else:
        return "Low Risk", "🟢", "risk-low"


def main():
    # Header
    st.markdown('<div class="main-header">🔥 Wildfire Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered wildfire risk assessment using weather and environmental data</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        ["synthetic", "uci"],
        help="Choose the dataset used for training"
    )
    
    # Model selection
    available_models = {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm"
    }
    
    selected_model_display = st.sidebar.selectbox(
        "Select Model",
        list(available_models.keys()),
        index=2,  # Default to XGBoost
        help="Choose the prediction model"
    )
    selected_model = available_models[selected_model_display]
    
    # Load model
    model = load_model(selected_model, dataset_name)
    
    if model is None:
        st.error(f"Model '{selected_model}' not found. Please train the model first using `python src/models.py`")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📈 Feature Importance", "ℹ️ About"])
    
    # TAB 1: Prediction
    with tab1:
        st.header("Make Wildfire Predictions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Method")
            input_method = st.radio(
                "Choose input method:",
                ["Manual Input", "Upload CSV"],
                horizontal=True
            )
            
            if input_method == "Manual Input":
                st.markdown("### Weather Conditions")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    temp_max = st.slider("Maximum Temperature (°C)", 0, 50, 30)
                    temp_min = st.slider("Minimum Temperature (°C)", 0, 40, 15)
                    temp_avg = st.slider("Average Temperature (°C)", 0, 45, 25)
                
                with col_b:
                    humidity = st.slider("Relative Humidity (%)", 0, 100, 40)
                    wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 15)
                    precipitation = st.slider("Precipitation (mm)", 0, 50, 0)
                
                with col_c:
                    ndvi = st.slider("NDVI (Vegetation Index)", 0.0, 1.0, 0.5)
                    elevation = st.slider("Elevation (m)", 0, 3000, 1000)
                    distance_to_road = st.slider("Distance to Road (km)", 0, 50, 5)
                
                st.markdown("### Temporal Information")
                col_d, col_e = st.columns(2)
                
                with col_d:
                    month = st.selectbox("Month", list(range(1, 13)), index=6)
                    day_of_week = st.selectbox("Day of Week", 
                                               ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                                "Friday", "Saturday", "Sunday"], index=0)
                
                # Create feature dictionary
                input_features = {
                    'temp_max': temp_max,
                    'temp_min': temp_min,
                    'temp_avg': temp_avg,
                    'relative_humidity': humidity,
                    'wind_speed': wind_speed,
                    'precipitation': precipitation,
                    'ndvi': ndvi,
                    'elevation': elevation,
                    'distance_to_road': distance_to_road,
                    'month': month,
                    'day_of_week': ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                    "Friday", "Saturday", "Sunday"].index(day_of_week),
                }
                
                # Add derived features
                input_features['temp_range'] = temp_max - temp_min
                input_features['dryness_index'] = temp_max / (humidity + 1)
                input_features['heat_humidity_index'] = temp_avg * (1 - humidity / 100)
                input_features['wind_fire_potential'] = wind_speed * (100 - humidity) / 100
                
                # Temporal features
                input_features['month_sin'] = np.sin(2 * np.pi * month / 12)
                input_features['month_cos'] = np.cos(2 * np.pi * month / 12)
                input_features['is_summer'] = 1 if month in [6, 7, 8] else 0
                
                # Create DataFrame
                input_df = pd.DataFrame([input_features])
                
                # Align with model features
                for feature in model.feature_names:
                    if feature not in input_df.columns:
                        input_df[feature] = 0
                
                input_df = input_df[model.feature_names]
                
                if st.button("🔥 Predict Fire Risk", type="primary", use_container_width=True):
                    # Preprocess
                    if model.scaler is not None:
                        input_scaled = model.scaler.transform(input_df)
                    else:
                        input_scaled = input_df.values
                    
                    # Predict
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0][1]
                    
                    # Display result in col2
                    with col2:
                        st.subheader("Prediction Result")
                        
                        risk_level, emoji, css_class = get_risk_level(probability)
                        
                        st.markdown(f"""
                        <div class="metric-card {css_class}">
                            <h2 style="text-align: center;">{emoji} {risk_level}</h2>
                            <h1 style="text-align: center; margin: 1rem 0;">{probability*100:.1f}%</h1>
                            <p style="text-align: center; color: #666;">Fire Occurrence Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.metric("Prediction", "Fire" if prediction == 1 else "No Fire")
                        st.metric("Confidence", f"{max(probability, 1-probability)*100:.1f}%")
                        
                        # Risk factors
                        st.markdown("### Key Risk Factors")
                        if probability > 0.5:
                            factors = []
                            if temp_max > 35:
                                factors.append("⚠️ High temperature")
                            if humidity < 30:
                                factors.append("⚠️ Low humidity")
                            if wind_speed > 20:
                                factors.append("⚠️ High wind speed")
                            if precipitation < 1:
                                factors.append("⚠️ Low precipitation")
                            
                            if factors:
                                for factor in factors:
                                    st.markdown(factor)
                            else:
                                st.markdown("✓ Moderate conditions")
                        else:
                            st.markdown("✓ Favorable conditions")
                            st.markdown("✓ Low fire risk")
            
            else:  # CSV Upload
                st.markdown("### Upload CSV File")
                uploaded_file = st.file_uploader(
                    "Upload a CSV file with weather data",
                    type=['csv'],
                    help="CSV should contain columns matching the model features"
                )
                
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    st.write("Data Preview:", df.head())
                    
                    if st.button("🔥 Predict All Rows", type="primary"):
                        # Align features
                        for feature in model.feature_names:
                            if feature not in df.columns:
                                df[feature] = 0
                        
                        X = df[model.feature_names]
                        
                        # Preprocess
                        if model.scaler is not None:
                            X_scaled = model.scaler.transform(X)
                        else:
                            X_scaled = X.values
                        
                        # Predict
                        predictions = model.predict(X_scaled)
                        probabilities = model.predict_proba(X_scaled)[:, 1]
                        
                        # Add to dataframe
                        df['prediction'] = predictions
                        df['fire_probability'] = probabilities
                        df['risk_level'] = df['fire_probability'].apply(
                            lambda p: get_risk_level(p)[0]
                        )
                        
                        st.success(f"Predictions completed for {len(df)} rows!")
                        st.write(df[['prediction', 'fire_probability', 'risk_level']])
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "wildfire_predictions.csv",
                            "text/csv"
                        )
    
    # TAB 2: Model Performance
    with tab2:
        st.header("Model Performance Metrics")
        
        metrics_df = load_evaluation_metrics(dataset_name)
        
        if metrics_df is not None:
            # Display metrics table
            st.subheader("Performance Comparison")
            
            # Format metrics for display
            display_df = metrics_df.copy()
            for col in display_df.columns:
                if col != 'model' and col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
            
            st.dataframe(display_df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # ROC Curve
                roc_path = config.RESULTS_DIR / f"evaluation_{dataset_name}" / f"{selected_model}_roc_curve.png"
                if roc_path.exists():
                    st.subheader("ROC Curve")
                    st.image(str(roc_path))
            
            with col2:
                # Confusion Matrix
                cm_path = config.RESULTS_DIR / f"evaluation_{dataset_name}" / f"{selected_model}_confusion_matrix.png"
                if cm_path.exists():
                    st.subheader("Confusion Matrix")
                    st.image(str(cm_path))
            
            # Precision-Recall Curve
            pr_path = config.RESULTS_DIR / f"evaluation_{dataset_name}" / f"{selected_model}_pr_curve.png"
            if pr_path.exists():
                st.subheader("Precision-Recall Curve")
                st.image(str(pr_path), use_column_width=True)
        else:
            st.warning("No evaluation metrics found. Please run evaluation first using `python src/evaluation.py`")
    
    # TAB 3: Feature Importance
    with tab3:
        st.header("Feature Importance Analysis")
        
        # Feature importance plot
        fi_path = config.RESULTS_DIR / f"evaluation_{dataset_name}" / f"{selected_model}_feature_importance.png"
        if fi_path.exists():
            st.subheader(f"{selected_model_display} - Top Features")
            st.image(str(fi_path), use_column_width=True)
            
            # Load and display feature importance table
            fi_csv_path = config.RESULTS_DIR / f"evaluation_{dataset_name}" / f"{selected_model}_feature_importance.csv"
            if fi_csv_path.exists():
                fi_df = pd.read_csv(fi_csv_path)
                st.subheader("Feature Importance Table")
                st.dataframe(fi_df.head(20), use_container_width=True)
        else:
            st.warning("Feature importance visualization not found. Please run evaluation first.")
        
        # SHAP analysis
        shap_path = config.RESULTS_DIR / f"evaluation_{dataset_name}" / f"{selected_model}_shap_summary.png"
        if shap_path.exists():
            st.subheader("SHAP Summary Plot")
            st.info("SHAP (SHapley Additive exPlanations) shows how each feature contributes to predictions")
            st.image(str(shap_path), use_column_width=True)
    
    # TAB 4: About
    with tab4:
        st.header("About This System")
        
        st.markdown("""
        ### Wildfire Prediction System
        
        This system uses machine learning to predict wildfire occurrence based on weather and environmental conditions.
        
        #### Features
        - **Multiple ML Models**: Compare Logistic Regression, Random Forest, XGBoost, and LightGBM
        - **Real-time Predictions**: Get instant fire risk assessments
        - **Batch Processing**: Upload CSV files for bulk predictions
        - **Model Explainability**: Understand which factors drive predictions using SHAP
        
        #### Input Features
        - **Weather**: Temperature, humidity, wind speed, precipitation
        - **Environmental**: Vegetation index (NDVI), elevation, proximity to roads
        - **Temporal**: Month, day of week, seasonal patterns
        
        #### Model Performance
        Models are evaluated using:
        - ROC-AUC (Area Under the ROC Curve)
        - Precision and Recall
        - F1 Score
        - Confusion Matrix
        
        #### Use Cases
        - Fire management and prevention
        - Resource allocation planning
        - Public warning systems
        - Risk assessment for insurance
        
        #### Limitations
        - Predictions based on historical patterns
        - Does not account for human activities
        - Performance varies by geographic region
        - Should supplement, not replace, expert judgment
        
        #### Technical Stack
        - **ML Frameworks**: scikit-learn, XGBoost, LightGBM
        - **Explainability**: SHAP
        - **Web Framework**: Streamlit
        - **Data Processing**: pandas, numpy
        
        ---
        
        **Version**: 1.0.0
        
        **Contact**: For questions or feedback, please contact the development team.
        """)


if __name__ == "__main__":
    main()
