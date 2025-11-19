"""
Interactive Streamlit Dashboard for AQI Anomaly Detection System

This dashboard provides:
1. Real-time anomaly detection and monitoring
2. Explainable AI insights (SHAP/LIME visualizations)
3. Historical anomaly browser with filtering
4. Alert management system
5. Model performance metrics

Author: TY Sem 5 AIML Student
Date: November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import sys
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

# Import custom modules
from alert_system import SeverityClassifier, AlertGenerator, AlertManager
from feature_engineering import engineer_features, get_required_feature_columns
from visualization import AnomalyVisualizer, ModelComparisonVisualizer, ExplainabilityVisualizer
from utils import PathManager, MetricsCalculator

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="AQI Anomaly Detection System",
    page_icon="⚠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

# Custom CSS
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(120deg, #e74c3c, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .anomaly-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .alert-card-critical {
        background: linear-gradient(135deg, #d31027 0%, #ea384d 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #c0392b;
    }
    
    .alert-card-high {
        background: linear-gradient(135deg, #f12711 0%, #f5af19 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #e67e22;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3498db;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    /* Feature importance styling */
    .feature-item {
        padding: 8px;
        margin: 5px 0;
        background-color: #f8f9fa;
        border-radius: 5px;
        border-left: 3px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data
def load_data():
    """Load processed data"""
    try:
        processed_data_path = DATA_DIR / 'City_Day.csv'
        if processed_data_path.exists():
            df = pd.read_csv(processed_data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    model_files = {
        'Isolation Forest': MODELS_DIR / 'isolation_forest_model.pkl',
        'LOF': MODELS_DIR / 'lof_model.pkl'
    }
    
    for name, path in model_files.items():
        if path.exists():
            models[name] = joblib.load(path)
    
    return models


@st.cache_data
def load_feature_names():
    """Load feature column names"""
    try:
        feature_cols_path = MODELS_DIR / 'feature_columns.pkl'
        if feature_cols_path.exists():
            return joblib.load(feature_cols_path)
    except:
        pass
    return None


@st.cache_resource
def load_label_encoder():
    """Load the saved label encoder used during training"""
    encoder_path = MODELS_DIR / 'label_encoder.pkl'
    if encoder_path.exists():
        try:
            return joblib.load(encoder_path)
        except Exception:
            return None
    return None


def load_results_summary():
    """Load detection results summary"""
    try:
        summary_path = RESULTS_DIR / 'detection_summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                return json.load(f)
    except:
        pass
    return None


def get_severity_level(anomaly_score, percentile_90=None):
    """
    Determine severity level of anomaly
    
    Args:
        anomaly_score: Anomaly score from model
        percentile_90: 90th percentile threshold
    
    Returns:
        Severity level string
    """
    if percentile_90 is None:
        percentile_90 = -0.1  # Default threshold
    
    if anomaly_score < percentile_90 * 2:
        return "Critical"
    elif anomaly_score < percentile_90 * 1.5:
        return "High"
    elif anomaly_score < percentile_90:
        return "Medium"
    else:
        return "Low"


def _season_from_month(month_value):
    """Convert month number to season index used during training"""
    if month_value in [12, 1, 2]:
        return 0  # Winter
    if month_value in [3, 4, 5]:
        return 1  # Spring
    if month_value in [6, 7, 8]:
        return 2  # Summer
    return 3  # Fall


def prepare_anomaly_features(df_input):
    """Build the exact 26-column feature matrix expected by anomaly models"""
    feature_cols = load_feature_names()
    if feature_cols is None:
        st.error("Feature metadata missing. Please run data preprocessing first.")
        return None
    if df_input is None or df_input.empty:
        return None

    df_prepped = df_input.copy()
    original_index = df_prepped.index
    df_prepped = df_prepped.reset_index(drop=True)
    df_prepped['_orig_idx'] = np.arange(len(df_prepped))

    # Ensure Date column exists and is parsed
    if 'Date' in df_prepped.columns:
        df_prepped['Date'] = pd.to_datetime(df_prepped['Date'], errors='coerce')
    else:
        df_prepped['Date'] = pd.Timestamp.now()
    df_prepped['Date'] = df_prepped['Date'].fillna(pd.Timestamp.now())

    # Guarantee required pollutant columns exist
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
                     'O3', 'Benzene', 'Toluene', 'Xylene']
    for col in pollutant_cols + ['AQI']:
        if col not in df_prepped.columns:
            df_prepped[col] = 0

    # Temporal features
    df_prepped['Month'] = df_prepped['Date'].dt.month
    df_prepped['DayOfWeek'] = df_prepped['Date'].dt.dayofweek
    df_prepped['DayOfYear'] = df_prepped['Date'].dt.dayofyear
    df_prepped['Quarter'] = df_prepped['Date'].dt.quarter
    df_prepped['Season'] = df_prepped['Month'].apply(_season_from_month)
    df_prepped['IsWeekend'] = (df_prepped['DayOfWeek'] >= 5).astype(int)

    # Sort for lag/rolling computation
    if 'City' in df_prepped.columns:
        df_prepped = df_prepped.sort_values(['City', 'Date'])
        group_obj = df_prepped.groupby('City')
    else:
        df_prepped = df_prepped.sort_values('Date')
        group_obj = None

    # Lag features
    if group_obj is not None:
        df_prepped['AQI_lag1'] = group_obj['AQI'].shift(1)
        df_prepped['AQI_lag7'] = group_obj['AQI'].shift(7)
        df_prepped['PM2.5_lag1'] = group_obj['PM2.5'].shift(1)
    else:
        df_prepped['AQI_lag1'] = df_prepped['AQI'].shift(1)
        df_prepped['AQI_lag7'] = df_prepped['AQI'].shift(7)
        df_prepped['PM2.5_lag1'] = df_prepped['PM2.5'].shift(1)

    df_prepped['AQI_lag1'].fillna(df_prepped['AQI'], inplace=True)
    df_prepped['AQI_lag7'].fillna(df_prepped['AQI'], inplace=True)
    df_prepped['PM2.5_lag1'].fillna(df_prepped['PM2.5'], inplace=True)

    # Rolling stats
    if group_obj is not None:
        df_prepped['AQI_rolling_mean_7'] = group_obj['AQI'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df_prepped['AQI_rolling_std_7'] = group_obj['AQI'].transform(
            lambda x: x.rolling(window=7, min_periods=1).std()
        )
    else:
        df_prepped['AQI_rolling_mean_7'] = df_prepped['AQI'].rolling(window=7, min_periods=1).mean()
        df_prepped['AQI_rolling_std_7'] = df_prepped['AQI'].rolling(window=7, min_periods=1).std()
    df_prepped['AQI_rolling_std_7'].fillna(0, inplace=True)

    # Ratio features
    df_prepped['PM_ratio'] = df_prepped['PM2.5'] / (df_prepped['PM10'] + 1e-6)
    df_prepped['NOx_NO2_ratio'] = df_prepped['NOx'] / (df_prepped['NO2'] + 1e-6)

    # City encoding
    if 'City' in df_prepped.columns:
        encoder = load_label_encoder()
        if encoder:
            class_to_idx = {cls: idx for idx, cls in enumerate(encoder.classes_)}
            df_prepped['City_Encoded'] = df_prepped['City'].map(lambda c: class_to_idx.get(c, 0)).astype(int)
        else:
            fallback_encoder = LabelEncoder()
            df_prepped['City_Encoded'] = fallback_encoder.fit_transform(df_prepped['City'].astype(str))
    else:
        df_prepped['City_Encoded'] = 0

    # Ensure every expected column exists
    for col in feature_cols:
        if col not in df_prepped.columns:
            df_prepped[col] = 0

    # Restore original order and index
    df_prepped = df_prepped.sort_values('_orig_idx')
    features = df_prepped[feature_cols].copy()
    features.index = original_index
    df_prepped.drop(columns=['_orig_idx'], inplace=True)

    return features.fillna(0)


def display_anomaly_card(row, score, severity, idx):
    """Display anomaly information card"""
    
    severity_colors = {
        "Critical": "alert-card-critical",
        "High": "alert-card-high"
    }
    
    card_class = severity_colors.get(severity, "metric-card")
    
    st.markdown(f"""
    <div class="{card_class}">
        <h4>Anomaly #{idx} - {severity}</h4>
        <p><strong>City:</strong> {row.get('City', 'Unknown')}</p>
        <p><strong>Date:</strong> {row.get('Date', 'Unknown')}</p>
        <p><strong>AQI:</strong> {row.get('AQI', 'N/A'):.0f}</p>
        <p><strong>Anomaly Score:</strong> {score:.4f}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    """Main dashboard application"""
    
    # Title
    st.markdown('<h1 class="main-title">AQI Anomaly Detection & Prediction System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Purpose:</strong> This system detects unusual air quality patterns using advanced machine learning 
        and provides explainable insights into why certain readings are flagged as anomalous.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://play-lh.googleusercontent.com/zYcmB8SKs2NMSMTBTBDqEqiF57MJNq__XW97SV3hoYfxBnPOizJwfhiE4KvwY4ziFI4", width=150)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["Dashboard", "Data Management", "Anomaly Explorer", "Explainable AI", 
             "Alert Center", "Model Performance", "AQI Prediction", "Future Forecast"]
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        # Load summary
        summary = load_results_summary()
        if summary:
            st.metric("Training Samples", f"{summary.get('train_size', 0):,}")
            st.metric("Test Samples", f"{summary.get('test_size', 0):,}")
            st.metric("Contamination Rate", f"{summary.get('contamination', 0.1)*100:.1f}%")
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        **Anomaly Detection:**
        - Isolation Forest
        - Local Outlier Factor (LOF)
        - Autoencoder Neural Network
        
        **Prediction Models (16):**
        - 7 Regression Models
        - 6 Classification Models
        - 3 Clustering Algorithms
        
        **Explainability:**
        - SHAP (Shapley Values)
        - LIME (Local Interpretability)
        """)
    
    # Main content based on page selection
    if page == "Dashboard":
        show_dashboard_page()
    elif page == "Data Management":
        show_data_management_page()
    elif page == "Anomaly Explorer":
        show_anomaly_explorer_page()
    elif page == "Explainable AI":
        show_explainable_ai_page()
    elif page == "Alert Center":
        show_alert_center_page()
    elif page == "Model Performance":
        show_model_performance_page()
    elif page == "AQI Prediction":
        show_aqi_prediction_page()
    elif page == "Future Forecast":
        show_future_forecast_page()


# ============================================================================
# Page 1: Dashboard
# ============================================================================

def show_dashboard_page():
    """Main dashboard overview"""
    st.header("System Overview")
    
    # Load data
    df = load_data()
    models = load_models()
    summary = load_results_summary()
    
    if df is None:
        st.error("Data not found. Please run data_preprocessing.py first.")
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Records</h3>
            <h2>{:,}</h2>
            <p>Air quality measurements</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Cities Monitored</h3>
            <h2>{}</h2>
            <p>Across India</p>
        </div>
        """.format(df['City'].nunique()), unsafe_allow_html=True)
    
    with col3:
        if summary and 'detectors' in summary:
            total_anomalies = sum(d['num_anomalies'] for d in summary['detectors'].values())
            st.markdown("""
            <div class="anomaly-card">
                <h3>Anomalies Detected</h3>
                <h2>{:,}</h2>
                <p>Unusual patterns found</p>
            </div>
            """.format(total_anomalies // len(summary['detectors'])), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="anomaly-card">
                <h3>Anomalies</h3>
                <h2>N/A</h2>
                <p>Run detection first</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Models Active</h3>
            <h2>{}</h2>
            <p>Detection algorithms</p>
        </div>
        """.format(len(models)), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("AQI Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        df['AQI'].hist(bins=50, ax=ax, color='#3498db', edgecolor='black')
        ax.set_xlabel('AQI Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of AQI Values')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Top Cities by Records")
        city_counts = df['City'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        city_counts.plot(kind='barh', ax=ax, color='#e74c3c')
        ax.set_xlabel('Number of Records')
        ax.set_title('Top 10 Cities')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # Recent data section
    st.markdown("---")
    st.subheader("Recent Data Sample")
    st.dataframe(df.tail(10), use_container_width=True)


# ============================================================================
# Page 2: Anomaly Explorer
# ============================================================================

def show_anomaly_explorer_page():
    """Explore detected anomalies"""
    st.header("Anomaly Explorer")
    
    st.markdown("""
    <div class="info-box">
        Browse and analyze detected anomalies with filtering options. 
        Each anomaly is scored and categorized by severity level.
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    models = load_models()
    
    if df is None or not models:
        st.warning("Please ensure data preprocessing and model training are complete.")
        return
    
    # Model selection
    model_name = st.selectbox("Select Detection Model", list(models.keys()))
    
    st.markdown("---")
    
    # Detect anomalies
    with st.spinner("Detecting anomalies..."):
        try:
            df_prepared = df.copy()
            feature_matrix = prepare_anomaly_features(df_prepared)
            if feature_matrix is None:
                st.error("Could not prepare anomaly features. Please rerun preprocessing.")
                return
            df_prepared[feature_matrix.columns] = feature_matrix
            X = feature_matrix.values
            
            # Predict
            model = models[model_name]
            predictions = model.predict(X)
            scores = model.score_samples(X)
            
            # Find anomalies
            anomaly_mask = (predictions == -1)
            anomaly_indices = np.where(anomaly_mask)[0]
            
            st.success(f"Detected {len(anomaly_indices)} anomalies ({len(anomaly_indices)/len(X)*100:.2f}%)")
            
            # Filter options
            st.subheader("Filter Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cities = ['All'] + sorted(df_prepared['City'].dropna().unique().tolist())
                selected_city = st.selectbox("City", cities)
            
            with col2:
                severity_options = ['All', 'Critical', 'High', 'Medium', 'Low']
                selected_severity = st.selectbox("Severity", severity_options)
            
            with col3:
                num_display = st.slider("Number to Display", 10, 100, 20)
            
            # Apply filters
            anomaly_df = df_prepared.iloc[anomaly_indices].copy()
            anomaly_df['Anomaly_Score'] = scores[anomaly_indices]
            
            # Calculate severity
            if anomaly_mask.any():
                percentile_90 = np.percentile(scores[anomaly_mask], 10)
            else:
                percentile_90 = np.percentile(scores, 10) if len(scores) else 0
            anomaly_df['Severity'] = anomaly_df['Anomaly_Score'].apply(
                lambda x: get_severity_level(x, percentile_90)
            )
            
            # Filter by city
            if selected_city != 'All':
                anomaly_df = anomaly_df[anomaly_df['City'] == selected_city]
            
            # Filter by severity
            if selected_severity != 'All':
                anomaly_df = anomaly_df[anomaly_df['Severity'] == selected_severity]
            
            # Sort by score
            anomaly_df = anomaly_df.sort_values('Anomaly_Score').head(num_display)
            
            st.markdown("---")
            st.subheader(f"Top {len(anomaly_df)} Anomalies")
            
            # Display anomaly cards
            for idx, (_, row) in enumerate(anomaly_df.iterrows(), 1):
                with st.expander(f"Anomaly #{idx} - {row['Severity']} - {row['City']} ({row['Date'].date()})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Pollutant Levels:**")
                        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
                        for pollutant in pollutants:
                            if pollutant in row:
                                st.write(f"- **{pollutant}:** {row[pollutant]:.2f}")
                    
                    with col2:
                        st.markdown("**Anomaly Details:**")
                        st.write(f"- **AQI:** {row['AQI']:.0f}")
                        st.write(f"- **Anomaly Score:** {row['Anomaly_Score']:.4f}")
                        st.write(f"- **Severity:** {row['Severity']}")
                        st.write(f"- **Month:** {row['Month']}")
                        st.write(f"- **Day of Week:** {row['DayOfWeek']}")
            
        except Exception as e:
            st.error(f"Error during anomaly detection: {str(e)}")


# ============================================================================
# Page 3: Explainable AI
# ============================================================================

def show_explainable_ai_page():
    """Show explainability visualizations"""
    st.header("Explainable AI Insights")
    
    st.markdown("""
    <div class="info-box">
        <strong>What is Explainable AI (XAI)?</strong><br>
        XAI helps us understand <em>why</em> a particular reading was flagged as anomalous. 
        We use SHAP (Shapley values) and LIME to identify which pollutants contributed most to the anomaly detection.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check for generated visualizations
    st.subheader("SHAP Feature Importance")
    
    shap_files = list(RESULTS_DIR.glob('shap_summary_*.png'))
    
    if shap_files:
        for shap_file in shap_files:
            model_name = shap_file.stem.replace('shap_summary_', '').replace('_', ' ').title()
            st.markdown(f"### {model_name}")
            st.image(str(shap_file), use_container_width=True)
    else:
        st.warning("SHAP visualizations not found. Please run src/explainable_ai.py first.")
    
    st.markdown("---")
    
    # Feature importance tables
    st.subheader("Feature Importance Rankings")
    
    importance_files = list(RESULTS_DIR.glob('feature_importance_*.csv'))
    
    if importance_files:
        tabs = st.tabs([f.stem.replace('feature_importance_', '').replace('_', ' ').title() 
                        for f in importance_files])
        
        for tab, importance_file in zip(tabs, importance_files):
            with tab:
                try:
                    importance_df = pd.read_csv(importance_file)
                    
                    # Display top 15
                    st.dataframe(
                        importance_df.head(15).style.background_gradient(subset=['Importance']),
                        use_container_width=True
                    )
                    
                    # Bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    importance_df.head(10).plot(
                        x='Feature', y='Importance', kind='barh', ax=ax, color='#3498db'
                    )
                    ax.set_xlabel('SHAP Importance')
                    ax.set_title('Top 10 Features by SHAP Importance')
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error loading importance data: {str(e)}")
    else:
        st.info("No feature importance data available yet.")


# ============================================================================
# Page 4: Alert Center
# ============================================================================

def show_alert_center_page():
    """Alert management system"""
    st.header("Alert Center")
    
    st.markdown("""
    <div class="alert-card-critical">
        <h3>Active Alerts</h3>
        <p>Real-time monitoring of critical and high-severity anomalies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    models = load_models()
    
    if df is None or not models:
        st.warning("System not initialized")
        return
    
    # Detect recent anomalies
    with st.spinner("Scanning for alerts..."):
        try:
            df_prepared = df.copy()
            feature_matrix = prepare_anomaly_features(df_prepared)
            if feature_matrix is None:
                st.error("Could not prepare anomaly features. Please rerun preprocessing.")
                return
            df_prepared[feature_matrix.columns] = feature_matrix
            X = feature_matrix.values
            
            model = list(models.values())[0]
            predictions = model.predict(X)
            scores = model.score_samples(X)
            
            anomaly_mask = (predictions == -1)
            anomaly_indices = np.where(anomaly_mask)[0]
            
            # Get recent anomalies
            recent_df = df_prepared.iloc[anomaly_indices].copy()
            recent_df['Anomaly_Score'] = scores[anomaly_indices]
            
            # Calculate severity
            if anomaly_mask.any():
                percentile_90 = np.percentile(scores[anomaly_mask], 10)
            else:
                percentile_90 = np.percentile(scores, 10) if len(scores) else 0
            recent_df['Severity'] = recent_df['Anomaly_Score'].apply(
                lambda x: get_severity_level(x, percentile_90)
            )
            
            # Filter critical and high
            critical_df = recent_df[recent_df['Severity'].str.contains('Critical|High')]
            critical_df = critical_df.sort_values('Anomaly_Score').head(20)
            
            # Display alerts
            st.subheader(f"{len(critical_df)} Critical/High Alerts")
            
            for idx, (_, row) in enumerate(critical_df.iterrows(), 1):
                severity_class = "alert-card-critical" if "Critical" in row['Severity'] else "alert-card-high"
                
                st.markdown(f"""
                <div class="{severity_class}">
                    <h4>Alert #{idx} - {row['Severity']}</h4>
                    <p><strong>Location:</strong> {row['City']}</p>
                    <p><strong>Date:</strong> {row['Date']}</p>
                    <p><strong>AQI:</strong> {row['AQI']:.0f}</p>
                    <p><strong>PM2.5:</strong> {row.get('PM2.5', 0):.1f} μg/m³</p>
                    <p><strong>Score:</strong> {row['Anomaly_Score']:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Summary statistics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Critical Alerts", 
                         len(critical_df[critical_df['Severity'].str.contains('Critical')]))
            with col2:
                st.metric("High Alerts", 
                         len(critical_df[critical_df['Severity'].str.contains('High')]))
            with col3:
                most_affected = critical_df['City'].value_counts().index[0] if len(critical_df) > 0 else "N/A"
                st.metric("Most Affected City", most_affected)
            
        except Exception as e:
            st.error(f"Error generating alerts: {str(e)}")


# ============================================================================
# Page 5: Model Performance
# ============================================================================

def show_model_performance_page():
    """Show model performance metrics"""
    st.header("Model Performance Analysis")
    
    # Load summary
    summary = load_results_summary()
    
    if summary is None:
        st.warning("No performance data available. Please run anomaly_detectors.py first.")
        return
    
    st.markdown("""
    <div class="success-box">
        <strong>Models Trained Successfully</strong><br>
        Comprehensive anomaly detection pipeline with multiple algorithms.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Training information
    st.subheader("Training Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Samples", f"{summary.get('train_size', 0):,}")
    with col2:
        st.metric("Test Samples", f"{summary.get('test_size', 0):,}")
    with col3:
        st.metric("Contamination Rate", f"{summary.get('contamination', 0.1)*100:.1f}%")
    
    st.markdown("---")
    
    # Model comparison
    st.subheader("Model Comparison")
    
    if 'detectors' in summary:
        comparison_data = []
        for name, metrics in summary['detectors'].items():
            comparison_data.append({
                'Model': name.replace('_', ' ').title(),
                'Anomalies Detected': metrics['num_anomalies'],
                'Detection Rate (%)': f"{metrics['anomaly_rate']:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Anomaly counts
        comparison_df.plot(x='Model', y='Anomalies Detected', kind='bar', ax=ax1, color='#e74c3c')
        ax1.set_title('Anomalies Detected by Model')
        ax1.set_ylabel('Count')
        ax1.grid(alpha=0.3)
        ax1.legend().remove()
        
        # Detection rates
        rates = [float(r.rstrip('%')) for r in comparison_df['Detection Rate (%)']]
        comparison_df_rates = comparison_df.copy()
        comparison_df_rates['Detection Rate'] = rates
        comparison_df_rates.plot(x='Model', y='Detection Rate', kind='bar', ax=ax2, color='#3498db')
        ax2.set_title('Detection Rate by Model')
        ax2.set_ylabel('Rate (%)')
        ax2.axhline(y=summary.get('contamination', 0.1)*100, color='red', 
                   linestyle='--', label='Expected Rate')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Comparison visualization
    st.markdown("---")
    st.subheader("Visual Comparison")
    
    comparison_viz_path = RESULTS_DIR / 'anomaly_detection_comparison.png'
    if comparison_viz_path.exists():
        st.image(str(comparison_viz_path), use_container_width=True)
    else:
        st.info("Comprehensive comparison visualization not yet generated.")
    
    # Prediction models performance
    st.markdown("---")
    st.subheader("Prediction Models Performance")
    
    # Load regression comparison
    reg_comp_path = MODELS_DIR / 'regression_comparison.csv'
    if reg_comp_path.exists():
        reg_df = pd.read_csv(reg_comp_path)
        
        st.markdown("**Regression Models (AQI Value Prediction)**")
        
        # Clean up model names for display
        reg_display = reg_df.copy()
        reg_display['Model Name'] = reg_display['model'].str.split('(').str[0]
        
        # Show key metrics
        display_cols = ['Model Name', 'r2', 'rmse', 'mae', 'mape']
        if all(col in reg_display.columns or col == 'Model Name' for col in display_cols):
            st.dataframe(reg_display[display_cols].round(4), use_container_width=True)
        
        # Visualize regression performance
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=reg_display['Model Name'],
            y=reg_df['r2'],
            name='R² Score',
            marker_color='lightblue',
            text=reg_df['r2'].round(4),
            textposition='outside'
        ))
        fig.update_layout(
            title='Regression Models - R² Score Comparison',
            xaxis_title='Model',
            yaxis_title='R² Score',
            height=400,
            yaxis_range=[0, 1.05]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Load classification comparison
    cls_comp_path = MODELS_DIR / 'classification_comparison.csv'
    if cls_comp_path.exists():
        cls_df = pd.read_csv(cls_comp_path)
        
        st.markdown("**Classification Models (AQI Category Prediction)**")
        
        # Clean up model names for display
        cls_display = cls_df.copy()
        cls_display['Model Name'] = cls_display['model'].str.split('(').str[0]
        
        # Show key metrics
        display_cols = ['Model Name', 'accuracy', 'precision', 'recall', 'f1']
        if all(col in cls_display.columns or col == 'Model Name' for col in display_cols):
            st.dataframe(cls_display[display_cols].round(4), use_container_width=True)
        
        # Visualize classification performance
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cls_display['Model Name'],
            y=cls_df['accuracy'],
            name='Accuracy',
            marker_color='lightgreen',
            text=cls_df['accuracy'].round(4),
            textposition='outside'
        ))
        fig.update_layout(
            title='Classification Models - Accuracy Comparison',
            xaxis_title='Model',
            yaxis_title='Accuracy',
            height=400,
            yaxis_range=[0, 1.05]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Clustering visualization
    st.markdown("---")
    st.subheader("Anomaly Detection Visualization")
    
    # Load dataset for anomaly viz
    df = load_data()
    if df is not None:
        # Load anomaly detection models
        clustering_info = [
            ('isolation_forest', 'Isolation Forest'),
            ('lof', 'LOF')
        ]
        
        tabs = st.tabs(["Isolation Forest", "LOF"])
        
        for idx, (tab, (model_name, display_name)) in enumerate(zip(tabs, clustering_info)):
            with tab:
                model_path = MODELS_DIR / f'{model_name}_model.pkl'
                if model_path.exists():
                    try:
                        anomaly_model = joblib.load(model_path)
                        
                        df_subset = df.head(1000).copy()
                        feature_matrix = prepare_anomaly_features(df_subset)
                        if feature_matrix is None:
                            st.warning("Feature metadata missing. Skipping visualization.")
                            continue
                        df_subset[feature_matrix.columns] = feature_matrix
                        X_subset = feature_matrix.values
                        
                        # Get anomaly labels (-1 for anomaly, 1 for normal)
                        labels = anomaly_model.predict(X_subset)
                        
                        # Create visualization using first 2 components (PCA-like)
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_subset)
                        
                        # Plot anomaly detection results
                        fig = go.Figure()
                        
                        # Separate normal and anomaly points
                        normal_mask = labels == 1
                        anomaly_mask = labels == -1
                        
                        # Plot normal points
                        fig.add_trace(go.Scatter(
                            x=X_pca[normal_mask, 0],
                            y=X_pca[normal_mask, 1],
                            mode='markers',
                            name='Normal',
                            marker=dict(
                                size=6,
                                color='lightblue',
                                opacity=0.5
                            )
                        ))
                        
                        # Plot anomaly points
                        fig.add_trace(go.Scatter(
                            x=X_pca[anomaly_mask, 0],
                            y=X_pca[anomaly_mask, 1],
                            mode='markers',
                            name='Anomaly',
                            marker=dict(
                                size=10,
                                color='red',
                                opacity=0.8,
                                symbol='x'
                            )
                        ))
                        
                        fig.update_layout(
                            title=f'{display_name} - Anomaly Detection (PCA Visualization)',
                            xaxis_title='First Principal Component',
                            yaxis_title='Second Principal Component',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Anomaly statistics
                        st.markdown(f"**Detection Statistics:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            n_normal = np.sum(labels == 1)
                            st.metric("Normal Points", n_normal)
                        with col2:
                            n_anomalies = np.sum(labels == -1)
                            st.metric("Anomalies Detected", n_anomalies)
                        with col3:
                            anomaly_pct = (n_anomalies / len(labels)) * 100
                            st.metric("Anomaly Rate", f"{anomaly_pct:.2f}%")
                        
                    except Exception as e:
                        st.warning(f"Could not visualize {model_name}: {str(e)}")
                else:
                    st.info(f"{model_name.upper()} model not found. Train clustering models first.")


# ============================================================================
# Page 6: AQI Prediction with Live Weather Data
# ============================================================================

def show_data_management_page():
    """Data Management page for fetching and training"""
    from pages.data_management import render_data_management_page
    render_data_management_page()


def show_aqi_prediction_page():
    """AQI Prediction page with live weather integration"""
    from pages.aqi_prediction import render_aqi_prediction_page
    render_aqi_prediction_page()


def show_future_forecast_page():
    """Future AQI Forecast page"""
    from pages.future_prediction import render_future_prediction_page
    render_future_prediction_page()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
