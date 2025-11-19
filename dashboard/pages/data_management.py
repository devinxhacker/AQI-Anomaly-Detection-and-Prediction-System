"""
Data Management Page - Fetch, Store, and Train Models
Allows users to fetch historical air quality data and train models on it
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import subprocess
import time

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

from weather_api import WeatherAPI, fetch_data_with_progress, get_aqi_category
from aqi_predictor import AQIPredictorSystem


# Indian cities for data fetching
DEFAULT_CITIES = [
    'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru',
    'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore',
    'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad',
    'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow',
    'Mumbai', 'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram', 'Visakhapatnam'
]


def render_data_management_page():
    """Render the Data Management dashboard page"""
    
    st.markdown("## Data Management & Model Training")
    st.markdown("Fetch live historical data, store it, and train models - all in one place!")
    
    # Initialize session state
    if 'fetched_data' not in st.session_state:
        st.session_state.fetched_data = None
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'data_saved' not in st.session_state:
        st.session_state.data_saved = False
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Fetch Data", 
        "Dataset Info",
        "Train Models", 
        "Training Status"
    ])
    
    # ========== TAB 1: FETCH DATA ==========
    with tab1:
        st.markdown("### Fetch Historical Air Quality Data")
        st.info("**Note**: Free API tier provides current data. Historical data is simulated with realistic variations.")
        
        # API Key Configuration
        with st.expander("API Configuration", expanded=False):
            st.markdown("""
            Get your free API key from [OpenWeatherMap](https://openweathermap.org/api)
            - Free tier: 60 calls/minute, 1,000,000 calls/month
            - Supports current air quality data worldwide
            """)
            
            temp_api_key = st.text_input("Enter API Key:", type="password", key="fetch_api_key")
            
        # Get API key
        api_key = None
        try:
            if hasattr(st, 'secrets') and 'OPENWEATHER_API_KEY' in st.secrets:
                api_key = st.secrets['OPENWEATHER_API_KEY']
        except:
            pass
        
        if not api_key:
            api_key = temp_api_key or os.getenv('OPENWEATHER_API_KEY')
        
        if not api_key or api_key == 'your_api_key_here':
            st.warning("Please configure your OpenWeatherMap API key above")
            return
        
        # Date Range Selection
        col1, col2 = st.columns(2)
        with col1:
            # Default: 1 year ago from today
            default_start = datetime.now() - timedelta(days=365)
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                max_value=datetime.now()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # City Selection
        st.markdown("#### Select Cities")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            use_default_cities = st.checkbox("Use default Indian cities (26 cities)", value=True)
        
        if use_default_cities:
            selected_cities = DEFAULT_CITIES
            st.success(f"Selected {len(selected_cities)} Indian cities")
        else:
            city_input = st.text_area(
                "Enter cities (one per line):",
                value="\n".join(DEFAULT_CITIES[:5]),
                height=150
            )
            selected_cities = [city.strip() for city in city_input.split('\n') if city.strip()]
        
        # Country Code
        country_code = st.text_input("Country Code", value="IN", max_chars=2, help="2-letter country code (e.g., IN for India)")
        
        # Calculate stats
        days_count = (end_date - start_date).days + 1
        total_records = len(selected_cities) * days_count
        estimated_time = (len(selected_cities) * 1.1) / 60  # seconds to minutes
        
        st.markdown(f"""
        **Fetch Summary:**
        - Cities: {len(selected_cities)}
        - Date Range: {days_count} days
        - Total Records: ~{total_records:,}
        - Estimated Time: ~{estimated_time:.1f} minutes
        """)
        
        # Fetch Button
        if st.button("Fetch Data", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(message, progress):
                status_text.text(message)
                progress_bar.progress(progress)
            
            try:
                with st.spinner("Fetching data from OpenWeatherMap..."):
                    df = fetch_data_with_progress(
                        api_key=api_key,
                        cities=selected_cities,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d'),
                        country_code=country_code,
                        progress_callback=update_progress
                    )
                    
                    st.session_state.fetched_data = df
                    st.session_state.data_saved = False
                    st.session_state.training_complete = False
                    
                    st.success(f"Successfully fetched {len(df):,} records!")
                    st.balloons()
                    
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.exception(e)
    
    # ========== TAB 2: DATASET INFO ==========
    with tab2:
        st.markdown("### Dataset Information")
        
        if st.session_state.fetched_data is not None:
            df = st.session_state.fetched_data
            
            # Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Cities", df['City'].nunique())
            with col3:
                st.metric("Date Range", f"{df['Date'].nunique()} days")
            with col4:
                avg_aqi = df['AQI'].mean()
                st.metric("Avg AQI", f"{avg_aqi:.1f}")
            
            # AQI Distribution
            st.markdown("#### AQI Distribution by Category")
            category_counts = df['AQI_Bucket'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="AQI Category Distribution",
                color=category_counts.index,
                color_discrete_map={
                    'Good': '#00e400',
                    'Satisfactory': '#ffff00',
                    'Moderate': '#ff7e00',
                    'Poor': '#ff0000',
                    'Very Poor': '#8f3f97',
                    'Severe': '#7e0023'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Time Series
            st.markdown("#### AQI Trends Over Time")
            daily_aqi = df.groupby('Date')['AQI'].mean().reset_index()
            fig = px.line(
                daily_aqi,
                x='Date',
                y='AQI',
                title="Average Daily AQI",
                labels={'AQI': 'Average AQI', 'Date': 'Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # City Comparison
            st.markdown("#### Average AQI by City")
            city_aqi = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(15)
            fig = px.bar(
                x=city_aqi.values,
                y=city_aqi.index,
                orientation='h',
                title="Top 15 Cities by Average AQI",
                labels={'x': 'Average AQI', 'y': 'City'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data Preview
            with st.expander("View Data Sample", expanded=False):
                st.dataframe(df.head(100), use_container_width=True)
            
            # Save to CSV
            st.markdown("#### Save Dataset")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                save_path = st.text_input(
                    "Save Location",
                    value="data/City_Day.csv"
                )
            
            with col2:
                st.write("")
                st.write("")
                if st.button("Save Dataset", type="primary"):
                    try:
                        # Ensure directory exists
                        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                        
                        # Save
                        df.to_csv(save_path, index=False)
                        st.session_state.data_saved = True
                        st.success(f"Data saved to {save_path}")
                    except Exception as e:
                        st.error(f"Error saving data: {str(e)}")
        
        else:
            st.info("👈 Fetch data first from the 'Fetch Data' tab")
    
    # ========== TAB 3: TRAIN MODELS ==========
    with tab3:
        st.markdown("### Train ML Models")
        
        if not st.session_state.data_saved:
            st.warning("Please save the dataset first (Dataset Info tab)")
        else:
            st.success("Dataset is ready for training")
            
            st.markdown("""
            #### Training Pipeline:
            1. **Data Preprocessing** - Clean and prepare data
            2. **Feature Engineering** - Create lag features, encode categories
            3. **Train Anomaly Detection** - Isolation Forest, LOF, Autoencoder
            4. **Train Prediction Models** - 7 Regression + 6 Classification models
            5. **Train Clustering** - K-Means, DBSCAN, Hierarchical
            6. **Model Evaluation** - Cross-validation and performance metrics
            
            **Estimated Time:** 10-15 minutes
            """)
            
            # Training options
            with st.expander("Training Options", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
                with col2:
                    cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            
            # Train Button
            if st.button("Train All Models", type="primary", use_container_width=True):
                progress_container = st.container()
                
                with progress_container:
                    st.markdown("### Training in Progress...")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Step 1: Data Preprocessing
                        status_text.text("Step 1/5: Data Preprocessing...")
                        progress_bar.progress(0.1)
                        
                        result = subprocess.run(
                            ['python3', 'src/data_preprocessing.py'],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        
                        if result.returncode != 0:
                            st.error("Data preprocessing failed")
                            st.code(result.stderr)
                            return
                        
                        progress_bar.progress(0.2)
                        st.success("Data preprocessing complete")
                        
                        # Step 2: Train Anomaly Detection
                        status_text.text("Step 2/5: Training Anomaly Detection Models...")
                        progress_bar.progress(0.3)
                        
                        result = subprocess.run(
                            ['python3', 'src/anomaly_detectors.py', '--train-all'],
                            capture_output=True,
                            text=True,
                            timeout=600
                        )
                        
                        if result.returncode != 0:
                            st.warning("Anomaly detection training had issues")
                            with st.expander("View errors"):
                                st.code(result.stderr)
                        else:
                            st.success("Anomaly detection models trained")
                        
                        progress_bar.progress(0.5)
                        
                        # Step 3: Train Prediction Models
                        status_text.text("Step 3/5: Training Prediction Models (16 algorithms)...")
                        progress_bar.progress(0.6)
                        
                        result = subprocess.run(
                            ['python3', 'src/aqi_predictor.py'],
                            capture_output=True,
                            text=True,
                            timeout=600
                        )
                        
                        if result.returncode != 0:
                            st.error("Prediction model training failed")
                            st.code(result.stderr)
                            return
                        
                        progress_bar.progress(0.8)
                        st.success("Prediction models trained (7 regression + 6 classification + 3 clustering)")
                        
                        # Step 4: Generate XAI
                        status_text.text("Step 4/5: Generating Explainable AI insights...")
                        progress_bar.progress(0.9)
                        
                        result = subprocess.run(
                            ['python3', 'src/explainable_ai.py'],
                            capture_output=True,
                            text=True,
                            timeout=600
                        )
                        
                        if result.returncode != 0:
                            st.warning("XAI generation had issues (optional)")
                        else:
                            st.success("XAI insights generated")
                        
                        # Complete
                        progress_bar.progress(1.0)
                        status_text.text("Training Complete!")
                        
                        st.session_state.training_complete = True
                        
                        st.success("All models trained successfully!")
                        st.balloons()
                        
                        # Show summary
                        st.markdown("""
                        ### Training Summary
                        - **Anomaly Detection**: Isolation Forest, LOF, Autoencoder
                        - **Regression Models**: Random Forest, Gradient Boosting, AdaBoost, Decision Tree, Linear, Ridge, KNN
                        - **Classification Models**: Random Forest, Gradient Boosting, AdaBoost, Decision Tree, Logistic, KNN, Naive Bayes
                        - **Clustering**: K-Means, DBSCAN, Hierarchical
                        
                        **Next Steps:**
                        - Explore anomaly detection results
                        - Make AQI predictions
                        - View model performance metrics
                        - Analyze clustering patterns
                        """)
                        
                    except subprocess.TimeoutExpired:
                        st.error("Training timeout - Process took too long")
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
                        st.exception(e)
    
    # ========== TAB 4: TRAINING STATUS ==========
    with tab4:
        st.markdown("### Training Status & Model Files")
        
        # Check for model files
        models_dir = Path('models')
        
        if models_dir.exists():
            model_files = list(models_dir.glob('*.joblib')) + list(models_dir.glob('*.pkl'))
            
            if model_files:
                st.success(f"Found {len(model_files)} model files")
                
                # Categorize models
                regression_models = [f for f in model_files if 'regressor' in f.name and 'best' not in f.name]
                classification_models = [f for f in model_files if 'classifier' in f.name and 'best' not in f.name]
                anomaly_models = [f for f in model_files if 'model.pkl' in f.name]
                clustering_models = [f for f in model_files if 'cluster' in f.name.lower()]
                other_models = [f for f in model_files if f not in regression_models + classification_models + anomaly_models + clustering_models]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Regression Models", len(regression_models))
                    with st.expander("View Files"):
                        for f in regression_models:
                            st.text(f"• {f.name}")
                
                with col2:
                    st.metric("Classification Models", len(classification_models))
                    with st.expander("View Files"):
                        for f in classification_models:
                            st.text(f"• {f.name}")
                
                with col3:
                    st.metric("Anomaly Detection", len(anomaly_models))
                    with st.expander("View Files"):
                        for f in anomaly_models:
                            st.text(f"• {f.name}")
                
                # Model size info
                total_size = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
                st.info(f"📦 Total model storage: {total_size:.1f} MB")
                
                # Load and show performance
                if (models_dir / 'regression_comparison.csv').exists():
                    st.markdown("#### Regression Model Performance")
                    reg_df = pd.read_csv(models_dir / 'regression_comparison.csv', index_col=0)
                    st.dataframe(reg_df.style.highlight_max(axis=0), use_container_width=True)
                
                if (models_dir / 'classification_comparison.csv').exists():
                    st.markdown("#### Classification Model Performance")
                    cls_df = pd.read_csv(models_dir / 'classification_comparison.csv', index_col=0)
                    st.dataframe(cls_df.style.highlight_max(axis=0), use_container_width=True)
            
            else:
                st.warning("No trained models found. Please train models first.")
        
        else:
            st.warning("Models directory not found. Please train models first.")
        
        # Training status
        if st.session_state.training_complete:
            st.success("Training completed in this session!")
        else:
            st.info("No training completed in this session yet")


# Render the page
if __name__ == "__main__":
    render_data_management_page()
