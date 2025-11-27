"""
AQI Prediction Page with Live Weather Data
Uses multiple ML models for real-time AQI prediction and forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

from weather_api import WeatherAPI, estimate_aqi_from_pollutants, get_aqi_category
from aqi_predictor import AQIPredictorSystem
from feature_engineering import prepare_single_prediction_features


def render_aqi_prediction_page():
    """Render the AQI Prediction dashboard page"""
    
    st.markdown("## Live Weather-Based AQI Prediction")
    st.markdown("Get real-time AQI predictions using live weather data and multiple ML models")
    
    # Initialize session state
    if 'live_data' not in st.session_state:
        st.session_state.live_data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    # Load predictor system
    @st.cache_resource
    def load_predictor():
        # Use correct path - go up two levels from pages/aqi_prediction.py to Project/, then to models/
        models_path = Path(__file__).resolve().parent.parent.parent / 'models'
        predictor = AQIPredictorSystem(models_dir=str(models_path))
        if not predictor.load_models():
            st.warning("Models not found. Please train models first by running: python src/aqi_predictor.py")
            return None
        return predictor
    
    predictor = load_predictor()
    
    # Initialize Weather API - safely check for API key
    api_key = None
    try:
        # Try to get from Streamlit secrets first
        if hasattr(st, 'secrets') and 'OPENWEATHER_API_KEY' in st.secrets:
            api_key = st.secrets['OPENWEATHER_API_KEY']
    except:
        pass
    
    # Fall back to environment variable
    if not api_key:
        api_key = os.getenv('OPENWEATHER_API_KEY')
    
    weather_api = WeatherAPI(api_key=api_key)
    
    # API Key Configuration
    with st.expander("OpenWeatherMap API Configuration", expanded=not api_key):
        st.markdown("""
        To use live weather data, you need a free API key from OpenWeatherMap:
        
        1. Visit [OpenWeatherMap API](https://openweathermap.org/api)
        2. Sign up for a free account (60 calls/minute)
        3. Get your API key from the dashboard
        4. Enter it below or add to `.streamlit/secrets.toml`
        """)
        
        temp_api_key = st.text_input("Enter API Key (temporary):", type="password")
        if temp_api_key:
            weather_api.api_key = temp_api_key
            st.success("API key configured!")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        city_input = st.text_input(
            "🌍 Enter City Name:",
            placeholder="e.g., Delhi, Mumbai, Bangalore",
            help="Enter any Indian city name"
        )
    
    with col2:
        country_code = st.text_input(
            "Country Code:",
            value="IN",
            help="2-letter country code"
        )
    
    # Fetch Data Button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        fetch_button = st.button("Fetch Live Data", type="primary", use_container_width=True)
    
    with col2:
        if st.session_state.live_data:
            if st.button("Clear Data", use_container_width=True):
                st.session_state.live_data = None
                st.session_state.predictions = None
                st.rerun()
    
    # Fetch live data
    if fetch_button and city_input:
        with st.spinner(f"Fetching live data for {city_input}..."):
            live_data = weather_api.get_live_aqi_data(city_input, country_code)
        
        if live_data:
            st.session_state.live_data = live_data
            st.success(f"Successfully fetched data for {city_input}!")
        else:
            st.error("Failed to fetch data. Please check city name and API key.")
    
    # Display data if available
    if st.session_state.live_data and predictor:
        live_data = st.session_state.live_data
        
        st.markdown("---")
        
        # Display current AQI
        category, code, color = get_aqi_category(live_data['actual_aqi'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "City",
                live_data['city']
            )
        
        with col2:
            st.metric(
                "Current AQI",
                f"{live_data['actual_aqi']:.1f}",
                delta=category
            )
        
        with col3:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background-color: {color}; 
                        border-radius: 10px; color: white;'>
                <h3 style='margin: 0;'>{category}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            timestamp = datetime.fromtimestamp(live_data['timestamp'])
            st.metric(
                "Last Updated",
                timestamp.strftime('%H:%M')
            )
        
        st.markdown("---")
        
        # Display pollutant levels
        st.markdown("### Current Pollutant Levels")
        
        pollutants_data = {
            'Pollutant': ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'NO', 'NH3', 'NOx'],
            'Value': [
                live_data['PM2.5'],
                live_data['PM10'],
                live_data['NO2'],
                live_data['CO'],
                live_data['SO2'],
                live_data['O3'],
                live_data['NO'],
                live_data['NH3'],
                live_data['NOx']
            ],
            'Unit': ['μg/m³', 'μg/m³', 'μg/m³', 'mg/m³', 'μg/m³', 'μg/m³', 'μg/m³', 'μg/m³', 'μg/m³']
        }
        
        # Create interactive bar chart
        fig = px.bar(
            pollutants_data,
            x='Pollutant',
            y='Value',
            color='Value',
            color_continuous_scale='Reds',
            title='Live Pollutant Concentrations',
            labels={'Value': 'Concentration'}
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ML Prediction Section
        st.markdown("### ML Model Predictions")
        
        # Prepare features for prediction
        # Need to add lag features and city encoding
        # For live data, we'll use median values as estimates
        
        col1, col2 = st.columns(2)
        
        with col1:
            aqi_lag = st.number_input(
                "Yesterday's AQI (estimate):",
                min_value=0.0,
                max_value=500.0,
                value=float(live_data['actual_aqi']),
                help="Estimated AQI from previous day"
            )
        
        with col2:
            pm25_lag = st.number_input(
                "Yesterday's PM2.5 (estimate):",
                min_value=0.0,
                max_value=500.0,
                value=float(live_data['PM2.5']),
                help="Estimated PM2.5 from previous day"
            )
        
        # City selection for encoding
        available_cities = [
            'Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal',
            'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam',
            'Gurugram', 'Guwahati', 'Hyderabad', 'Jaipur', 'Jorapokhar', 'Kochi',
            'Kolkata', 'Lucknow', 'Mumbai', 'Patna', 'Shillong', 'Talcher',
            'Thiruvananthapuram', 'Visakhapatnam'
        ]
        
        selected_city = st.selectbox(
            "Select nearest training city:",
            available_cities,
            index=available_cities.index('Delhi') if 'Delhi' in available_cities else 0,
            help="Choose the closest city from our training dataset"
        )
        
        if st.button("Predict AQI", type="primary", use_container_width=True):
            with st.spinner("Running predictions with all models..."):
                # Prepare pollutants dictionary with AQI estimate
                pollutants = {
                    'PM2.5': float(live_data['PM2.5']),
                    'PM10': float(live_data['PM10']),
                    'NO': float(live_data['NO']),
                    'NO2': float(live_data['NO2']),
                    'NOx': float(live_data['NOx']),
                    'NH3': float(live_data['NH3']),
                    'CO': float(live_data['CO']),
                    'SO2': float(live_data['SO2']),
                    'O3': float(live_data['O3']),
                    'Benzene': float(live_data['Benzene']),
                    'Toluene': float(live_data['Toluene']),
                    'Xylene': float(live_data['Xylene']),
                    'AQI': float(aqi_lag)  # Use yesterday's AQI for lag features
                }
                
                # Use feature engineering utility to create proper feature set with one-hot encoding
                features_df = prepare_single_prediction_features(
                    pollutants=pollutants,
                    city=selected_city,
                    date=datetime.now(),
                    use_onehot_city=True,  # Use one-hot encoding to match trained model
                    simple_features=True   # Use simple features matching trained model
                )
                
                # Select only required features
                if predictor.feature_columns:
                    # Ensure all required columns exist
                    missing_cols = set(predictor.feature_columns) - set(features_df.columns)
                    for col in missing_cols:
                        features_df[col] = 0
                    features_df = features_df[predictor.feature_columns]
                
                # Make prediction (model trained WITHOUT scaling)
                prediction = predictor.predict(features_df)
                
                st.session_state.predictions = prediction
        
        # Display predictions
        if st.session_state.predictions:
            prediction = st.session_state.predictions
            
            st.markdown("---")
            st.markdown("### Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 30px; border-radius: 15px; text-align: center; color: white;'>
                    <h4 style='margin: 0;'>Real AQI</h4>
                    <h1 style='margin: 10px 0; font-size: 3rem;'>{live_data['actual_aqi']:.1f}</h1>
                    <p style='margin: 0;'>From API</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            padding: 30px; border-radius: 15px; text-align: center; color: white;'>
                    <h4 style='margin: 0;'>Predicted AQI</h4>
                    <h1 style='margin: 10px 0; font-size: 3rem;'>{prediction['predicted_aqi']:.1f}</h1>
                    <p style='margin: 0;'>ML Model</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                difference = abs(prediction['predicted_aqi'] - live_data['actual_aqi'])
                accuracy = max(0, 100 - (difference / live_data['actual_aqi']) * 100) if live_data['actual_aqi'] > 0 else 0
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                            padding: 30px; border-radius: 15px; text-align: center; color: white;'>
                    <h4 style='margin: 0;'>Accuracy</h4>
                    <h1 style='margin: 10px 0; font-size: 3rem;'>{accuracy:.1f}%</h1>
                    <p style='margin: 0;'>Error: ±{difference:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Predicted category
            pred_category, pred_code, pred_color = get_aqi_category(prediction['predicted_aqi'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style='background-color: {color}; padding: 20px; border-radius: 10px; 
                            text-align: center; color: white;'>
                    <h3>Actual Category</h3>
                    <h2>{category}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background-color: {pred_color}; padding: 20px; border-radius: 10px; 
                            text-align: center; color: white;'>
                    <h3>Predicted Category</h3>
                    <h2>{pred_category}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Health Advisory
            st.markdown("---")
            st.markdown("### Health Advisory")
            
            health_advisories = {
                'Good': "Air quality is satisfactory, and air pollution poses little or no risk.",
                'Satisfactory': "Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor exertion.",
                'Moderate': "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
                'Poor': "Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects.",
                'Very Poor': "Health alert: The risk of health effects is increased for everyone. Avoid outdoor exposure.",
                'Severe': "Health warning: Everyone is more likely to be affected by serious health effects. Stay indoors!"
            }
            
            st.info(f"**{pred_category}:** {health_advisories.get(pred_category, 'No advisory available')}")
            
            # Model confidence gauge
            st.markdown("---")
            st.markdown("### Model Confidence")
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = accuracy,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Prediction Accuracy"},
                delta = {'reference': 95},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "lightgreen"}
                    ],
                    'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif not city_input and not st.session_state.live_data:
        # Show instructions
        st.info("Enter a city name above and click 'Fetch Live Data' to get started!")
        
        st.markdown("---")
        st.markdown("### Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Live Weather Data**
            - Real-time air quality from OpenWeatherMap
            - Worldwide city support
            - Accurate pollutant measurements
            """)
        
        with col2:
            st.markdown("""
            **Multiple ML Models**
            - Random Forest, Gradient Boosting
            - KNN, Decision Tree
            - Linear Regression, AdaBoost
            - Ensemble predictions
            """)
        
        with col3:
            st.markdown("""
            **Advanced Analytics**
            - Real-time AQI calculation
            - Accuracy metrics
            - Health advisories
            - Interactive visualizations
            """)


# Render the page
if __name__ == "__main__":
    render_aqi_prediction_page()
