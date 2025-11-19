"""
Future AQI Prediction Page
Predict AQI for future dates using trained ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import joblib

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))

from aqi_predictor import AQIPredictorSystem
from weather_api import WeatherAPI, get_aqi_category
from feature_engineering import engineer_features, prepare_single_prediction_features


DEFAULT_CITIES = ['Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Kolkata', 'Hyderabad']


def render_future_prediction_page():
    """Render future AQI prediction page"""
    
    st.markdown("## Future AQI Prediction")
    st.markdown("Predict air quality for upcoming dates using ML models")
    
    # Load predictor
    @st.cache_resource
    def load_predictor():
        models_path = Path(__file__).resolve().parent.parent.parent / 'models'
        predictor = AQIPredictorSystem(models_dir=str(models_path))
        if not predictor.load_models():
            return None
        return predictor
    
    predictor = load_predictor()
    
    if predictor is None:
        st.error("Prediction models not loaded. Please train models first.")
        return
    
    # Input Section
    # Initialize Weather API
    api_key = None
    try:
        if hasattr(st, 'secrets') and 'OPENWEATHER_API_KEY' in st.secrets:
            api_key = st.secrets['OPENWEATHER_API_KEY']
    except:
        pass
    if not api_key:
        api_key = os.getenv('OPENWEATHER_API_KEY')
    
    weather_api = WeatherAPI(api_key=api_key) if api_key else None
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_city = st.selectbox("Select City", DEFAULT_CITIES)
        
    with col2:
        forecast_days = st.slider("Forecast Days", 1, 30, 7)
    
    # Auto-fetch current conditions
    st.markdown("### Current Air Quality Conditions")
    
    auto_fetch = st.checkbox("Auto-fetch current conditions from API", value=True)
    
    current_data = None
    if auto_fetch and weather_api:
        with st.spinner(f"Fetching current data for {selected_city}..."):
            current_data = weather_api.get_live_aqi_data(selected_city, 'IN')
        
        if current_data:
            st.success(f"Fetched current AQI: {current_data['actual_aqi']:.1f}")
            # Use fetched values as defaults
            pm25_default = current_data.get('PM2.5', 50)
            pm10_default = current_data.get('PM10', 75)
            no2_default = current_data.get('NO2', 40)
            co_default = current_data.get('CO', 1.0)
            so2_default = current_data.get('SO2', 20)
            o3_default = current_data.get('O3', 50)
        else:
            st.info("Using default values. Enter manually if needed.")
            pm25_default = 50.0
            pm10_default = 75.0
            no2_default = 40.0
            co_default = 1.0
            so2_default = 20.0
            o3_default = 50.0
    else:
        st.info("Manual input mode. Enter current pollutant values.")
        pm25_default = 50.0
        pm10_default = 75.0
        no2_default = 40.0
        co_default = 1.0
        so2_default = 20.0
        o3_default = 50.0
    
    st.markdown("**Adjust values if needed:**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pm25 = st.number_input("PM2.5 (μg/m³)", 0.0, 500.0, float(pm25_default), 1.0)
    with col2:
        pm10 = st.number_input("PM10 (μg/m³)", 0.0, 600.0, float(pm10_default), 1.0)
    with col3:
        no2 = st.number_input("NO2 (μg/m³)", 0.0, 400.0, float(no2_default), 1.0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        co = st.number_input("CO (mg/m³)", 0.0, 50.0, float(co_default), 0.1)
    with col2:
        so2 = st.number_input("SO2 (μg/m³)", 0.0, 400.0, float(so2_default), 1.0)
    with col3:
        o3 = st.number_input("O3 (μg/m³)", 0.0, 300.0, float(o3_default), 1.0)
    
    # Predict button
    if st.button("Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating forecast..."):
            try:
                # Generate predictions for future dates
                forecast_data = []
                start_date = datetime.now()
                
                current_aqi = (pm25 + pm10 + no2 + co*10 + so2 + o3) / 6  # Simple estimate
                aqi_lag1 = current_aqi
                pm25_lag1 = pm25
                
                for day in range(forecast_days):
                    pred_date = start_date + timedelta(days=day+1)
                    
                    # Add some realistic variation
                    daily_variation = 1 + (np.random.randn() * 0.15)
                    
                    # Prepare pollutants dict
                    pollutants = {
                        'PM2.5': float(max(0, pm25 * daily_variation)),
                        'PM10': float(max(0, pm10 * daily_variation)),
                        'NO': float(max(0, 10 * daily_variation)),
                        'NO2': float(max(0, no2 * daily_variation)),
                        'NOx': float(max(0, (no2 * 1.2) * daily_variation)),
                        'NH3': float(max(0, 15 * daily_variation)),
                        'CO': float(max(0, co * daily_variation)),
                        'SO2': float(max(0, so2 * daily_variation)),
                        'O3': float(max(0, o3 * daily_variation)),
                        'Benzene': float(max(0, 1.0 * daily_variation)),
                        'Toluene': float(max(0, 5.0 * daily_variation)),
                        'Xylene': float(max(0, 2.0 * daily_variation)),
                        'AQI': float(aqi_lag1)
                    }
                    
                    # Use feature engineering utility to prepare features with one-hot encoding
                    features_df = prepare_single_prediction_features(
                        pollutants=pollutants,
                        city=selected_city,
                        date=pred_date,
                        use_onehot_city=True,  # Use one-hot encoding to match trained model
                        simple_features=True   # Use simple features matching trained model
                    )
                    
                    # Select only required feature columns
                    feature_cols = predictor.feature_columns
                    
                    # Ensure all required columns exist
                    for col in feature_cols:
                        if col not in features_df.columns:
                            features_df[col] = 0
                    
                    features_for_pred = features_df[feature_cols]
                    
                    # Make prediction (model trained WITHOUT scaling)
                    prediction = predictor.predict(features_for_pred)
                    
                    forecast_data.append({
                        'Date': pred_date.strftime('%Y-%m-%d'),
                        'Day': pred_date.strftime('%A'),
                        'AQI': prediction['predicted_aqi'],
                        'Category': prediction['predicted_bucket_name'],
                        'PM2.5': pollutants['PM2.5'],
                        'PM10': pollutants['PM10']
                    })
                    
                    # Update lag features for next prediction
                    aqi_lag1 = prediction['predicted_aqi']
                    pm25_lag1 = pollutants['PM2.5']
                
                forecast_df = pd.DataFrame(forecast_data)
                
                # Display results
                st.success(f"Generated {forecast_days}-day forecast for {selected_city}")
                
                # Forecast chart
                st.markdown("### AQI Forecast")
                
                fig = go.Figure()
                
                # Add AQI trace
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['AQI'],
                    mode='lines+markers',
                    name='Predicted AQI',
                    line=dict(color='royalblue', width=3),
                    marker=dict(size=10)
                ))
                
                # Add category zones
                fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Good")
                fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Satisfactory")
                fig.add_hrect(y0=100, y1=200, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Moderate")
                fig.add_hrect(y0=200, y1=300, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Poor")
                fig.add_hrect(y0=300, y1=400, fillcolor="purple", opacity=0.1, line_width=0, annotation_text="Very Poor")
                
                fig.update_layout(
                    title=f"AQI Forecast for {selected_city}",
                    xaxis_title="Date",
                    yaxis_title="AQI",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.markdown("### Detailed Forecast")
                
                # Color code by category
                def get_category_color(category):
                    colors = {
                        'Good': '#00e400',
                        'Satisfactory': '#ffff00',
                        'Moderate': '#ff7e00',
                        'Poor': '#ff0000',
                        'Very Poor': '#8f3f97',
                        'Severe': '#7e0023'
                    }
                    return colors.get(category, '#FFFFFF')
                
                # Display styled table
                for idx, row in forecast_df.iterrows():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                    
                    with col1:
                        st.markdown(f"**{row['Date']}** ({row['Day']})")
                    
                    with col2:
                        st.metric("AQI", f"{row['AQI']:.0f}")
                    
                    with col3:
                        color = get_category_color(row['Category'])
                        st.markdown(
                            f"<div style='background-color:{color};padding:8px;border-radius:5px;text-align:center;color:black;font-weight:bold'>{row['Category']}</div>",
                            unsafe_allow_html=True
                        )
                    
                    with col4:
                        st.text(f"PM2.5: {row['PM2.5']:.1f} | PM10: {row['PM10']:.1f}")
                
                # Download option
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Download Forecast (CSV)",
                    data=csv,
                    file_name=f"aqi_forecast_{selected_city}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    render_future_prediction_page()
