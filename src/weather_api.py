"""
Weather API Integration Module for Real-Time AQI Data
Fetches live air quality and weather data from OpenWeatherMap API

Features:
- Real-time air quality data retrieval
- Historical air quality data (date range support)
- City coordinate lookup
- Standard AQI calculation from pollutant values
- Support for worldwide cities
- Batch data fetching for multiple cities
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class WeatherAPI:
    """Class to handle OpenWeatherMap API interactions"""
    
    def __init__(self, api_key=None):
        """
        Initialize Weather API client
        
        Args:
            api_key (str): OpenWeatherMap API key. If None, loads from .env file
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.geo_url = "http://api.openweathermap.org/geo/1.0"
        
        if not self.api_key or self.api_key == 'your_api_key_here':
            logger.warning("OpenWeatherMap API key not configured properly")
        
    def get_air_pollution(self, lat, lon):
        """
        Get current air pollution data for given coordinates
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            dict: Air pollution data or None if error
        """
        if not self.api_key or self.api_key == 'your_api_key_here':
            logger.error("Invalid API key")
            return None
            
        url = f"{self.base_url}/air_pollution"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching air pollution data: {e}")
            return None
    
    def get_city_coordinates(self, city_name, country_code=None):
        """
        Get coordinates for a city name
        
        Args:
            city_name (str): Name of the city
            country_code (str): Optional 2-letter country code (e.g., 'IN' for India)
            
        Returns:
            tuple: (lat, lon) or (None, None) if not found
        """
        if not self.api_key or self.api_key == 'your_api_key_here':
            logger.error("Invalid API key")
            return None, None
            
        url = f"{self.geo_url}/direct"
        
        # Build query with country code if provided
        query = f"{city_name},{country_code}" if country_code else city_name
        
        params = {
            'q': query,
            'limit': 1,
            'appid': self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                logger.info(f"Found coordinates for {city_name}: ({data[0]['lat']}, {data[0]['lon']})")
                return data[0]['lat'], data[0]['lon']
            
            logger.warning(f"City not found: {city_name}")
            return None, None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching coordinates: {e}")
            return None, None
    
    def get_live_aqi_data(self, city_name, country_code='IN'):
        """
        Get live AQI data for a city
        
        Args:
            city_name (str): Name of the city
            country_code (str): 2-letter country code (default: 'IN' for India)
            
        Returns:
            dict: Processed pollutant data ready for prediction, or None if error
        """
        # Get coordinates
        lat, lon = self.get_city_coordinates(city_name, country_code)
        
        if lat is None or lon is None:
            logger.error(f"Could not find coordinates for {city_name}")
            return None
        
        # Get air pollution data
        pollution_data = self.get_air_pollution(lat, lon)
        
        if not pollution_data or 'list' not in pollution_data:
            logger.error(f"No pollution data available for {city_name}")
            return None
        
        # Extract components
        components = pollution_data['list'][0]['components']
        
        # Calculate actual AQI from pollutant concentrations
        pm25 = components.get('pm2_5', 0)
        pm10 = components.get('pm10', 0)
        no2 = components.get('no2', 0)
        so2 = components.get('so2', 0)
        co = components.get('co', 0)
        o3 = components.get('o3', 0)
        
        # Calculate AQI
        actual_aqi = estimate_aqi_from_pollutants(pm25, pm10, no2, so2, co, o3)
        
        # Map API components to model features
        # OpenWeatherMap provides: co, no, no2, o3, so2, pm2_5, pm10, nh3
        processed_data = {
            'PM2.5': components.get('pm2_5', 0),
            'PM10': components.get('pm10', 0),
            'NO': components.get('no', 0),
            'NO2': components.get('no2', 0),
            'NOx': components.get('no', 0) + components.get('no2', 0),  # Approximate
            'NH3': components.get('nh3', 0),
            'CO': components.get('co', 0) / 1000,  # Convert from μg/m³ to mg/m³
            'SO2': components.get('so2', 0),
            'O3': components.get('o3', 0),
            'Benzene': 0,  # Not provided by API, use 0 as default
            'Toluene': 0,  # Not provided by API
            'Xylene': 0,   # Not provided by API
            'timestamp': pollution_data['list'][0]['dt'],
            'city': city_name,
            'actual_aqi': actual_aqi,
            'latitude': lat,
            'longitude': lon
        }
        
        logger.info(f"Successfully fetched live data for {city_name}: AQI={actual_aqi:.1f}")
        return processed_data


def estimate_aqi_from_pollutants(pm25, pm10, no2, so2, co, o3):
    """
    Calculate AQI using standard Indian AQI calculation formulas
    Uses the maximum AQI among all pollutants (sub-index method)
    
    Args:
        pm25 (float): PM2.5 concentration (μg/m³)
        pm10 (float): PM10 concentration (μg/m³)
        no2 (float): NO2 concentration (μg/m³)
        so2 (float): SO2 concentration (μg/m³)
        co (float): CO concentration (μg/m³)
        o3 (float): O3 concentration (μg/m³)
        
    Returns:
        float: Calculated AQI value (0-500)
    """
    
    def calculate_sub_index(concentration, breakpoints):
        """Calculate sub-index for a pollutant using linear interpolation"""
        for i in range(len(breakpoints) - 1):
            c_low, c_high, aqi_low, aqi_high = breakpoints[i]
            if c_low <= concentration <= c_high:
                # Linear interpolation formula
                sub_aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low
                return sub_aqi
        # If concentration exceeds highest breakpoint
        return breakpoints[-1][3]  # Return highest AQI
    
    # Indian AQI Breakpoints: (C_low, C_high, AQI_low, AQI_high)
    pm25_breakpoints = [
        (0, 30, 0, 50),
        (30, 60, 51, 100),
        (60, 90, 101, 200),
        (90, 120, 201, 300),
        (120, 250, 301, 400),
        (250, 500, 401, 500)
    ]
    
    pm10_breakpoints = [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 250, 101, 200),
        (250, 350, 201, 300),
        (350, 430, 301, 400),
        (430, 600, 401, 500)
    ]
    
    no2_breakpoints = [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 180, 101, 200),
        (180, 280, 201, 300),
        (280, 400, 301, 400),
        (400, 600, 401, 500)
    ]
    
    so2_breakpoints = [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 380, 101, 200),
        (380, 800, 201, 300),
        (800, 1600, 301, 400),
        (1600, 2400, 401, 500)
    ]
    
    co_breakpoints = [
        (0, 1, 0, 50),
        (1, 2, 51, 100),
        (2, 10, 101, 200),
        (10, 17, 201, 300),
        (17, 34, 301, 400),
        (34, 50, 401, 500)
    ]
    
    o3_breakpoints = [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 168, 101, 200),
        (168, 208, 201, 300),
        (208, 748, 301, 400),
        (748, 1000, 401, 500)
    ]
    
    # Calculate sub-indices for all pollutants
    sub_indices = []
    
    if pm25 > 0:
        sub_indices.append(calculate_sub_index(pm25, pm25_breakpoints))
    if pm10 > 0:
        sub_indices.append(calculate_sub_index(pm10, pm10_breakpoints))
    if no2 > 0:
        sub_indices.append(calculate_sub_index(no2, no2_breakpoints))
    if so2 > 0:
        sub_indices.append(calculate_sub_index(so2, so2_breakpoints))
    if co > 0:
        sub_indices.append(calculate_sub_index(co / 1000, co_breakpoints))  # Convert to mg/m³
    if o3 > 0:
        sub_indices.append(calculate_sub_index(o3, o3_breakpoints))
    
    # AQI is the maximum of all sub-indices
    if sub_indices:
        aqi = max(sub_indices)
        return round(min(aqi, 500), 2)  # Cap at 500 and round to 2 decimals
    else:
        return 0


def get_aqi_category(aqi_value):
    """
    Get AQI category and color based on AQI value
    
    Args:
        aqi_value (float): AQI value
        
    Returns:
        tuple: (category_name, category_code, color_code)
    """
    if aqi_value <= 50:
        return 'Good', 0, '#00e400'
    elif aqi_value <= 100:
        return 'Satisfactory', 1, '#ffff00'
    elif aqi_value <= 200:
        return 'Moderate', 2, '#ff7e00'
    elif aqi_value <= 300:
        return 'Poor', 3, '#ff0000'
    elif aqi_value <= 400:
        return 'Very Poor', 4, '#8f3f97'
    else:
        return 'Severe', 5, '#7e0023'


def fetch_historical_data(api_key, cities, start_date, end_date, country_code='IN'):
    """
    Fetch historical air quality data for multiple cities over a date range
    
    Note: OpenWeatherMap free tier only supports current + 5-day forecast.
    For historical data, this function simulates by fetching current data
    and creating synthetic historical data (for demonstration purposes).
    
    For real historical data, you would need:
    - OpenWeatherMap Professional/Enterprise subscription
    - Or alternative APIs like AQICN, WAQI, etc.
    
    Args:
        api_key (str): OpenWeatherMap API key
        cities (list): List of city names
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        country_code (str): Country code (default: 'IN')
        
    Returns:
        DataFrame: Historical data in City_Day.csv format
    """
    logger.info(f"Fetching data for {len(cities)} cities from {start_date} to {end_date}")
    
    api = WeatherAPI(api_key=api_key)
    all_data = []
    
    # Convert dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = pd.date_range(start, end, freq='D')
    
    for city in cities:
        logger.info(f"Fetching data for {city}...")
        
        # Get current data as baseline
        current_data = api.get_live_aqi_data(city, country_code)
        
        if not current_data:
            logger.warning(f"Skipping {city} - no data available")
            continue
        
        # Create daily records (simulated with variation)
        for date in date_range:
            # Add realistic daily variation to pollutant values
            variation = 1 + (np.random.randn() * 0.2)  # ±20% variation
            
            record = {
                'City': city,
                'Date': date.strftime('%Y-%m-%d'),
                'PM2.5': max(0, current_data['PM2.5'] * variation),
                'PM10': max(0, current_data['PM10'] * variation),
                'NO': max(0, current_data.get('NO', 0) * variation),
                'NO2': max(0, current_data['NO2'] * variation),
                'NOx': max(0, current_data.get('NOx', 0) * variation),
                'NH3': max(0, current_data.get('NH3', 0) * variation),
                'CO': max(0, current_data['CO'] * variation),
                'SO2': max(0, current_data['SO2'] * variation),
                'O3': max(0, current_data['O3'] * variation),
                'Benzene': max(0, current_data.get('Benzene', 0) * variation),
                'Toluene': max(0, current_data.get('Toluene', 0) * variation),
                'Xylene': max(0, current_data.get('Xylene', 0) * variation),
            }
            
            # Calculate AQI
            record['AQI'] = estimate_aqi_from_pollutants(
                record['PM2.5'], record['PM10'], record['NO2'],
                record['SO2'], record['CO'], record['O3']
            )
            
            # Get AQI bucket
            category, code, _ = get_aqi_category(record['AQI'])
            record['AQI_Bucket'] = category
            
            all_data.append(record)
        
        # Rate limiting - 60 calls/minute for free tier
        time.sleep(1.1)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    logger.info(f"Fetched {len(df)} records for {len(cities)} cities")
    
    return df


def fetch_data_with_progress(api_key, cities, start_date, end_date, 
                            country_code='IN', progress_callback=None):
    """
    Fetch historical data with progress updates for UI integration
    
    Args:
        api_key (str): API key
        cities (list): List of cities
        start_date (str): Start date 'YYYY-MM-DD'
        end_date (str): End date 'YYYY-MM-DD'
        country_code (str): Country code
        progress_callback (callable): Function to call with progress updates
        
    Returns:
        DataFrame: Historical data
    """
    api = WeatherAPI(api_key=api_key)
    all_data = []
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = pd.date_range(start, end, freq='D')
    
    total_operations = len(cities) * len(date_range)
    completed = 0
    
    for city_idx, city in enumerate(cities):
        if progress_callback:
            progress_callback(f"Fetching data for {city}...", 
                            completed / total_operations)
        
        current_data = api.get_live_aqi_data(city, country_code)
        
        if not current_data:
            logger.warning(f"Skipping {city}")
            completed += len(date_range)
            continue
        
        for date in date_range:
            variation = 1 + (np.random.randn() * 0.2)
            
            record = {
                'City': city,
                'Date': date.strftime('%Y-%m-%d'),
                'PM2.5': max(0, current_data['PM2.5'] * variation),
                'PM10': max(0, current_data['PM10'] * variation),
                'NO': max(0, current_data.get('NO', 0) * variation),
                'NO2': max(0, current_data['NO2'] * variation),
                'NOx': max(0, current_data.get('NOx', 0) * variation),
                'NH3': max(0, current_data.get('NH3', 0) * variation),
                'CO': max(0, current_data['CO'] * variation),
                'SO2': max(0, current_data['SO2'] * variation),
                'O3': max(0, current_data['O3'] * variation),
                'Benzene': max(0, current_data.get('Benzene', 0) * variation),
                'Toluene': max(0, current_data.get('Toluene', 0) * variation),
                'Xylene': max(0, current_data.get('Xylene', 0) * variation),
            }
            
            record['AQI'] = estimate_aqi_from_pollutants(
                record['PM2.5'], record['PM10'], record['NO2'],
                record['SO2'], record['CO'], record['O3']
            )
            
            category, _, _ = get_aqi_category(record['AQI'])
            record['AQI_Bucket'] = category
            
            all_data.append(record)
            completed += 1
        
        time.sleep(1.1)  # Rate limiting
    
    if progress_callback:
        progress_callback("Complete!", 1.0)
    
    df = pd.DataFrame(all_data)
    return df


# Example usage
if __name__ == "__main__":
    # Test the API
    api = WeatherAPI()
    
    # Test city
    test_city = "Delhi"
    print(f"\nFetching live AQI data for {test_city}...")
    
    data = api.get_live_aqi_data(test_city)
    
    if data:
        print(f"\n[SUCCESS] Data fetched!")
        print(f"City: {data['city']}")
        print(f"Actual AQI: {data['actual_aqi']:.2f}")
        print(f"PM2.5: {data['PM2.5']:.2f} μg/m³")
        print(f"PM10: {data['PM10']:.2f} μg/m³")
        print(f"NO2: {data['NO2']:.2f} μg/m³")
        print(f"CO: {data['CO']:.2f} mg/m³")
        print(f"O3: {data['O3']:.2f} μg/m³")
        
        category, code, color = get_aqi_category(data['actual_aqi'])
        print(f"\nAQI Category: {category} (Code: {code})")
    else:
        print("\n[ERROR] Failed to fetch data. Please check your API key.")
