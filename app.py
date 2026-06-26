from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import math
from datetime import datetime
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Constants
LOCATIONS = [
    "Coimbatore", "Kancheepuram", "Thiruvarur", "Kudankulam", "Neyveli",
    "Kamuthi", "Tiruchirappalli", "Salem", "Tuticorin", "Tiruchengode"
]

# Dictionary to store model objects once loaded
models = {}

# Dictionary to store feature scalers for each location
scalers = {}

# Feature statistics for normalization (will be loaded from file if available)
feature_stats = {}

# Load feature statistics from file if available
try:
    feature_stats_path = 'content/model_results/feature_statistics.json'
    if os.path.exists(feature_stats_path):
        with open(feature_stats_path, 'r') as f:
            feature_stats = json.load(f)
        print(f"Loaded feature statistics from {feature_stats_path}")
except Exception as e:
    print(f"Error loading feature statistics: {e}")
    # Initialize with some reasonable defaults for key features
    feature_stats = {
        'temperature': {'mean': 28.0, 'std': 4.0, 'min': 15.0, 'max': 42.0},
        'humidity': {'mean': 65.0, 'std': 15.0, 'min': 20.0, 'max': 100.0},
        'wind_speed': {'mean': 10.0, 'std': 5.0, 'min': 0.0, 'max': 30.0},
        'pressure': {'mean': 1010.0, 'std': 5.0, 'min': 995.0, 'max': 1025.0},
        'solar_radiation': {'mean': 600.0, 'std': 300.0, 'min': 0.0, 'max': 1200.0},
        'cloud_cover': {'mean': 40.0, 'std': 30.0, 'min': 0.0, 'max': 100.0}
    }

# Load model metrics
metrics_df = pd.read_csv('content/model_results/model_summary.csv')
metrics = {}
for _, row in metrics_df.iterrows():
    metrics[row['Location']] = {
        'MSE': row['MSE'],
        'RMSE': row['RMSE'],
        'R2': row['R²'],
        'MAPE': row['MAPE']
    }

# Calculate average metrics
avg_metrics = {
    'avg_r2': metrics_df['R²'].mean(),
    'avg_rmse': metrics_df['RMSE'].mean(),
    'avg_mse': metrics_df['MSE'].mean(),
    'avg_mape': metrics_df['MAPE'].mean()
}

# Feature importance for each model
feature_importance = {}
for location in LOCATIONS:
    try:
        feature_df = pd.read_csv(f'content/model_results/{location}_feature_importance.csv')
        feature_importance[location] = feature_df.to_dict('records')
    except Exception as e:
        print(f"Error loading feature importance for {location}: {e}")
        feature_importance[location] = []

# Function to prepare features based on the feature importance CSV for model prediction
def prepare_features(user_inputs, current_datetime=None, normalize=True):
    if current_datetime is None:
        current_datetime = datetime.now()
    
    # Extract basic inputs - only peak_power, mounting_type, and panel_material are required now
    # Set reasonable defaults for other parameters
    temperature = user_inputs.get('temperature', 25.0)
    humidity = user_inputs.get('humidity', 65.0)
    wind_speed = user_inputs.get('wind_speed', 10.0)
    pressure = user_inputs.get('pressure', 1013.0)
    solar_rad = user_inputs.get('solar_radiation', 800.0)
    time_of_day = user_inputs.get('time_of_day', 12)  # Default to noon
    cloud_cover = user_inputs.get('cloud_cover', 20.0)
    
    # Extract panel specifications - these are the only required inputs now
    peak_power = user_inputs.get('peak_power', 100.0)
    panel_material = user_inputs.get('panel_material', 'crystSi')
    mounting_type = user_inputs.get('mounting_type', 'fixed')
    
    # Print message indicating we're using the simplified model
    print(f"Using simplified prediction model with only 3 inputs: peak_power={peak_power}kW, mounting_type={mounting_type}, panel_material={panel_material}")
    print(f"Using defaults for environmental parameters: temp={temperature}°C, wind={wind_speed}km/h, time={time_of_day}h")
    
    # Validate and constrain input values to reasonable ranges
    def constrain_value(value, feature_name):
        if feature_name in feature_stats:
            stats = feature_stats[feature_name]
            # Use min-max values from statistics if available, otherwise use defaults
            min_val = stats.get('min', 0.0)
            max_val = stats.get('max', float('inf'))
            return max(min_val, min(max_val, value))
        return value
    
    # Apply constraints to numerical inputs
    temperature = constrain_value(temperature, 'temperature')
    humidity = constrain_value(humidity, 'humidity')
    wind_speed = constrain_value(wind_speed, 'wind_speed')
    pressure = constrain_value(pressure, 'pressure')
    solar_rad = constrain_value(solar_rad, 'solar_radiation')
    cloud_cover = constrain_value(cloud_cover, 'cloud_cover')
    
    # Calculate all possible features that might be used by the model
    features = {}
    
    # Basic features from user inputs
    features['T2m'] = temperature
    features['WS10m'] = wind_speed
    features['T2m_WS10m'] = temperature * wind_speed
    
    # Time-based features
    hour = int(time_of_day)
    month = current_datetime.month
    features['hour'] = hour
    features['hour_sin'] = math.sin(hour * 2 * math.pi / 24)
    features['hour_cos'] = math.cos(hour * 2 * math.pi / 24)
    features['month'] = month
    features['month_sin'] = math.sin(month * 2 * math.pi / 12)
    features['month_cos'] = math.cos(month * 2 * math.pi / 12)
    
    # Date features from current date
    day = current_datetime.day
    year = current_datetime.year
    day_of_year = current_datetime.timetuple().tm_yday
    day_of_week = current_datetime.weekday()
    
    features['day'] = day
    features['month'] = month
    features['year'] = year
    features['day_of_year'] = day_of_year
    features['day_of_week'] = day_of_week
    
    features['day_sin'] = math.sin(day_of_year * 2 * math.pi / 365)
    features['day_cos'] = math.cos(day_of_year * 2 * math.pi / 365)
    
    # Solar position features (approximate)
    # In a real application, you'd use a more accurate solar position calculation
    solar_zenith = 90 - (90 * (1 - abs(hour - 12) / 12))
    solar_azimuth = 180 * (hour / 24)
    
    features['solar_zenith'] = solar_zenith
    features['solar_azimuth'] = solar_azimuth
    features['is_daytime'] = 1 if (hour >= 6 and hour <= 18) else 0
    
    # Mounting type (one-hot encoded)
    features['mounting_type_fixed'] = 1 if mounting_type == 'fixed' else 0
    features['mounting_type_inclined_axis'] = 1 if mounting_type == 'inclined_axis' else 0
    features['mounting_type_two_axis'] = 1 if mounting_type == 'two_axis' else 0
    features['mounting_type_single_axis_vertical'] = 1 if mounting_type == 'single_axis_vertical' else 0
    
    # Technology type (one-hot encoded)
    features['technology_crystSi'] = 1 if panel_material == 'crystSi' else 0
    features['technology_CIS'] = 1 if panel_material == 'CIS' else 0
    features['technology_CdTe'] = 1 if panel_material == 'CdTe' else 0
    
    # Peak power
    features['peak_power'] = float(peak_power)
    
    # Add derived features for enhanced prediction
    
    # 1. Technology efficiency factors (W/m²)
    tech_efficiency = {
        'crystSi': 200,  # Crystalline Silicon
        'CIS': 160,      # Copper Indium Selenide
        'CdTe': 175      # Cadmium Telluride
    }
    features['tech_efficiency'] = tech_efficiency.get(panel_material, 150)
    
    # 2. Mounting efficiency multipliers
    mounting_efficiency = {
        'fixed': 1.0,           # baseline
        'inclined_axis': 1.1,   # 10% improvement
        'single_axis_vertical': 1.15,  # 15% improvement
        'two_axis': 1.3         # 30% improvement with dual-axis tracking
    }
    features['mounting_efficiency'] = mounting_efficiency.get(mounting_type, 1.0)
    
    # 3. Calculate theoretical maximum power
    features['theoretical_max_power'] = features['peak_power'] * features['tech_efficiency'] * features['mounting_efficiency']
    
    # 4. Temperature derating factor (efficiency decreases with higher temperature)
    # Different panel technologies have different temperature coefficients
    temp_coefficients = {
        'crystSi': -0.0045,  # -0.45% per °C
        'CdTe': -0.0025,     # -0.25% per °C
        'CIS': -0.0030,      # -0.30% per °C
    }
    temp_coef = temp_coefficients.get(panel_material, -0.0040)  # Default -0.40% per °C
    ref_temp = 25.0     # Reference temperature
    features['temp_coefficient'] = temp_coef
    features['temp_derating'] = 1.0 + temp_coef * (temperature - ref_temp)
    features['temp_derating'] = max(0.7, min(1.1, features['temp_derating']))  # Clip to reasonable values
    
    # 5. Apply temperature derating to theoretical max power
    features['theoretical_max_power'] *= features['temp_derating']
    
    # 6. Add seasonal adjustment factor
    month_factors = {
        1: 0.65, 2: 0.75, 3: 0.85, 4: 0.95, 5: 1.05, 6: 1.10,  # Jan-Jun
        7: 1.10, 8: 1.05, 9: 0.95, 10: 0.85, 11: 0.75, 12: 0.65  # Jul-Dec
    }
    features['seasonal_factor'] = month_factors.get(month, 1.0)
    features['seasonal_adjusted_power'] = features['theoretical_max_power'] * features['seasonal_factor']
    
    # 7. Wind cooling effect on panels
    features['wind_cooling'] = 1.0 + 0.005 * min(20, wind_speed)
    features['seasonal_adjusted_power'] *= features['wind_cooling']
    
    # 8. Cloud cover impact (direct relationship)
    features['cloud_impact'] = 1.0 - (cloud_cover / 100.0) * 0.8  # 0% clouds = 1.0, 100% clouds = 0.2
    features['seasonal_adjusted_power'] *= features['cloud_impact']

    # 9. Location specific adjustments for user-selected location
    if 'location' in user_inputs:
        location = user_inputs['location']
        # Define location-specific metadata
        location_metadata = {
            'Coimbatore': {
                'elevation': 411,        # meters above sea level
                'aerosol_factor': 0.92,  # Air clarity factor (1.0 = clear)
                'terrain_factor': 0.94   # Combined terrain impact
            },
            'Kancheepuram': {
                'elevation': 85,
                'aerosol_factor': 0.88,
                'terrain_factor': 0.96
            },
            'Thiruvarur': {
                'elevation': 8,
                'aerosol_factor': 0.94,
                'terrain_factor': 0.98
            },
            'Kudankulam': {
                'elevation': 15,
                'aerosol_factor': 0.96,
                'terrain_factor': 0.99
            },
            'Neyveli': {
                'elevation': 53,
                'aerosol_factor': 0.85,
                'terrain_factor': 0.89
            },
            'Kamuthi': {
                'elevation': 29,
                'aerosol_factor': 0.95,
                'terrain_factor': 0.96
            },
            'Tiruchirappalli': {
                'elevation': 88,
                'aerosol_factor': 0.91,
                'terrain_factor': 0.94
            },
            'Salem': {
                'elevation': 278,
                'aerosol_factor': 0.9,
                'terrain_factor': 0.95
            },
            'Tuticorin': {
                'elevation': 4,
                'aerosol_factor': 0.94,
                'terrain_factor': 0.97
            },
            'Tiruchengode': {
                'elevation': 226,
                'aerosol_factor': 0.92,
                'terrain_factor': 0.95
            }
        }
        
        # Get location metadata or use defaults
        metadata = location_metadata.get(location, {
            'elevation': 100, 
            'aerosol_factor': 0.9, 
            'terrain_factor': 0.95
        })
        
        # Apply location-specific adjustments
        features['elevation'] = metadata['elevation']
        features['aerosol_factor'] = metadata['aerosol_factor']
        features['terrain_factor'] = metadata['terrain_factor']
        
        # Elevation adjustment (0.5% increase per 100m)
        features['elevation_adjustment'] = 1.0 + (metadata['elevation'] / 100) * 0.005
        
        # Combined location adjustment
        features['location_adjustment'] = metadata['aerosol_factor'] * metadata['terrain_factor'] * features['elevation_adjustment']
        
        # Apply to theoretical max power
        features['seasonal_adjusted_power'] *= features['location_adjustment']
    
    # Create DataFrame from features dictionary
    df = pd.DataFrame([features])
    
    # Apply normalization if required
    if normalize:
        # Identify numerical features
        numerical_features = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
                # Skip location-specific string columns
                if col not in ['location_name', 'location_cat']:
                    numerical_features.append(col)
        
        # Apply normalization using feature statistics
        for feature in numerical_features:
            if feature in feature_stats:
                stats = feature_stats[feature]
                # Z-score normalization
                mean = stats.get('mean', df[feature].mean())
                std = stats.get('std', df[feature].std() or 1.0)  # Avoid division by zero
                df[feature] = (df[feature] - mean) / std
    
    return df

# Function to load model on demand
def load_model(location):
    if location in models:
        return models[location]
    
    # Try loading joblib model first, then xgb, then pkl
    for ext in ['joblib', 'xgb', 'pkl']:
        model_path = f'content/trained_models/{location}_model.{ext}'
        if os.path.exists(model_path):
            try:
                if ext == 'joblib':
                    model = joblib.load(model_path)
                else:  # 'pkl' or 'xgb'
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                models[location] = model
                print(f"Loaded {location} model from {model_path}")
                
                # Try to load scaler if it exists
                scaler_path = f'content/trained_models/{location}_scaler.joblib'
                if os.path.exists(scaler_path):
                    try:
                        scaler = joblib.load(scaler_path)
                        scalers[location] = scaler
                        print(f"Loaded scaler for {location}")
                    except Exception as e:
                        print(f"Error loading scaler for {location}: {e}")
                
                return model
            except Exception as e:
                print(f"Error loading {location} model from {model_path}: {e}")
    
    return None

@app.route('/')
def home():
    return render_template('home.html', locations=LOCATIONS, metrics=metrics)

@app.route('/predict-form')
def prediction_form():
    # Get location from query parameter if specified
    location = request.args.get('location', '')
    return render_template('index.html', locations=LOCATIONS, selected_location=location, metrics=metrics)

@app.route('/dashboard')
def dashboard():
    # Get selected location from query parameter if specified
    selected_location = request.args.get('selected_location', '')
    
    # If selected location is not valid, set to empty (to show overview dashboard)
    if selected_location and selected_location not in LOCATIONS:
        selected_location = ''
    
    return render_template(
        'dashboard.html', 
        locations=LOCATIONS, 
        selected_location=selected_location, 
        metrics=metrics,
        avg_metrics=avg_metrics
    )

@app.route('/cost-analysis')
def cost_analysis():
    return render_template('cost_analysis.html', locations=LOCATIONS)

@app.route('/land-analysis')
def land_analysis():
    return render_template('land_analysis.html', locations=LOCATIONS)

# Store the last prediction to allow retrieval across pages
last_prediction = None

@app.route('/predict', methods=['POST'])
def predict():
    global last_prediction  # Access the global variable
    data = request.form
    location = data.get('location')
    
    if location not in LOCATIONS:
        return jsonify({"error": "Invalid location selected"}), 400
    
    # Extract input features and prepare them for prediction
    try:
        # Get current datetime for time features
        current_datetime = datetime.now()
        
        # Get only required user inputs (simplified model)
        input_data = {
            # Required inputs
            'peak_power': float(data.get('peak_power', 100.0)),
            'panel_material': data.get('panel_material', 'crystSi'),
            'mounting_type': data.get('mounting_type', 'fixed'),
            # Add location for location-specific adjustments
            'location': location
        }
        
        # Optional environmental parameters - will use defaults if not provided
        if 'temperature' in data and data['temperature']:
            input_data['temperature'] = float(data.get('temperature'))
        if 'humidity' in data and data['humidity']:
            input_data['humidity'] = float(data.get('humidity'))
        if 'wind_speed' in data and data['wind_speed']:
            input_data['wind_speed'] = float(data.get('wind_speed'))
        if 'pressure' in data and data['pressure']:
            input_data['pressure'] = float(data.get('pressure'))
        if 'solar_radiation' in data and data['solar_radiation']:
            input_data['solar_radiation'] = float(data.get('solar_radiation'))
        if 'time_of_day' in data and data['time_of_day']:
            input_data['time_of_day'] = float(data.get('time_of_day'))
        if 'cloud_cover' in data and data['cloud_cover']:
            input_data['cloud_cover'] = float(data.get('cloud_cover'))
        
        # Prepare features for the model with normalization
        input_df = prepare_features(input_data, current_datetime, normalize=True)
        
        # Extract time information for enhanced prediction and energy calculations
        hour = int(input_data.get('time_of_day', 12))  # Default to noon if not provided
        month = current_datetime.month
        
        # Load model for the selected location
        model = load_model(location)
        
        if model is None:
            return jsonify({"error": f"Failed to load model for {location}"}), 500
        
        try:
            # Make prediction with uncertainty estimate
            prediction_with_uncertainty = predict_with_uncertainty(model, input_df, location)
            prediction = prediction_with_uncertainty['prediction']
            uncertainty = {
                'std_dev': prediction_with_uncertainty['std_dev'],
                'lower_bound': prediction_with_uncertainty['lower_bound'],
                'upper_bound': prediction_with_uncertainty['upper_bound'],
                'confidence': prediction_with_uncertainty['confidence']
            }
            
            print(f"Prediction for {location}: {prediction} (±{uncertainty['std_dev']:.2f})")
        except Exception as model_error:
            # If prediction fails, print detailed error and available features
            print(f"Model prediction error: {model_error}")
            print(f"Input features: {list(input_df.columns)}")
            
            # Use enhanced physics-based fallback prediction
            
            # 1. Start with default solar radiation and apply cloud cover reduction
            solar_rad = input_data.get('solar_radiation', 800.0)
            cloud_factor = 1.0 - (input_data.get('cloud_cover', 20.0) / 100.0) * 0.8  # Cloud cover impact (100% clouds still allows ~20% diffuse light)
            effective_irradiance = solar_rad * cloud_factor
            
            # 2. Apply panel technology efficiency
            tech_efficiency = {
                'crystSi': 0.2,  # 20% efficiency for crystalline silicon
                'CIS': 0.16,     # 16% efficiency for CIS
                'CdTe': 0.175    # 17.5% efficiency for CdTe
            }
            panel_efficiency = tech_efficiency.get(input_data['panel_material'], 0.17)  # Default 17%
            
            # 3. Apply temperature derating
            temp = input_data.get('temperature', 25.0)
            temp_coef = -0.004  # Typical temperature coefficient (-0.4% per °C above 25°C)
            temp_factor = 1.0 + temp_coef * (temp - 25.0)
            temp_factor = max(0.7, min(1.1, temp_factor))  # Limit to reasonable range
            
            # 4. Apply mounting type advantage
            mounting_advantage = {
                'fixed': 1.0,           # baseline
                'inclined_axis': 1.1,   # 10% improvement
                'single_axis_vertical': 1.15,  # 15% improvement
                'two_axis': 1.3         # 30% improvement
            }
            tracking_factor = mounting_advantage.get(input_data['mounting_type'], 1.0)
            
            # 5. Calculate power per m² and scale by panel size
            power_per_m2 = effective_irradiance * panel_efficiency * temp_factor * tracking_factor
            
            # 6. Convert peak_power (kW) to panel area (m²) using average power density
            # Typical solar panels generate ~150-200 W/m²
            avg_power_density = {
                'crystSi': 200,  # W/m²
                'CIS': 160,      # W/m²
                'CdTe': 175      # W/m²
            }
            power_density = avg_power_density.get(input_data['panel_material'], 180)
            
            # Calculate panel area: peak_power (W) / power_density (W/m²)
            panel_area = (input_data['peak_power'] * 1000) / power_density  # m²
            
            # 7. Calculate total power output
            prediction = power_per_m2 * panel_area
            
            # 8. Apply time-of-day solar curve adjustment
            if 6 <= hour <= 18:
                time_factor = 1.0 - abs(hour - 12) / 8.0  # 0.5 at 8am/4pm, 0 at 6am/6pm
                time_factor = max(0.05, time_factor)  # Ensure at least 5% even at dawn/dusk
                prediction *= time_factor
            else:
                # At night, power is minimal (only if some systems have lights/etc)
                prediction = 0.001 * panel_area  # Negligible power
            
            # 9. Apply seasonal adjustment
            month_factors = {
                1: 0.65, 2: 0.75, 3: 0.85, 4: 0.95, 5: 1.05, 6: 1.1,  # Jan-Jun
                7: 1.1, 8: 1.05, 9: 0.95, 10: 0.85, 11: 0.75, 12: 0.65  # Jul-Dec
            }
            seasonal_factor = month_factors.get(month, 1.0)
            prediction *= seasonal_factor
            
            # Create a default uncertainty range for the fallback prediction
            uncertainty = {
                'std_dev': prediction * 0.12, # Assume 12% standard deviation
                'lower_bound': prediction * 0.75, # 25% lower bound
                'upper_bound': prediction * 1.25, # 25% upper bound
                'confidence': 0.90 # 90% confidence interval for fallback
            }
            
            print(f"Using enhanced physics-based fallback prediction: {prediction}")
        
        # Calculate derived metrics (PV system performance indicators)
        
        # 1. Get theoretical maximum power from processed features
        theoretical_max = input_df['theoretical_max_power'].values[0]
        
        # 2. Calculate capacity factor (ratio of actual to theoretical max)
        capacity_factor = prediction / theoretical_max if theoretical_max > 0 else 0
        capacity_factor = min(1.0, max(0, capacity_factor))  # Clip to valid range
        
        # 3. Calculate energy production estimates
        
        # Get location-specific solar resource data
        location_peak_hours = {
            'Coimbatore': 5.2,
            'Kancheepuram': 5.0,
            'Thiruvarur': 5.1, 
            'Kudankulam': 5.5,
            'Neyveli': 5.0,
            'Kamuthi': 5.6,
            'Tiruchirappalli': 5.2,
            'Salem': 5.1,
            'Tuticorin': 5.4,
            'Tiruchengode': 5.2
        }
        
        # Use location-specific value or default
        equivalent_peak_hours = location_peak_hours.get(location, 5.0)
        
        # Daily energy production (kWh) adjusted for time of day
        if 6 <= hour <= 18:
            # For daytime predictions, scale using solar curve
            solar_curve_weight = 1.0 - abs(hour - 12) / 8.0  # Peak at noon
            solar_curve_weight = max(0.05, solar_curve_weight)  # Minimum 5% at dawn/dusk
            
            # Scale to full-day equivalent (kWh)
            daily_energy = (prediction / solar_curve_weight) * equivalent_peak_hours / 1000
        else:
            # For nighttime, estimate based on theoretical maximum
            daily_energy = theoretical_max * equivalent_peak_hours / 1000
        
        # Annual energy based on daily, with seasonal adjustment
        month_factors = {
            1: 0.65, 2: 0.75, 3: 0.85, 4: 0.95, 5: 1.05, 6: 1.1,  # Jan-Jun
            7: 1.1, 8: 1.05, 9: 0.95, 10: 0.85, 11: 0.75, 12: 0.65  # Jul-Dec
        }
        monthly_factor = month_factors.get(month, 1.0)
        annual_energy = (daily_energy / monthly_factor) * sum(month_factors.values()) * (365 / 12)
        
        # 4. Calculate financial and environmental benefits
        
        # CO2 emission reduction (using average grid emission factor for India - 0.82 kg CO2/kWh)
        co2_offset_daily = daily_energy * 0.82  # kg CO2
        co2_offset_annual = annual_energy * 0.82  # kg CO2
        
        # Financial savings (using average tariff of 6 Rs/kWh)
        financial_savings_daily = daily_energy * 6  # Rupees
        financial_savings_annual = annual_energy * 6  # Rupees
        
        # Get metrics for this location
        location_metrics = metrics.get(location, {})
        
        # Store prediction in result
        result = {
            'location': location,
            'prediction': float(prediction),
            'power_kw': float(prediction) / 1000,
            'uncertainty': uncertainty,
            'input_values': input_data,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': location_metrics,
            'feature_importance': feature_importance.get(location, []),
            'theoretical_max_w': float(theoretical_max),
            'capacity_factor': float(capacity_factor),
            'daily_energy_kwh': float(daily_energy),
            'annual_energy_kwh': float(annual_energy),
            'co2_offset_daily_kg': float(co2_offset_daily),
            'co2_offset_annual_kg': float(co2_offset_annual),
            'financial_savings_daily_inr': float(financial_savings_daily),
            'financial_savings_annual_inr': float(financial_savings_annual)
        }
        
        # Store the last prediction for retrieval across pages
        last_prediction = result
        
        return redirect(url_for('results'))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/results')
def results():
    return render_template('results.html', locations=LOCATIONS)

# API endpoints for the dashboard
@app.route('/api/model_metrics')
def api_model_metrics():
    return jsonify(metrics)

@app.route('/api/feature_importance/<location>')
def api_feature_importance(location):
    if location not in LOCATIONS:
        return jsonify({"error": "Invalid location"}), 400
    
    return jsonify(feature_importance.get(location, []))

# Store the last prediction to allow retrieval across pages
last_prediction = None

@app.route('/api/last_prediction')
def api_last_prediction():
    """API endpoint to get the last prediction made"""
    global last_prediction
    if last_prediction is None:
        return jsonify({"error": "No prediction has been made yet"}), 404
    return jsonify(last_prediction)

# Calculate prediction with uncertainty estimate
def predict_with_uncertainty(model, input_df, location):
    """Make prediction with uncertainty estimate using bootstrapping technique"""
    try:
        # Base prediction
        base_prediction = model.predict(input_df)[0]
        
        # Create a series of perturbed input features for bootstrapping
        # This simulates uncertainty in the input parameters
        n_bootstrap = 30  # Increased from 20 for better sampling
        perturbations = []
        
        # Generate perturbed input data
        for i in range(n_bootstrap):
            perturbed_df = input_df.copy()
            
            # Perturb all numerical features with small random variations
            for col in perturbed_df.columns:
                if pd.api.types.is_numeric_dtype(perturbed_df[col]) and col != 'peak_power':
                    # Add small random variation (±5%)
                    perturb_factor = 1.0 + np.random.uniform(-0.05, 0.05)
                    perturbed_df[col] = perturbed_df[col] * perturb_factor
            
            # Make sure to perturb key derived features for the simplified model
            key_features = ['tech_efficiency', 'mounting_efficiency', 'theoretical_max_power', 
                           'seasonal_adjusted_power', 'temp_derating']
            
            for feature in key_features:
                if feature in perturbed_df.columns:
                    perturb_factor = 1.0 + np.random.uniform(-0.03, 0.03)  # ±3% variation
                    perturbed_df[feature] = perturbed_df[feature] * perturb_factor
            
            # Make prediction with perturbed features
            perturbed_prediction = model.predict(perturbed_df)[0]
            perturbations.append(perturbed_prediction)
        
        # Calculate prediction statistics
        predictions = np.array([base_prediction] + perturbations)
        mean_prediction = predictions.mean()
        std_dev = predictions.std()
        
        # Calculate 95% confidence interval
        lower_bound = mean_prediction - 1.96 * std_dev
        upper_bound = mean_prediction + 1.96 * std_dev
        
        # Apply location-specific adjustment to uncertainty based on model performance
        # Locations with better metrics have tighter confidence intervals
        if location in metrics:
            location_r2 = metrics[location].get('R2', 0.8)
            # Scale the uncertainty based on R² score (higher R² = lower uncertainty)
            uncertainty_factor = 1.5 * (1 - location_r2)  # e.g., R²=0.99 gives factor of 0.015
            uncertainty_factor = max(0.02, min(0.3, uncertainty_factor))  # Keep between 2% and 30%
            
            # Apply the location-specific adjustment
            std_dev = std_dev * (1 + uncertainty_factor)
            lower_bound = mean_prediction - 1.96 * std_dev
            upper_bound = mean_prediction + 1.96 * std_dev
        
        # Ensure we don't predict negative power
        lower_bound = max(0, lower_bound)
        
        # Apply theoretical max power as an upper bound (with some margin)
        if 'theoretical_max_power' in input_df.columns:
            theo_max = input_df['theoretical_max_power'].values[0]
            upper_bound = min(upper_bound, theo_max * 1.3)  # Allow up to 30% over theoretical max
        
        return {
            'prediction': float(mean_prediction),
            'std_dev': float(std_dev),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'confidence': 0.95  # 95% confidence interval
        }
    
    except Exception as e:
        print(f"Error in uncertainty estimation: {e}")
        import traceback
        traceback.print_exc()
        
        # Return base prediction with a default uncertainty range
        try:
            prediction = model.predict(input_df)[0]
            
            # If we can access theoretical max power, use it to estimate uncertainty
            if 'theoretical_max_power' in input_df.columns:
                theo_max = input_df['theoretical_max_power'].values[0]
                std_dev = prediction * 0.1  # 10% standard deviation as default
                lower_bound = max(0, prediction - 1.96 * std_dev)
                upper_bound = min(theo_max * 1.3, prediction + 1.96 * std_dev)
            else:
                # Simple percentage-based fallback
                std_dev = prediction * 0.12  # 12% standard deviation
                lower_bound = max(0, prediction * 0.8)  # 20% below prediction
                upper_bound = prediction * 1.2  # 20% above prediction
            
            return {
                'prediction': float(prediction),
                'std_dev': float(std_dev),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'confidence': 0.95
            }
        except Exception as pred_error:
            print(f"Error in base prediction: {pred_error}")
            # If even base prediction fails, return a fallback
            return {
                'prediction': 0.0,
                'std_dev': 0.0,
                'lower_bound': 0.0,
                'upper_bound': 0.0,
                'confidence': 0.0
            }

if __name__ == '__main__':
    app.run(debug=True) 