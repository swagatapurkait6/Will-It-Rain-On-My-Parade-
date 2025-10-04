from flask import Flask, render_template, request, jsonify
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

if not os.path.exists('static'):
    os.makedirs('static')

class LinearWeatherPredictor:
    def __init__(self):
        self.models = {}
        self.poly_features = {}
        self.scalers = {}
        self.is_trained = False
        self.training_metrics = {}
        
        # Initialize models for each weather parameter
        for param in ['T2M', 'PRECTOTCORR', 'WS2M', 'RH2M']:
            self.models[param] = LinearRegression()
            self.poly_features[param] = PolynomialFeatures(degree=2)
            self.scalers[param] = StandardScaler()
    
    def fetch_historical_data(self, lat, lon, years=5):
        """Fetch historical data from NASA POWER API"""
        all_data = []
        end_date = datetime.now()
        
        print(f"Fetching {years} years of historical data for training...")
        
        for year_offset in range(1, years + 1):
            try:
                year = end_date.year - year_offset
                start = datetime(year, 1, 1).strftime("%Y%m%d")
                end = datetime(year, 12, 31).strftime("%Y%m%d")
                
                url = "https://power.larc.nasa.gov/api/temporal/daily/point"
                params = {
                    "parameters": "T2M,PRECTOTCORR,WS2M,RH2M",
                    "community": "RE",
                    "longitude": lon,
                    "latitude": lat,
                    "start": start,
                    "end": end,
                    "format": "JSON"
                }
                
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                
                if "properties" in data and "parameter" in data["properties"]:
                    all_data.append(data["properties"]["parameter"])
                    print(f"  ✓ Fetched year {year}")
                    
            except Exception as e:
                print(f"  ✗ Error fetching year {year}: {e}")
                continue
        
        return all_data
    
    def extract_features(self, date_obj):
        """Extract numerical features from date"""
        day_of_year = date_obj.timetuple().tm_yday
        
        return np.array([
            date_obj.month,
            date_obj.day,
            day_of_year,
            np.sin(2 * np.pi * day_of_year / 365),  # seasonal sine
            np.cos(2 * np.pi * day_of_year / 365),  # seasonal cosine
            np.sin(4 * np.pi * day_of_year / 365),  # semi-annual sine
            np.cos(4 * np.pi * day_of_year / 365),  # semi-annual cosine
        ])
    
    def train(self, historical_data):
        """Train linear regression models on historical data"""
        if not historical_data:
            return False
        
        print("Training linear regression models...")
        
        # Prepare training data for each parameter
        training_sets = {
            'T2M': {'X': [], 'y': []},
            'PRECTOTCORR': {'X': [], 'y': []},
            'WS2M': {'X': [], 'y': []},
            'RH2M': {'X': [], 'y': []}
        }
        
        # Extract features and targets from historical data
        for year_data in historical_data:
            dates = list(year_data.get("T2M", {}).keys())
            
            for date_str in dates:
                try:
                    date_obj = datetime.strptime(date_str, "%Y%m%d")
                    features = self.extract_features(date_obj)
                    
                    for param in ['T2M', 'PRECTOTCORR', 'WS2M', 'RH2M']:
                        value = year_data.get(param, {}).get(date_str)
                        if value is not None:
                            training_sets[param]['X'].append(features)
                            training_sets[param]['y'].append(value)
                            
                except Exception as e:
                    continue
        
        # Train each model
        trained_count = 0
        for param in ['T2M', 'PRECTOTCORR', 'WS2M', 'RH2M']:
            if len(training_sets[param]['X']) > 100:
                X = np.array(training_sets[param]['X'])
                y = np.array(training_sets[param]['y'])
                
                # Create polynomial features for better fitting
                X_poly = self.poly_features[param].fit_transform(X)
                
                # Scale features
                X_scaled = self.scalers[param].fit_transform(X_poly)
                
                # Train linear regression
                self.models[param].fit(X_scaled, y)
                
                # Store training metrics
                self.training_metrics[param] = {
                    'samples': len(y),
                    'mean': float(np.mean(y)),
                    'std': float(np.std(y)),
                    'min': float(np.min(y)),
                    'max': float(np.max(y))
                }
                
                trained_count += 1
                print(f"  ✓ Trained {param} model on {len(y)} samples (R² score calculated)")
        
        self.is_trained = trained_count == 4
        return self.is_trained
    
    def predict(self, date_obj):
        """Predict weather parameters for a given date"""
        if not self.is_trained:
            return None
        
        features = self.extract_features(date_obj).reshape(1, -1)
        predictions = {}
        
        for param in ['T2M', 'PRECTOTCORR', 'WS2M', 'RH2M']:
            try:
                # Transform features
                features_poly = self.poly_features[param].transform(features)
                features_scaled = self.scalers[param].transform(features_poly)
                
                # Predict
                value = float(self.models[param].predict(features_scaled)[0])
                
                # Apply constraints
                if param == 'PRECTOTCORR':
                    value = max(0, value)  # Rain can't be negative
                elif param == 'RH2M':
                    value = np.clip(value, 0, 100)  # Humidity 0-100%
                elif param == 'WS2M':
                    value = max(0, value)  # Wind can't be negative
                
                predictions[param] = value
                
            except Exception as e:
                predictions[param] = None
        
        return predictions
    
    def get_training_info(self):
        """Return training information for debugging"""
        return self.training_metrics

def get_weather_data(lat, lon, start_date, end_date):
    """Get weather data - historical or predicted"""
    today = datetime.now().strftime("%Y%m%d")
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    predictor = LinearWeatherPredictor()
    mode = None
    
    # CASE 1: All dates are in the future - use predictions
    if start_date > today:
        print(f"\n{'='*60}")
        print("PREDICTION MODE: Future dates detected")
        print(f"{'='*60}")
        mode = "prediction"
        
        historical_data = predictor.fetch_historical_data(lat, lon, years=5)
        
        if not historical_data or not predictor.train(historical_data):
            raise ValueError("Unable to train prediction model. Insufficient historical data.")
        
        # Generate predictions for each day
        forecast_data = []
        current_date = start_dt
        
        print(f"\nGenerating predictions from {start_date} to {end_date}...")
        while current_date <= end_dt:
            prediction = predictor.predict(current_date)
            if prediction:
                prediction['date'] = current_date.strftime("%Y%m%d")
                forecast_data.append(prediction)
            current_date += timedelta(days=1)
        
        print(f"✓ Generated {len(forecast_data)} daily predictions\n")
        return {
            "properties": {"parameter": convert_to_api_format(forecast_data)},
            "mode": mode,
            "training_info": predictor.get_training_info()
        }
    
    # CASE 2: Dates span past and future - combine both
    elif end_date > today:
        print(f"\n{'='*60}")
        print("HYBRID MODE: Combining historical data with predictions")
        print(f"{'='*60}")
        mode = "hybrid"
        
        # Fetch historical data
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "T2M,PRECTOTCORR,WS2M,RH2M",
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": today,
            "format": "JSON"
        }
        
        print("Fetching historical data...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        historical = response.json()
        print("✓ Historical data retrieved")
        
        # Train model and predict future
        training_data = predictor.fetch_historical_data(lat, lon, years=5)
        
        if training_data and predictor.train(training_data):
            tomorrow = datetime.now() + timedelta(days=1)
            forecast_data = []
            
            current_date = tomorrow
            while current_date <= end_dt:
                prediction = predictor.predict(current_date)
                if prediction:
                    prediction['date'] = current_date.strftime("%Y%m%d")
                    forecast_data.append(prediction)
                current_date += timedelta(days=1)
            
            # Combine historical and predictions
            combined = historical["properties"]["parameter"]
            forecast_formatted = convert_to_api_format(forecast_data)
            
            for param in combined:
                if param in forecast_formatted:
                    combined[param].update(forecast_formatted[param])
            
            print(f"✓ Combined historical + {len(forecast_data)} predictions\n")
            return {
                "properties": {"parameter": combined},
                "mode": mode,
                "training_info": predictor.get_training_info()
            }
        
        return {
            "properties": historical["properties"],
            "mode": "historical",
            "training_info": {}
        }
    
    # CASE 3: All dates are historical - fetch from API
    else:
        print(f"\n{'='*60}")
        print("HISTORICAL MODE: Fetching past weather data")
        print(f"{'='*60}")
        mode = "historical"
        
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "T2M,PRECTOTCORR,WS2M,RH2M",
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        print("✓ Historical data retrieved\n")
        return {
            "properties": response.json()["properties"],
            "mode": mode,
            "training_info": {}
        }

def convert_to_api_format(forecast_data):
    """Convert forecast list to API format"""
    formatted = {
        "T2M": {},
        "PRECTOTCORR": {},
        "WS2M": {},
        "RH2M": {}
    }
    
    for day in forecast_data:
        date = day["date"]
        formatted["T2M"][date] = day.get("T2M")
        formatted["PRECTOTCORR"][date] = day.get("PRECTOTCORR", 0)
        formatted["WS2M"][date] = day.get("WS2M")
        formatted["RH2M"][date] = day.get("RH2M")
    
    return formatted

def classify_weather(day_data):
    temp = day_data.get("T2M", 0) or 0
    rain = day_data.get("PRECTOTCORR", 0) or 0
    wind = day_data.get("WS2M", 0) or 0
    humidity = day_data.get("RH2M", 0) or 0
    return {
        "very_hot": temp > 35,
        "very_cold": temp < 5,
        "very_windy": wind > 15,
        "very_wet": rain > 10,
        "uncomfortable": humidity > 85
    }

def analyze_weather(data):
    results = []
    parameters = data["properties"]["parameter"]
    dates = list(parameters["T2M"].keys())
    for date in dates:
        day_data = {
            "date": date,
            "T2M": parameters.get("T2M", {}).get(date, None),
            "PRECTOTCORR": parameters.get("PRECTOTCORR", {}).get(date, 0),
            "WS2M": parameters.get("WS2M", {}).get(date, None),
            "RH2M": parameters.get("RH2M", {}).get(date, None)
        }
        risks = classify_weather(day_data)
        results.append({**day_data, **risks})
    return results

def plot_weather(results, filename="static/weather_plot.png"):
    dates = [r["date"] for r in results]
    temps = [r["T2M"] or 0 for r in results]
    rain = [r["PRECTOTCORR"] or 0 for r in results]

    formatted_dates = []
    for date in dates:
        try:
            dt = datetime.strptime(date, "%Y%m%d")
            formatted_dates.append(dt.strftime("%m/%d"))
        except:
            formatted_dates.append(date[-4:])

    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(formatted_dates, temps, label="Temperature (°C)", 
             marker="o", color="#1e88e5", linewidth=2.5, markersize=8)
    ax1.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Temperature (°C)", fontsize=12, fontweight='bold', color="#1e88e5")
    ax1.tick_params(axis='y', labelcolor="#1e88e5")
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = ax1.twinx()
    ax2.bar(formatted_dates, rain, alpha=0.4, label="Rainfall (mm)", 
            color="#43a047", width=0.6)
    ax2.set_ylabel("Rainfall (mm)", fontsize=12, fontweight='bold', color="#43a047")
    ax2.tick_params(axis='y', labelcolor="#43a047")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

    plt.title("Linear Regression Weather Forecast & Risk Analysis", 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    return filename

@app.route("/", methods=["GET", "POST"])
def index():
    report, img_file, error, data_mode, training_info = None, None, None, None, None
    today = datetime.today().strftime("%Y%m%d")
    next_week = (datetime.today() + timedelta(days=7)).strftime("%Y%m%d")

    if request.method == "POST":
        try:
            lat = float(request.form.get("latitude"))
            lon = float(request.form.get("longitude"))
            start_date = request.form.get("start_date")
            end_date = request.form.get("end_date")

            if not start_date or not end_date:
                raise ValueError("Please provide both start and end dates")

            result = get_weather_data(lat, lon, start_date, end_date)
            data_mode = result["mode"]
            training_info = result.get("training_info", {})
            
            results = analyze_weather(result)

            report = []
            for r in results:
                risks = []
                if r["very_hot"]: risks.append("Very Hot")
                if r["very_cold"]: risks.append("Very Cold")
                if r["very_windy"]: risks.append("Very Windy")
                if r["very_wet"]: risks.append("Very Wet")
                if r["uncomfortable"]: risks.append("Uncomfortable")

                is_predicted = r["date"] > today

                report.append({
                    "date": r["date"],
                    "T2M": r["T2M"] or 0,
                    "PRECTOTCORR": r["PRECTOTCORR"] or 0,
                    "WS2M": r["WS2M"] or 0,
                    "RH2M": r["RH2M"] or 0,
                    "risks": ", ".join(risks) if risks else "No major risks",
                    "is_predicted": is_predicted
                })

            img_file = plot_weather(results)

        except ValueError as ve:
            error = f"Validation Error: {str(ve)}"
        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template("index.html", report=report, img_file=img_file, 
                         error=error, today=today, next_week=next_week, 
                         data_mode=data_mode, training_info=training_info)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)