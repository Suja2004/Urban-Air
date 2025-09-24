# Urban-Air Intelligence

A Streamlit-based dashboard for monitoring air quality (PM2.5, CO₂, etc.), generating short-term forecasts, and visualizing alerts.  
Supports real-time data ingestion, forecasting, and interactive filtering of network alerts.

---

## 🚀 Features
- Real-time AQI monitoring (PM2.5, CO₂, etc.)
- Auto-refresh dashboard
- 3-hour AQI forecast with trend analysis
- Recent network alerts with filtering & details
- Interactive charts and visualizations

---

## 📦 Setup & Installation

### 1. Clone this repository
```
git clone https://github.com/suja2004/Urban-Air.git
cd Urban-Air-main
```
### 2. Create a virtual environment
```
# Windows
python -m venv .venv
.venv\Scripts\activate
```
```
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```
### 3. Install dependencies
```
pip install -r requirements.txt
```

### Run the App
```
streamlit run app.py
```

The app will launch in your browser at:
```
http://localhost:8501
```

### ⚙️ Configuration
* Auto-refresh: Toggle in the UI to refresh data every 10s.
* Forecasting: Uses simple trend-based forecasting for PM2.5.
* Azure IoT Hub (optional): If you connect devices, configure credentials in a .env file. Example:
 ```
 IOTHUB_CONNECTION_STRING="your-azure-iot-hub-connection-string"
```

### 📊 Dashboard Sections
* Current AQI: Real-time PM2.5 and CO₂ readings
* Forecast (Next 3 Hours): Predictive AQI with health status indicators
* Recent Network Alerts: Last 20 alerts, with filtering and expandable details
* Visualization: Interactive charts for historical and forecast data

### 🛠️ Developed for **AINNOVATION 2025 II Hackathon**
