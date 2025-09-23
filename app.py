import streamlit as st
import asyncio
import json
import random
import os
import time
import threading
import queue
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import joblib
from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device.common.transport_exceptions import ConnectionDroppedError
import nest_asyncio
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title=" Pro - Air Quality Monitor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Futuristic Dark Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    :root {
        --primary-color: #00D4AA;
        --secondary-color: #0077BE;
        --glow-color: rgba(0, 212, 170, 0.5);
        --dark-bg: #0A0F1C;
        --card-bg: #1A1F2E;
        --border-color: #2A3B4F;
        --text-primary: #FFFFFF;
        --text-secondary: #8B949E;
        --success-color: #2ECC71;
        --warning-color: #F1C40F;
        --danger-color: #E74C3C;
    }
    
    .main .block-container {
        padding: 1rem 2rem;
        max-width: 100%;
    }
    
    body {
        background-color: var(--dark-bg);
        color: var(--text-primary);
    }
    
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 3.5rem;
        text-align: center;
        letter-spacing: 3px;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--card-bg) 0%, rgba(26, 31, 46, 0.8) 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px var(--glow-color);
        border-color: var(--primary-color);
    }
    
    .street-lamp-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 10px;
        transition: all 0.3s ease;
    }
    
    .street-lamp-container:hover {
        transform: scale(1.1);
    }
    
    .lamp-post {
        width: 8px;
        height: 70px;
        background: linear-gradient(180deg, #555 0%, #333 100%);
        margin: 0 auto;
        border-radius: 4px;
    }
    
    .lamp-head {
        width: 40px;
        height: 24px;
        border-radius: 50% 50% 10px 10px;
        margin-top: -3px;
        box-shadow: 0 0 30px currentColor, inset 0 0 10px rgba(255,255,255,0.3);
        border: 2px solid currentColor;
        position: relative;
        background: linear-gradient(45deg, currentColor, rgba(255,255,255,0.2));
    }
    
    .lamp-glow {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        opacity: 0.4;
        background: radial-gradient(circle, currentColor 5%, transparent 70%);
        margin-top: -25px;
        animation: pulse 3s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        from {
            opacity: 0.2;
            transform: scale(0.9);
        }
        to {
            opacity: 0.5;
            transform: scale(1.1);
        }
    }
    
    .lamp-label {
        color: var(--text-primary);
        font-size: 0.9rem;
        text-align: center;
        margin-top: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        text-shadow: 0 0 5px rgba(0,0,0,0.8);
        padding: 4px 8px;
        background: rgba(0,0,0,0.6);
        border-radius: 8px;
        backdrop-filter: blur(5px);
    }
    
    /* Tab styling */
    div[role="tablist"] {
        gap: 10px;
        margin-bottom: 1rem;
    }
    
    button[role="tab"] {
        background-color: transparent;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-secondary);
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
        padding: 16px;
        font-weight: 500;
    }
    
    button[role="tab"]:hover {
        background-color: var(--card-bg);
        color: var(--primary-color);
        border-color: var(--primary-color);
    }
    
    button[role="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        box-shadow: 0 0 15px var(--glow-color);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--card-bg);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
        animation: blink 2s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    
    .status-good { background-color: var(--success-color); }
    .status-moderate { background-color: var(--warning-color); }
    .status-poor { background-color: #E67E22; }
    .status-hazardous { background-color: var(--danger-color); }
            
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
CONNECTION_STRING = os.getenv("IOTHUB_DEVICE_CONNECTION_STRING")

COMPREHENSIVE_DEVICES = [
    {
        "device_id": "ECO-NODE-01",
        "location": {"area": "Lalbagh Park", "zone": "Green Zone"},
        "target_quality": "good"
    },
    {
        "device_id": "ECO-NODE-02",
        "location": {"area": "JP Nagar", "zone": "Residential"},
        "target_quality": "moderate"
    },
    {
        "device_id": "ECO-NODE-03",
        "location": {"area": "Commercial Street", "zone": "Commercial"},
        "target_quality": "poor"
    },
    {
        "device_id": "ECO-NODE-04",
        "location": {"area": "Peenya Industrial", "zone": "Industrial"},
        "target_quality": "hazardous"
    },
    {
        "device_id": "ECO-NODE-05",
        "location": {"area": "Outer Ring Road", "zone": "Highway"},
        "target_quality": "poor"
    },
    {
        "device_id": "ECO-NODE-06",
        "location": {"area": "Electronic City", "zone": "IT Campus"},
        "target_quality": "moderate"
    }
]

# Feature names that match the training columns 
FEATURE_ORDER = [
    'Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO',
    'Proximity_to_Industrial_Areas', 'Population_Density'
]

# --- IoT Hub Communication (Background Thread) ---


def iot_hub_sender(message_queue: queue.Queue):
    """Runs in a background thread to send messages from the queue to Azure IoT Hub."""

    async def sender_loop():
        st.toast("🚀 IoT Hub sender thread started...", icon="🚀")

        # Check if connection string is configured 
        if not CONNECTION_STRING or "YourIoTHubConnectionString" in CONNECTION_STRING:
            error_msg = "🔴 OFFLINE (Connection String not configured)"
            print(f"IOT HUB SENDER: {error_msg}")
            st.session_state.iot_hub_status = error_msg
            st.toast(error_msg, icon="❌")
            return

        st.session_state.iot_hub_status = "🟡 CONNECTING..."
        st.toast("Connecting to Azure IoT Hub...", icon="☁️")

        try:
            device_client = IoTHubDeviceClient.create_from_connection_string(
                CONNECTION_STRING)
            await device_client.connect()
            st.session_state.iot_hub_status = "🟢 CONNECTED"
            st.toast("✅ Successfully connected to IoT Hub!", icon="✅")
        except Exception as e:
            error_msg = "🔴 CONNECTION ERROR"
            print(f"IOT HUB SENDER: {error_msg} - {e}")
            st.session_state.iot_hub_status = error_msg
            st.toast(f"❌ IoT Hub Connection Failed: {e}", icon="❌")
            return

        # Main sender loop
        while True:
            try:
                message_to_send = message_queue.get_nowait()
                if message_to_send:
                    json_message = json.dumps(message_to_send)
                    await device_client.send_message(json_message)
                    st.session_state.messages_sent_to_hub += 1
            except queue.Empty:
                await asyncio.sleep(1)
            except ConnectionDroppedError:
                error_msg = "🔴 CONNECTION DROPPED. RECONNECTING..."
                print(error_msg)
                st.session_state.iot_hub_status = error_msg
                st.toast(error_msg, icon="🔁")
                try:
                    await device_client.connect()
                    st.session_state.iot_hub_status = "🟢 RECONNECTED"
                except Exception as e:
                    print(f"Failed to reconnect: {e}")
            except Exception as e:
                error_msg = "🔴 UNEXPECTED ERROR"
                print(f"{error_msg}: {e}")
                st.session_state.iot_hub_status = error_msg
                st.toast(f"❌ IoT Hub Error: {e}", icon="❌")
                try:
                    await device_client.shutdown()
                except:
                    pass
                return

    nest_asyncio.apply()
    asyncio.run(sender_loop())

# --- Device Simulation Class with Integrated ML Model ---


class AirQualityTestDevice:
    def __init__(self, device_config):
        self.config = device_config
        self.device_id = device_config["device_id"]
        self.target_quality = device_config["target_quality"]
        self.last_predicted_quality = "moderate"
        self.is_active = False

        # Try to load ML model
        try:
            base_path = os.path.dirname(__file__)
            model_path = os.path.join(
                base_path, 'model', 'air_quality_model.pkl')
            scaler_path = os.path.join(base_path, 'model', 'scaler.pkl')
            encoder_path = os.path.join(
                base_path, 'model', 'label_encoder.pkl')

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            self.model_loaded = True
            print(f"✅ ML Model loaded successfully for {self.device_id}")
        except Exception as e:
            print(f"⚠️  Could not load ML model for {self.device_id}: {e}")
            self.model_loaded = False

        self.prime_device_state()

    def prime_device_state(self):
        """Initialize the device with a baseline reading"""
        initial_data = self.generate_targeted_sensor_data()
        self.last_predicted_quality = self.predict_air_quality(
            initial_data).lower()
        self.is_active = True

    def get_target_ranges(self):
        """Define sensor value ranges for different air quality levels"""
        target_ranges = {
            "good": {
                "PM2.5": (2, 15),
                "NO2": (3, 25),
                "CO": (0.2, 2.0)
            },
            "moderate": {
                "PM2.5": (12, 40),
                "NO2": (20, 55),
                "CO": (1.5, 4.5)
            },
            "poor": {
                "PM2.5": (35, 70),
                "NO2": (40, 85),
                "CO": (3.0, 7.0)
            },
            "hazardous": {
                "PM2.5": (80, 300),
                "NO2": (70, 150),
                "CO": (5.0, 12.0)
            }
        }
        return target_ranges.get(self.target_quality, target_ranges["moderate"])

    def generate_targeted_sensor_data(self):
        """Generate realistic sensor data based on target quality level"""
        ranges = self.get_target_ranges()

        # Add some seasonal and time-based variations
        current_hour = datetime.now().hour
        is_rush_hour = current_hour in [7, 8, 9, 17, 18, 19, 20]
        rush_multiplier = 1.3 if is_rush_hour else 1.0

        return {
            "Temperature": round(random.uniform(16, 44), 2),
            "Humidity": round(random.uniform(20, 90), 2),
            "PM2.5": max(1.0, round(random.uniform(*ranges["PM2.5"]) * rush_multiplier, 2)),
            "PM10": max(2.0, round(random.uniform(20, 200) * rush_multiplier, 2)),
            "NO2": max(1.0, round(random.uniform(*ranges["NO2"]) * rush_multiplier, 2)),
            "SO2": max(0.5, round(random.uniform(5, 50), 2)),
            "CO": max(0.1, round(random.uniform(*ranges["CO"]) * rush_multiplier, 2)),
            "Proximity_to_Industrial_Areas": round(random.uniform(1, 15), 2),
            "Population_Density": random.randint(300, 2000)
        }

    def predict_air_quality(self, sensor_data):
        """Predict air quality using ML model or fallback to rule-based system"""
        if not self.model_loaded:
            # Fallback rule-based prediction
            pm25 = sensor_data["PM2.5"]
            if pm25 <= 15:
                return "good"
            elif pm25 <= 40:
                return "moderate"
            elif pm25 <= 70:
                return "poor"
            else:
                return "hazardous"

        try:
            # Use ML model for prediction
            features_df = pd.DataFrame([sensor_data], columns=FEATURE_ORDER)
            scaled_features = self.scaler.transform(features_df)
            prediction = self.model.predict(scaled_features)
            predicted_quality = self.label_encoder.inverse_transform(prediction)[
                0]
            return predicted_quality
        except Exception as e:
            print(f"Error during prediction for {self.device_id}: {e}")
            # Fallback to moderate if prediction fails
            return "moderate"

# --- Session State Initialization ---


def initialize_session_state():
    """Initialize all session state variables"""
    if 'devices' not in st.session_state:
        st.session_state.devices = [AirQualityTestDevice(
            config) for config in COMPREHENSIVE_DEVICES]
        st.session_state.historical_data = []

        # Initialize quality counts
        st.session_state.quality_counts = {
            "good": 0, "moderate": 0, "poor": 0, "hazardous": 0}
        for device in st.session_state.devices:
            st.session_state.quality_counts[device.last_predicted_quality] += 1

        st.session_state.total_alerts = 0
        st.session_state.last_update = datetime.now()
        st.session_state.iot_hub_status = "⚪ PENDING"
        st.session_state.messages_sent_to_hub = 0
        st.session_state.message_queue = queue.Queue()

        # Start IoT Hub sender thread
        sender_thread = threading.Thread(
            target=iot_hub_sender,
            args=(st.session_state.message_queue,),
            daemon=True
        )
        sender_thread.start()

# --- Core Data Update Logic ---


def update_dashboard_data():
    """Update sensor data for all devices and handle quality changes"""
    for device in st.session_state.devices:
        # Generate new sensor reading
        sensor_data = device.generate_targeted_sensor_data()
        predicted_quality = device.predict_air_quality(sensor_data).lower()

        # Check if quality changed
        if predicted_quality != device.last_predicted_quality:
            # Prepare message for IoT Hub
            message_payload = {
                "deviceId": device.device_id,
                "timestamp": datetime.now().isoformat(),
                **sensor_data
            }

            # Queue message for IoT Hub
            st.session_state.message_queue.put(message_payload)

            # Add to historical data
            log_entry = {
                "area": device.config["location"]["area"],
                "quality": predicted_quality,
                **message_payload
            }
            st.session_state.historical_data.append(log_entry)

            # Maintain historical data limit
            if len(st.session_state.historical_data) > 1000:
                st.session_state.historical_data.pop(0)

            # Update quality counts
            st.session_state.quality_counts[device.last_predicted_quality] -= 1
            st.session_state.quality_counts[predicted_quality] += 1

            # Update alert count and device state
            st.session_state.total_alerts += 1
            device.last_predicted_quality = predicted_quality

    st.session_state.last_update = datetime.now()

# --- Charting Functions ---


def create_quality_distribution_chart():
    """Create a donut chart showing air quality distribution across all devices"""
    labels = list(st.session_state.quality_counts.keys())
    values = list(st.session_state.quality_counts.values())
    colors = ["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"]

    fig = go.Figure(data=[go.Pie(
        labels=[l.title() for l in labels],
        values=values,
        marker_colors=colors,
        hole=0.65,
        textinfo='label+percent',
        textfont=dict(size=14, family="Inter, sans-serif"),
        insidetextorientation='radial'
    )])

    fig.update_layout(
        title_text="<b>Air Quality Distribution</b>",
        title_font_family="Orbitron, sans-serif",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=400,
        annotations=[dict(
            text=f'<b>{len(st.session_state.devices)}<br>NODES</b>',
            x=0.5, y=0.5,
            font_size=24,
            showarrow=False,
            font_family="Orbitron, sans-serif"
        )]
    )
    return fig


def create_sensor_trends_chart():
    """Create time series charts for key sensor readings"""
    if not st.session_state.historical_data:
        return go.Figure().update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_text="Awaiting Data..."
        )

    # Get last 100 data points
    df = pd.DataFrame(st.session_state.historical_data[-100:])
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            'PM2.5 Levels (µg/m³)',
            'Nitrogen Dioxide (NO₂) Levels',
            'Temperature (°C) & Humidity (%)'
        )
    )

    # Add traces
    fig.add_trace(go.Scatter(
        x=df['timestamp_dt'], y=df['PM2.5'],
        name='PM2.5', mode='lines',
        line=dict(width=2, color='#E74C3C'),
        fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.2)'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['timestamp_dt'], y=df['NO2'],
        name='NO2', mode='lines',
        line=dict(width=2, color='#F1C40F'),
        fill='tozeroy', fillcolor='rgba(241, 196, 15, 0.2)'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df['timestamp_dt'], y=df['Temperature'],
        name='Temperature', mode='lines',
        line=dict(width=2, color='#00A9FF')
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=df['timestamp_dt'], y=df['Humidity'],
        name='Humidity', mode='lines',
        line=dict(width=2, color='#2ECC71', dash='dot')
    ), row=3, col=1)

    fig.update_layout(
        title_text="<b>Real-time Sensor Telemetry</b>",
        title_font_family="Orbitron, sans-serif",
        template="plotly_dark",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,           
            xanchor="center",
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis3_title="Time"
    )

    return fig


def create_street_lamp_svg(device):
    """Create an animated street lamp visualization for each device"""
    quality_colors = {
        "good": "#2ECC71",
        "moderate": "#F1C40F",
        "poor": "#E67E22",
        "hazardous": "#E74C3C"
    }
    color = quality_colors.get(device.last_predicted_quality, "#9E9E9E")

    return f"""
    <div class="street-lamp-container" style="color: {color};">
        <div class="lamp-head" style="background: {color};"></div>
        <div class="lamp-post"></div>
        <div class="lamp-glow"></div>
        <div class="lamp-label">{device.config['location']['area']}</div>
    </div>
    """

# --- Main Application Layout ---


def main():
    """Main application function"""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-title">Urban Air</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Environmental Monitoring System for Smart Cities</p>',
                unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.markdown("## ⚙️ SYSTEM CONTROLS")

    auto_refresh = st.sidebar.checkbox(
        "Enable Auto-Refresh (10s)", value=True)

    if st.sidebar.button("REFRESH DATA", type="primary", use_container_width=True):
        with st.spinner("Updating sensor data..."):
            update_dashboard_data()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("## SYSTEM UPDATE")

    # Connection status warning
    if not CONNECTION_STRING:
        st.sidebar.warning(
            "⚠️ IoT Hub Connection String is not set in your .env file.", icon="⚠️")

    # Status metrics
    # status_color = "🟢" if "CONNECTED" in st.session_state.iot_hub_status else "🔴"
    # st.sidebar.markdown(
    #     f"**IoT Hub Status:** {st.session_state.iot_hub_status}")

    # col1, col2 = st.sidebar.columns(2)
    # with col1:
    #     st.metric("Active Nodes", len(st.session_state.devices))
    #     st.metric("Messages Sent", st.session_state.messages_sent_to_hub)
    # with col2:
    #     st.metric("Total Alerts", st.session_state.total_alerts)
    #     st.metric("Queue Size", st.session_state.message_queue.qsize())

    st.sidebar.write(
        f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(
        ["Status", "System Analytics", "Event Log"])

    with tab1:
        st.subheader("Live Nodes")

        # Street lamp visualization
        cols = st.columns(6)
        for i, device in enumerate(st.session_state.devices):
            with cols[i]:
                st.markdown(create_street_lamp_svg(
                    device), unsafe_allow_html=True)

        st.markdown("---")

        # Node details cards
        st.subheader("📡 Node Details")
        card_cols = st.columns(3)

        for i, device in enumerate(st.session_state.devices):
            with card_cols[i % 3]:
                quality_emoji = {
                    "good": "🟢",
                    "moderate": "🟡",
                    "poor": "🟠",
                    "hazardous": "🔴"
                }
                emoji = quality_emoji.get(device.last_predicted_quality, "⚪")

                st.markdown(f"""
                <div class="metric-card">
                    <h4>🏢 {device.config['location']['area']}</h4>
                    <p><strong>Status:</strong> {emoji} {device.last_predicted_quality.title()}</p>
                    <p><strong>Zone Type:</strong> {device.config['location']['zone']}</p>
                    <p><strong>Device ID:</strong> {device.device_id}</p>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.subheader("📈 System Analytics")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.plotly_chart(create_quality_distribution_chart(),
                            use_container_width=True)

        with col2:
            st.plotly_chart(create_sensor_trends_chart(),
                            use_container_width=True)

        # Additional metrics
        if st.session_state.historical_data:
            st.markdown("---")
            st.subheader("📊 Key Performance Indicators")

            recent_data = pd.DataFrame(st.session_state.historical_data[-50:])

            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

            with kpi_col1:
                avg_pm25 = recent_data['PM2.5'].mean()
                st.metric("Avg PM2.5", f"{avg_pm25:.1f} µg/m³")

            with kpi_col2:
                avg_temp = recent_data['Temperature'].mean()
                st.metric("Avg Temperature", f"{avg_temp:.1f}°C")

            with kpi_col3:
                avg_humidity = recent_data['Humidity'].mean()
                st.metric("Avg Humidity", f"{avg_humidity:.1f}%")

            with kpi_col4:
                avg_no2 = recent_data['NO2'].mean()
                st.metric("Avg NO2", f"{avg_no2:.1f} ppm")

    with tab3:
        st.subheader("📋 Recent Events")

        if st.session_state.historical_data:
            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                selected_quality = st.selectbox(
                    "Filter by Quality:",
                    ["All", "Good", "Moderate", "Poor", "Hazardous"]
                )
            with col2:
                max_entries = st.slider("Max entries to show:", 5, 50, 20)

            # Filter data
            filtered_data = st.session_state.historical_data
            if selected_quality != "All":
                filtered_data = [
                    entry for entry in filtered_data
                    if entry['quality'].title() == selected_quality
                ]

            # Display events
            st.markdown("### Event Timeline")
            for entry in reversed(filtered_data[-max_entries:]):
                quality_emoji = {
                    "good": "🟢",
                    "moderate": "🟡",
                    "poor": "🟠",
                    "hazardous": "🔴"
                }
                emoji = quality_emoji.get(entry['quality'], "⚪")

                timestamp = datetime.fromisoformat(entry['timestamp'])
                time_str = timestamp.strftime('%H:%M:%S')
                date_str = timestamp.strftime('%Y-%m-%d')

                # Create expandable event card
                with st.expander(f"{emoji} {entry['area']} - {entry['quality'].title()} at {time_str}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write("**Environmental Data:**")
                        st.write(f"PM2.5: {entry.get('PM2.5', 'N/A')} µg/m³")
                        st.write(f"PM10: {entry.get('PM10', 'N/A')} µg/m³")
                        st.write(f"NO2: {entry.get('NO2', 'N/A')} ppm")

                    with col2:
                        st.write("**Weather Conditions:**")
                        st.write(
                            f"Temperature: {entry.get('Temperature', 'N/A')}°C")
                        st.write(f"Humidity: {entry.get('Humidity', 'N/A')}%")
                        st.write(f"CO: {entry.get('CO', 'N/A')} ppm")

                    with col3:
                        st.write("**Location Info:**")
                        st.write(f"Device: {entry.get('deviceId', 'N/A')}")
                        st.write(f"Date: {date_str}")
                        st.write(
                            f"Industrial Proximity: {entry.get('Proximity_to_Industrial_Areas', 'N/A')} km")
        else:
            st.info(
                "🔄 Awaiting network activity... Data will appear here once sensors start reporting.")
            st.markdown("""
            **What you'll see here:**
            - Real-time air quality changes
            - Sensor readings with timestamps  
            - Environmental condition details
            - Device location information
            """)

    # Auto-refresh functionality
    if auto_refresh:
        # Show countdown
        placeholder = st.empty()
        for remaining in range(10, 0, -1):
            # placeholder.text(f"🔄 Auto-refresh in {remaining} seconds...")
            time.sleep(1)

        placeholder.empty()
        update_dashboard_data()
        st.rerun()

# --- Additional Utility Functions ---


def export_data():
    """Export historical data to CSV"""
    if st.session_state.historical_data:
        df = pd.DataFrame(st.session_state.historical_data)
        return df.to_csv(index=False)
    return None


def get_system_health():
    """Calculate overall system health metrics"""
    if not st.session_state.devices:
        return 0

    quality_weights = {"good": 4, "moderate": 3, "poor": 2, "hazardous": 1}
    total_score = sum(quality_weights[device.last_predicted_quality]
                      for device in st.session_state.devices)
    max_possible = len(st.session_state.devices) * 4

    return (total_score / max_possible) * 100

# --- Error Handling and Logging ---


def log_error(error_msg, exception=None):
    """Log errors with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] ERROR: {error_msg}"
    if exception:
        log_entry += f" - {str(exception)}"
    print(log_entry)

# --- Performance Monitoring ---


def monitor_performance():
    """Monitor system performance metrics"""
    metrics = {
        "active_devices": len(st.session_state.devices),
        "data_points": len(st.session_state.historical_data),
        "queue_size": st.session_state.message_queue.qsize(),
        "system_health": get_system_health(),
        "uptime": (datetime.now() - st.session_state.last_update).total_seconds()
    }
    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Application Error: {str(e)}")
        log_error("Application startup failed", e)
        st.markdown("""
        **Troubleshooting Steps:**
        1. Check if all required packages are installed
        2. Verify your .env file contains the correct IoT Hub connection string
        3. Ensure ML model files are in the correct 'model/' directory
        4. Restart the application
        
        **Need Help?**
        Contact your system administrator or check the application logs.
        """)
