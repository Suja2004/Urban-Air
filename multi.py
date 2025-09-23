import nest_asyncio
import numpy as np
import joblib
from azure.iot.device.aio import IoTHubDeviceClient
import asyncio
import json
import random
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


# --- Configuration ---
CONNECTION_STRING = os.getenv("IOTHUB_DEVICE_CONNECTION_STRING")

# Define street lamps specifically designed to test all air quality categories
COMPREHENSIVE_DEVICES = [
    {
        "device_id": "StreetLamp_Clean_Park_001",
        "location": {"lat": 12.9698, "lon": 77.5970, "area": "Lalbagh Park", "zone": "Green Zone"},
        "target_quality": "good",
        "description": "Park area with clean air"
    },
    {
        "device_id": "StreetLamp_Residential_002",
        "location": {"lat": 12.9352, "lon": 77.6245, "area": "JP Nagar", "zone": "Residential"},
        "target_quality": "moderate",
        "description": "Typical residential area"
    },
    {
        "device_id": "StreetLamp_Commercial_003",
        "location": {"lat": 12.9716, "lon": 77.5946, "area": "Commercial Street", "zone": "Commercial"},
        "target_quality": "poor",
        "description": "Busy commercial area with traffic"
    },
    {
        "device_id": "StreetLamp_Industrial_004",
        "location": {"lat": 12.8500, "lon": 77.6700, "area": "Peenya Industrial", "zone": "Industrial"},
        "target_quality": "hazardous",
        "description": "Heavy industrial zone"
    },
    {
        "device_id": "StreetLamp_Highway_005",
        "location": {"lat": 13.0200, "lon": 77.5800, "area": "Outer Ring Road", "zone": "Highway"},
        "target_quality": "poor",
        "description": "Major highway with heavy traffic"
    },
    {
        "device_id": "StreetLamp_IT_Campus_006",
        "location": {"lat": 12.9450, "lon": 77.6950, "area": "Electronic City Phase 1", "zone": "IT Campus"},
        "target_quality": "moderate",
        "description": "Tech park with moderate pollution"
    }
]


class AirQualityTestDevice:
    """Street lamp device designed to generate specific air quality predictions."""

    def __init__(self, device_config):
        self.config = device_config
        self.device_id = device_config["device_id"]
        self.target_quality = device_config["target_quality"]
        self.last_predicted_quality = None  # Will be set during priming

        # Load ML model
        try:
            base_path = os.path.dirname(__file__)
            self.model = joblib.load(os.path.join(
                base_path, 'model', 'air_quality_model.pkl'))
            self.scaler = joblib.load(os.path.join(
                base_path, 'model', 'scaler.pkl'))
            self.label_encoder = joblib.load(os.path.join(
                base_path, 'model', 'label_encoder.pkl'))
            print(f"✅ Model loaded for {self.device_id}")

            # --- NEW: Prime the device with an initial state ---
            self.prime_device_state()

        except FileNotFoundError:
            print(
                f"❌ Model files not found for {self.device_id}. Ensure they are in a 'model' sub-folder.")
            raise

    def prime_device_state(self):
        """Generates an initial prediction to set a valid starting state."""
        print(f"    priming state for {self.device_id}...")
        initial_sensor_data = self.generate_targeted_sensor_data()
        self.last_predicted_quality = self.predict_air_quality(
            initial_sensor_data)
        status_line = format_quality_status(
            self.last_predicted_quality, self.target_quality)
        print(
            f"   -> Initial state for {self.device_id:<32} set to {status_line}")

    def get_target_ranges(self):
        """Get sensor ranges designed to produce specific air quality predictions."""
        target_ranges = {
            "good": {"pm2_5": (2, 15), "pm10": (5, 25), "no2": (3, 25), "so2": (1, 12), "co": (0.2, 2.0), "proximity": (15, 30), "population": (100, 500)},
            "moderate": {"pm2_5": (12, 40), "pm10": (20, 60), "no2": (20, 55), "so2": (8, 30), "co": (1.5, 4.5), "proximity": (5, 20), "population": (400, 1000)},
            "poor": {"pm2_5": (35, 70), "pm10": (50, 110), "no2": (40, 85), "so2": (20, 50), "co": (3.0, 7.0), "proximity": (1, 10), "population": (800, 1600)},
            "hazardous": {"pm2_5": (80, 300), "pm10": (120, 500), "no2": (70, 150), "so2": (35, 80), "co": (5.0, 12.0), "proximity": (0.2, 5), "population": (1200, 2500)}
        }
        return target_ranges.get(self.target_quality, target_ranges["moderate"])

    def generate_targeted_sensor_data(self):
        """Generate sensor data targeted to produce specific air quality prediction."""
        ranges = self.get_target_ranges()
        hour = datetime.now().hour

        rush_hour = hour in [7, 8, 9, 17, 18, 19, 20]
        rush_multiplier = 1.2 if rush_hour else 1.0
        weekend_multiplier = 0.85 if datetime.now().weekday() >= 5 else 1.0
        final_multiplier = rush_multiplier * \
            weekend_multiplier * (0.9 + random.random() * 0.2)

        sensor_data = {
            "temperature": round(random.uniform(16, 44), 2),
            "humidity": round(random.uniform(20, 90), 2),
            "pm2_5": max(1.0, round(random.uniform(*ranges["pm2_5"]) * final_multiplier, 2)),
            "pm10": max(2.0, round(random.uniform(*ranges["pm10"]) * final_multiplier, 2)),
            "no2": max(1.0, round(random.uniform(*ranges["no2"]) * final_multiplier, 2)),
            "so2": max(0.5, round(random.uniform(*ranges["so2"]) * final_multiplier, 2)),
            "co": max(0.1, round(random.uniform(*ranges["co"]) * final_multiplier, 2)),
            "proximityToIndustrialAreas": round(random.uniform(*ranges["proximity"]), 2),
            "populationDensity": random.randint(*ranges["population"])
        }
        return sensor_data

    def predict_air_quality(self, sensor_data):
        """Use ML model to predict air quality."""
        try:
            feature_values = [
                sensor_data["temperature"], sensor_data["humidity"], sensor_data["pm2_5"],
                sensor_data["pm10"], sensor_data["no2"], sensor_data["so2"],
                sensor_data["co"], sensor_data["proximityToIndustrialAreas"],
                sensor_data["populationDensity"]
            ]
            features = np.array([feature_values])

            scaled_features = self.scaler.transform(features)
            prediction = self.model.predict(scaled_features)
            return self.label_encoder.inverse_transform(prediction)[0]
        except Exception as e:
            print(f"❌ Prediction error for {self.device_id}: {e}")
            return "unknown"


def format_quality_status(predicted, target):
    """Format air quality status with color indicators."""
    status_icons = {"good": "🟢", "moderate": "🟡",
                    "poor": "🟠", "hazardous": "🔴"}
    icon = status_icons.get(predicted, "⚪")
    target_icon = status_icons.get(target, "⚪")
    match_status = "✓" if predicted == target else "✗"
    return f"{icon} {predicted.upper():<10} (Target: {target_icon} {target.upper()}) {match_status}"


async def run_comprehensive_simulation():
    """Run comprehensive air quality testing simulation."""
    print("Initializing devices and priming their states...")
    devices = [AirQualityTestDevice(config)
               for config in COMPREHENSIVE_DEVICES]
    print("-" * 50)

    if not CONNECTION_STRING or "HostName" not in CONNECTION_STRING:
        print("❌ IoT Hub connection string is not valid. Halting simulation.")
        return

    device_client = IoTHubDeviceClient.create_from_connection_string(
        CONNECTION_STRING)
    try:
        print("🔌 Connecting to IoT Hub...")
        await device_client.connect()
        print("✅ Connected!")
        print("\n" + "="*100 + "\n🧪 SIMULATION LOOP STARTED\n" + "="*100 + "\n")

        cycle_count = 0
        while True:
            cycle_count += 1
            print(
                f"📊 CYCLE {cycle_count} - {datetime.now().strftime('%H:%M:%S')}\n" + "-" * 100)

            messages_sent = 0
            for device in devices:
                sensor_data = device.generate_targeted_sensor_data()
                predicted_quality = device.predict_air_quality(sensor_data)
                status_line = format_quality_status(
                    predicted_quality, device.target_quality)

                if predicted_quality != device.last_predicted_quality and predicted_quality != "unknown":
                    messages_sent += 1
                    telemetry_data = {
                        "deviceId": device.device_id, "timestamp": datetime.now().isoformat(),
                        "location": device.config["location"], "targetQuality": device.target_quality,
                        "predictedAirQuality": predicted_quality, "description": device.config["description"],
                        **sensor_data
                    }
                    message = json.dumps(telemetry_data)
                    await device_client.send_message(message)
                    device.last_predicted_quality = predicted_quality
                    print(
                        f"📤 {device.device_id:<32} | {device.config['location']['area']:<20} | {status_line}")
                else:
                    print(
                        f"⚪ {device.device_id:<32} | {device.config['location']['area']:<20} | {status_line} (No change)")

            print(
                f"\nCycle Summary: {messages_sent}/{len(devices)} devices sent updates.")
            print("="*100 + "\n")
            await asyncio.sleep(random.uniform(15, 25))

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
    finally:
        if device_client and device_client.connected:
            print("\n🔌 Disconnecting...")
            await device_client.shutdown()


if __name__ == "__main__":
    nest_asyncio.apply()
    try:
        asyncio.run(run_comprehensive_simulation())
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")
