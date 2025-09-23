import asyncio
import json
import random
import os
from datetime import datetime
from azure.iot.device.aio import IoTHubDeviceClient
import joblib
import numpy as np
import nest_asyncio
from dotenv import load_dotenv
load_dotenv()
# --- Configuration ---
CONNECTION_STRING = os.getenv("IOTHUB_DEVICE_CONNECTION_STRING")
DEVICE_ID = "RaspberryPi"  # Define Device ID to be included in the payload

# --- Load the Trained Model and Preprocessing Objects ---
try:
    model = joblib.load('model/air_quality_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    print("✅ Successfully loaded model and preprocessors.")
except FileNotFoundError:
    print("❌ Error: Model files not found. Ensure 'air_quality_model.pkl', 'scaler.pkl', and 'label_encoder.pkl' are in the same directory.")
    exit()


async def main():
    """
    Connects to Azure IoT Hub, simulates sensor data, uses a local ML model
    for classification, and sends telemetry only when the predicted air quality changes.
    """
    # Create the device client from the connection string.
    device_client = IoTHubDeviceClient.create_from_connection_string(
        CONNECTION_STRING)

    try:
        print("Connecting to IoT Hub...")
        await device_client.connect()
        print("✅ Device connected. Starting telemetry simulation...")

        # State variable to track the last sent air quality status.
        # This helps in reducing unnecessary data transmission.
        last_predicted_quality = None

        while True:
            # --- 1. Simulate Sensor Readings ---
            # Ranges are expanded to generate more varied air quality predictions.
            temperature = round(random.uniform(15.0, 45.0), 2)
            humidity = round(random.uniform(20.0, 100.0), 2)
            pm2_5 = round(random.uniform(5.0, 350.0), 2)
            pm10 = round(random.uniform(10.0, 450.0), 2)
            no2 = round(random.uniform(10.0, 100.0), 2)
            so2 = round(random.uniform(5.0, 50.0), 2)
            co = round(random.uniform(0.5, 8.0), 2)
            proximity_to_industrial = round(random.uniform(0.5, 20.0), 2)
            population_density = random.randint(200, 2000)

            # --- 2. Use the ML Model for Local Classification ---
            try:
                real_time_features = np.array([[
                    temperature, humidity, pm2_5, pm10, no2, so2, co,
                    proximity_to_industrial, population_density
                ]])

                scaled_features = scaler.transform(real_time_features)
                prediction_encoded = model.predict(scaled_features)
                predicted_air_quality = label_encoder.inverse_transform(prediction_encoded)[
                    0]

            except Exception as e:
                print(f"\n❌ Error during model prediction: {e}")
                # Skip this cycle if prediction fails
                await asyncio.sleep(10)
                continue

            # --- 3. Send Data Only on State Change ---
            # This logic makes the device "smarter" by only sending data when the
            # air quality category actually changes, saving bandwidth and cost.
            if predicted_air_quality != last_predicted_quality:
                print(
                    f"\n❗️ CHANGE DETECTED: New Predicted Air Quality is '{predicted_air_quality}'")
                last_predicted_quality = predicted_air_quality  # Update state

                # Construct the message payload, ensuring deviceId is included
                telemetry_data = {
                    "deviceId": DEVICE_ID,  # CRITICAL: For Stream Analytics grouping
                    "timestamp": datetime.now().isoformat(),
                    "temperature": temperature,
                    "humidity": humidity,
                    "pm2_5": pm2_5,
                    "pm10": pm10,
                    "no2": no2,
                    "so2": so2,
                    "co": co,
                    "proximityToIndustrialAreas": proximity_to_industrial,
                    "populationDensity": population_density,
                    "predictedAirQuality": predicted_air_quality
                }

                message = json.dumps(telemetry_data)

                print(f"   -> Sending message to IoT Hub...")
                print(f"Generated message:\n{message}")
                await device_client.send_message(message)
                print(f"   -> Message sent.")

            else:
                # Quietly indicate that the script is still running without spamming the console.
                print(f"Same Prediction.", end="", flush=True)

            # Wait before the next simulation cycle.
            await asyncio.sleep(10)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # Ensure the client is shut down gracefully.
        if device_client and device_client.connected:
            print("\nShutting down the device client.")
            await device_client.shutdown()

if __name__ == "__main__":
    nest_asyncio.apply()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
