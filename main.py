import asyncio
import json
import random
import os
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from azure.iot.device.aio import IoTHubDeviceClient
from dotenv import load_dotenv

load_dotenv()
# --- Configuration ---
CONNECTION_STRING = os.getenv("IOTHUB_DEVICE_CONNECTION_STRING")
DEVICE_ID = "RaspberryPi"

# This event will be used to signal the simulation to stop gracefully.
stop_event = asyncio.Event()


# --- Model Loading ---

# Load the model and preprocessors once at startup for efficiency.
try:
    model = joblib.load('model/air_quality_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    print("✅ Model and preprocessors loaded successfully.")
except FileNotFoundError:
    print("❌ FATAL: Model files not found. The /predict endpoint will not work.")
    model = None


# --- Background Simulation Logic ---

async def run_iot_simulation():
    """
    A long-running background task that simulates an IoT device, predicts air
    quality, and sends data to Azure IoT Hub only when the state changes.
    """
    if not CONNECTION_STRING or "HostName" not in CONNECTION_STRING:
        print("⚠️ WARNING: IoT Hub connection string is not configured. Simulation will not send data.")
        return

    device_client = IoTHubDeviceClient.create_from_connection_string(
        CONNECTION_STRING)
    try:
        await device_client.connect()
        print("✅ Background Simulation: Connected to IoT Hub.")
        last_predicted_quality = None

        while not stop_event.is_set():
            # 1. Simulate Sensor Readings
            temperature = round(random.uniform(15.0, 45.0), 2)
            # ... (all other sensor simulations)
            humidity = round(random.uniform(20.0, 100.0), 2)
            pm2_5 = round(random.uniform(5.0, 350.0), 2)
            pm10 = round(random.uniform(10.0, 450.0), 2)
            no2 = round(random.uniform(10.0, 100.0), 2)
            so2 = round(random.uniform(5.0, 50.0), 2)
            co = round(random.uniform(0.5, 8.0), 2)
            proximity_to_industrial = round(random.uniform(0.5, 20.0), 2)
            population_density = random.randint(200, 2000)

            # 2. Use the ML Model for Prediction
            features = np.array([[temperature, humidity, pm2_5, pm10, no2,
                                so2, co, proximity_to_industrial, population_density]])
            scaled_features = scaler.transform(features)
            prediction_encoded = model.predict(scaled_features)
            predicted_air_quality = label_encoder.inverse_transform(prediction_encoded)[
                0]

            # 3. Send Data Only on State Change
            if predicted_air_quality != last_predicted_quality:
                print(
                    f"\n🔄 Simulation: State changed to '{predicted_air_quality}'. Sending message.")
                last_predicted_quality = predicted_air_quality
                telemetry_data = {
                    "deviceId": DEVICE_ID,
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
                await device_client.send_message(message)
            else:
                print(".", end="", flush=True)

            # Wait for 10 seconds or until the stop event is set
            await asyncio.sleep(10)

    except Exception as e:
        print(f"\n❌ ERROR in background simulation: {e}")
    finally:
        if device_client.connected:
            await device_client.shutdown()
        print("🛑 Background Simulation: Disconnected and stopped.")


# --- FastAPI Lifespan Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown of the background simulation task.
    """
    print("🚀 Server starting up...")
    # Start the simulation task when the server starts
    simulation_task = asyncio.create_task(run_iot_simulation())
    yield  # The server is now running
    print("👋 Server shutting down...")
    # Signal the simulation to stop and wait for it to finish
    stop_event.set()
    await simulation_task


# --- FastAPI App and API Endpoints ---

app = FastAPI(
    title="Smart Lamp API & IoT Simulator",
    description="A unified server that provides an on-demand prediction API and runs a continuous IoT device simulation in the background.",
    version="2.0.0",
    lifespan=lifespan
)

# Pydantic models for API data validation


class SensorData(BaseModel):
    temperature: float = Field(..., example=35.5)
    humidity: float = Field(..., example=85.2)
    pm2_5: float = Field(..., example=150.7)
    pm10: float = Field(..., example=220.1)
    no2: float = Field(..., example=75.3)
    so2: float = Field(..., example=30.8)
    co: float = Field(..., example=4.5)
    proximityToIndustrialAreas: float = Field(..., example=2.1)
    populationDensity: int = Field(..., example=1500)


class PredictionResponse(BaseModel):
    predictedAirQuality: str


@app.post("/predict", response_model=PredictionResponse, tags=["On-Demand Prediction"])
def predict(data: SensorData):
    """Receives sensor data and returns a single air quality prediction."""
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    features = np.array([[data.temperature, data.humidity, data.pm2_5, data.pm10, data.no2,
                        data.so2, data.co, data.proximityToIndustrialAreas, data.populationDensity]])
    scaled_features = scaler.transform(features)
    prediction_encoded = model.predict(scaled_features)
    predicted_quality = label_encoder.inverse_transform(prediction_encoded)[0]
    return {"predictedAirQuality": predicted_quality}


@app.get("/simulation/status", tags=["Background Simulation"])
def get_simulation_status():
    """Check the status of the background simulation task."""
    if stop_event.is_set():
        return {"status": "Stopped"}
    return {"status": "Running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
