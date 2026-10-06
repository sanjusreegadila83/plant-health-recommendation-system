"""Built-in plant care guidance and optional external service integrations."""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

_CARE_GUIDANCE = {
    "Healthy": {
        "Treatment": [
            "Continue regular watering, adjusting for the plant and season.",
            "Maintain adequate sunlight for the plant species.",
            "Maintain proper soil nutrition; avoid unnecessary treatments.",
        ],
        "Prevention": [
            "Monitor leaves regularly for changes.",
            "Keep tools and pots clean and ensure good air circulation.",
        ],
        "Possible causes": ["No concerning condition was indicated by the model."],
    },
    "Leaf Disease": {
        "Treatment": [
            "Remove heavily affected leaves and dispose of them appropriately.",
            "Improve air circulation around the plant.",
            "Avoid excessive watering; water the soil rather than wetting foliage.",
            "Monitor disease progression.",
            "Consider an appropriate plant-safe treatment after confirming the cause.",
        ],
        "Prevention": [
            "Avoid crowding plants and keep foliage dry where practical.",
            "Clean gardening tools between plants.",
            "Check new plants and leaves regularly for symptoms.",
        ],
        "Possible causes": [
            "The predicted label may be associated with a fungal, bacterial, or environmental issue."
        ],
    },
    "Severe Disease": {
        "Treatment": [
            "Isolate the affected plant if practical and safe for the other plants.",
            "Remove severely affected portions with clean tools.",
            "Check nearby plants for similar symptoms.",
            "Consult a local agriculture expert if the problem continues.",
        ],
        "Prevention": [
            "Disinfect tools and avoid moving contaminated plant material.",
            "Improve spacing and monitor nearby plants.",
            "Confirm the cause before applying a treatment.",
        ],
        "Possible causes": [
            "The model assigned a severe-category label; the exact cause needs expert confirmation."
        ],
    },
}

_WEATHER_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


def get_recommendations(health_status: str) -> dict[str, list[str]]:
    """Return built-in guidance for a supported health category."""
    return _CARE_GUIDANCE.get(health_status, _CARE_GUIDANCE["Leaf Disease"])


def get_ai_recommendation(
    predicted_class: str, health_status: str, confidence: float
) -> tuple[str | None, str]:
    """Ask Gemini for a concise explanation when configured; report failures for fallback."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        return None, "Gemini API key is not configured; built-in recommendations remain available."

    try:
        from google import genai
    except ImportError as error:
        return None, f"Gemini support is unavailable; built-in recommendations remain available. Details: {error}"
    try:
        from google.genai.errors import APIError
    except ImportError as error:
        return None, f"Gemini support is unavailable; built-in recommendations remain available. Details: {error}"

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=(
                "Give beginner-friendly plant care guidance based on this image-model result. "
                "Include headings for Disease explanation, Possible causes, Treatment suggestions, "
                "Prevention suggestions, and Basic care instructions. Do not claim certainty; "
                "say a local agriculture expert should confirm the issue. The result is "
                f"predicted label '{predicted_class}', application category '{health_status}', "
                f"model confidence {confidence:.1%}. Do not invent a plant species."
            ),
        )
        text = getattr(response, "text", None)
        if not text or not text.strip():
            return None, "Gemini returned no explanation; use the built-in recommendations below."
        return text.strip(), "Gemini explanation generated."
    except (APIError, TimeoutError, OSError, ValueError, TypeError, RuntimeError) as error:
        return None, (
            "Gemini could not be reached; built-in recommendations remain available. "
            f"Details: {error}"
        )


def get_weather(latitude: float, longitude: float) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch current conditions from the public Open-Meteo API."""
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        current = payload["current"]
        code = int(current["weather_code"])
        return {
            "temperature": float(current["temperature_2m"]),
            "humidity": float(current["relative_humidity_2m"]),
            "rain": float(current.get("rain", current.get("precipitation", 0.0))),
            "description": _WEATHER_DESCRIPTIONS.get(code, "Variable conditions"),
        }, None
    except (requests.RequestException, ValueError, KeyError, TypeError) as error:
        return None, f"Weather service unavailable; you can continue without it. Details: {error}"


def get_environment_advice(weather: dict[str, Any]) -> list[str]:
    """Provide cautious care prompts from current conditions, not a disease diagnosis."""
    advice: list[str] = []
    if weather["humidity"] >= 80 or weather["rain"] >= 2:
        advice.append(
            "Humidity or rain is high: avoid additional overhead watering and allow foliage to dry."
        )
    if weather["temperature"] >= 32:
        advice.append("Hot conditions are reported: check soil moisture more often and avoid midday stress.")
    elif weather["temperature"] <= 5:
        advice.append("Cold conditions are reported: protect sensitive plants from chilling where possible.")
    if not advice:
        advice.append("No strong weather-related adjustment is suggested by these current conditions.")
    return advice
