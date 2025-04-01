import requests
from typing import Dict, Any


class WeatherService:
    def __init__(self, api_url: str = "https://your-backend-api.com/api/weather"):
        self.api_url = api_url

    def get_weather(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get weather data from backend API"""
        if not weather_data.get("is_weather_query", False):
            return {"success": False, "message": "Not a weather query"}

        try:
            # Prepare data for backend
            payload = {
                "query_type": "weather",
                "location": weather_data.get("location", ""),
                "time_period": weather_data.get("time_period", "today"),
                "original_query": weather_data.get("original_query", ""),
                "language": "de"
            }

            # Send request to backend
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                return {
                    "success": True,
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "message": f"Backend error: {response.status_code}"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error: {str(e)}"
            }

    def check_status(self) -> str:
        """Check if backend API is available"""
        try:
            backend_url = self.api_url.split('/api/')[0] + "/health"
            response = requests.get(backend_url, timeout=5)
            if response.status_code == 200:
                return "available"
            else:
                return f"error: {response.status_code}"
        except Exception as e:
            return f"error: {str(e)}"
