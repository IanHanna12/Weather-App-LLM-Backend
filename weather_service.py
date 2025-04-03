import requests
from typing import Dict, Any


class WeatherService:
    def __init__(self, api_url: str = "http://localhost:8080/api/weather"):
        self.api_url = api_url

    def get_weather(self, weather_data):
        try:
            import requests

            location = weather_data.get("location", "Heilbronn")
            url = f"{self.api_url}{location}"

            print(f"Requesting weather data from: {url}")

            response = requests.get(url)
            print(f"Backend response status: {response.status_code}")

            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                print(f"Backend error response: {response.text}")
                return {"success": False, "message": f"Backend error: {response.status_code}"}
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return {"success": False, "message": f"Error: {str(e)}"}

