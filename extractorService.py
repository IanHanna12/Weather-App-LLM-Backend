import requests
import json
import re


class WeatherExtractor:
    """Rule-based weather information extractor (fallback)"""

    def __init__(self):
        self.weather_words = [
            "wetter", "temperatur", "regen", "schnee", "sonne", "wind",
            "kalt", "warm", "gewitter", "niederschlag", "bewölkt", "wolken",
            "grad", "celsius", "tag", "day", "forecast", "vorhersage"
        ]

        self.cities = [
            "berlin", "hamburg", "münchen", "köln", "frankfurt", "stuttgart",
            "düsseldorf", "dresden", "leipzig", "hannover", "nürnberg",
            "dortmund", "essen", "bremen", "bonn", "mannheim"
        ]

        self.today_words = ["heute", "jetzt", "aktuell"]
        self.tomorrow_words = ["morgen", "tomorrow"]
        self.week_words = ["woche", "tage", "übermorgen", "week", "days", "two", "drei", "vier", "fünf"]

    def extract(self, text):
        if not text:
            return {"is_weather_query": False, "location": None, "time_period": "today"}

        text = text.lower()

        # More lenient weather query detection - if a city is mentioned, it's likely a weather query
        is_weather = any(word in text for word in self.weather_words)

        # Find location
        location = None
        for city in self.cities:
            if city in text:
                location = city
                # If we found a city, it's more likely to be a weather query
                is_weather = True
                break

        # Determine time period
        time_period = "today"  # default

        if any(word in text for word in self.tomorrow_words):
            time_period = "tomorrow"
        elif any(word in text for word in self.week_words):
            time_period = "week"

        # Check for "in X days" pattern
        days_match = re.search(r'in\s+(\d+)\s+tag', text)
        if days_match:
            days = int(days_match.group(1))
            if days == 1:
                time_period = "tomorrow"
            elif days > 1:
                time_period = "week"

        # Check for "two day" pattern
        if "two day" in text or "2 day" in text or "zwei tag" in text:
            time_period = "week"

        return {
            "is_weather_query": is_weather,
            "location": location,
            "time_period": time_period
        }


class OllamaExtractor:
    """LLM-based weather information extractor using Ollama"""

    def __init__(self, model="llama2", fallback_extractor=None):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        self.fallback_extractor = fallback_extractor

    def is_available(self):
        """Check if Ollama is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def extract(self, text):
        """Extract weather information from text using Ollama"""
        if not text:
            return {"is_weather_query": False, "location": None, "time_period": "today"}

        # If Ollama is not available, use fallback
        if not self.is_available():
            print("Ollama is not available, using fallback extractor")
            if self.fallback_extractor:
                return self.fallback_extractor.extract(text)
            return {"is_weather_query": False, "location": None, "time_period": "today"}

        # Create a prompt that defines the slots we want to extract
        prompt = f"""
        Analyze this query about weather: "{text}"

        Extract the following information:
        1. Is this a weather-related query? (yes/no)
        2. Location mentioned (city name or "none" if not specified)
        3. Time period (today, tomorrow, week, or specific date)

        Format your response exactly like this:
        WEATHER_QUERY: yes/no
        LOCATION: [extracted location or "none"]
        TIME_PERIOD: [extracted time period]
        """

        try:
            # Call Ollama API
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=5
            )

            # Check if request was successful
            if response.status_code != 200:
                raise Exception(f"Ollama API returned status code {response.status_code}")

            # Parse the response
            result = response.json().get("response", "")

            # Extract the structured information
            weather_query = re.search(r'WEATHER_QUERY:\s*(yes|no)', result, re.IGNORECASE)
            location = re.search(r'LOCATION:\s*([^\n]+)', result, re.IGNORECASE)
            time_period = re.search(r'TIME_PERIOD:\s*([^\n]+)', result, re.IGNORECASE)

            # Create the result dictionary
            extracted_data = {
                "is_weather_query": weather_query and weather_query.group(1).lower() == "yes",
                "location": (location and location.group(1).strip() != "none") and location.group(1).strip() or None,
                "time_period": (time_period and time_period.group(1).strip()) or "today"
            }

            # Normalize time period
            if extracted_data["time_period"] not in ["today", "tomorrow", "week"]:
                # If it's a specific date or other format, map to one of our standard periods
                if "week" in extracted_data["time_period"].lower() or any(
                        day in extracted_data["time_period"].lower() for day in
                        ["montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag"]):
                    extracted_data["time_period"] = "week"
                elif "tomorrow" in extracted_data["time_period"].lower() or "morgen" in extracted_data[
                    "time_period"].lower():
                    extracted_data["time_period"] = "tomorrow"
                else:
                    extracted_data["time_period"] = "today"

            print(f"Ollama extraction result: {extracted_data}")
            return extracted_data

        except Exception as e:
            print(f"Error using Ollama for extraction: {e}")
            # Fall back to rule-based extractor if available
            if self.fallback_extractor:
                return self.fallback_extractor.extract(text)
            return {"is_weather_query": False, "location": None, "time_period": "today"}


class HybridExtractor:
    """Combines LLM-based and rule-based extraction for best results"""

    def __init__(self, model="llama2"):
        self.rule_based = WeatherExtractor()
        self.ollama = OllamaExtractor(model=model, fallback_extractor=self.rule_based)

    def extract(self, text):
        """Extract weather information using the best available method"""
        # First try with Ollama
        result = self.ollama.extract(text)

        # If Ollama couldn't find a location but thinks it's a weather query,
        # try the rule-based approach as well and merge results
        if result["is_weather_query"] and result["location"] is None:
            rule_result = self.rule_based.extract(text)
            if rule_result["location"] is not None:
                result["location"] = rule_result["location"]

        return result
