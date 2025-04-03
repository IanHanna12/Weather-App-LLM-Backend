import re


class WeatherExtractor:
    def __init__(self):
        self.weather_words = [
            "wetter", "temperatur", "regen", "schnee", "sonne", "wind",
            "kalt", "warm", "gewitter", "niederschlag", "bewölkt", "wolken",
            "grad", "celsius", "vorhersage"
        ]

        self.cities = [
            "berlin", "hamburg", "münchen", "köln", "frankfurt", "stuttgart",
            "düsseldorf", "dresden", "leipzig", "hannover", "nürnberg",
            "dortmund", "essen", "bremen", "bonn", "mannheim", "heilbronn"
        ]

        self.today_words = ["heute", "jetzt", "aktuell"]
        self.tomorrow_words = ["morgen"]
        self.week_words = ["woche", "tage", "übermorgen"]

    def extract(self, text):
        if not text:
            return {"is_weather_query": False, "location": None, "time_period": "today"}

        text = text.lower()

        is_weather = False
        for word in self.weather_words:
            if word in text:
                is_weather = True
                break

        # Ort (Stadt) finden
        location = None
        for city in self.cities:
            if city in text:
                location = city
                is_weather = True
                break

        # Wenn keine Stadt in der Liste gefunden wurde, nach Mustern suchen
        if location is None and is_weather:
            # Muster: "Wetter [Stadt]"
            pattern = r'wetter\s+([a-zäöüß]+)'
            match = re.search(pattern, text)
            if match:
                potential_city = match.group(1)
                if (potential_city not in self.weather_words and
                        potential_city not in self.today_words and
                        potential_city not in self.tomorrow_words and
                        potential_city not in self.week_words):
                    location = potential_city

        # Zeitraum bestimmen
        time_period = "today"
        for word in self.tomorrow_words:
            if word in text:
                time_period = "tomorrow"
                break

        for word in self.week_words:
            if word in text:
                time_period = "week"
                break

        return {
            "is_weather_query": is_weather,
            "location": location,
            "time_period": time_period
        }
