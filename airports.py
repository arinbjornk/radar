# Dictionary of major airports with their coordinates (lat, lon)
# Key is ICAO code (e.g. EGLL), but we can also support IATA lookup if needed.
# OpenSky returns ICAO codes.

AIRPORTS = {
    # Europe
    "EGLL": {"lat": 51.4700, "lon": -0.4543, "name": "London Heathrow", "iata": "LHR"},
    "EGKK": {"lat": 51.1481, "lon": -0.1903, "name": "London Gatwick", "iata": "LGW"},
    "LFPG": {"lat": 49.0097, "lon": 2.5479, "name": "Paris Charles de Gaulle", "iata": "CDG"},
    "LFPO": {"lat": 48.7262, "lon": 2.3652, "name": "Paris Orly", "iata": "ORY"},
    "EDDF": {"lat": 50.0379, "lon": 8.5622, "name": "Frankfurt", "iata": "FRA"},
    "EDDM": {"lat": 48.3537, "lon": 11.7750, "name": "Munich", "iata": "MUC"},
    "EHAM": {"lat": 52.3105, "lon": 4.7683, "name": "Amsterdam Schiphol", "iata": "AMS"},
    "LEMD": {"lat": 40.4839, "lon": -3.5680, "name": "Madrid Barajas", "iata": "MAD"},
    "LEBL": {"lat": 41.2974, "lon": 2.0833, "name": "Barcelona El Prat", "iata": "BCN"},
    "LIRF": {"lat": 41.8003, "lon": 12.2389, "name": "Rome Fiumicino", "iata": "FCO"},
    "LSGG": {"lat": 46.2370, "lon": 6.1092, "name": "Geneva", "iata": "GVA"},
    "LSZH": {"lat": 47.4582, "lon": 8.5555, "name": "Zurich", "iata": "ZRH"},
    "EIDW": {"lat": 53.4213, "lon": -6.2701, "name": "Dublin", "iata": "DUB"},
    "EKCH": {"lat": 55.6180, "lon": 12.6508, "name": "Copenhagen", "iata": "CPH"},
    "ENGM": {"lat": 60.1975, "lon": 11.1004, "name": "Oslo", "iata": "OSL"},
    "ESSA": {"lat": 59.6498, "lon": 17.9238, "name": "Stockholm Arlanda", "iata": "ARN"},
    "EFHK": {"lat": 60.3172, "lon": 24.9633, "name": "Helsinki", "iata": "HEL"},
    "LOWW": {"lat": 48.1103, "lon": 16.5697, "name": "Vienna", "iata": "VIE"},
    "BIKF": {"lat": 63.9850, "lon": -22.6056, "name": "Keflavik", "iata": "KEF"},

    # North America
    "KJFK": {"lat": 40.6413, "lon": -73.7781, "name": "New York JFK", "iata": "JFK"},
    "KEWR": {"lat": 40.6895, "lon": -74.1745, "name": "Newark", "iata": "EWR"},
    "KLGA": {"lat": 40.7769, "lon": -73.8740, "name": "LaGuardia", "iata": "LGA"},
    "KLAX": {"lat": 33.9416, "lon": -118.4085, "name": "Los Angeles", "iata": "LAX"},
    "KSFO": {"lat": 37.6213, "lon": -122.3790, "name": "San Francisco", "iata": "SFO"},
    "KORD": {"lat": 41.9742, "lon": -87.9073, "name": "Chicago O'Hare", "iata": "ORD"},
    "KATL": {"lat": 33.6407, "lon": -84.4277, "name": "Atlanta Hartsfield-Jackson", "iata": "ATL"},
    "KMIA": {"lat": 25.7959, "lon": -80.2870, "name": "Miami", "iata": "MIA"},
    "KDFW": {"lat": 32.8998, "lon": -97.0403, "name": "Dallas/Fort Worth", "iata": "DFW"},
    "KDEN": {"lat": 39.8561, "lon": -104.6737, "name": "Denver", "iata": "DEN"},
    "KSEA": {"lat": 47.4502, "lon": -122.3118, "name": "Seattle-Tacoma", "iata": "SEA"},
    "KBOS": {"lat": 42.3656, "lon": -71.0096, "name": "Boston Logan", "iata": "BOS"},
    "CYYZ": {"lat": 43.6777, "lon": -79.6248, "name": "Toronto Pearson", "iata": "YYZ"},
    "CYVR": {"lat": 49.1947, "lon": -123.1762, "name": "Vancouver", "iata": "YVR"},

    # Asia / Middle East
    "OMDB": {"lat": 25.2532, "lon": 55.3657, "name": "Dubai International", "iata": "DXB"},
    "OTHH": {"lat": 25.2731, "lon": 51.6080, "name": "Doha Hamad", "iata": "DOH"},
    "WSSS": {"lat": 1.3644, "lon": 103.9915, "name": "Singapore Changi", "iata": "SIN"},
    "RJTT": {"lat": 35.5494, "lon": 139.7798, "name": "Tokyo Haneda", "iata": "HND"},
    "RJAA": {"lat": 35.7720, "lon": 140.3929, "name": "Tokyo Narita", "iata": "NRT"},
    "VHHH": {"lat": 22.3080, "lon": 113.9185, "name": "Hong Kong", "iata": "HKG"},
    "RKSI": {"lat": 37.4602, "lon": 126.4407, "name": "Seoul Incheon", "iata": "ICN"},
    "ZBAA": {"lat": 40.0799, "lon": 116.6031, "name": "Beijing Capital", "iata": "PEK"},
    "VIDP": {"lat": 28.5562, "lon": 77.1000, "name": "Delhi Indira Gandhi", "iata": "DEL"},
    "VTBS": {"lat": 13.6900, "lon": 100.7501, "name": "Bangkok Suvarnabhumi", "iata": "BKK"},

    # Oceania
    "YSSY": {"lat": -33.9399, "lon": 151.1753, "name": "Sydney", "iata": "SYD"},
    "YMML": {"lat": -37.6690, "lon": 144.8410, "name": "Melbourne", "iata": "MEL"},
    "NZAA": {"lat": -37.0082, "lon": 174.7950, "name": "Auckland", "iata": "AKL"},

    # South America
    "SBGR": {"lat": -23.4356, "lon": -46.4731, "name": "São Paulo/Guarulhos", "iata": "GRU"},
    "SKBO": {"lat": 4.7016, "lon": -74.1469, "name": "Bogota El Dorado", "iata": "BOG"},
    "SCEL": {"lat": -33.3907, "lon": -70.7926, "name": "Santiago", "iata": "SCL"},
    "SAEZ": {"lat": -34.8222, "lon": -58.5358, "name": "Buenos Aires Ezeiza", "iata": "EZE"},

    # Africa
    "FAOR": {"lat": -26.1367, "lon": 28.2411, "name": "Johannesburg O.R. Tambo", "iata": "JNB"},
    "HECA": {"lat": 30.1219, "lon": 31.4056, "name": "Cairo", "iata": "CAI"},
    "HKJK": {"lat": -1.3192, "lon": 36.9275, "name": "Nairobi Jomo Kenyatta", "iata": "NBO"},
    "DNAA": {"lat": 9.0068, "lon": 7.2631, "name": "Abuja", "iata": "ABV"}, # Fixed from Lagos which is DNMM
    "DNMM": {"lat": 6.5774, "lon": 3.3210, "name": "Lagos", "iata": "LOS"}
}

def get_airport(code):
    # Try direct ICAO match
    if code in AIRPORTS:
        return AIRPORTS[code]
    
    # Try IATA match (slower but works)
    for icao, data in AIRPORTS.items():
        if data.get('iata') == code:
            return data
            
    return None