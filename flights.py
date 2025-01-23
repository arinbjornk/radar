import requests
import json
import os
import time
from datetime import datetime

def get_all_flights(username=None, password=None):
    """
    Get all flights with caching using OpenSky REST API
    Returns: list of processed flight data with necessary plotting info
    """
    cache_file = 'flight_cache.json'
    
    # Check if cache exists and is fresh (less than 60 seconds old)
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            
        # Check if cache is fresh (less than 60 seconds old)
        if time.time() - cache_data['timestamp'] < 60*10:
            print("Using cached flight data")
            return cache_data['flights']
    
    # If no cache or cache is stale, fetch new data
    try:
        # Prepare API URL
        base_url = "https://opensky-network.org/api/states/all"
        
        # Make API request (with auth if provided)
        auth = (username, password) if username and password else None
        response = requests.get(base_url, auth=auth)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or 'states' not in data:
            print("No flight data received")
            return []
        
        # Process and store only the data we need for plotting
        flights = []
        for state in data['states']:
            # Only include if we have coordinates
            if state[5] and state[6]:  # longitude and latitude are not null
                flight = {
                    'icao24': state[0],
                    'callsign': state[1].strip() if state[1] else 'N/A',
                    'origin_country': state[2],
                    'longitude': float(state[5]),
                    'latitude': float(state[6]),
                    'altitude': float(state[7] if state[7] else state[13] or 0),  # baro_altitude or geo_altitude
                    'velocity': float(state[9]) if state[9] else 0,
                    'heading': float(state[10]) if state[10] else 0,
                    'on_ground': bool(state[8]),
                    'vertical_rate': float(state[11]) if state[11] else 0
                }
                flights.append(flight)
        
        # Cache the data
        cache_data = {
            'timestamp': time.time(),
            'flights': flights
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        print(f"Retrieved {len(flights)} flights from API")
        return flights
        
    except Exception as e:
        print(f"Error fetching flight data: {e}")
        # If there's an error, try to use cached data even if it's old
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)['flights']
        return []

# Example usage:
if __name__ == "__main__":
    # Get all flights
    flights = get_all_flights()
    
    # Or with authentication for better rate limits
    # flights = get_all_flights(username="your_username", password="your_password")
    
    # Print some sample data
    for flight in flights[:5]:  # First 5 flights
        print(f"Flight {flight['callsign']}: "
              f"Pos({flight['longitude']:.2f}, {flight['latitude']:.2f}), "
              f"Alt: {flight['altitude']:.0f}m, "
              f"Heading: {flight['heading']}°")
