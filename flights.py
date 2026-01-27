import requests
import json
import os
import time
import random
import logging
from datetime import datetime, timedelta
from airports import AIRPORTS

def get_flight_track(icao24):
    """
    Fetch historical track for a specific aircraft (breadcrumbs)
    """
    try:
        url = "https://opensky-network.org/api/tracks/all"
        params = {
            "icao24": icao24,
            "time": 0
        }
        
        logging.debug(f"Fetching flight track for {icao24}: {url}")
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # data['path'] is a list of [time, lat, lon, alt, heading, speed]
            path = data.get('path', [])
            if path:
                logging.debug(f"Found {len(path)} track points")
                # Extract (lat, lon) tuples. API returns (time, lat, lon, ...)
                # We want (lon, lat) for our plotter usually, but let's stick to API naming first
                # API: path point is [time, latitude, longitude, baro_altitude, true_track, on_ground]
                # Return list of {'lat': x, 'lon': y}
                return [{'lat': p[1], 'lon': p[2]} for p in path]
            else:
                logging.debug("Track path was empty")
        else:
            logging.error(f"Error fetching flight track: {response.status_code} {response.text}")
            
    except Exception as e:
        logging.error(f"Exception fetching flight track: {e}")
    
    return []

def get_flight_details(icao24):
    """
    Fetch detailed flight info (route) for a specific aircraft
    """
    try:
        # Look at last 24 hours
        now = int(time.time())
        end_time = now
        begin_time = now - 86400
        
        url = "https://opensky-network.org/api/flights/aircraft"
        params = {
            "icao24": icao24,
            "begin": begin_time,
            "end": end_time
        }
        
        logging.debug(f"Fetching flight details for {icao24}: {url} params={params}")
        
        response = requests.get(url, params=params, timeout=10)
        logging.debug(f"Flight details response: {response.status_code}")
        
        if response.status_code == 200:
            flights = response.json()
            logging.debug(f"Found {len(flights)} flights for {icao24}")
            if flights:
                logging.debug(f"Most recent flight data: {flights[0]}")
                return flights[0]
            else:
                logging.debug("Flights list was empty")
        else:
            logging.error(f"Error fetching flight details: {response.text}")
                
    except Exception as e:
        logging.error(f"Exception fetching flight details: {e}")
        print(f"Error fetching flight details: {e}")
    
    return None

def get_all_flights(username=None, password=None):
    """
    Get all flights with caching using OpenSky REST API
    Returns: list of processed flight data with necessary plotting info
    """
    cache_file = 'flight_cache.json'
    
    # Check if cache exists and is fresh (less than 300 seconds old)
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            
        # Check if cache is fresh (less than 300 seconds old)
        if time.time() - cache_data['timestamp'] < 300:
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
                callsign = state[1].strip() if state[1] else 'N/A'
                
                # We no longer mock routes here. Real routes are fetched on demand.
                origin = None
                dest = None

                flight = {
                    'icao24': state[0],
                    'callsign': callsign,
                    'origin_country': state[2],
                    'longitude': float(state[5]),
                    'latitude': float(state[6]),
                    'altitude': float(state[7] if state[7] else state[13] or 0),  # baro_altitude or geo_altitude
                    'velocity': float(state[9]) if state[9] else 0,
                    'heading': float(state[10]) if state[10] else 0,
                    'on_ground': bool(state[8]),
                    'vertical_rate': float(state[11]) if state[11] else 0,
                    'origin_airport': origin,
                    'destination_airport': dest
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
