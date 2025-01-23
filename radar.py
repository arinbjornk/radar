import geopandas as gpd
import numpy as np
import sys
import argparse
from shapely.geometry import MultiPolygon, Polygon, LineString
from flights import get_all_flights
from datetime import datetime, timedelta
import requests

# ANSI color codes
BLUE = '\033[38;5;39m'      # Vibrant azure blue for track
GREEN = '\033[38;5;10m'     # Bright lime green for current position
RESET = '\033[0m'           # Reset to default
GRAY = '\033[38;5;245m'     # Medium gray for basemap (not too dim)
BRIGHT_WHITE = '\033[97m'   # Bright white for highlights
CYAN = '\033[38;5;51m'      # Electric cyan (alternative track color)
YELLOW = '\033[38;5;227m'   # Warm yellow (alternative highlight)

# Improved map and aircraft characters
MAP_CHAR = '·'         # Lighter character for map
AIRCRAFT_CHARS = {     # More prominent aircraft symbols
    'N': '▲',
    'NE': '▲',
    'E': '▶',
    'SE': '▼',
    'S': '▼',
    'SW': '▼',
    'W': '◀',
    'NW': '▲'
}

def smooth_track(points, min_distance=0.1):
    """
    Smooth the track by removing points that are too close together
    and interpolating points that are too far apart
    """
    if not points or len(points) < 2:
        return points
        
    smoothed = [points[0]]
    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        
        # Calculate distance between points
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if dist < min_distance:
            continue  # Skip points that are too close
            
        if dist > min_distance * 5:  # If points are too far apart, interpolate
            steps = int(dist / min_distance)
            for j in range(1, steps):
                t = j / steps
                x = x1 + (x2-x1) * t
                y = y1 + (y2-y1) * t
                smoothed.append((x, y))
                
        smoothed.append((x2, y2))
    
    return smoothed

def get_direction_char(x1, y1, x2, y2, prev_x=None, prev_y=None, next_x=None, next_y=None):
    """
    Get the appropriate line character based on the direction between points,
    using compact characters to avoid overlap
    """
    dx = x2 - x1
    dy = y2 - y1
    
    # Use smaller, more compact characters
    if abs(dx) > abs(dy) * 2:
        return '·'  # Horizontal movement (middle dot)
    elif abs(dy) > abs(dx) * 2:
        return '⋅'  # Vertical movement (alternative dot)
    
    # For diagonal movements
    # if dx > 0 and dy < 0:
    #     return '∙'  # Diagonal up-right (bullet dot)
    # elif dx > 0 and dy > 0:
    #     return '∙'  # Diagonal down-right
    # elif dx < 0 and dy < 0:
    #     return '∙'  # Diagonal up-left
    # elif dx < 0 and dy > 0:
    #     return '∙'  # Diagonal down-left
    
    return '·'  # Fallback for small movements

def get_flight_track(callsign):
    """
    Get the track of a specific flight using OpenSky API
    """
    # Get current flight data to find the aircraft's ICAO24
    flights = get_all_flights((35, 71, -25, 45))  # Europe bbox
    target_flight = None
    for flight in flights:
        if flight['callsign'].strip() == callsign.strip():
            target_flight = flight
            break
    
    if not target_flight:
        return None, None

    # Get historical track data
    try:
        # Make request to OpenSky tracks API
        url = f"https://opensky-network.org/api/tracks/all"
        params = {
            "icao24": target_flight['icao24'],
            "time": 0  # 0 means get the current/latest track
        }
        
        # If you have OpenSky credentials, use them like this:
        # response = requests.get(url, params=params, auth=('username', 'password'))
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            track_data = response.json()
            print(f"Got track data: {track_data}")  # Debug print
            # Extract track points from the path array
            # Each point in path is [time, lat, lon, baro_altitude, true_track, on_ground]
            track_points = []
            if 'path' in track_data:
                track_points = [(point[2], point[1]) for point in track_data['path'] 
                              if point[1] is not None and point[2] is not None]
                print(f"Extracted {len(track_points)} track points")  # Debug print
            else:
                print("No path data in response")
                
            if not track_points:  # If no track points, at least show current position
                track_points = [(target_flight['longitude'], target_flight['latitude'])]
            return track_points, target_flight
        else:
            print(f"Error getting track data: {response.status_code}")
            # If we can't get the track, at least return the current position
            return [(target_flight['longitude'], target_flight['latitude'])], target_flight
            
    except Exception as e:
        print(f"Error getting flight track: {e}")
        return [(target_flight['longitude'], target_flight['latitude'])], target_flight

def plot_radar_map(width=120, height=40, map_chars=' x', flight_char='✈', target_callsign=None, region='europe', info=False):
    """
    Creates an ASCII map with flights plotted on top
    If target_callsign is provided, shows only that flight with its track
    
    Args:
        width (int): Width of the map in characters
        height (int): Height of the map in characters
        map_chars (str): Characters to use for the map
        flight_char (str): Character to use for flights
        target_callsign (str): Callsign of a specific flight to track
        region (str): Region to display (europe, north_america, south_america, 
                     asia, africa, australia, antarctica)
    """
    # Bounding boxes for different regions (min_lon, min_lat, max_lon, max_lat)
    REGION_BBOXES = {
        'europe': (-25, 35, 45, 71),
        'north_america': (-168, 5, -52, 83),
        'south_america': (-92, -56, -32, 13),
        'asia': (26, -11, 190, 81),
        'africa': (-26, -35, 60, 38),
        'australia': (112, -55, 180, -10),
        'antarctica': (-180, -90, 180, -60)
    }
    bbox = REGION_BBOXES.get(region, REGION_BBOXES['europe'])
    canvas = [[map_chars[0] for _ in range(width)] for _ in range(height)]
    
    # Load and plot the map
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    world = world.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    
    def plot_coords(coords):
        x_coords = ((coords[:, 0] - bbox[0]) / (bbox[2] - bbox[0]) * width).astype(int)
        y_coords = ((bbox[3] - coords[:, 1]) / (bbox[3] - bbox[1]) * height).astype(int)
        
        for x, y in zip(x_coords, y_coords):
            if 0 <= x < width and 0 <= y < height:
                canvas[y][x] = f"{GRAY}{MAP_CHAR}{RESET}"  # Gray dots for map

    # Plot map
    for idx, country in world.iterrows():
        try:
            if country.geometry is None:
                continue
            
            if isinstance(country.geometry, MultiPolygon):
                for polygon in country.geometry.geoms:
                    coords = np.array(polygon.exterior.coords)
                    plot_coords(coords)
            else:
                coords = np.array(country.geometry.exterior.coords)
                plot_coords(coords)
                    
        except (AttributeError, ValueError) as e:
            print(f"Error processing country {country['NAME']}: {e}")
            continue

    # Get flights data and shuffle to show different flights each time
    flights = get_all_flights((bbox[1], bbox[3], bbox[0], bbox[2]))
    np.random.shuffle(flights)
    
    if target_callsign:
        # Get track for specific flight
        track_points, target_flight = get_flight_track(target_callsign)
        
        if track_points and target_flight:
            # Plot track with directional lines
            track_points = smooth_track(track_points)
            print(f"After smoothing: {len(track_points)} points")  # Debug print
            
            # Draw track line by line
            for i in range(len(track_points)-1):
                x1 = int(((track_points[i][0] - bbox[0]) / 
                        (bbox[2] - bbox[0]) * width))
                y1 = int(((bbox[3] - track_points[i][1]) / 
                        (bbox[3] - bbox[1]) * height))
                
                x2 = int(((track_points[i+1][0] - bbox[0]) / 
                        (bbox[2] - bbox[0]) * width))
                y2 = int(((bbox[3] - track_points[i+1][1]) / 
                        (bbox[3] - bbox[1]) * height))
                
                if (0 <= x1 < width and 0 <= y1 < height and 
                    0 <= x2 < width and 0 <= y2 < height):
                    # Draw line character based on direction
                    line_char = get_direction_char(x1, y1, x2, y2)
                    canvas[y1][x1] = f"{BLUE}{line_char}{RESET}"  # Blue line for track
                    
            # Draw final point of track
            if track_points:
                x = int(((track_points[-1][0] - bbox[0]) / 
                        (bbox[2] - bbox[0]) * width))
                y = int(((bbox[3] - track_points[-1][1]) / 
                        (bbox[3] - bbox[1]) * height))
                if 0 <= x < width and 0 <= y < height:
                    canvas[y][x] = f"{BLUE}•{RESET}"  # Mark end of track
                x1 = int(((track_points[i][0] - bbox[0]) / 
                         (bbox[2] - bbox[0]) * width))
                y1 = int(((bbox[3] - track_points[i][1]) / 
                         (bbox[3] - bbox[1]) * height))
                
                x2 = int(((track_points[i+1][0] - bbox[0]) / 
                         (bbox[2] - bbox[0]) * width))
                y2 = int(((bbox[3] - track_points[i+1][1]) / 
                         (bbox[3] - bbox[1]) * height))
                
                if 0 <= x1 < width and 0 <= y1 < height:
                    line_char = get_direction_char(x1, y1, x2, y2)
                    canvas[y1][x1] = f"{BLUE}{line_char}{RESET}"  # Blue line for track
            
            # Plot current position with direction
            x = int(((target_flight['longitude'] - bbox[0]) / 
                     (bbox[2] - bbox[0]) * width))
            y = int(((bbox[3] - target_flight['latitude']) / 
                     (bbox[3] - bbox[1]) * height))
            
            if 0 <= x < width and 0 <= y < height:
                if target_flight['heading'] is not None:
                    # Convert heading to cardinal direction
                    heading = target_flight['heading']
                    if heading < 22.5 or heading >= 337.5:
                        direction = 'E'
                    elif heading < 67.5:
                        direction = 'NE'
                    elif heading < 112.5:
                        direction = 'N'
                    elif heading < 157.5:
                        direction = 'NW'
                    elif heading < 202.5:
                        direction = 'W'
                    elif heading < 247.5:
                        direction = 'SW'
                    elif heading < 292.5:
                        direction = 'S'
                    else:
                        direction = 'SE'
                    flight_symbol = AIRCRAFT_CHARS[direction]
                else:
                    flight_symbol = AIRCRAFT_CHARS['E']  # Default direction
                
                if info:
                    canvas[y][x] = f"{GREEN}{flight_symbol}{RESET} {YELLOW}{flight['callsign']}{RESET}"
                else:
                    canvas[y][x] = f"{GREEN}{flight_symbol}{RESET}"  # Green for current position
            
            flights = [target_flight]  # Only show info for target flight
        else:
            print(f"Flight {target_callsign} not found")
            flights = []
    else:
        # Filter flights to only those within bounding box
        filtered_flights = []
        for flight in flights:
            if (bbox[0] <= flight['longitude'] <= bbox[2] and 
                bbox[1] <= flight['latitude'] <= bbox[3]):
                filtered_flights.append(flight)
                if len(filtered_flights) >= 5:
                    break
                    
        flights = filtered_flights  # Update main flights list

        # Plot first 5 flights within bounding box
        for flight in flights:
            x = int(((flight['longitude'] - bbox[0]) / 
                     (bbox[2] - bbox[0]) * width))
            y = int(((bbox[3] - flight['latitude']) / 
                     (bbox[3] - bbox[1]) * height))
            
            if 0 <= x < width and 0 <= y < height:
                if flight['heading'] is not None:
                    direction_chars = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘']
                    direction_idx = int(((flight['heading'] + 22.5) % 360) / 45)
                    flight_symbol = direction_chars[direction_idx]
                else:
                    flight_symbol = flight_char
                    
                if info:
                    canvas[y][x] = f"{GREEN}{flight_symbol}{RESET} {YELLOW}{flight['callsign']}{RESET}"
                else:
                    canvas[y][x] = f"{GREEN}{flight_symbol}{RESET}"  # Green for current position

    # Convert canvas to string
    map_str = '\n'.join([''.join(row) for row in canvas])
    
    # Print flight details below the map
    flight_info = f"\n{BRIGHT_WHITE}Active Flights:{RESET}\n"
    for flight in flights:
        if flight['callsign'] != 'N/A':
            # Add aircraft category description
            category_desc = {
                0: "No info",
                2: "Light",
                3: "Small",
                4: "Large",
                5: "High Vortex Large",
                6: "Heavy",
                7: "High Performance",
                8: "Rotorcraft",
                9: "Glider",
                10: "Lighter-than-air"
            }.get(flight.get('category', 0), "Unknown")
            
            # Format airspeed from m/s to knots
            speed_kts = int(flight['velocity'] * 1.944) if flight.get('velocity') else 0
            
            flight_info += (f"{YELLOW}{flight['callsign']}{RESET} "
                          f"FL{int(flight['altitude']/100):03d} "  # Flight level
                          f"{speed_kts}kts "
                          f"Hdg:{int(flight['heading'])}° "
                          f"{flight['latitude']:.2f}°N, {flight['longitude']:.2f}°E\n")
    
    return map_str + flight_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display a real-time radar map of flights",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-c', '--callsign',
        help="Callsign of a specific flight to track (e.g. SWA123)"
    )
    parser.add_argument(
        '-r', '--region',
        default='europe',
        help="""Region to display. Options:
    europe (default)
    north_america
    south_america
    asia
    africa
    australia
    antarctica"""
    )
    parser.add_argument(
        '-i', '--info',
        action='store_true',
        help="Show callsigns next to aircraft markers"
    )
    
    args = parser.parse_args()
    radar_display = plot_radar_map(target_callsign=args.callsign, region=args.region, info=args.info)
    print(radar_display)
