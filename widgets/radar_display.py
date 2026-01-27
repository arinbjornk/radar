from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text
import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon
import math
from airports import AIRPORTS

# Constants
REGION_BBOXES = {
    'europe': (-25, 35, 45, 71),
    'north_america': (-168, 5, -52, 83),
    'south_america': (-92, -56, -32, 13),
    'asia': (26, -11, 190, 81),
    'africa': (-26, -35, 60, 38),
    'australia': (112, -55, 180, -10),
    'antarctica': (-180, -90, 180, -60)
}

AIRCRAFT_CHARS = {
    'N': '↑', 'NE': '↗', 'E': '→', 'SE': '↘',
    'S': '↓', 'SW': '↙', 'W': '←', 'NW': '↖'
}

class RadarDisplayWidget(Static):
    """Widget for displaying the radar map with flights"""

    map_region = reactive('europe')
    tracked_flight = reactive(None)
    can_focus = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.map_width = 110
        self.map_height = 35
        self.map_char = '.'
        
        # Viewport state
        self.zoom_level = 1.0
        self.center_lat = None
        self.center_lon = None
        
        # Caching state
        self._background_cache = None
        self._last_render_params = None # (region, zoom, center_lat, center_lon)
        self._world_data = None
        
        # Display mode
        self.single_flight_mode = False
        self.track_history = []

    def on_mount(self):
        """Called when widget is mounted"""
        self._center_on_region()
        # Initialize empty display
        self.update_flights([])

    def smooth_track(self, points, min_distance=0.1):
        """
        Smooth the track by removing points that are too close together
        and interpolating points that are too far apart.
        Points is a list of dicts {'lat': lat, 'lon': lon}
        """
        if not points or len(points) < 2:
            return points
            
        smoothed = [points[0]]
        for i in range(1, len(points)):
            p1 = points[i-1]
            p2 = points[i]
            
            # Calculate distance between points
            dist = math.sqrt((p2['lon']-p1['lon'])**2 + (p2['lat']-p1['lat'])**2)
            
            if dist < min_distance:
                continue  # Skip points that are too close
                
            if dist > min_distance * 5:  # If points are too far apart, interpolate
                steps = int(dist / min_distance)
                for j in range(1, steps):
                    t = j / steps
                    lat = p1['lat'] + (p2['lat']-p1['lat']) * t
                    lon = p1['lon'] + (p2['lon']-p1['lon']) * t
                    smoothed.append({'lat': lat, 'lon': lon})
                    
            smoothed.append(p2)
        
        return smoothed

    def get_direction_char(self, x1, y1, x2, y2):
        """
        Get the appropriate line character based on the direction between points
        """
        dx = x2 - x1
        dy = y2 - y1
        
        # Use smaller, more compact characters
        if abs(dx) > abs(dy) * 2:
            return '·'  # Horizontal movement
        elif abs(dy) > abs(dx) * 2:
            return '⋅'  # Vertical movement
        
        return '·'  # Fallback

    def watch_map_region(self, new_region: str):
        """Called when region changes"""
        self.zoom_level = 1.0
        self._center_on_region()
        self._background_cache = None

    def watch_tracked_flight(self, new_tracked_flight: str):
        """Called when tracked flight changes"""
        # If we lose tracking, disable single flight mode and clear history
        if not new_tracked_flight:
            self.single_flight_mode = False
            self.track_history = []

    def _center_on_region(self):
        """Center the map on the current region's center point"""
        if self.map_region not in REGION_BBOXES:
            return
            
        bbox = REGION_BBOXES[self.map_region]
        self.center_lat = (bbox[1] + bbox[3]) / 2
        self.center_lon = (bbox[0] + bbox[2]) / 2

    def get_current_bbox(self):
        """Calculate current bounding box based on zoom and center"""
        region_bbox = REGION_BBOXES[self.map_region]

        # Calculate base width/height of region
        base_width = region_bbox[2] - region_bbox[0]
        base_height = region_bbox[3] - region_bbox[1]

        # Apply zoom
        current_width = base_width / self.zoom_level
        current_height = base_height / self.zoom_level

        # Calculate bounds around center point
        min_lon = self.center_lon - current_width / 2
        max_lon = self.center_lon + current_width / 2
        min_lat = self.center_lat - current_height / 2
        max_lat = self.center_lat + current_height / 2

        return (min_lon, min_lat, max_lon, max_lat)

    def to_mercator(self, lon, lat):
        """Convert lon/lat to Mercator coordinates"""
        # Handle numpy arrays or scalars
        lat = np.clip(lat, -89.9, 89.9)
        x = lon
        y = np.degrees(np.log(np.tan(np.pi/4 + np.radians(lat)/2)))
        return x, y

    def _load_world_data(self):
        """Lazy load world data"""
        if self._world_data is None:
            try:
                url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
                self._world_data = gpd.read_file(url)
            except Exception:
                self._world_data = None

    def _render_background(self, bbox):
        """Render the static map elements to a character grid"""
        self._load_world_data()
        
        # Create empty canvas
        canvas = [[' ' for _ in range(self.map_width)] for _ in range(self.map_height)]
        
        if self._world_data is None:
            return canvas

        # Mercator bounds
        min_lon, min_lat, max_lon, max_lat = bbox
        min_mx, min_my = self.to_mercator(min_lon, min_lat)
        max_mx, max_my = self.to_mercator(max_lon, max_lat)
        merc_bbox = (min_mx, min_my, max_mx, max_my)

        # Clip and draw
        try:
            # Spatial index query for performance
            world_clip = self._world_data.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

            for _, country in world_clip.iterrows():
                if country.geometry is None:
                    continue

                geoms = [country.geometry]
                if isinstance(country.geometry, MultiPolygon):
                    geoms = country.geometry.geoms

                for polygon in geoms:
                    coords = np.array(polygon.exterior.coords)
                    self._plot_coords(coords, canvas, merc_bbox)
        except Exception:
            pass
            
        return canvas

    def _plot_coords(self, coords, canvas, merc_bbox):
        """Plot coordinates on canvas using Mercator projection"""
        min_mx, min_my, max_mx, max_my = merc_bbox
        
        lons = coords[:, 0]
        lats = coords[:, 1]
        mx, my = self.to_mercator(lons, lats)
        
        width_mx = max_mx - min_mx
        height_my = max_my - min_my
        
        if width_mx == 0 or height_my == 0:
            return

        # Vectorized calculation
        x_coords = ((mx - min_mx) / width_mx * self.map_width).astype(int)
        y_coords = ((max_my - my) / height_my * self.map_height).astype(int)
        
        # Draw lines between consecutive points
        for i in range(len(x_coords) - 1):
            self._draw_pixel_line(canvas, x_coords[i], y_coords[i], x_coords[i+1], y_coords[i+1])

    def _draw_pixel_line(self, canvas, x0, y0, x1, y1):
        """Draw a line between two pixel coordinates using Bresenham's algorithm"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if 0 <= x0 < self.map_width and 0 <= y0 < self.map_height:
                canvas[y0][x0] = self.map_char

            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def update_flights(self, flights):
        """Update the display with new flight data"""
        bbox = self.get_current_bbox()
        
        # Check if we need to re-render background
        current_params = (self.map_region, self.zoom_level, self.center_lat, self.center_lon)
        if self._background_cache is None or self._last_render_params != current_params:
            self._background_cache = self._render_background(bbox)
            self._last_render_params = current_params

        # Copy background
        canvas = [row[:] for row in self._background_cache]
        
        # Mercator bounds for flight plotting
        min_lon, min_lat, max_lon, max_lat = bbox
        min_mx, min_my = self.to_mercator(min_lon, min_lat)
        max_mx, max_my = self.to_mercator(max_lon, max_lat)
        merc_bbox = (min_mx, min_my, max_mx, max_my)

        # Determine which flights to show
        visible_flights = []
        target_flight_obj = None

        if self.single_flight_mode and self.tracked_flight:
             for flight in flights:
                 if flight['callsign'].strip() == self.tracked_flight:
                     target_flight_obj = flight
                     break
             
             if target_flight_obj:
                 visible_flights = [target_flight_obj]
                 self._draw_route(target_flight_obj, canvas, merc_bbox)
                 self._plot_single_flight(target_flight_obj, canvas, merc_bbox)
        else:
            # Show all flights (filtered by viewport)
            for flight in flights:
                if not (bbox[0] <= flight['longitude'] <= bbox[2] and 
                       bbox[1] <= flight['latitude'] <= bbox[3]):
                    continue
                    
                visible_flights.append(flight)
                self._plot_single_flight(flight, canvas, merc_bbox)

        # Convert to Text
        self.update(self._canvas_to_rich_text(canvas))
        return visible_flights

    def _draw_route(self, flight, canvas, merc_bbox):
        """Draw the route line and airports"""
        origin_code = flight.get('origin_airport')
        dest_code = flight.get('destination_airport')
        
        # 1. Draw History (Past)
        if self.track_history and len(self.track_history) > 1:
            # Smooth the track
            smoothed_history = self.smooth_track(self.track_history)
            
            # Draw line segments from track history
            for i in range(len(smoothed_history) - 1):
                p1 = smoothed_history[i]
                p2 = smoothed_history[i+1]
                
                # Convert to canvas coords to calculate direction char
                min_mx, min_my, max_mx, max_my = merc_bbox
                width_mx = max_mx - min_mx
                height_my = max_my - min_my
                
                mx1, my1 = self.to_mercator(p1['lon'], p1['lat'])
                x1 = int(((mx1 - min_mx) / width_mx * self.map_width))
                y1 = int(((max_my - my1) / height_my * self.map_height))
                
                mx2, my2 = self.to_mercator(p2['lon'], p2['lat'])
                x2 = int(((mx2 - min_mx) / width_mx * self.map_width))
                y2 = int(((max_my - my2) / height_my * self.map_height))
                
                char = self.get_direction_char(x1, y1, x2, y2)
                
                self._plot_line_mercator(
                    p1['lon'], p1['lat'], 
                    p2['lon'], p2['lat'], 
                    canvas, merc_bbox, char=char
                )
                
            # Draw line from last track point to current position
            last_p = smoothed_history[-1]
            self._plot_line_mercator(
                last_p['lon'], last_p['lat'],
                flight['longitude'], flight['latitude'],
                canvas, merc_bbox, char='·'
            )
        elif origin_code:
            # Fallback: Straight line from Origin -> Plane if no track data
            origin = AIRPORTS.get(origin_code)
            if origin:
                self._plot_line_mercator(
                    origin['lon'], origin['lat'], 
                    flight['longitude'], flight['latitude'], 
                    canvas, merc_bbox, char='*'
                )

        # 2. Draw Future (Plane -> Dest)
        if dest_code:
            dest = AIRPORTS.get(dest_code)
            if dest:
                self._plot_line_mercator(
                    flight['longitude'], flight['latitude'],
                    dest['lon'], dest['lat'], 
                    canvas, merc_bbox, char='·'
                )
        
        # 3. Labels
        if origin_code:
            origin = AIRPORTS.get(origin_code)
            if origin:
                self._plot_text_mercator(origin['lon'], origin['lat'], origin_code, canvas, merc_bbox)
        
        if dest_code:
            dest = AIRPORTS.get(dest_code)
            if dest:
                self._plot_text_mercator(dest['lon'], dest['lat'], dest_code, canvas, merc_bbox)

    def _plot_line_mercator(self, lon1, lat1, lon2, lat2, canvas, merc_bbox, char='.'):
        """Draw a line between two geo coordinates"""
        # Simple interpolation (straight line on Mercator map)
        steps = 20
        lons = np.linspace(lon1, lon2, steps)
        lats = np.linspace(lat1, lat2, steps)
        
        coords = np.column_stack((lons, lats))
        
        # Manually plot points
        min_mx, min_my, max_mx, max_my = merc_bbox
        width_mx = max_mx - min_mx
        height_my = max_my - min_my
        
        mx, my = self.to_mercator(lons, lats)
        
        x_coords = ((mx - min_mx) / width_mx * self.map_width).astype(int)
        y_coords = ((max_my - my) / height_my * self.map_height).astype(int)
        
        for x, y in zip(x_coords, y_coords):
             if 0 <= x < self.map_width and 0 <= y < self.map_height:
                 if canvas[y][x] == ' ': # Don't overwrite existing features
                     canvas[y][x] = char

    def _plot_text_mercator(self, lon, lat, text, canvas, merc_bbox):
        """Plot text label at coordinate"""
        min_mx, min_my, max_mx, max_my = merc_bbox
        width_mx = max_mx - min_mx
        height_my = max_my - min_my
        
        mx, my = self.to_mercator(lon, lat)
        x = int(((mx - min_mx) / width_mx * self.map_width))
        y = int(((max_my - my) / height_my * self.map_height))
        
        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            # Center text
            start_x = x - len(text) // 2
            for i, char in enumerate(text):
                px = start_x + i
                if 0 <= px < self.map_width:
                    canvas[y][px] = char

    def _plot_single_flight(self, flight, canvas, merc_bbox):
        min_mx, min_my, max_mx, max_my = merc_bbox
        width_mx = max_mx - min_mx
        height_my = max_my - min_my

        fx, fy = self.to_mercator(flight['longitude'], flight['latitude'])
        
        x = int(((fx - min_mx) / width_mx * self.map_width))
        y = int(((max_my - fy) / height_my * self.map_height))

        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            heading = flight.get('heading', 0)
            # Calculate direction symbol
            idx = int(((heading + 22.5) % 360) / 45)
            direction = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][idx]
            symbol = AIRCRAFT_CHARS[direction]
            
            # Highlight tracked flight
            if self.tracked_flight and flight['callsign'].strip() == self.tracked_flight:
                canvas[y][x] = 'T' # Placeholder for tracked
            else:
                canvas[y][x] = symbol

    def _canvas_to_rich_text(self, canvas) -> Text:
        """Convert canvas to Rich Text with colors"""
        text = Text()
        
        # Add Header
        if self.single_flight_mode and self.tracked_flight:
             text.append(f"TRACKING: {self.tracked_flight} | Zoom: {self.zoom_level:.1f}x\n", style="bold #E0AF68")
        else:
             text.append(f"RADAR: {self.map_region.upper()} | Zoom: {self.zoom_level:.1f}x\n", style="bold #9ECE6A")
        
        for row in canvas:
            for char in row:
                if char == '.':
                    text.append(char, style="#414868") # Map background dots
                elif char in AIRCRAFT_CHARS.values():
                    text.append(char, style="bold #7DCFFF") # Aircraft
                elif char == 'T':
                    text.append('✈', style="bold #F7768E reverse") # Tracked
                elif char == '•':
                    text.append('·', style="#565f89") # Future route (displayed as dot but darker)
                elif char == '·' or char == '⋅':
                    text.append(char, style="bold #7AA2F7") # Past route (blue)
                elif char == '*':
                    text.append('*', style="bold #E0AF68") # Fallback past route
                elif char.isalnum() and char not in AIRCRAFT_CHARS.values(): # Airport codes
                    text.append(char, style="bold #c0caf5 on #7AA2F7")
                else:
                    text.append(char)
            text.append('\n')
            
        return text

    # Control methods
    def zoom_in(self):
        self.zoom_level = min(self.zoom_level * 1.5, 20.0)
        
    def zoom_out(self):
        self.zoom_level = max(self.zoom_level / 1.5, 0.5)

    def pan(self, direction):
        step_lat = (REGION_BBOXES[self.map_region][3] - REGION_BBOXES[self.map_region][1]) / self.zoom_level * 0.1
        step_lon = (REGION_BBOXES[self.map_region][2] - REGION_BBOXES[self.map_region][0]) / self.zoom_level * 0.1
        
        if direction == 'north': self.center_lat += step_lat
        elif direction == 'south': self.center_lat -= step_lat
        elif direction == 'west': self.center_lon -= step_lon
        elif direction == 'east': self.center_lon += step_lon