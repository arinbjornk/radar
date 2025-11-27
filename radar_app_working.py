from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Static, Label
from textual.binding import Binding
from datetime import datetime
from rich.text import Text
import asyncio
import geopandas as gpd
import numpy as np
import random
from shapely.geometry import MultiPolygon
from flights import get_all_flights


class RadarApp(App):
    """Flight radar application using Textual"""

    CSS = """
    Screen {
        layout: vertical;
    }

    #main-container {
        layout: horizontal;
        height: 1fr;
    }

    #radar-display {
        width: 3fr;
        border: solid green;
        margin: 1;
    }

    #flight-info {
        width: 1fr;
        border: solid blue;
        margin: 1;
    }

    #status-bar {
        height: 3;
        background: $primary;
        color: white;
        content-align: center middle;
    }
    """

    BINDINGS = [
        Binding("r", "change_region", "Region"),
        Binding("t", "track_flight", "Track"),
        Binding("q", "quit", "Quit"),

        # Zoom controls
        Binding("plus,equal", "zoom_in", "Zoom In"),
        Binding("minus", "zoom_out", "Zoom Out"),

        # Pan controls
        Binding("up", "pan_north", "Pan North"),
        Binding("down", "pan_south", "Pan South"),
        Binding("left", "pan_west", "Pan West"),
        Binding("right", "pan_east", "Pan East"),

        # Utility
        Binding("c", "center_map", "Center"),
        Binding("f", "fit_region", "Fit Region"),
    ]

    def __init__(self):
        super().__init__()
        self.regions = ['europe', 'north_america', 'south_america', 'asia', 'africa', 'australia', 'antarctica']
        self.current_region_index = 0
        self.tracked_flight = None
        self.current_flights = []  # Store current flights for display

        # Region bounding boxes (min_lon, min_lat, max_lon, max_lat)
        self.region_bboxes = {
            'europe': (-25, 35, 45, 71),
            'north_america': (-168, 5, -52, 83),
            'south_america': (-92, -56, -32, 13),
            'asia': (26, -11, 190, 81),
            'africa': (-26, -35, 60, 38),
            'australia': (112, -55, 180, -10),
            'antarctica': (-180, -90, 180, -60)
        }

        # Zoom and pan state
        self.zoom_level = 1.0
        self.center_lat = None
        self.center_lon = None
        self.follow_mode = False

        # Persistent active flights list
        self.active_flights = []  # Flights currently being displayed
        self.max_active_flights = 8  # Maximum flights to show

        # Map and flight rendering settings
        self.map_width = 110   # Much larger to fill the green panel
        self.map_height = 35   # Much larger to fill the green panel
        self.map_char = '·'
        self.aircraft_chars = {
            'N': '↑', 'NE': '↗', 'E': '→', 'SE': '↘',
            'S': '↓', 'SW': '↙', 'W': '←', 'NW': '↖'
        }

        # Initialize map center to current region
        self._center_on_region()

    def _center_on_region(self):
        """Center the map on the current region's center point"""
        bbox = self.region_bboxes[self.regions[self.current_region_index]]
        self.center_lat = (bbox[1] + bbox[3]) / 2  # Average of min and max lat
        self.center_lon = (bbox[0] + bbox[2]) / 2  # Average of min and max lon

    def get_current_bbox(self):
        """Calculate current bounding box based on zoom and center"""
        region_bbox = self.region_bboxes[self.regions[self.current_region_index]]

        # Calculate base width/height of region
        base_width = region_bbox[2] - region_bbox[0]  # max_lon - min_lon
        base_height = region_bbox[3] - region_bbox[1]  # max_lat - min_lat

        # Apply zoom (higher zoom = smaller area)
        current_width = base_width / self.zoom_level
        current_height = base_height / self.zoom_level

        # Calculate bounds around center point
        min_lon = self.center_lon - current_width / 2
        max_lon = self.center_lon + current_width / 2
        min_lat = self.center_lat - current_height / 2
        max_lat = self.center_lat + current_height / 2

        return (min_lon, min_lat, max_lon, max_lat)

    def _get_pan_step(self):
        """Calculate pan step size based on current zoom"""
        region_bbox = self.region_bboxes[self.regions[self.current_region_index]]
        base_height = region_bbox[3] - region_bbox[1]
        return (base_height / self.zoom_level) * 0.15  # 15% of current view

    def _clamp_to_region_bounds(self):
        """Ensure center point doesn't pan outside reasonable bounds"""
        region_bbox = self.region_bboxes[self.regions[self.current_region_index]]

        # Add padding to prevent panning completely outside region
        padding_lat = (region_bbox[3] - region_bbox[1]) * 0.7
        padding_lon = (region_bbox[2] - region_bbox[0]) * 0.7

        self.center_lat = max(min(self.center_lat, region_bbox[3] + padding_lat),
                             region_bbox[1] - padding_lat)
        self.center_lon = max(min(self.center_lon, region_bbox[2] + padding_lon),
                             region_bbox[0] - padding_lon)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container(id="main-container"):
            yield Static("", id="radar-display")
            yield Static("", id="flight-info")

        yield Static("", id="status-bar")

    def on_mount(self):
        """Called when app is mounted"""
        self.set_interval(10.0, self.refresh_displays)
        self.refresh_displays()

    def refresh_displays(self):
        """Refresh both radar and flight info displays"""
        # Update radar display
        radar_content = self.create_radar_content()
        self.query_one("#radar-display", Static).update(radar_content)

        # Update flight info
        flight_content = self.create_flight_info_content()
        self.query_one("#flight-info", Static).update(flight_content)

        # Update status bar
        status_content = self.create_status_content()
        self.query_one("#status-bar", Static).update(status_content)

    def create_radar_content(self) -> Text:
        """Create radar display content with real map"""
        content = Text()

        # Minimal header
        region_name = self.regions[self.current_region_index].upper()
        if self.tracked_flight:
            header = f"🎯 {region_name} - TRACKING {self.tracked_flight}"
        else:
            header = f"📡 {region_name} RADAR"

        content.append(header + "\n", style="bold green")

        # Generate the actual map (takes up most of the space)
        try:
            map_content = self.generate_map()
            content.append(map_content)
        except Exception as e:
            content.append(f"Map error: {str(e)}\n", style="red")
            # Minimal fallback
            content.append("Map unavailable", style="dim")

        return content

    def to_mercator(self, lon, lat):
        """Convert lon/lat to Mercator coordinates"""
        # Handle numpy arrays or scalars
        lat = np.clip(lat, -89.9, 89.9) # Clip to avoid infinity
        x = lon
        y = np.degrees(np.log(np.tan(np.pi/4 + np.radians(lat)/2)))
        return x, y

    def generate_map(self) -> Text:
        """Generate the actual radar map with flights"""
        bbox = self.get_current_bbox()
        
        # Calculate Mercator bounds for the viewport
        min_lon, min_lat, max_lon, max_lat = bbox
        min_mx, min_my = self.to_mercator(min_lon, min_lat)
        max_mx, max_my = self.to_mercator(max_lon, max_lat)
        merc_bbox = (min_mx, min_my, max_mx, max_my)

        # Create canvas
        canvas = [[' ' for _ in range(self.map_width)] for _ in range(self.map_height)]

        # Load and plot map borders
        try:
            url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            world = gpd.read_file(url)
            world = world.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

            # Plot country borders
            for idx, country in world.iterrows():
                try:
                    if country.geometry is None:
                        continue

                    if isinstance(country.geometry, MultiPolygon):
                        for polygon in country.geometry.geoms:
                            coords = np.array(polygon.exterior.coords)
                            self._plot_coords(coords, canvas, merc_bbox)
                    else:
                        coords = np.array(country.geometry.exterior.coords)
                        self._plot_coords(coords, canvas, merc_bbox)
                except (AttributeError, ValueError):
                    continue

        except Exception as e:
            # If map loading fails, just continue with flights
            pass

        # Get and plot flights
        try:
            region_bbox = self.region_bboxes[self.regions[self.current_region_index]]
            flights = get_all_flights((region_bbox[1], region_bbox[3], region_bbox[0], region_bbox[2]))

            if self.tracked_flight:
                # Show only tracked flight
                target_flight = None
                for flight in flights:
                    if flight['callsign'].strip() == self.tracked_flight.strip():
                        target_flight = flight
                        break

                if target_flight:
                    self._plot_flight(target_flight, canvas, merc_bbox)
                    flights_to_show = [target_flight]
                else:
                    flights_to_show = []
            else:
                # Update active flights list intelligently
                flights_to_show = self._update_active_flights(flights, bbox)

                # Plot all active flights
                for flight in flights_to_show:
                    self._plot_flight(flight, canvas, merc_bbox)

        except Exception as e:
            flights_to_show = []

        # Convert canvas to Rich Text
        map_text = self._canvas_to_rich_text(canvas)

        # Store flights data for the flight info panel to use
        self.current_flights = flights_to_show

        return map_text

    def _update_active_flights(self, all_flights, current_bbox):
        """Intelligently update the active flights list to maintain consistency"""
        # Create a lookup dict for current flights by callsign for fast access
        current_flights_dict = {flight['callsign'].strip(): flight for flight in all_flights if flight['callsign'] and flight['callsign'] != 'N/A'}

        # Step 1: Check which active flights are still visible and update their data
        still_visible = []
        for active_flight in self.active_flights:
            callsign = active_flight['callsign'].strip()
            if callsign in current_flights_dict:
                updated_flight = current_flights_dict[callsign]
                # Check if still in current view
                if (current_bbox[0] <= updated_flight['longitude'] <= current_bbox[2] and
                    current_bbox[1] <= updated_flight['latitude'] <= current_bbox[3]):
                    still_visible.append(updated_flight)

        # Step 2: Find new flights that could be added (not already active and in view)
        active_callsigns = {flight['callsign'].strip() for flight in still_visible}
        available_new_flights = []
        for flight in all_flights:
            if (flight['callsign'] and flight['callsign'] != 'N/A' and
                flight['callsign'].strip() not in active_callsigns and
                current_bbox[0] <= flight['longitude'] <= current_bbox[2] and
                current_bbox[1] <= flight['latitude'] <= current_bbox[3]):
                available_new_flights.append(flight)

        # Step 3: Fill up to max_active_flights with new flights if needed
        active_flights_result = still_visible[:]
        slots_available = self.max_active_flights - len(active_flights_result)

        if slots_available > 0 and available_new_flights:
            # Randomly select from available new flights to fill remaining slots
            random.shuffle(available_new_flights)
            active_flights_result.extend(available_new_flights[:slots_available])

        # Update the persistent active flights list
        self.active_flights = active_flights_result
        return active_flights_result

    def _plot_coords(self, coords, canvas, merc_bbox):
        """Plot coordinates on canvas using Mercator projection"""
        min_mx, min_my, max_mx, max_my = merc_bbox
        
        # Project coordinates
        lons = coords[:, 0]
        lats = coords[:, 1]
        mx, my = self.to_mercator(lons, lats)
        
        # Scale to canvas
        # Check for zero division or empty range
        width_mx = max_mx - min_mx
        height_my = max_my - min_my
        
        if width_mx == 0 or height_my == 0:
            return

        x_coords = ((mx - min_mx) / width_mx * self.map_width).astype(int)
        y_coords = ((max_my - my) / height_my * self.map_height).astype(int)

        for x, y in zip(x_coords, y_coords):
            if 0 <= x < self.map_width and 0 <= y < self.map_height:
                canvas[y][x] = self.map_char

    def _draw_line(self, canvas, x1, y1, x2, y2):
        """Draw a line between two points on the canvas"""
        # Simple line drawing algorithm to fill gaps between border points
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Handle cases where points are the same or very close
        if dx == 0 and dy == 0:
            if 0 <= x1 < self.map_width and 0 <= y1 < self.map_height:
                canvas[y1][x1] = self.map_char
            return

        x, y = x1, y1
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1

        # Plot starting point
        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            canvas[y][x] = self.map_char

        # Handle different slope cases
        if dx > dy:
            # More horizontal than vertical
            error = dx / 2
            while x != x2:
                error -= dy
                if error < 0:
                    y += y_inc
                    error += dx
                x += x_inc
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    canvas[y][x] = self.map_char
        else:
            # More vertical than horizontal
            error = dy / 2
            while y != y2:
                error -= dx
                if error < 0:
                    x += x_inc
                    error += dy
                y += y_inc
                if 0 <= x < self.map_width and 0 <= y < self.map_height:
                    canvas[y][x] = self.map_char

    def _plot_flight(self, flight, canvas, merc_bbox):
        """Plot a single flight on the canvas using Mercator projection"""
        min_mx, min_my, max_mx, max_my = merc_bbox
        
        # Project flight coordinates
        fx, fy = self.to_mercator(flight['longitude'], flight['latitude'])
        
        width_mx = max_mx - min_mx
        height_my = max_my - min_my
        
        if width_mx == 0 or height_my == 0:
            return

        x = int(((fx - min_mx) / width_mx * self.map_width))
        y = int(((max_my - fy) / height_my * self.map_height))

        if 0 <= x < self.map_width and 0 <= y < self.map_height:
            if flight['heading'] is not None:
                # Get direction character based on heading
                heading = flight['heading']
                dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                idx = int(((heading + 22.5) % 360) / 45)
                direction = dirs[idx]
                flight_symbol = self.aircraft_chars[direction]
            else:
                flight_symbol = self.aircraft_chars['E']  # Default direction

            canvas[y][x] = flight_symbol

    def _canvas_to_rich_text(self, canvas) -> Text:
        """Convert canvas to Rich Text with colors"""
        text = Text()

        for row in canvas:
            for char in row:
                if char == self.map_char:
                    text.append(char, style="white")
                elif char in self.aircraft_chars.values():
                    text.append(char, style="bold green")
                else:
                    text.append(char)
            text.append('\n')

        return text

    def create_flight_info_content(self) -> Text:
        """Create flight info panel content with real data"""
        content = Text()

        content.append("FLIGHT INFO\n", style="bold cyan")
        content.append("═" * 15 + "\n", style="cyan")
        content.append("\n")

        # Get current region flights
        try:
            region_bbox = self.region_bboxes[self.regions[self.current_region_index]]
            flights = get_all_flights((region_bbox[1], region_bbox[3], region_bbox[0], region_bbox[2]))

            # Use current view bbox for filtering visible flights
            current_bbox = self.get_current_bbox()

            if self.tracked_flight:
                # Show detailed info for tracked flight
                target_flight = None
                for flight in flights:
                    if flight['callsign'].strip() == self.tracked_flight.strip():
                        target_flight = flight
                        break

                if target_flight:
                    content.append(f"Callsign:\n{target_flight['callsign']}\n\n", style="bold yellow")

                    # Flight details
                    alt_display = f"FL{int(target_flight['altitude']/100):03d}" if target_flight['altitude'] else "N/A"
                    speed_display = f"{int(target_flight['velocity'] * 1.944)}kts" if target_flight['velocity'] else "N/A"
                    heading_display = f"{int(target_flight['heading'])}°" if target_flight['heading'] else "N/A"

                    content.append(f"{alt_display}  {speed_display}\n", style="white")
                    content.append(f"HDG: {heading_display}\n", style="white")
                    content.append(f"{target_flight['latitude']:.2f}°N\n{target_flight['longitude']:.2f}°E\n", style="green")
                    content.append(f"\nCountry:\n{target_flight['origin_country']}\n", style="dim")
                else:
                    content.append(f"Flight {self.tracked_flight}\nnot found", style="red")
            else:
                # Show region overview
                content.append(f"Region:\n{self.regions[self.current_region_index].title()}\n\n", style="green")

                # Count flights in current view
                visible_flights = []
                for flight in flights:
                    if (current_bbox[0] <= flight['longitude'] <= current_bbox[2] and
                        current_bbox[1] <= flight['latitude'] <= current_bbox[3]):
                        visible_flights.append(flight)

                content.append(f"Visible flights: {len(visible_flights)}\n", style="white")
                content.append(f"Total in region: {len(flights)}\n", style="white")

                # Show sample flights
                content.append("\nVisible flights:\n", style="dim")
                for i, flight in enumerate(visible_flights[:3]):
                    if flight['callsign'] and flight['callsign'] != 'N/A':
                        content.append(f"{flight['callsign'][:8]}\n", style="green")

        except Exception as e:
            content.append(f"Error loading\nflight data:\n{str(e)[:20]}...", style="red")

        # Add active flights section
        content.append("\n" + "─" * 15 + "\n", style="cyan")
        content.append("ACTIVE FLIGHTS\n", style="bold yellow")
        content.append("─" * 15 + "\n", style="cyan")

        if hasattr(self, 'current_flights') and self.current_flights:
            for flight in self.current_flights:
                if flight['callsign'] and flight['callsign'] != 'N/A':
                    alt_display = f"FL{int(flight['altitude']/100):03d}" if flight['altitude'] else "N/A"
                    speed_display = f"{int(flight['velocity'] * 1.944)}kts" if flight['velocity'] else "N/A"
                    content.append(f"✈{flight['callsign'][:8]}\n", style="green")
                    content.append(f"  {alt_display} {speed_display}\n", style="white")
        else:
            content.append("No flights visible\nin current view", style="dim")

        return content

    def create_status_content(self) -> str:
        """Create status bar content"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        region = self.regions[self.current_region_index].title()
        tracked = f"Tracking: {self.tracked_flight}" if self.tracked_flight else "Scanning"
        zoom_info = f"Zoom: {self.zoom_level:.1f}x"
        controls = "[←↑↓→] Pan [+/-] Zoom [C] Center [F] Fit [R] Region [T] Track [Q] Quit"

        return f"{region} | {tracked} | {zoom_info} | {timestamp} | {controls}"

    def action_zoom_in(self):
        """Zoom in on the map"""
        self.zoom_level = min(self.zoom_level * 1.5, 10.0)  # Max 10x zoom
        self.notify(f"Zoom: {self.zoom_level:.1f}x")
        self.refresh_displays()

    def action_zoom_out(self):
        """Zoom out of the map"""
        self.zoom_level = max(self.zoom_level / 1.5, 0.1)  # Min 0.1x zoom
        self.notify(f"Zoom: {self.zoom_level:.1f}x")
        self.refresh_displays()

    def action_pan_north(self):
        """Pan map north"""
        step = self._get_pan_step()
        self.center_lat = min(self.center_lat + step, 85.0)  # Don't go past north pole
        self._clamp_to_region_bounds()
        self.refresh_displays()

    def action_pan_south(self):
        """Pan map south"""
        step = self._get_pan_step()
        self.center_lat = max(self.center_lat - step, -85.0)  # Don't go past south pole
        self._clamp_to_region_bounds()
        self.refresh_displays()

    def action_pan_west(self):
        """Pan map west"""
        step = self._get_pan_step()
        self.center_lon = self.center_lon - step
        # Handle wrapping around the international date line
        if self.center_lon < -180:
            self.center_lon += 360
        self._clamp_to_region_bounds()
        self.refresh_displays()

    def action_pan_east(self):
        """Pan map east"""
        step = self._get_pan_step()
        self.center_lon = self.center_lon + step
        # Handle wrapping around the international date line
        if self.center_lon > 180:
            self.center_lon -= 360
        self._clamp_to_region_bounds()
        self.refresh_displays()

    def action_center_map(self):
        """Center on tracked flight or region default"""
        if self.tracked_flight and hasattr(self, 'current_flights') and self.current_flights:
            # Center on tracked flight
            for flight in self.current_flights:
                if flight['callsign'].strip() == self.tracked_flight:
                    self.center_lat = flight['latitude']
                    self.center_lon = flight['longitude']
                    self.notify(f"Centered on {self.tracked_flight}")
                    break
            else:
                # Flight not found in current flights, center on region
                self._center_on_region()
                self.notify("Flight not visible, centered on region")
        else:
            # Center on region
            self._center_on_region()
            self.notify(f"Centered on {self.regions[self.current_region_index].title()}")

        self.refresh_displays()

    def action_fit_region(self):
        """Reset zoom and center to show entire region"""
        self.zoom_level = 1.0
        self._center_on_region()
        self.notify(f"Fit to {self.regions[self.current_region_index].title()}")
        self.refresh_displays()

    def action_change_region(self):
        """Cycle through available regions"""
        self.current_region_index = (self.current_region_index + 1) % len(self.regions)
        new_region = self.regions[self.current_region_index]

        # Reset zoom and center for new region
        self.zoom_level = 1.0
        self._center_on_region()

        # Clear active flights list for new region
        self.active_flights = []

        self.refresh_displays()
        self.notify(f"Region changed to: {new_region.title()}")

    def action_track_flight(self):
        """Toggle flight tracking or prompt for callsign"""
        if self.tracked_flight:
            self.tracked_flight = None
            self.notify("Tracking disabled")
            self.refresh_displays()
        else:
            # Show available flights for tracking
            try:
                bbox = self.region_bboxes[self.regions[self.current_region_index]]
                flights = get_all_flights((bbox[1], bbox[3], bbox[0], bbox[2]))

                # Find flights in current region
                region_flights = []
                for flight in flights:
                    if (bbox[0] <= flight['longitude'] <= bbox[2] and
                        bbox[1] <= flight['latitude'] <= bbox[3]):
                        if flight['callsign'] and flight['callsign'] != 'N/A':
                            region_flights.append(flight['callsign'].strip())

                if region_flights:
                    # For now, track the first available flight
                    # In a full implementation, you'd show an input dialog
                    self.tracked_flight = region_flights[0]
                    self.notify(f"Now tracking: {self.tracked_flight}")
                    self.refresh_displays()
                else:
                    self.notify("No flights available to track in this region")

            except Exception as e:
                self.notify(f"Error loading flights: {str(e)}")

    def action_clear_tracking(self):
        """Clear flight tracking"""
        self.tracked_flight = None
        self.notify("Tracking cleared")
        self.refresh_displays()

    def action_quit(self):
        """Quit the application"""
        self.exit()


if __name__ == "__main__":
    app = RadarApp()
    app.run()