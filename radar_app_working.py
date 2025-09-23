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
        Binding("r", "change_region", "Change Region"),
        Binding("t", "track_flight", "Track Flight"),
        Binding("q", "quit", "Quit"),
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

        # Map and flight rendering settings
        self.map_width = 70   # Much larger width to fill panel
        self.map_height = 25  # Much larger height to fill panel
        self.map_char = '·'
        self.aircraft_chars = {
            'N': '↑', 'NE': '↗', 'E': '→', 'SE': '↘',
            'S': '↓', 'SW': '↙', 'W': '←', 'NW': '↖'
        }

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

    def generate_map(self) -> Text:
        """Generate the actual radar map with flights"""
        bbox = self.region_bboxes[self.regions[self.current_region_index]]

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
                            self._plot_coords(coords, canvas, bbox)
                    else:
                        coords = np.array(country.geometry.exterior.coords)
                        self._plot_coords(coords, canvas, bbox)
                except (AttributeError, ValueError):
                    continue

        except Exception as e:
            # If map loading fails, just continue with flights
            pass

        # Get and plot flights
        try:
            flights = get_all_flights((bbox[1], bbox[3], bbox[0], bbox[2]))

            if self.tracked_flight:
                # Show only tracked flight
                target_flight = None
                for flight in flights:
                    if flight['callsign'].strip() == self.tracked_flight.strip():
                        target_flight = flight
                        break

                if target_flight:
                    self._plot_flight(target_flight, canvas, bbox)
                    flights_to_show = [target_flight]
                else:
                    flights_to_show = []
            else:
                # Show multiple flights (limit to 8 for clarity)
                random.shuffle(flights)
                flights_to_show = []
                for flight in flights:
                    if (bbox[0] <= flight['longitude'] <= bbox[2] and
                        bbox[1] <= flight['latitude'] <= bbox[3]):
                        flights_to_show.append(flight)
                        self._plot_flight(flight, canvas, bbox)
                        if len(flights_to_show) >= 8:
                            break

        except Exception as e:
            flights_to_show = []

        # Convert canvas to Rich Text
        map_text = self._canvas_to_rich_text(canvas)

        # Store flights data for the flight info panel to use
        self.current_flights = flights_to_show

        return map_text

    def _plot_coords(self, coords, canvas, bbox):
        """Plot coordinates on canvas"""
        x_coords = ((coords[:, 0] - bbox[0]) / (bbox[2] - bbox[0]) * self.map_width).astype(int)
        y_coords = ((bbox[3] - coords[:, 1]) / (bbox[3] - bbox[1]) * self.map_height).astype(int)

        for x, y in zip(x_coords, y_coords):
            if 0 <= x < self.map_width and 0 <= y < self.map_height:
                canvas[y][x] = self.map_char

    def _plot_flight(self, flight, canvas, bbox):
        """Plot a single flight on the canvas"""
        x = int(((flight['longitude'] - bbox[0]) / (bbox[2] - bbox[0]) * self.map_width))
        y = int(((bbox[3] - flight['latitude']) / (bbox[3] - bbox[1]) * self.map_height))

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
            bbox = self.region_bboxes[self.regions[self.current_region_index]]
            flights = get_all_flights((bbox[1], bbox[3], bbox[0], bbox[2]))

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

                # Count flights in region
                region_flights = []
                for flight in flights:
                    if (bbox[0] <= flight['longitude'] <= bbox[2] and
                        bbox[1] <= flight['latitude'] <= bbox[3]):
                        region_flights.append(flight)

                content.append(f"Active flights: {len(region_flights)}\n", style="white")
                content.append(f"Total tracked: {len(flights)}\n", style="white")

                # Show sample flights
                content.append("\nSample flights:\n", style="dim")
                for i, flight in enumerate(region_flights[:3]):
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
        tracked = f"Tracking: {self.tracked_flight}" if self.tracked_flight else "Tracking: None"
        controls = "[R]egion [T]rack [Q]uit"

        return f"Region: {region} | {tracked} | {timestamp} | {controls}"

    def action_change_region(self):
        """Cycle through available regions"""
        self.current_region_index = (self.current_region_index + 1) % len(self.regions)
        new_region = self.regions[self.current_region_index]
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