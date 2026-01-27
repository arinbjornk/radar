from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Static, Label, Input
from textual.binding import Binding
from datetime import datetime
from widgets.radar_display import RadarDisplayWidget
from widgets.flight_info import FlightInfoWidget
from flights import get_all_flights, get_flight_details, get_flight_track
from collections import defaultdict
import random
import logging

# Set up logging
logging.basicConfig(
    filename='debug.log', 
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

class StatusBar(Static):
    """Status bar showing current state and controls"""

    def __init__(self, region: str = "europe", tracked_flight: str = None, **kwargs):
        super().__init__(**kwargs)
        self.current_region = region
        self.current_tracked_flight = tracked_flight

    def compose(self) -> ComposeResult:
        status_text = self._build_status_text()
        yield Label(status_text, id="status-text")

    def _build_status_text(self) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if self.current_tracked_flight:
            tracked = f"[{'#F7768E'}]Tracking: {self.current_tracked_flight}[/]" 
        else:
            tracked = f"[{'#565f89'}]Tracking: None[/]"
            
        # Key controls with Rich markup
        controls = (
            f"[{'#E0AF68'} reverse] R [/] Region  "
            f"[{'#E0AF68'} reverse] T [/] Track/Search  "
            f"[{'#E0AF68'} reverse] +/- [/] Zoom  "
            f"[{'#E0AF68'} reverse] Arrows [/] Pan  "
            f"[{'#E0AF68'} reverse] Q [/] Quit"
        )
        
        region_display = f"Region: [{'#9ECE6A'}]{self.current_region.title()}[/]"
        
        return f" {region_display}  │  {tracked}  │  {timestamp}  │  {controls}"

    def update_status(self, region: str = None, tracked_flight: str = None):
        if region is not None:
            self.current_region = region
        if tracked_flight is not None:
            self.current_tracked_flight = tracked_flight

        status_text = self._build_status_text()
        self.query_one("#status-text", Label).update(status_text)


class RadarApp(App):
    """Main radar application"""

    CSS = """
    Screen {
        layout: vertical;
        background: #1e1e1e;
    }

    #main-container {
        layout: horizontal;
        height: 1fr;
    }

    #radar-display {
        width: 3fr;
        border: solid #9ECE6A;
        margin: 1;
    }

    #flight-info {
        width: 1fr;
        border: solid #7AA2F7;
        margin: 1;
    }
    
    #search-input {
        display: none;
        height: 3;
        margin: 1;
        border: solid #E0AF68;
        background: #1e1e1e;
        color: #a9b1d6;
    }
    
    #search-input.visible {
        display: block;
    }

    #status-bar {
        height: 3;
        background: #282c34;
        color: #a9b1d6;
        content-align: center middle;
    }
    """

    BINDINGS = [
        Binding("r", "change_region", "Change Region"),
        Binding("t", "track_flight", "Track Flight"),
        Binding("q", "quit", "Quit"),
        Binding("plus,equal", "zoom_in", "Zoom In"),
        Binding("minus", "zoom_out", "Zoom Out"),
        Binding("up", "pan_north", "Pan North"),
        Binding("down", "pan_south", "Pan South"),
        Binding("left", "pan_west", "Pan West"),
        Binding("right", "pan_east", "Pan East"),
    ]

    def __init__(self):
        super().__init__()
        self.regions = ['europe', 'north_america', 'south_america', 'asia', 'africa', 'australia', 'antarctica']
        self.current_region_index = 0
        self.tracked_flight = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container(id="main-container"):
            yield RadarDisplayWidget(id="radar-display")
            yield FlightInfoWidget(id="flight-info")
            
        yield Input(placeholder="Enter flight number (e.g. BAW123)", id="search-input")

        yield StatusBar(
            region=self.regions[self.current_region_index],
            tracked_flight=self.tracked_flight,
            id="status-bar"
        )

    def on_mount(self):
        """Called when app is mounted - set up timer for auto-refresh"""
        self.set_interval(5.0, self.refresh_displays)
        # Initial load
        self.call_after_refresh(self.refresh_displays)
        # Ensure search input doesn't steal focus on startup
        self.set_focus(None)

    def _apply_density_filter(self, flights):
        """Downsample flights based on density, preserving remote flights"""
        if not flights:
            return []

        # Grid configuration (degrees)
        GRID_SIZE = 2.0
        MAX_PER_CELL = 2

        grid = defaultdict(list)
        filtered_flights = []
        
        # Always keep tracked flight
        tracked_obj = None
        
        for flight in flights:
            # Separate tracked flight
            if self.tracked_flight and flight['callsign'].strip() == self.tracked_flight:
                tracked_obj = flight
                continue

            # Bucketize by grid cell
            lat_idx = int(flight['latitude'] // GRID_SIZE)
            lon_idx = int(flight['longitude'] // GRID_SIZE)
            grid[(lat_idx, lon_idx)].append(flight)

        # Process grid
        for cell_flights in grid.values():
            if len(cell_flights) > MAX_PER_CELL:
                # High density: sample a few
                filtered_flights.extend(random.sample(cell_flights, MAX_PER_CELL))
            else:
                # Low density: keep all
                filtered_flights.extend(cell_flights)
        
        # Add tracked flight back if it exists
        if tracked_obj:
            filtered_flights.append(tracked_obj)
            
        return filtered_flights

    def refresh_displays(self):
        """Refresh both radar and flight info displays"""
        try:
            # 1. Get flight data
            flights = get_all_flights()
            
            # 2. Filter for display density
            filtered_flights = self._apply_density_filter(flights)
            
            radar_display = self.query_one("#radar-display", RadarDisplayWidget)
            flight_info = self.query_one("#flight-info", FlightInfoWidget)

            # 3. Update radar and get visible flights
            visible_flights = radar_display.update_flights(filtered_flights)
            
            # 4. Update info panel
            flight_info.update_flights(visible_flights)
            
            # 5. Update status bar time
            self.query_one("#status-bar", StatusBar).update_status()
            
        except Exception as e:
            self.notify(f"Error refreshing displays: {str(e)}")

    def action_change_region(self):
        """Cycle through available regions"""
        self.current_region_index = (self.current_region_index + 1) % len(self.regions)
        new_region = self.regions[self.current_region_index]

        # Update radar display region
        radar_display = self.query_one("#radar-display", RadarDisplayWidget)
        radar_display.map_region = new_region

        # Update flight info region
        flight_info = self.query_one("#flight-info", FlightInfoWidget)
        flight_info.map_region = new_region

        # Update status bar
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_status(region=new_region)

        self.notify(f"Region changed to: {new_region.title()}")
        
        # Trigger immediate refresh
        self.refresh_displays()

    def on_key(self, event):
        """Handle key events"""
        logging.debug(f"Key pressed: {event.key}")
        if event.key == "escape":
            logging.debug("Escape key caught")
            input_widget = self.query_one("#search-input")
            if input_widget.has_class("visible"):
                logging.debug("Closing search input from Escape")
                # Close search input
                input_widget.remove_class("visible")
                self.set_focus(None)
            elif self.tracked_flight:
                logging.debug("Clearing tracking from Escape")
                # Stop tracking
                self.tracked_flight = None
                radar = self.query_one("#radar-display", RadarDisplayWidget)
                radar.single_flight_mode = False
                radar.tracked_flight = None
                self.query_one("#flight-info", FlightInfoWidget).tracked_flight = None
                self.query_one("#status-bar", StatusBar).update_status(tracked_flight=None)
                self.notify("Tracking cleared")
                self.refresh_displays()

    def action_track_flight(self):
        """Toggle search input"""
        logging.debug("Action track_flight triggered")
        input_widget = self.query_one("#search-input")
        input_widget.toggle_class("visible")
        
        if input_widget.has_class("visible"):
            logging.debug("Showing search input and requesting focus")
            input_widget.value = ""  # Clear previous input
            # Use call_later to prevent the 't' keypress from being captured by the input
            self.call_later(input_widget.focus)
        else:
            logging.debug("Hiding search input and releasing focus")
            # If closing, remove focus
            self.set_focus(None)

    def on_input_submitted(self, message: Input.Submitted):
        """Handle search submission"""
        logging.debug(f"Input submitted: {message.value}")
        input_widget = self.query_one("#search-input")
        input_widget.remove_class("visible")
        self.set_focus(None)
        
        callsign = message.value.upper().strip()
        if not callsign:
             # Clear tracking
             self.tracked_flight = None
             self.query_one("#radar-display", RadarDisplayWidget).single_flight_mode = False
             self.notify("Tracking disabled")
        else:
             self.tracked_flight = callsign
             radar_display = self.query_one("#radar-display", RadarDisplayWidget)
             radar_display.single_flight_mode = True
             self.notify(f"Searching for {callsign}...")
             
             # Check if flight is currently available to center map
             flights = get_all_flights()
             target = next((f for f in flights if f['callsign'].strip() == callsign), None)
             if target:
                 radar_display.center_lat = target['latitude']
                 radar_display.center_lon = target['longitude']
                 
                 # 1. Fetch Historical Track (The "Blue Line")
                 track_points = get_flight_track(target['icao24'])
                 if track_points:
                     radar_display.track_history = track_points
                     self.notify(f"Track history found ({len(track_points)} points)")
                 else:
                     radar_display.track_history = []
                     self.notify("No track history available")

                 # 2. Fetch Route Info (Origin/Dest)
                 self.notify(f"Fetching details for {callsign}...")
                 details = get_flight_details(target['icao24'])
                 if details:
                     target['origin_airport'] = details.get('estDepartureAirport')
                     target['destination_airport'] = details.get('estArrivalAirport')
                     if target['origin_airport'] and target['destination_airport']:
                         self.notify(f"Route found: {target['origin_airport']} -> {target['destination_airport']}")
                     else:
                         self.notify("Route info incomplete")
                 else:
                     self.notify("No detailed route info found")
                     
             else:
                 self.notify(f"Flight {callsign} not currently found")

        # Propagate to widgets
        self.query_one("#radar-display", RadarDisplayWidget).tracked_flight = self.tracked_flight
        self.query_one("#flight-info", FlightInfoWidget).tracked_flight = self.tracked_flight
        self.query_one("#status-bar", StatusBar).update_status(tracked_flight=self.tracked_flight)
        
        input_widget.value = ""
        self.refresh_displays()

    def action_zoom_in(self):
        self.query_one("#radar-display", RadarDisplayWidget).zoom_in()
        self.refresh_displays()

    def action_zoom_out(self):
        self.query_one("#radar-display", RadarDisplayWidget).zoom_out()
        self.refresh_displays()
    
    def action_pan_north(self):
        self.query_one("#radar-display", RadarDisplayWidget).pan('north')
        self.refresh_displays()

    def action_pan_south(self):
        self.query_one("#radar-display", RadarDisplayWidget).pan('south')
        self.refresh_displays()

    def action_pan_west(self):
        self.query_one("#radar-display", RadarDisplayWidget).pan('west')
        self.refresh_displays()

    def action_pan_east(self):
        self.query_one("#radar-display", RadarDisplayWidget).pan('east')
        self.refresh_displays()

    def action_quit(self):
        """Quit the application"""
        self.exit()


if __name__ == "__main__":
    app = RadarApp()
    app.run()