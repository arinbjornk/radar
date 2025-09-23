from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Static, Label
from textual.binding import Binding
from datetime import datetime
from widgets.radar_display import RadarDisplayWidget
from widgets.flight_info import FlightInfoWidget


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
        tracked = f"Tracking: {self.current_tracked_flight}" if self.current_tracked_flight else "Tracking: None"
        controls = "[R]egion [T]rack [Q]uit"
        return f"Region: {self.current_region.title()} | {tracked} | {timestamp} | {controls}"

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
    }

    #main-container {
        layout: horizontal;
        height: 1fr;
    }

    #radar-display {
        width: 3fr;
        border: solid white;
        margin: 1;
    }

    #flight-info {
        width: 1fr;
        border: solid white;
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container(id="main-container"):
            yield RadarDisplayWidget(id="radar-display")
            yield FlightInfoWidget(id="flight-info")

        yield StatusBar(
            region=self.regions[self.current_region_index],
            tracked_flight=self.tracked_flight,
            id="status-bar"
        )

    def on_mount(self):
        """Called when app is mounted - set up timer for auto-refresh"""
        self.set_interval(10.0, self.refresh_displays)
        # Initial load
        self.refresh_displays()

    def refresh_displays(self):
        """Refresh both radar and flight info displays"""
        try:
            radar_display = self.query_one("#radar-display", RadarDisplayWidget)
            flight_info = self.query_one("#flight-info", FlightInfoWidget)

            radar_display.refresh_radar()
            flight_info.refresh_flight_info()
        except Exception as e:
            self.notify(f"Error refreshing displays: {str(e)}")

    def action_change_region(self):
        """Cycle through available regions"""
        self.current_region_index = (self.current_region_index + 1) % len(self.regions)
        new_region = self.regions[self.current_region_index]

        # Update radar display region
        radar_display = self.query_one("#radar-display", RadarDisplayWidget)
        radar_display.region = new_region

        # Update flight info region
        flight_info = self.query_one("#flight-info", FlightInfoWidget)
        flight_info.region = new_region

        # Update status bar
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_status(region=new_region)

        self.notify(f"Region changed to: {new_region.title()}")

    def action_track_flight(self):
        """Open dialog to input flight callsign for tracking"""
        # For now, just show a notification - we'll implement the input dialog later
        self.notify("Flight tracking input dialog - coming soon!")

    def action_quit(self):
        """Quit the application"""
        self.exit()


if __name__ == "__main__":
    app = RadarApp()
    app.run()