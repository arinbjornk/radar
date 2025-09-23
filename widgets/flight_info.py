from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text


class FlightInfoWidget(Static):
    """Widget for displaying flight information"""

    region = reactive('europe')
    tracked_flight = reactive(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_mount(self):
        """Called when widget is mounted"""
        self.refresh_flight_info()

    def watch_region(self, new_region: str):
        """Called when region changes"""
        self.refresh_flight_info()

    def watch_tracked_flight(self, new_tracked_flight: str):
        """Called when tracked flight changes"""
        self.refresh_flight_info()

    def refresh_flight_info(self):
        """Refresh the flight information display"""
        content = Text()
        content.append("FLIGHT INFO\n", style="bold cyan")
        content.append("=" * 15 + "\n", style="white")
        content.append("\n")

        if self.tracked_flight:
            content.append(f"Tracking:\n{self.tracked_flight}\n", style="yellow")
        else:
            content.append(f"Region: {self.region.title()}\n", style="green")

        content.append("\nFlight data will\nbe displayed here...\n", style="dim")

        self.update(content)