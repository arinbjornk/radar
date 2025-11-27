from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text


class RadarDisplayWidget(Static):
    """Widget for displaying the radar map with flights"""

    map_region = reactive('europe')
    tracked_flight = reactive(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_mount(self):
        """Called when widget is mounted"""
        self.refresh_radar()

    def watch_map_region(self, new_region: str):
        """Called when region changes"""
        self.refresh_radar()

    def watch_tracked_flight(self, new_tracked_flight: str):
        """Called when tracked flight changes"""
        self.refresh_radar()

    def refresh_radar(self):
        """Refresh the radar display"""
        # Simplified radar display for now
        content = Text()
        content.append("=" * 50 + "\n", style="white")
        content.append(f"RADAR DISPLAY - {self.map_region.upper()}\n", style="bold green")
        content.append("=" * 50 + "\n", style="white")
        content.append("\n")

        if self.tracked_flight:
            content.append(f"Tracking: {self.tracked_flight}\n", style="yellow")
        else:
            content.append("Showing all flights in region\n", style="cyan")

        content.append("\n")
        content.append("Flight radar map will be rendered here...\n", style="white")
        content.append("(Map rendering temporarily simplified)\n", style="dim")

        self.update(content)