from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text


class FlightInfoWidget(Static):
    """Widget for displaying flight information"""

    map_region = reactive('europe')
    tracked_flight = reactive(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visible_flights = []

    def on_mount(self):
        """Called when widget is mounted"""
        self.refresh_flight_info()

    def watch_map_region(self, new_region: str):
        """Called when region changes"""
        self.refresh_flight_info()

    def watch_tracked_flight(self, new_tracked_flight: str):
        """Called when tracked flight changes"""
        self.refresh_flight_info()

    def update_flights(self, flights):
        """Update the list of visible flights"""
        self.visible_flights = flights
        self.refresh_flight_info()

    def refresh_flight_info(self):
        """Refresh the flight information display"""
        content = Text()
        content.append("FLIGHT INFO\n", style="bold #7DCFFF")
        content.append("=" * 15 + "\n", style="#565f89")
        content.append("\n")

        if self.tracked_flight:
            content.append(f"Tracking:\n{self.tracked_flight}\n", style="#E0AF68")
            # Find tracked flight details
            target = next((f for f in self.visible_flights if f['callsign'].strip() == self.tracked_flight), None)
            if target:
                # Route info
                origin = target.get('origin_airport')
                dest = target.get('destination_airport')
                if origin and dest:
                     content.append(f"\nRoute:\n{origin} -> {dest}\n", style="bold #9ECE6A")
                elif target.get('origin_country'):
                     content.append(f"\nOrigin Country:\n{target['origin_country']}\n", style="#9ECE6A")
                
                content.append(f"\nAlt: {target.get('altitude', 'N/A')}m\n", style="#c0caf5")
                content.append(f"Spd: {target.get('velocity', 'N/A')}m/s\n", style="#c0caf5")
                content.append(f"Hdg: {target.get('heading', 'N/A')}°\n", style="#c0caf5")
            else:
                 content.append("\n(Not in view)\n", style="dim #F7768E")
        else:
            content.append(f"Region: {self.map_region.title()}\n", style="#9ECE6A")
            content.append(f"Visible: {len(self.visible_flights)}\n", style="#c0caf5")
            
            if self.visible_flights:
                content.append("\nNearby:\n", style="#565f89")
                for f in self.visible_flights[:10]:
                    callsign = f.get('callsign', 'N/A').strip()
                    if callsign:
                        content.append(f"{callsign}\n", style="#7DCFFF")

        self.update(content)