from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Header, Static, Label
from textual.binding import Binding


class SimpleRadarApp(App):
    """Simple test app to verify Textual works"""

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
        background: blue;
        color: white;
    }
    """

    BINDINGS = [
        Binding("r", "change_region", "Change Region"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Container(id="main-container"):
            yield Static("Radar display area", id="radar-display")
            yield Static("Flight info area", id="flight-info")

        yield Static("Status bar - Press R to change region, Q to quit", id="status-bar")

    def action_change_region(self):
        """Test region change action"""
        self.notify("Region changed!")

    def action_quit(self):
        """Quit the application"""
        self.exit()


if __name__ == "__main__":
    app = SimpleRadarApp()
    app.run()