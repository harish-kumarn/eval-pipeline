import yaml
import json
from pathlib import Path
from typing import Dict, Any

from rich.console import Console

console = Console()

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_metrics_config(config_path: str) -> Dict[str, Any]:
    """Load the metrics configuration JSON, returning an empty dict on failure."""
    if not config_path:
        return {}
    config_path = Path(config_path)
    if not config_path.exists():
        console.print(
            f"[bold yellow]Metrics config {config_path} not found. Using all metrics.[/bold yellow]"
        )
        return {}
    console.log(f"[dim]Loading metrics config from {config_path}[/dim]")
    with open(config_path, "r") as f:
        return json.load(f)