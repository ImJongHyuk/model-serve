"""Application state management utilities."""

import time
from typing import Dict, Any


# Global application state
_app_state: Dict[str, Any] = {}


def initialize_app_state() -> None:
    """Initialize application state with default values."""
    _app_state.update({
        "startup_time": time.time(),
        "request_count": 0
    })


def get_app_state() -> Dict[str, Any]:
    """Get application state.
    
    Returns:
        Application state dictionary
    """
    return _app_state


def increment_request_count() -> None:
    """Increment the request counter."""
    _app_state["request_count"] = _app_state.get("request_count", 0) + 1


def get_request_count() -> int:
    """Get the current request count.
    
    Returns:
        Number of requests processed
    """
    return _app_state.get("request_count", 0)


def clear_app_state() -> None:
    """Clear application state (for testing)."""
    _app_state.clear()