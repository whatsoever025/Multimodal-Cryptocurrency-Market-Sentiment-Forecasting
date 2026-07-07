"""Shared utility functions (logging setup, duration formatting)."""

import logging


def setup_logging(level=logging.INFO) -> None:
    """Configure the root logger with a standard timestamp format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def format_duration(seconds: float) -> str:
    """Convert seconds to a human-readable string (e.g. 1.2s, 2.5m, 1.5h)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


if __name__ == "__main__":
    """Test utility functions."""
    print("Testing format_duration()...")
    assert format_duration(5.2) == "5.2s"
    assert format_duration(150.0) == "2.5m"
    assert format_duration(3600.0) == "1.0h"
    print("✓ All utility tests passed!")
