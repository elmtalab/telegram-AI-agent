"""Structlog and OpenTelemetry wiring."""

def get_logger(name: str):
    import logging
    return logging.getLogger(name)
