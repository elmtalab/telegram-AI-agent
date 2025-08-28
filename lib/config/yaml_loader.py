"""Safe YAML loader."""
import yaml

def load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}
