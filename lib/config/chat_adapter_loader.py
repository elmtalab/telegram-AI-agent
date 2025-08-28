from dataclasses import dataclass
from typing import Any, Dict

from .yaml_loader import load_yaml


@dataclass
class AdapterConfig:
    """Typed view over ``chat_adapter.yaml``.

    Only the keys required by the exercises are exposed.  The raw mapping is
    retained so that lookups of shared defaults (such as regular expression
    references) remain possible without having to duplicate the structure
    in Python.
    """

    raw: Dict[str, Any]
    clean: Dict[str, Any]
    detect_urls: Dict[str, Any]
    detect_media_types: Dict[str, Any]
    attachments_guard: Dict[str, Any]
    locale: Dict[str, Any]
    sticky: Dict[str, Any]
    hints: Dict[str, Any]


def load_chat_adapter_config(path: str) -> AdapterConfig:
    """Load ``chat_adapter.yaml`` and return an :class:`AdapterConfig`.

    Parameters
    ----------
    path:
        File system path to the YAML configuration file.
    """

    raw = load_yaml(path)
    adapter = raw.get("adapter", {})
    preprocess = adapter.get("preprocess", {})
    return AdapterConfig(
        raw=raw,
        clean=preprocess.get("clean", {}),
        detect_urls=preprocess.get("detect_urls", {}),
        detect_media_types=preprocess.get("detect_media_types", {}),
        attachments_guard=preprocess.get("attachments_guard", {}),
        locale=adapter.get("locale", {}),
        sticky=adapter.get("memory", {}).get("sticky_carry", {}),
        hints=adapter.get("hints", {}),
    )
