"""Modelo X — Framework de Neuroplasticidad Geométrica para LLMs (prototipo Fase 2)."""
from .config import Config, get_device
from .train import Trainer

__all__ = ["Config", "get_device", "Trainer"]
