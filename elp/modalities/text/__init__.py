"""Text LLM ELP adapters: Runge-Kutta / Anderson / geodesic updates and reversible patches."""

from .strategies import TextLoopConfig, explicit_runge_kutta, geodesic_extrapolate, iterate_window
from .wrapper import ElasticRecurrencePatch, WindowLoopPatch

__all__ = [
    "ElasticRecurrencePatch",
    "TextLoopConfig",
    "WindowLoopPatch",
    "explicit_runge_kutta",
    "geodesic_extrapolate",
    "iterate_window",
]
