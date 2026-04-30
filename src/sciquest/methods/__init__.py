"""Method-aware scientific reasoning support for SciQuest."""

from .resolver import resolve_method_stack, auto_recommend_method_stack
from .registry import MethodRegistry

__all__ = ["MethodRegistry", "resolve_method_stack", "auto_recommend_method_stack"]
