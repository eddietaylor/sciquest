"""Method-aware scientific reasoning support for SciQuest."""

from .registry import MethodRegistry
from .resolver import auto_recommend_method_stack, resolve_method_stack

__all__ = ["MethodRegistry", "resolve_method_stack", "auto_recommend_method_stack"]
