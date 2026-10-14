import importlib
import pkgutil
from pathlib import Path
from .head import PredictiveHead, PREDICTIVE_HEADS, register_head, get_pred_head

# auto import all head modules
_dir = Path(__file__).resolve().parent
for _, _module, _ in pkgutil.iter_modules([str(_dir)]):
    if _module != "head":
        importlib.import_module(f".{_module}", package=__name__)

__all__ = ["PredictiveHead", "PREDICTIVE_HEADS", "register_head", "get_pred_head"]
