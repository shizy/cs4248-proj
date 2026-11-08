import importlib
import pkgutil
from pathlib import Path
from .pooler import HiddenStatePooler, HIDDEN_POOLERS, register_pooler, get_hidden_pooler

# auto import all pooler modules
_dir = Path(__file__).resolve().parent
for _, _module, _ in pkgutil.iter_modules([str(_dir)]):
    if _module != "pooler":
        importlib.import_module(f".{_module}", package=__name__)

__all__ = ["HiddenStatePooler", "HIDDEN_POOLERS", "register_pooler", "get_hidden_pooler"]
