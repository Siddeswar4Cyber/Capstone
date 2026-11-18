"""Federated IDS components, organized by function.

Historical flat Python imports remain aliases of the canonical modules. This
keeps existing operator commands and integrations usable during migration without
maintaining duplicate implementations or eagerly importing optional runtimes.
"""
import importlib
import importlib.abc
import importlib.util
import sys

from ppfcl_nids.deployment import MODULE_LOCATIONS


class _HistoricalModuleLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass  # The canonical module was executed by import_module.

    def get_code(self, fullname):
        # Preserve historical ``python -m`` commands in a fresh interpreter.
        return compile(
            f"import runpy\nrunpy.run_module({self.target!r}, run_name='__main__')\n",
            f"<compatibility entry point {fullname}>", "exec",
        )


class _HistoricalModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefix, separator, name = fullname.rpartition(".")
        if prefix != __name__ or not separator or name == "dashboard":
            return None
        destination = MODULE_LOCATIONS.get(name)
        if destination is None:
            return None
        loader = _HistoricalModuleLoader(f"{__name__}.{destination}")
        return importlib.util.spec_from_loader(fullname, loader)


if not any(isinstance(finder, _HistoricalModuleFinder) for finder in sys.meta_path):
    sys.meta_path.append(_HistoricalModuleFinder())
