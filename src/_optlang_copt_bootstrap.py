"""Lazy COBRApy registration installed by the optlang-copt wheel.

This module is imported by a small ``.pth`` startup hook. It deliberately
does not import COBRApy, optlang, or coptpy at interpreter startup. Instead it
waits until COBRApy loads its solver registry and registers COPT immediately
after that module has initialized.
"""

from __future__ import absolute_import

import importlib.abc
import importlib.machinery
import sys


_OPTLANG_TARGET = "optlang"
_COBRA_TARGET = "cobra.util.solver"
_TARGETS = {_OPTLANG_TARGET, _COBRA_TARGET}


def _adapter():
    adapter = sys.modules.get("optlang_copt")
    if adapter is not None and not hasattr(adapter, "register_with_optlang"):
        # optlang_copt itself is currently importing optlang. Its __init__
        # will finish registration after the circular import unwinds.
        return None
    if adapter is None:
        import optlang_copt as adapter
    return adapter


def _register_optlang():
    adapter = _adapter()
    if adapter is not None:
        adapter.register_with_optlang()


def _register_cobra():
    adapter = _adapter()
    if adapter is not None:
        adapter.register_with_cobra()


class _AfterImportLoader(importlib.abc.Loader):
    def __init__(self, loader, finder, fullname):
        self._loader = loader
        self._finder = finder
        self._fullname = fullname

    def create_module(self, spec):
        create_module = getattr(self._loader, "create_module", None)
        return None if create_module is None else create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        if self._fullname == _OPTLANG_TARGET:
            _register_optlang()
        else:
            _register_cobra()
            try:
                sys.meta_path.remove(self._finder)
            except ValueError:
                pass


class _CobraSolverFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _AfterImportLoader(spec.loader, self, fullname)
        return spec


if _OPTLANG_TARGET in sys.modules:
    _register_optlang()
if _COBRA_TARGET in sys.modules:
    _register_cobra()
elif not any(isinstance(item, _CobraSolverFinder) for item in sys.meta_path):
    sys.meta_path.insert(0, _CobraSolverFinder())
