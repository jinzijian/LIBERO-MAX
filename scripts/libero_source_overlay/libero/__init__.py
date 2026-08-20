"""Expose a source checkout as the top-level :mod:`libero` package.

Some recent ``hf-libero`` wheels install a regular top-level package that
shadows LIBERO-Plus and LIBERO-PRO source trees, whose outer ``libero``
directory is a namespace package.  The MAX launchers put this small regular
package first on ``PYTHONPATH`` and point ``LIBERO_SOURCE_PACKAGE_ROOT`` at the
source tree's outer ``libero`` directory.  Imports such as
``libero.libero.benchmark`` then resolve against the frozen benchmark source
instead of an unrelated wheel.
"""

from os import environ
from pathlib import Path


_root = environ.get("LIBERO_SOURCE_PACKAGE_ROOT")
if not _root:
    raise RuntimeError("LIBERO_SOURCE_PACKAGE_ROOT is required")
_path = Path(_root).expanduser().resolve()
if not (_path / "libero" / "__init__.py").is_file():
    raise RuntimeError("invalid LIBERO source package root: %s" % _path)

__path__ = [str(_path)]
