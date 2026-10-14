# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import os
import re
import sys


sys.path.insert(0, os.path.abspath("../.."))

from sphinx.util.typing import stringify_annotation
from sphinxawesome_theme.postprocess import Icons

import oircal


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "oircal"
copyright = "2026, UCSD Optical-Infrared Laboratory"
author = "UCSD Optical-Infrared Laboratory"

# Version
version = oircal.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

napoleon_custom_sections = [
    ("Returns", "params_style"),
]

# Resolves :ref:/:doc: targets embedded in third-party (scipy/numpy)
# docstrings pulled in via autodoc.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

autosummary_generate = True  # Generate autosummary stubs automatically
autodoc_default_options = {
    "members": True,  # Include all class members
    "private-members": True,  # Include private members
    "undoc-members": False,  # Include undocumented members
    "show-inheritance": True,  # Show class inheritance
    # Do not document ``from x import y`` names as if they belonged here (avoids
    # duplicate object descriptions across package re-exports).
    "imported-members": False,
}
# Signatures show no annotations. Parameter types come from the docstring or, if it
# gives none, from the annotation (see `add_param_types` below).
autodoc_typehints = "none"
autodoc_inherit_docstrings = False

# Bare `name` is a Python cross-reference to any object type (function, class, ...).
# A leading dot (`.name`, :func:`.name`, :class:`.name`) searches all objects ending
# in that name, so the full module path is not needed.
default_role = "py:obj"

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://sphinxawesome.xyz/how-to/configure/
html_theme = "sphinxawesome_theme"


pygments_style = "friendly"
pygments_style_dark = "friendly"
html_permalinks_icon = Icons.permalinks_icon
html_static_path = ["_static"]
html_css_files = ["custom.css"]

add_module_names = False

#### Sphinx custom formatting ####


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip third-party/stdlib objects merely imported into an oircal module."""
    if skip:
        return True
    # e.g. numpy, numba or typing objects imported into a module were never "at home"
    # there, and documenting them under every module that imports them produces
    # duplicate-object-description / ambiguous-cross-reference warnings.
    if what in ("class", "function", "exception", "data", "module"):
        obj_mod = getattr(obj, "__module__", None)
        if obj_mod and not obj_mod.startswith("oircal"):
            return True
    return False


def add_param_types(app, what, name, obj, options, lines):
    """
    Add a ``:type:`` field for each documented parameter whose docstring gives no type.

    The type is taken from the signature annotation, and ``optional`` is appended when the
    parameter has a default value. Runs after napoleon has converted the docstring to fields.
    """
    if what not in ("function", "method", "class"):
        return
    try:
        sig = inspect.signature(getattr(obj, "py_func", obj))
    except (TypeError, ValueError):
        return
    params = []
    for line in lines:
        m = re.match(r":param (?:\S+ )?\*{0,2}(\w+):", line)
        if m:
            params.append(m.group(1))
    typed = {m.group(1) for line in lines if (m := re.match(r":type \*{0,2}(\w+):", line))}
    new_lines = []
    for p in params:
        if p in typed or p not in sig.parameters:
            continue
        param = sig.parameters[p]
        parts = []
        if param.annotation is not inspect.Parameter.empty:
            parts.append(stringify_annotation(param.annotation, mode="smart"))
        if param.default is not inspect.Parameter.empty:
            parts.append("optional")
        if parts:
            new_lines.append(f":type {p}: {', '.join(parts)}")
    if new_lines:
        lines.extend(new_lines)


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
    # Priority > 500 so this runs after napoleon's docstring conversion.
    app.connect("autodoc-process-docstring", add_param_types, priority=600)
