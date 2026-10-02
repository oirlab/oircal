# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys, os
sys.path.insert(0, os.path.abspath("../.."))

import oircal


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'oircal'
copyright = '2026, UCSD Optical-Infrared Laboratory'
author = 'UCSD Optical-Infrared Laboratory'

# Version
version = oircal.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

napoleon_custom_sections = [
    ('Returns', 'params_style')
]

# Resolves :ref:/:doc: targets embedded in third-party (scipy/numpy)
# docstrings pulled in via autodoc.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

autosummary_generate = True  # Generate autosummary stubs automatically
autodoc_default_options = {
    'members': True,          # Include all class members
    'private-members': True, # Include private members
    'undoc-members': False,    # Include undocumented members
    'show-inheritance': True, # Show class inheritance
    # Do not document ``from x import y`` names as if they belonged here (avoids
    # duplicate object descriptions across package re-exports).
    'imported-members': False,
}
# Avoid pulling third-party / typing objects into the inventory via annotations.
autodoc_typehints = 'none'
autodoc_inherit_docstrings = False

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://sphinxawesome.xyz/how-to/configure/
html_theme = 'sphinxawesome_theme'
from sphinxawesome_theme.postprocess import Icons
pygments_style = "friendly"
pygments_style_dark = "friendly"
html_permalinks_icon = Icons.permalinks_icon
html_static_path = ['_static']
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


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
