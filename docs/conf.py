import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Strom'
copyright = '2026, Jan Balanya'
author = 'Jan Balanya'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',
]

myst_enable_extensions = ['dollarmath']

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    'drafts/**', 'old_case_study.md',
]
html_static_path = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
