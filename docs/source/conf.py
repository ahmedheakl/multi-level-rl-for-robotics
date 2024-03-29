"""Configuration file for the Sphinx documentation builder"""
import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

project = "Highrl"
copyright = "2022, Ahmed Heakl"
author = "Ahmed Heakl"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "rst2pdf.pdfbuilder",
    "sphinxcontrib.napoleon",
]
pdf_documents = [
    ("index", "highrldocs", "Highl Docs", "Ahmed Heakl"),
]
pdf_stylesheets = ["twocolumn"]
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    # 'special-members': '__init__',
    # 'undoc-members': True,
    # 'exclude-members': '__weakref__'
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
