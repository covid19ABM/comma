# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'comma - COvid Mental-health Model with Agents'
authors = ["Eva Viviani <e.viviani@esciencecenter.nl>",
           "Ji Qi <j.qi@esciencecenter.nl>"]

authors_ = ', '.join(authors)
author = authors_
copyright = f'2023, {author}'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['autoapi.extension', 'myst_parser',
              'nbsphinx', 'sphinx_gallery.load_style']
source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoapi_dirs = ['../comma/']
nbsphinx_execute = 'never'

html_static_path = ['_static']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
