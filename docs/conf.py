# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('..'))


__version__ = "None"

# read version without importing kubric (to avoid missing bpy dependency problems)
with open('../kubric/version.py') as f:
  exec(f.read(), globals())

# -- Project information -----------------------------------------------------

project = "Kubric"
copyright = "2021 The Kubric Authors"
author = 'The Kubric Authors'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Autodoc ----------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autodoc_mock_imports = ["bpy", "OpenEXR"]


# -- Napoleon ----------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True


# -- Linkcode ----------------------------

def linkcode_resolve(domain, info):
  # try to create a corresponding github link for each source-file
  # adapted from Lasagne https://github.com/Lasagne/Lasagne/blob/master/docs/conf.py#L114
  def find_source():
    obj = sys.modules[info['module']]
    for part in info['fullname'].split('.'):
      obj = getattr(obj, part)
    import inspect
    import os
    fn = inspect.getsourcefile(obj)
    fn = os.path.relpath(fn, start="..")
    source, lineno = inspect.getsourcelines(obj)
    return fn, lineno, lineno + len(source) - 1

  if domain != 'py' or not info['module']:
    return None
  try:
    filename = '%s#L%d-L%d' % find_source()
  except Exception:
    filename = info['module'].replace('.', '/') + '.py'
  tag = 'main' if 'dev' in release else ('v' + release)
  return "https://github.com/google-research/kubric/blob/%s/%s" % (tag, filename)

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
