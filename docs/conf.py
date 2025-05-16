import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Heart Sound Analysis'
copyright = '2024, Yusuke Yano'
author = 'Yusuke Yano'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static'] 