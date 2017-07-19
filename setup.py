# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# fetch version from within python_gclda_package module
with open(os.path.join('gclda', 'version.py')) as f:
    exec(f.read())

config = {
    'description': 'Python gcLDA Implementation',
    'author': 'Timothy Rubin',
    'url': 'https://github.com/timothyrubin/python_gclda/',
    'download_url': 'https://github.com/timothyrubin/python_gclda/',
    'author_email': 'tim.rubin@gmail.com',
    'version': __version__,
    'install_requires': ['numpy','scipy','matplotlib'],
    'packages': ['gclda'],
    'name': 'gclda'
}

setup(**config)
