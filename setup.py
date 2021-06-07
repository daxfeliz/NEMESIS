from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Python modules and scripts used for NEMESIS project.'


def readme():
    with open('README.md') as f:
        return f.read()

# Setting up

INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'astropy',
    'matplotlib','wotan','transitleastsquares','pandas','astroquery','fbpca']

EXTRAS_REQUIRE = {
    'all':['exoplanet','corner']
}

setup(
    name="nemesis_daxfeliz",
    version=VERSION,
    author="Dax L. Feliz",
    author_email="<dax.feliz@vanderbilt.edu>",
    url='https://github.com/daxfeliz/nemesis',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=readme(),
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    keywords=['python', 'astronomy', 'photometry', 'TESS'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    license='MIT',
)
