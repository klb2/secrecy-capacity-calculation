from setuptools import setup, find_packages

from secrecy_capacity import __version__, __author__, __email__

with open("README.md", encoding="utf8") as rm:
    long_desc = rm.read()

setup(
    name = "secrecy_capacity",
    version = __version__,
    author = __author__,
    author_email = __email__,
    description = "Implementation of various algorithms to calculate the secrecy capacity of MIMO channels",
    keywords = ["communications", "wiretap channel", "secrecy capacity"],
    long_description=long_desc,
    long_description_content_type="text/markdown",
    license='GPLv3',
    url='https://github.com/klb2/secrecy-capacity-calculation',
    project_urls={
        'Documentation': "https://secrecy-capacity-calculation.readthedocs.io",
        'Source Code': 'https://github.com/klb2/secrecy-capacity-calculation'
        },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Telecommunications Industry",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering",
        ],
    packages=find_packages(),
    tests_require=['pytest', 'tox'],
    install_requires=['numpy', 'scipy'],
)
