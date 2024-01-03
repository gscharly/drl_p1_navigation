import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="drl_p1_navigation",
    version="0.0.1",
    description="Udacity DRL Course - Value Based Methods - P1 Navitation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gscharly/drl_p1_navigation",
    author="Carlos Gomez",
    author_email="carlos.gomez.sanchez94@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3.6",
    ],
    # find_namespace_packages will recurse through the directories and find all the packages
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    zip_safe=False,
    install_requires=[
        "torch==1.10.2"
    ],
    python_requires=">=3.6.1,<=3.6.12"
)
