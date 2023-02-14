#! /usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="CNN_tool",
    version="1.0.0",
    description="Generate trained CNN from images folder",
    author="Krzysztof Ragan",
    author_email="krzysztof.ragan1@gmail.com",
    python_requires=">=3.10.*",
    packages=find_packages(),
    install_requires=["click", "tensorflow"],
    scripts=["main/cli.py"],
    entry_points = {
        "console_scripts": ["create_CNN=main.cli:create_CNN"]
    },
)
