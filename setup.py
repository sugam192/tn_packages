# setup.py
from setuptools import setup, find_packages

setup(
    name="tamingnifty",  # your package name
    version="1.0.2",
    description="A library for converting OHLC to Noiseless Charts (Renko and PNF)",
    author="Sugam Gupta",
    packages=find_packages(),
    install_requires=[
        "pymongo",
        "requests",
        "pandas",
        "slack_sdk",
        "python-dotenv",
        # add other dependencies here
    ],
    include_package_data=True,
    python_requires=">=3.7",
)