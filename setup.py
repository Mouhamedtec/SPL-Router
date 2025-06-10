from setuptools import setup, find_packages

setup(
    name="spl-router",
    version="0.1.0",
    description="A routing engine built on OSMnx for OpenStreetMap data processing and route calculation.",
    author="Mouhamedtec",
    fullname="Hassani Mohammed",
    packages=find_packages(),
    install_requires=[
        "osmnx>=1.3.0",
        "networkx>=2.8.0",
        "pandas>=1.5.0",
        "matplotlib>=3.5.0",
        "folium>=0.12.0",
        "numpy>=1.21.0"
    ],
    python_requires=">=3.7",
)