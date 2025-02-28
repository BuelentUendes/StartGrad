from setuptools import setup, find_packages

setup(
    name="Startgrad",
    version="0.1",
    packages=find_packages(where="src") + find_packages(where="utils"),
    package_dir={
        "src": "src",
        "utils": "utils",
    },
)