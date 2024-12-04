from setuptools import setup, find_packages
import os


def parse_requirements(filename):
    # Ensure the path is relative to the setup.py directory
    here = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(here, filename)
    try:
        with open(filepath) as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return []


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="cebra_sergej",  # The name of your package
    version="0.1.0",  # Initial version
    description="A package for analyzing and organizing CEBRA-related work",
    long_description=long_description,
    author="Sergej Maul",
    author_email="maulser@googlemail.com",
    url="https://github.com/Ch3fUlrich/CEBRA_own",  # Optional: URL for your project
    packages=["core"],
    include_package_data=True,  # Includes files specified in MANIFEST.in
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "dev": parse_requirements("requirements_dev.txt"),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change if using a different license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Ensure compatibility with specific Python versions
)
