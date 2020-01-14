import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
packages=setuptools.find_packages(),
setuptools.setup(
    name='pyECToolkit',
    version='0.0.1',
    author="Marcus Schwarting",
    author_email="mschwarting@anl.gov",
    packages=setuptools.find_packages(),
    description="Toolkit for ElectroCAT Datahub interaction, data analysis, and machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.15.4",
        "pandas>=0.23.4"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering"
    ],
    keywords=[],
    license="Apache License, Version 2.0",
    url="https://github.com/meschw04/pyECToolkit.git"
)
