from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MLStats",
    version="0.1.4",
    description="MLStats is a comprehensive Python library designed to seamlessly integrate established statistical methods into machine learning projects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Brritany/MLStats',
    project_urls={
        'Tracker': 'https://github.com/Brritany/MLStats/issues',
    },
    author="Yong-Zhen Huang",
    author_email="m946111005@tmu.edu.tw",
    packages=find_packages(),
    keywords=['python', 'statistical', 'Delong test', 'Bootstrapping'],
    install_requires=[
        "pandas", "numpy", "scipy"
    ],
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
