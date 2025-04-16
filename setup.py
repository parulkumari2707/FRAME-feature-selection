from setuptools import setup, find_packages
#from frame.version import __version__

import os

# Read version manually from version.py without importing frame
def read_version():
    version_file = os.path.join("frame", "version.py")
    with open(version_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Unable to find version string.")

setup(
    name="frame-feature-selector",
    version=read_version(),
    author="Parul Kumari",
    author_email="parulkumari2707@gmail.com",
    description="FRAME: A Hybrid Feature Selection Library Combining Forward Selection and RFE with XGBoost",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/parulkumari2707/FRAME-FEATURE-SELECTOR",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "frame-selector=frame.frame_selector:main"
        ]
    },
    keywords="feature-selection machine-learning sklearn xgboost forward-selection rfe hybrid",
    project_urls={
        "Bug Tracker": "https://github.com/parulkumari2707/FRAME-FEATURE-SELECTOR/issues",
        "Documentation": "https://github.com/parulkumari2707/FRAME-FEATURE-SELECTOR#readme",
    },
)