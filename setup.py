from setuptools import setup, find_packages

setup(
    name="ivim-dwi",
    version="1.0.0",
    author="Leonar Steven Prieto-Gonzalez",
    description="Robust IVIM and Kurtosis parameter estimation for DWI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ivim-robust",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pydicom",
        "nibabel",
        "tqdm",
        "joblib",
        "pandas",
        "scikit-learn",
        "xgboost"
    ],
)
