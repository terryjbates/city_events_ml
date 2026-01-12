from setuptools import setup, find_packages

# from city_events_ml import __version__


setup(
    name="city-events-ml",
    version="0.1.0",
    description="SF safety event frequency modeling pipelines",
    long_description=open("README.md", encoding="utf-8").read(),
    # version=__version__,
    author="Terry Bates",
    # packages=find_packages(),
    packages=["city_events_ml"],
    install_requires=[
        # core deps only; keep this conservative
        "numpy",
        "pandas",
        "scikit-learn",
    ],
    extras_require={
        "skops": ["skops>=0.11,<0.14"],
    },
    python_requires=">=3.12",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development",
        "Operating System :: POSIX",
        "Operating System :: Unix",
    ],
    license="Apache-2.0",
)
