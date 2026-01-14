from __future__ import annotations

import re
from pathlib import Path

from setuptools import setup, find_packages

# from city_events_ml import __version__


def read_version() -> str:
    init_py = Path(__file__).parent / "city_events_ml" / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not m:
        raise RuntimeError("Could not find __version__ in city_events_ml/__init__.py")
    return m.group(1)


setup(
    name="city-events-ml",
    version=read_version(),
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
        "test": ["pytest", "pytest-cov"],
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
