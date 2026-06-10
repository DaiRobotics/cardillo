from pathlib import Path

from setuptools import setup, find_packages

name = "cardillo"
version = "1.0.0"
author = "Tianxiang Dai"
author_email = (
    "dai@inm.uni-stuttgart.de",
)
url = "https://github.com/DaiRobotics/cardillo.git"
description = (
    "Research fork of Cardillo for reproducing the numerical experiments "
    "reported in the accompanying manuscript."
)

long_description = Path("README.md").read_text(encoding="utf-8")
long_description_content_type = "text/markdown"

license = "BSD-3-Clause"

setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    install_requires=[
        "numpy>=2.2.6",
        "scipy>=1.15.3",
        "tqdm>=4.62.3",
        "dill>=0.3.7",
        "vtk>=9.3.0",
        "scipy_dae>=0.1.0",
        "jax>=0.9.0",
        "numba>=0.64.0",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
)
