"""Packaging metadata for the repository-local Isaac Lab extension."""

from pathlib import Path

import toml
from setuptools import find_packages, setup


EXTENSION_ROOT = Path(__file__).resolve().parent
METADATA = toml.load(EXTENSION_ROOT / "config" / "extension.toml")["package"]

setup(
    name="bylw_isaac_p5",
    version=METADATA["version"],
    description=METADATA["description"],
    author=METADATA["author"],
    maintainer=METADATA["maintainer"],
    url=METADATA["repository"],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[],
    zip_safe=False,
)
