from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def _read_readme() -> str:
    readme_path = ROOT / "README.md"
    if not readme_path.exists():
        return ""
    return readme_path.read_text(encoding="utf-8")


def _read_requirements() -> list[str]:
    req_path = ROOT / "requirements.txt"
    if not req_path.exists():
        return []
    requirements: list[str] = []
    for raw_line in req_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            continue
        # Handle non-standard "pkg @ index-url https://..." lines by keeping the package.
        if " @ " in line and "index-url" in line:
            pkg = line.split("@", 1)[0].strip()
            if pkg:
                requirements.append(pkg)
            continue
        requirements.append(line)
    return requirements


setup(
    name="vectorial-neuro-modulators",
    version="0.1.0",
    description=(
        "Research-oriented toolkit for spiking neural networks and mean-field models."
    ),
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    author="Eviatar Bas",
    license="MIT",
    packages=find_packages(
        exclude=(
            "tests",
            "scripts",
            "configs",
            "simulations",
            "skills",
            "style",
        )
    ),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=_read_requirements(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,
)
