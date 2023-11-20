"""Setup of IPCV tools package."""

import setuptools

name = "ipcv_tools"
packages = setuptools.find_packages(include=name)
setuptools.setup(
    name=name,
    version="0.0.1",
    python_requires=">=3.6",
    install_requires=[
        "matplotlib",
        "numpy",
        "harvesters",
        "opencv-python",
        "pygame",
        "PyQt5",
        "pyqtgraph",
        "scikit-image",
        "scipy",
    ],
    packages=packages,
    extras_require={
        "dev": [
            "black",
            "pytest",
            "mypy==1.0.0",
            "isort",
            "flake8",
            "flake8-docstrings",
            "pydoclint",
            "pygraphviz",
            "import_deps",
            "PyQt5-stubs",
            "doit",
        ]
    },
)
