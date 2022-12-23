"""Setup of IPCV tools package."""

import setuptools

setuptools.setup(
    name="ipcv_tools",
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
    tests_require=["pytest"],
)
