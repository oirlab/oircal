from pathlib import Path

from Cython.Build import cythonize
import numpy as np
from setuptools import Extension, setup


# Compile every .pyx under oircal/ (e.g. oircal/ramp/fitramp_cython_Brandt.pyx).
extensions = [
    Extension(
        name=".".join(pyx.with_suffix("").parts),
        sources=[str(pyx)],
        include_dirs=[np.get_include()],
    )
    for pyx in sorted(Path("oircal").rglob("*.pyx"))
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
    ),
)
