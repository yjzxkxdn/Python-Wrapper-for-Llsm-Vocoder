from setuptools import setup


setup(
    cffi_modules=["src/pyllsm/_build.py:ffibuilder"],
)
