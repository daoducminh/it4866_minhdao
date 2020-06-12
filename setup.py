from setuptools import setup, find_packages

setup(
    name='BSTLD',
    version='1.0',
    description='Detect Traffic Light using Faster R-CNN',
    author='minhdao',
    packages=find_packages(exclude=[
        'docs', 'tests', 'static', 'templates', '.gitignore', 'README.md'
    ]),
    install_requires=[
        'tensorflow',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'scipy',
        'jupyterlab',
        'contextlib2',
        'Cython',
        'tf_slim',
        'pillow',
        'lxml',
        'pylint',
        'autopep8',
        'rope',
        'pyyaml'
    ],
)
