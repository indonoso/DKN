from setuptools import setup

setup(
    name='DKN',
    version='0.5.3',
    url='https://github.com/indonoso/DKN.git',
    author='Ivania Donoso-Guzmán based on fork',
    description='DKN using BERT',
    packages=['DKN'],
    install_requires=[
        'tensorflow',
        'gensim',
        'numpy',
        'pandas',
        'scikit-learn',
        'transformers',
        'torch',
    ],
)
