from setuptools import setup, find_packages
import os

VERSION = '0.9.6'
DESCRIPTION = "a transfer learning approach that explicitly models changes in transcriptional variance using a combination of variational autoencoders and normalizing flows"

setup(
    name='deepcellpredictor',
    version=VERSION,
    description='transfer learning approach',
    long_description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        'torch==1.11.0',
        'anndata==0.8.0',
        'pytorch_lightning==1.7.7',
        'scanpy==1.9.1',
        'scipy==1.6',
        'scvi-tools'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    python_requires=">=3.6",
)

