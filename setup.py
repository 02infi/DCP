from setuptools import setup, find_packages
import os

VERSION = '0.0.1'
DESCRIPTION = "a transfer learning approach that explicitly models changes in transcriptional variance using a combination of variational autoencoders and normalizing flows"

setup(
    name='DCP',
    version=VERSION,
    description='transfer learning approach',
    long_description='DESCRIPTION',
    packages=find_packages(),
    install_requires=[
        'numpy','pandas','os','math','matplotlib','torch','anndata','pytorch_lightning','typing','scanpy','scipy','scvi'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
  python_requires=">=3.6",
)

