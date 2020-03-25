from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path


here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

setup(
    name='infomemes',
    version='0.1.0',
    description='Infomemes simualtions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Luiz Tauffer',
    author_email='luiz@taufferconsulting.com',
    url='https://github.com/luiztauffer/infomemes',
    keywords='',
    install_requires=[
        'numpy', 'scipy', 'pandas', 'matplotlib', 'alive_progress'
    ],
    entry_points={
        'console_scripts': ['infomemes=infomemes.simulation:main'],
    }
)
